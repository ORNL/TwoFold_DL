from transformers import BertModel, BertConfig
from transformers import PreTrainedModel, PretrainedConfig

from .modules import PairRepresentation, CrossPairRepresentation
from .utils import get_extended_attention_mask
from .structure import Structure, CrossStructure, compute_weighted_FAPE, IPAConfig
from .structure import compute_kabsch_RMSD, compute_weighted_RMSD, compute_residue_dist, compute_residue_CN_dist, compute_residue_CNC_angle
from .structure import computeAllAtomCoordinates
from .residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_order_with_x
)

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

import pickle, numpy as np # for dumping items

class ProteinLigandConfig(PretrainedConfig):
    model_type = 'bert' # this is required for tokenizer selection

    def __init__(
        self,
        seq_config=BertConfig(),
        smiles_config=BertConfig(),
        n_cross_attention=3,
        linear_mem_attn=True,
        query_chunk_size_seq=512,
        key_chunk_size_seq=1024,
        query_chunk_size_smiles=512,
        key_chunk_size_smiles=1024,
        **kwargs
    ):

        self.smiles_config = smiles_config
        if isinstance(smiles_config, BertConfig):
            self.smiles_config = self.smiles_config.to_dict()

        self.seq_config = seq_config
        if isinstance(seq_config, BertConfig):
            self.seq_config = self.seq_config.to_dict()

        self.n_cross_attention = n_cross_attention

        # to estimate memory usage with deepspeed ZERO stage3, the larger of the two hidden dimensions
        self.hidden_size = self.seq_config['hidden_size']

        self.linear_mem_attn = linear_mem_attn

        self.query_chunk_size_seq = query_chunk_size_seq
        self.key_chunk_size_seq = key_chunk_size_seq

        self.query_chunk_size_smiles = query_chunk_size_smiles
        self.key_chunk_size_smiles = key_chunk_size_smiles

        super().__init__(**kwargs)

class ProteinLigandConfigStructure(ProteinLigandConfig):
    def __init__(
        self,
        seq_ipa_config = IPAConfig(),
        smiles_ipa_config = IPAConfig(),
        num_ipa_layers=8,
        num_rigid_groups=8,
        width_resnet=128,
        depth_resnet=5,
        num_embeddings=30, # >= number of amino acids
        num_atoms=14, # max number of heavy atoms in a residue
        enable_cross=True,
        seq_vocab=None,
        **kwargs
    ):

        self.seq_ipa_config = seq_ipa_config
        if isinstance(seq_ipa_config, IPAConfig):
            self.seq_ipa_config = self.seq_ipa_config.to_dict()

        self.smiles_ipa_config = smiles_ipa_config
        if isinstance(smiles_ipa_config, IPAConfig):
            self.smiles_ipa_config = self.smiles_ipa_config.to_dict()

        self.num_ipa_layers = num_ipa_layers
        self.num_rigid_groups = num_rigid_groups
        self.width_resnet = width_resnet
        self.depth_resnet = depth_resnet
        self.num_embeddings = num_embeddings
        self.num_atoms = num_atoms

        self.enable_cross = enable_cross
        self.seq_vocab = seq_vocab

        super().__init__(**kwargs)

class StructurePrediction(PreTrainedModel):
    config_class = ProteinLigandConfigStructure
    supports_gradient_checkpointing = True
    main_input_name = "input_ids_1" # estimate FLOPs from the protein sequence, which typically has more tokens

    def __init__(self, config):
        super().__init__(config)    

        self.pair_representation = PairRepresentation(config)
        self.structure = Structure(config)

        if config.enable_cross:
            self.cross_pair_representation = CrossPairRepresentation(config)
            self.cross_structure = CrossStructure(config)

        self.gradient_checkpointing = False
        self.enable_cross = config.enable_cross

        self.default_frame = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None
        self.input_ids_to_aatype = None

    def forward(
            self,
            input_ids_1=None,
            inputs_embeds_1=None,
            attention_mask_1=None,
            input_ids_2=None,
            inputs_embeds_2=None,
            attention_mask_2=None,
            labels_receptor_frames_xyz=None,
            labels_receptor_frames_rot=None,
            labels_receptor_xyz=None,
            labels_ligand_frames_xyz=None,
            labels_ligand_frames_rot=None,
            labels_ligand_xyz=None,
            labels_ligand_token_mask=None,
            labels_receptor_token_mask=None,
            return_coordinates=True,
            return_dict=False,
            **kwargs,
    ):
        pair_representation_output = self.pair_representation(
            input_ids_1=input_ids_1,
            inputs_embeds_1=inputs_embeds_1,
            attention_mask_1=attention_mask_1,
            input_ids_2=input_ids_2,
            inputs_embeds_2=inputs_embeds_2,
            attention_mask_2=attention_mask_2,
        )

        pair_representation_seq, pair_representation_smiles = pair_representation_output[:2]
        hidden_seq, hidden_smiles = pair_representation_output[2:4]

        hidden_smiles, hidden_seq, xyz_ligand, rot_ligand, xyz_receptor, rot_receptor, rotation_angles = self.structure(
                hidden_seq,
                hidden_smiles,
                attention_mask_1,
                attention_mask_2,
                pair_representation_seq,
                pair_representation_smiles,
            )

        if labels_receptor_frames_xyz is not None or labels_ligand_xyz is not None:
            if labels_ligand_xyz is None or labels_receptor_frames_xyz is None or labels_ligand_token_mask is None or labels_receptor_token_mask is None:
                raise ValueError("Need both ligand and receptor coordinates.")

            # mask non-atom coordinates
            mask_receptor = attention_mask_1*labels_receptor_token_mask
            mask_ligand = attention_mask_2*labels_ligand_token_mask

            # auxiliary loss on Calpha + ligand
            loss_receptor = compute_kabsch_RMSD(labels_receptor_frames_xyz, xyz_receptor, mask_receptor, dclamp=None)
            loss_ligand = compute_kabsch_RMSD(labels_ligand_frames_xyz, xyz_ligand, mask_ligand, dclamp=None)
            #loss_receptor_dist = compute_residue_dist(xyz_receptor, mask_receptor, dclamp=None)
            #print(f'loss_receptor_dist 1: {loss_receptor_dist:.3f}')
            #loss = 0.5*(loss_receptor+loss_ligand) + loss_receptor_dist * 0.5
            #loss = (loss_receptor + loss_ligand + loss_receptor_dist) / 3
            loss = (loss_receptor + loss_ligand) / 2

        if self.enable_cross:
            pair_representation_output = self.cross_pair_representation(
                hidden_states_1=hidden_seq,
                hidden_states_2=hidden_smiles,
                attention_mask_1=attention_mask_1,
                attention_mask_2=attention_mask_2,
            )
            pair_representation_cross, hidden_seq, hidden_smiles = pair_representation_output

            hidden_smiles, hidden_seq, xyz_ligand, rot_ligand, xyz_receptor, rot_receptor, rotation_angles = self.cross_structure(
                hidden_seq,
                hidden_smiles,
                attention_mask_1,
                attention_mask_2,
                rot_receptor,
                xyz_receptor,
                rot_ligand,
                xyz_ligand,
                pair_representation_cross,
                rotation_angles,
            )

            # generate sidechain coordinates
            #rotation_angles[:] = 1
            #print(rotation_angles[0, 1:11])
            receptor_feat = self.computeAllAtomCoordinates(input_ids_1,
                xyz_receptor,
                rot_receptor,
                rotation_angles,
            )

            #print(receptor_feat[0, 2])
        

        outputs = dict()
        if return_coordinates:
            outputs['ligand_frames_xyz'] = xyz_ligand
            outputs['ligand_frames_rot'] = rot_ligand
            outputs['receptor_frames_xyz'] = xyz_receptor
            outputs['receptor_frames_rot'] = rot_receptor

            if self.enable_cross:
                outputs['receptor_xyz'] = receptor_feat

        if not return_dict:
            outputs = tuple(outputs.values())

        if len(outputs) == 1:
            outputs = outputs[0]

        if labels_receptor_frames_xyz is not None or labels_ligand_xyz is not None:
            if labels_ligand_xyz is None or labels_receptor_frames_xyz is None or labels_ligand_token_mask is None or labels_receptor_token_mask is None:
                raise ValueError("Need both ligand and receptor coordinates.")

            if self.enable_cross:
                # ligand frames with a single atom
                ligand_feat = xyz_ligand.unsqueeze(2)
                ligand_feat = torch.cat([ligand_feat,
                    torch.zeros(*(ligand_feat.shape[:2] + (self.config.num_atoms-1,) + ligand_feat.shape[3:]),
                                device=ligand_feat.device, dtype=ligand_feat.dtype)], 2)

                weight = torch.cat([mask_receptor,mask_ligand],1)
                labels_feat = torch.cat([labels_receptor_xyz, labels_ligand_xyz], 1)
                feat = torch.cat([receptor_feat, ligand_feat], 1)

                use_fape = False
                if use_fape:
                    labels_frames_xyz = torch.cat([labels_receptor_frames_xyz, labels_ligand_frames_xyz], 1)
                    labels_frames_rot = torch.cat([labels_receptor_frames_rot, labels_ligand_frames_rot], 1)
                    frames_xyz = torch.cat([xyz_receptor, xyz_ligand], 1)
                    frames_rot = torch.cat([rot_receptor, rot_ligand], 1)
                    loss = (2*loss + compute_weighted_FAPE(labels_feat, labels_frames_xyz, labels_frames_rot, feat, frames_xyz, frames_rot, weight))/3
                else:
                    # flatten atom coordinates
                    non_nan = (~torch.any(torch.isnan(labels_feat),dim=-1)).type(torch.int64)
                    weight = weight[:,:,None]*non_nan

                    # normalize so that both molecules are weighted equally
                    seq_len = mask_receptor.shape[1]
                    norm_seq = torch.sum(weight[:,:seq_len], [-1,-2], keepdim=True)
                    norm_smiles = torch.sum(weight[:,seq_len:], [-1,-2], keepdim=True)
                    weight_seq = weight[:,:seq_len].type(feat.dtype)
                    weight_smiles = weight[:,seq_len:].type(feat.dtype)
                    weight_seq = torch.where(norm_seq > 0, weight_seq/norm_seq, weight_seq)
                    weight_smiles = torch.where(norm_smiles > 0, weight_smiles/norm_smiles, weight_smiles)
                    weight = torch.cat([weight_seq, weight_smiles], 1)

                    labels_feat = torch.nan_to_num(labels_feat)
                    labels_feat = labels_feat.reshape(*labels_feat.shape[:1], -1, *labels_feat.shape[-1:])
                    feat = feat.reshape(*feat.shape[:1], -1, *feat.shape[-1:])
                    weight = weight.reshape(*weight.shape[:1], -1)
                    #weight_smiles = weight_smiles.reshape(*weight_smiles.shape[:1], -1)
                    #weight_seq = weight_seq.reshape(*weight_seq.shape[:1], -1)

                    loss_kabsch = compute_kabsch_RMSD(labels_feat, feat, weight, dclamp=None)
                    loss_receptor_CN_dist = compute_residue_CN_dist(receptor_feat, mask_receptor, dclamp=None, losstype='bottom')
                    loss_receptor_CNC_angle = compute_residue_CNC_angle(receptor_feat, mask_receptor)
                    #print(f'loss_receptor_dist 2: {loss_receptor_dist:.3f}')
                    loss = (2 * loss + 3 * loss_kabsch + loss_receptor_CN_dist) / 6
                    # 1/6 ligand self, 1/6 protein self, 1/6 CN dist, 1/2 overall kabsch

#                    dumper = {'input_ids_1': input_ids_1.cpu().detach().numpy(),
#                              'receptor_feat': receptor_feat.cpu().detach().numpy(),
#                              'labels_receptor_xyz': labels_receptor_xyz.cpu().detach().numpy(),
#                              'loss': loss.cpu().detach().numpy(),
#                              'loss_this': compute_kabsch_RMSD(labels_feat, feat, weight).cpu().detach().numpy(),
#                              'loss_receptor_CA': compute_kabsch_RMSD(labels_receptor_frames_xyz, xyz_receptor, mask_receptor).cpu().detach().numpy(),
#                              'loss_ligand': compute_kabsch_RMSD(labels_ligand_frames_xyz, xyz_ligand, mask_ligand).cpu().detach().numpy(),}
#                              #'loss_receptor_all': compute_kabsch_RMSD(labels_receptor_xyz, receptor_feat, weight_seq).cpu().detach().numpy()}
#                    pickle.dump(dumper, open(f'random_dump/dump_{np.random.randint(1000)}.pkl', 'wb'))
#                    # Basically take input_id_1, receptor_feat, and loss

            return (loss, outputs)
        else:
            return outputs

    def computeAllAtomCoordinates(self, 
                input_ids_1,
                xyz_receptor,
                rot_receptor,
                rotation_angles,
                ):
        self._init_residue_constants(rotation_angles.dtype, rotation_angles.device)
        aatypes = torch.tensor(self.input_ids_to_aatype[input_ids_1], device=input_ids_1.device)#, requires_grad=False)
        return computeAllAtomCoordinates(
                aatypes,
                xyz_receptor,
                rot_receptor,
                rotation_angles,
                self.default_frame,
                self.group_idx,
                self.atom_mask,
                self.lit_positions)

    def _init_residue_constants(self, float_dtype, device):
        if self.default_frame is None:
            self.default_frame = torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None:    
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None:
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None:
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
    
        if self.input_ids_to_aatype is None:
            input_ids_to_aatype = torch.zeros(len(self.config.seq_vocab.keys()))
            for k in self.config.seq_vocab.keys():
                if k in restype_order_with_x.keys():
                    input_ids_to_aatype[self.config.seq_vocab[k]] = restype_order_with_x[k]
                else:
                    input_ids_to_aatype[self.config.seq_vocab[k]] = 20
                #print(f'{k} ({self.config.seq_vocab[k]}) -> {input_ids_to_aatype[self.config.seq_vocab[k]]}')
    
            self.input_ids_to_aatype = torch.tensor(input_ids_to_aatype, dtype=torch.long, device=device, requires_grad=False) 
            #print(self.input_ids_to_aatype) 

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        self.structure.gradient_checkpointing_enable()
        self.pair_representation.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.structure.gradient_checkpointing_disable()
        self.pair_representation.gradient_checkpointing_disable()

    def freeze_protein(self):
        self.pair_representation.freeze_protein()
        self.structure.freeze_protein()

    def freeze_ligand(self):
        self.pair_representation.freeze_ligand()
        self.structure.freeze_ligand()
