import torch
import torch.nn as nn
import numpy as np
from transformers import PretrainedConfig, BertConfig

from .IPCA import InvariantPointCrossAttention, LinearMemInvariantPointCrossAttention
from .building_blocks import MLP

from torch.utils.checkpoint import checkpoint
from packaging import version
from .residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)



class IPAConfig(PretrainedConfig):
    def __init__(
        self,
        bert_config = BertConfig(),
        num_ipa_heads=12,
        num_points=8,
        num_intermediate_layers=3,
        query_chunk_size=512,
        key_chunk_size=1024,
        **kwargs,
        ):

        self.bert_config = bert_config
        if isinstance(bert_config, BertConfig):
            self.bert_config = self.bert_config.to_dict()

        self.num_ipa_heads = num_ipa_heads
        self.num_points = num_points
        self.num_intermediate_layers = 3

        self.query_chunk_size=query_chunk_size
        self.key_chunk_size=key_chunk_size

        super().__init__(**kwargs)

def ijk_to_R(ijk):
    # make rotation matrix from vector component of unnormalized quaternion
    one = torch.ones(*(ijk.shape[:-1] + (1,)), dtype=ijk.dtype, device=ijk.device)
    rijk = torch.cat([one, ijk], dim=-1)

    r, i, j, k = torch.unbind(rijk, -1)
    two_s = 2.0 / (rijk * rijk).sum(-1)

    # convert to rotation matrix
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )

    return o.reshape(rijk.shape[:-1] + (3, 3))

class FrameTranslation(nn.Module):
    # Similar to Alg 23 currently without the quaternion part
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.bert_config['hidden_size'], 3) # x,y,z

    def forward(self, hidden_states):
        translation = self.linear(hidden_states)

        return translation

class FrameRotation(nn.Module):
    # Similar to Alg 23 currently without the quaternion part
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.bert_config['hidden_size'], 3) # x,y,z

    def forward(self, hidden_states):
        ijk = self.linear(hidden_states)

        return ijk_to_R(ijk)

def compute_weighted_RMSD(T_true, T_pred, weight, unit=10.0, dclamp=10.0, eps = 1e-4, reduction='mean'):
    dist = torch.sqrt(((T_pred - T_true)**2).sum(-1) + eps)
    weight = weight.type(dist.dtype)
    if dclamp is not None:
        dist = torch.clamp(dist, max=dclamp)
    dist = weight*dist

    loss = torch.sum(dist, dim=-1) / weight.sum(dim=-1) / unit

    # mini-batch reduction
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'gmean':
        loss = torch.exp(torch.mean(torch.log(loss)))
    else:
        raise ValueError

    return loss

def compute_weighted_FAPE(x_true, t_true, R_true, x_pred, t_pred, R_pred, weight, unit=10.0, dclamp=10.0, eps = 1e-6, reduction='mean'):
    non_nan = (~torch.any(torch.isnan(x_true),dim=-1)).type(torch.int64)[:,:,None]
    x_true = torch.nan_to_num(x_true)

    outer_true = torch.einsum('bijmk,bjkl->bijml',x_true[:,:,None] - t_true[:,None,:,None,:], R_true)
    outer_pred = torch.einsum('bijmk,bjkl->bijml',x_pred[:,:,None] - t_pred[:,None,:,None,:], R_pred)

    dist = torch.sqrt(torch.sum((outer_pred - outer_true)**2, dim=-1) + eps)

    # ignore non-occupied coordinates in ground truth (they are set to nan)
    dist = torch.where(non_nan==0, torch.zeros_like(dist), dist)
    weight = weight[:,:,None,None]*weight[:,None,:,None]*non_nan

    weight = weight.type(dist.dtype)
    if dclamp is not None:
        dist = torch.clamp(dist, max=dclamp)
    dist = weight*dist

    loss = dist.sum([-3,-2,-1]) / weight.sum([-3,-2,-1]) / unit

    # mini-batch reduction
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'gmean':
        loss = torch.exp(torch.mean(torch.log(loss)))
    else:
        raise ValueError

    return loss

def compute_kabsch_RMSD(T_true, T_pred, weight, unit=10.0, dclamp=10.0, eps = 1e-4, reduction='mean'):
    # T_true, T_pred is (N_batch, N_max_len_of_batch, 3)
    # weight is the mask of size (N_batch, N_max_len_of_batch)

    # First center with mean coordinate of meaningful tokens, which involves scaling up the mean coordinates w.r.t mean of weight.
    # And then we zero the non-meaningful ones - zero coordinates in SVD will not affect the rotation matrix

    weight = weight.type(T_true.dtype)

    T_true_cm = (T_true * weight[:, :, None]).mean(axis=1)[:,None,:] / (weight.mean(1)[:, None, None])
    T_pred_cm = (T_pred * weight[:, :, None]).mean(axis=1)[:,None,:] / (weight.mean(1)[:, None, None])

    T_true_cen = T_true - T_true_cm
    T_pred_cen = T_pred - T_pred_cm

    # Kabsch method
    C = torch.matmul(torch.transpose(T_pred_cen * weight[:, :, None], -2, -1), T_true_cen * weight[:, :, None])
    with torch.autocast('cuda',enabled=False):
        # svd doesn't support fp16/bf16 on AMD
        V, S, W = torch.linalg.svd(C.float())
    V_prime = V * torch.tensor([1, 1, -1], device=V.device, dtype=V.dtype)[None, :, None] # Alternate version of V if d < 0
    d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0

    V = torch.where(d[:, None, None], V_prime, V) # We have to broadcast d to select the right V's
    # Create Rotation matrix U
    U = torch.matmul(V, W)
    T_pred_align = torch.matmul(T_pred_cen, U) # Rotate P
    diff = T_pred_align - T_true_cen
    dist = torch.sqrt((diff**2).sum(-1) + eps)     # (N_batch, N_max_len_of_batch)

    if dclamp is not None:
        dist = torch.clamp(dist, max=dclamp)
    dist = weight*dist

    loss = torch.sum(dist, dim=-1) / weight.sum(dim=-1) / unit

    # mini-batch reduction
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'gmean':
        loss = torch.exp(torch.mean(torch.log(loss)))
    else:
        raise ValueError

    return loss

def compute_residue_dist(T_pred, weight, unit=10.0, dclamp=10.0, eps = 1e-4, reduction='mean'):
    # T_pred is predicted xyz of frames(N_batch, N_max_len_of_batch, 3)
    # weight is the mask of size (N_batch, N_max_len_of_batch)

    ca_ca = 3.80209737096 # This is a constant, taken from OpenFold

    weight = weight.type(T_pred.dtype)
    weight = weight[:, 1:] * weight[:, :-1]

    diff = torch.square(torch.sqrt(((T_pred[:, 1:, :] - T_pred[:, :-1, :])**2).sum(-1)) - ca_ca)

    if dclamp is not None:
        dist = torch.clamp(dist, max=dclamp)
    dist = diff * weight

    loss = torch.sum(dist, dim=-1) / weight.sum(dim=-1) / unit

    # mini-batch reduction
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'gmean':
        loss = torch.exp(torch.mean(torch.log(loss)))
    else:
        raise ValueError

    return loss

def compute_residue_CN_dist(T_pred, weight, unit=10.0, dclamp=10.0, eps = 1e-4, reduction='mean', losstype='bottom'):
    # T_pred is predicted features (atom coordinates) of shape (N_batch, N_max_len_of_batch, 14, 3)
    # weight is the mask of size (N_batch, N_max_len_of_batch)
    # Loss type 'bottom' is copied from AlphaFold 2, where loss is flat bottom L1 loss max(|T - c_n - 12 * c_n_s|, 0)
    # Loss type 'square': loss is L2 loss square((T - c_n))

    c_n = 1.3296 # This is a constant, taken from OpenFold (19/20 of 1.329 and 1/20 of 1.341 [for proline]
    c_n_s = 0.0141 # Weighted average of the sigma of bond length

    weight = weight.type(T_pred.dtype)
    weight = weight[:, 1:] * weight[:, :-1]

    # Calculate dist between C of res N and N of res N+1
    if losstype == 'square':
        diff = torch.square(torch.sqrt(((T_pred[:, 1:, 0, :] - T_pred[:, :-1, 2, :])**2).sum(-1) + eps) - c_n)
    elif losstype == 'bottom':
        diff = torch.maximum(torch.abs(torch.sqrt(((T_pred[:, 1:, 0, :] - T_pred[:, :-1, 2, :])**2).sum(-1) + eps) - c_n) - 12 * c_n_s,
                             torch.zeros_like(weight))
    
    if dclamp is not None:
        dist = torch.clamp(dist, max=dclamp)
    dist = diff * weight

    loss = torch.sum(dist, dim=-1) / weight.sum(dim=-1) / unit
       
    # mini-batch reduction
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'gmean':
        loss = torch.exp(torch.mean(torch.log(loss)))
    else:
        raise ValueError

    return loss

def compute_residue_CNC_angle(T_pred, weight, eps = 1e-4, reduction='mean'):
    # T_pred is predicted features (atom coordinates) of shape (N_batch, N_max_len_of_batch, 14, 3)
    # weight is the mask of size (N_batch, N_max_len_of_batch)
    # Loss type 'bottom' is copied from AlphaFold 2, where loss is flat bottom L1 loss max(|T - c_n - 12 * c_n_s|, 0)
    # Loss type 'square': loss is L2 loss square((T - c_n))

    cos_c_n_ca = -0.51665 # This is a constant, taken from OpenFold (19/20 of -0.5203 and 1/20 of -0.0353 [for proline])
    cos_c_n_ca_s = 0.03509 # Weighted average of the sigma of bond angle

    weight = weight.type(T_pred.dtype)
    weight = weight[:, 1:] * weight[:, :-1]

    # Calculate dist between C of res N, N of res N+1, and CA of res N+1
    diff = torch.maximum(torch.abs(angle_3point(T_pred[:, :-1, 2, :], T_pred[:, 1:, 0, :], T_pred[:, 1:, 1, :])  - cos_c_n_ca) - 12 * cos_c_n_ca_s,
                         torch.zeros_like(weight))
    
    dist = diff * weight

    loss = torch.sum(dist, dim=-1) / weight.sum(dim=-1) # this in radian
       
    # mini-batch reduction
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'gmean':
        loss = torch.exp(torch.mean(torch.log(loss)))
    else:
        raise ValueError

    return loss

class PlacementIteration(nn.Module):
    # Similar to AF2 FoldIteration (Alg 20, lines 6 - 10),
    # only that we're placing the ligand with a rigid protein frame
    def __init__(self, config, other_config=None, aniso=True, linear_mem=False, **kwargs):
        super().__init__()

        self.is_cross_attention = other_config is not None

        if not self.is_cross_attention:
            # self attention
            if linear_mem:
                self.ipa = LinearMemInvariantPointCrossAttention(config, config, **kwargs, is_cross_attention=False)
            else:
                self.ipa = InvariantPointCrossAttention(config, config, **kwargs, is_cross_attention=False)

            self.dropout_self = nn.Dropout(config.bert_config['hidden_dropout_prob'])
            self.norm_self = nn.LayerNorm(config.bert_config['hidden_size'])
        else:
            # cross attention
            if linear_mem:
                self.ipca = LinearMemInvariantPointCrossAttention(config, other_config, **kwargs)
            else:
                self.ipca = InvariantPointCrossAttention(config, other_config, **kwargs)

            self.dropout_cross = nn.Dropout(config.bert_config['hidden_dropout_prob'])
            self.norm_cross = nn.LayerNorm(config.bert_config['hidden_size'])

        self.intermediate = MLP(config.bert_config['hidden_size'], config.num_intermediate_layers)
        self.norm_intermediate = nn.LayerNorm(config.bert_config['hidden_size'])
        self.dropout_intermediate = nn.Dropout(config.bert_config['hidden_dropout_prob'])

        self.frame_translation = FrameTranslation(config)
        self.frame_rotation = FrameRotation(config)

        self.aniso = aniso

    def forward(
        self,
        hidden_states,
        attention_mask,
        pair_representation,
        rigid_rotations,
        rigid_translations,
        query_chunk_size = 1024,
        key_chunk_size = 4096,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_rigid_rotations=None,
        encoder_rigid_translations=None,
    ):
        if not self.is_cross_attention:
            ipa_output = self.ipa(hidden_states,
                  attention_mask=attention_mask,
                  pair_representation=pair_representation,
                  rigid_rotations=rigid_rotations,
                  rigid_translations=rigid_translations,
                  query_chunk_size=query_chunk_size,
                  key_chunk_size=key_chunk_size)

            hidden_states = self.norm_self(hidden_states + self.dropout_self(ipa_output))
        else:
            ipca_output = self.ipca(hidden_states,
                  encoder_hidden_states=encoder_hidden_states,
                  encoder_attention_mask=encoder_attention_mask,
                  pair_representation=pair_representation,
                  rigid_rotations=rigid_rotations,
                  rigid_translations=rigid_translations,
                  encoder_rigid_rotations=encoder_rigid_rotations,
                  encoder_rigid_translations=encoder_rigid_translations,
                  query_chunk_size=query_chunk_size,
                  key_chunk_size=key_chunk_size)

            hidden_states = self.norm_cross(hidden_states + self.dropout_cross(ipca_output))

        # transition
        hidden_states = hidden_states + self.dropout_intermediate(self.intermediate(hidden_states))
        hidden_states = self.norm_intermediate(hidden_states)

        # update spatial transforms
        if self.aniso:
            R = self.frame_rotation(hidden_states)
        else:
            # enforce rotation=const.
            R = self.frame_rotation(torch.mean(hidden_states,dim=1,keepdim=True))

            if version.parse(torch.__version__) < version.parse('1.8.0'):
                # WAR for missing broadcast in einsum
                R = R.repeat(1,hidden_states.shape[1],1,1)

        rigid_translations = torch.einsum('bnij,bnj->bni', R, rigid_translations)
        rigid_rotations = torch.einsum('bnij,bnjk->bnik', R, rigid_rotations)

        rigid_translations = rigid_translations + self.frame_translation(hidden_states)

        return hidden_states, rigid_translations, rigid_rotations

class SidechainRotation(nn.Module):
    # ResNet (Algorithm 20, lines 11-14)
    def __init__(self, config):
        super().__init__()

        self.initial_linear = nn.Linear(config.seq_config['hidden_size'], config.width_resnet)
        self.initial_relu = nn.ReLU()
        self.linear = nn.ModuleList([nn.Linear(config.width_resnet, config.width_resnet) for _ in range(config.depth_resnet-1)])
        self.relu = nn.ModuleList([nn.ReLU() for _ in range(config.depth_resnet)])
        self.final_linear = nn.Linear(config.width_resnet, config.num_rigid_groups-1)

    def forward(self,
                hidden_states,
                rotation_angles):
        act = self.initial_relu(self.initial_linear(hidden_states))
        for linear, relu in zip(self.linear, self.relu):
            act = act + relu(linear(act))
        rotation_angles = rotation_angles + self.final_linear(act)

        return rotation_angles

def make_rot_X(alpha):
    # convert to rotation matrix
    o = torch.stack(
        (
            torch.ones_like(alpha), torch.zeros_like(alpha), torch.zeros_like(alpha),
            torch.zeros_like(alpha), torch.cos(alpha), -torch.sin(alpha),
            torch.zeros_like(alpha), torch.sin(alpha), torch.cos(alpha),
        ),
        -1,
    )
    return o.reshape(alpha.shape + (3, 3))

def angle_3point(T1, T2, T3, eps = 1e-4):
    # T1 is C, T2 is N, and T3 is CA
    # Each is (N_batch, N_max_len_of_batch-1, 1, 3)
    NtoC = T1 - T2
    CAtoN = T3 - T2
    angles = torch.einsum('bni, bnj -> bn', NtoC, CAtoN) / ((NtoC**2).sum(-1) + eps) / ((CAtoN**2).sum(-1) + eps)
    return angles # This in (N_batch, N_max_len_of_batch-1)

def computeAllAtomCoordinates(
    # This is merging the torsion_angles_to_frames and 
    # frames_and_literature_positions_to_atom14_pos functions in OpenFold.

    input_ids, # a form of aatypes, [*, N]
    frame_xyz, # [*, N, 3]
    frame_rot, # [*, N, 3, 3]
    rotation_angles, # [*, N, 7]
    default_frames, # intra-residue frames, from library, 4x4 matrices [21, 8, 4, 4]
    group_idx, # [21, 14]
    atom_mask, # [21, 14]
    lit_positions # relative locations based on intra-residue frames, [21, 14, 3]
    ):


    # Algorithm 24, but we use polar coordinates [0, 2*pi]
    # TODO maybe add a term for polar coordinates beyond this range

    default_4x4 = default_frames[input_ids, ...] # [*, N, 8, 4, 4]
    default_r = default_4x4[..., :3, :3]         # [*, N, 8, 3, 3]
    default_t = default_4x4[..., :3, 3]          # [*, N, 8, 3] 
    # in OpenFold default_r is a Rigid class which includes both the default_r and default_t here

    # [1, 1, 1], zeros is correct as we want bb_rot to be zero radian
    bb_rot = rotation_angles.new_zeros((((1,) * len(rotation_angles.shape)))) 

    # [*, N, 8]
    rotation_angles = torch.cat([bb_rot.expand(*rotation_angles.shape[:-1], -1), rotation_angles], dim=-1)

    #print(rotation_angles)

    # TODO Generate local rotation matrix through make_rot_X and alpha
    all_rots = make_rot_X(rotation_angles) # [*, N, 8, 3, 3]

    #print(all_rots)

    # TODO Rotate default rotation with local rotation matrix
    all_frames_r = torch.einsum('bnaij, bnajk -> bnaik', default_r, all_rots)
    all_frames_t = default_t
    #all_frames_r = torch.einsum('bnaij, bnajk -> bnaik', all_rots, default_r)
    #all_frames_t = torch.einsum('bnaij, bnaj -> bnai', all_rots, default_t)

    #print(all_frames_r[:, 1:-1], all_frames_t[:, 1:-1])

    #all_frames_to_bb_r = all_frames_r
    #all_frames_to_bb_t = all_frames_t

    # TODO Calculate all frames to back bone
    # change frame-to-frame to frame-to-backbone starting from chi2, kind of tedious but will do
    chi2_frame_to_frame_r = all_frames_r[:, :, 5]
    chi2_frame_to_frame_t = all_frames_t[:, :, 5]
    chi3_frame_to_frame_r = all_frames_r[:, :, 6]
    chi3_frame_to_frame_t = all_frames_t[:, :, 6]
    chi4_frame_to_frame_r = all_frames_r[:, :, 7]
    chi4_frame_to_frame_t = all_frames_t[:, :, 7]

    chi1_frame_to_bb_r = all_frames_r[:, :, 4] # Gives [*, N, 3, 3]
    chi1_frame_to_bb_t = all_frames_t[:, :, 4] # Gives [*, N, 3]

    #chi2_frame_to_bb_r = torch.einsum('bnij, bnjk -> bnik', chi1_frame_to_bb_r, chi2_frame_to_frame_r)
    #chi2_frame_to_bb_t = torch.einsum('bnj, bnij -> bni',   chi1_frame_to_bb_t, chi2_frame_to_frame_r) + chi2_frame_to_frame_t
    #chi3_frame_to_bb_r = torch.einsum('bnij, bnjk -> bnik', chi2_frame_to_bb_r, chi3_frame_to_frame_r)
    #chi3_frame_to_bb_t = torch.einsum('bnj, bnij -> bni',   chi2_frame_to_bb_t, chi3_frame_to_frame_r) + chi3_frame_to_frame_t
    #chi4_frame_to_bb_r = torch.einsum('bnij, bnjk -> bnik', chi3_frame_to_bb_r, chi4_frame_to_frame_r)
    #chi4_frame_to_bb_t = torch.einsum('bnj, bnij -> bni',   chi3_frame_to_bb_t, chi4_frame_to_frame_r) + chi4_frame_to_frame_t
    #chi2_frame_to_bb_r = torch.einsum('bnij, bnjk -> bnik', chi2_frame_to_frame_r, chi1_frame_to_bb_r)
    #chi2_frame_to_bb_t = torch.einsum('bnij, bnj -> bni',   chi2_frame_to_frame_r, chi1_frame_to_bb_t) + chi2_frame_to_frame_t
    #chi3_frame_to_bb_r = torch.einsum('bnij, bnjk -> bnik', chi3_frame_to_frame_r, chi2_frame_to_bb_r)
    #chi3_frame_to_bb_t = torch.einsum('bnij, bnj -> bni',   chi3_frame_to_frame_r, chi2_frame_to_bb_t) + chi3_frame_to_frame_t
    #chi4_frame_to_bb_r = torch.einsum('bnij, bnjk -> bnik', chi4_frame_to_frame_r, chi3_frame_to_bb_r)
    #chi4_frame_to_bb_t = torch.einsum('bnij, bnj -> bni',   chi4_frame_to_frame_r, chi3_frame_to_bb_t) + chi4_frame_to_frame_t
    chi2_frame_to_bb_r = torch.einsum('bnij, bnjk -> bnik', chi1_frame_to_bb_r, chi2_frame_to_frame_r)
    chi2_frame_to_bb_t = torch.einsum('bnij, bnj -> bni',   chi1_frame_to_bb_r, chi2_frame_to_frame_t) + chi1_frame_to_bb_t
    chi3_frame_to_bb_r = torch.einsum('bnij, bnjk -> bnik', chi2_frame_to_bb_r, chi3_frame_to_frame_r)
    chi3_frame_to_bb_t = torch.einsum('bnij, bnj -> bni',   chi2_frame_to_bb_r, chi3_frame_to_frame_t) + chi2_frame_to_bb_t
    chi4_frame_to_bb_r = torch.einsum('bnij, bnjk -> bnik', chi3_frame_to_bb_r, chi4_frame_to_frame_r)
    chi4_frame_to_bb_t = torch.einsum('bnij, bnj -> bni',   chi3_frame_to_bb_r, chi4_frame_to_frame_t) + chi3_frame_to_bb_t


    all_frames_to_bb_r = torch.cat(
        [
            all_frames_r[:, :, :5],
            chi2_frame_to_bb_r.unsqueeze(2),
            chi3_frame_to_bb_r.unsqueeze(2),
            chi4_frame_to_bb_r.unsqueeze(2),
        ],
        dim=-3
    )
    all_frames_to_bb_t = torch.cat(
        [
            all_frames_t[:, :, :5],
            chi2_frame_to_bb_t.unsqueeze(2),
            chi3_frame_to_bb_t.unsqueeze(2),
            chi4_frame_to_bb_t.unsqueeze(2),
        ],
        dim=-2
    )

    # TODO Calculate all frames to global
    all_frames_to_global_r = torch.einsum('bnij, bnajk -> bnaik', frame_rot, all_frames_to_bb_r)
    all_frames_to_global_t = torch.einsum('bnij, bnaj -> bnai', frame_rot, all_frames_to_bb_t) + frame_xyz[:,:,None,:]
    #all_frames_to_global_r = torch.einsum('bnaij, bnjk -> bnaik', all_frames_to_bb_r, frame_rot)
    #all_frames_to_global_t = torch.einsum('bnaij, bnj -> bnai',   all_frames_to_bb_r, frame_xyz) + all_frames_to_bb_t

    # TODO Get masks and calculate all side chain atom locations 
    # (this is a part of frames_and_literature_positions_to_atom14_pos)

    # [*, N, 14]
    group_mask = group_idx[input_ids, ...]

    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3]
    )

    #print(all_frames_to_bb_r[:, 1:-1])
    #print(all_frames_to_bb_t[:, 1:-1])
    #print(group_mask[:, 1:-1])

    #print(input_ids)

    # [*, N, 14, 8, 3, 3]
    t_atoms_to_global_r = all_frames_to_global_r[..., None, :, :, :] * group_mask[..., None, None]
    #t_atoms_to_global_r = all_frames_to_bb_r[..., None, :, :, :] * group_mask[..., None, None]
    # [*, N, 14, 8, 3]
    t_atoms_to_global_t = all_frames_to_global_t[..., None, :, :] * group_mask[..., None]
    #t_atoms_to_global_t = all_frames_to_bb_t[..., None, :, :] * group_mask[..., None]

    # [*, N, 14]
    atom_mask = atom_mask[input_ids, ...]#.unsqueeze(-1) 
 
    # [*, N, 14, 3]
    lit_positions = lit_positions[input_ids, ...]
    # [*, N, 14, 8, 3]
    #pred_positions = torch.einsum('bnaj, bnakij -> bnaki', lit_positions, t_atoms_to_global_r) + t_atoms_to_global_t
    pred_positions = torch.einsum('bnakij, bnaj -> bnaki', t_atoms_to_global_r, lit_positions) + t_atoms_to_global_t
    pred_positions = pred_positions * atom_mask[..., None, None]
    # [*, N, 14, 3]
    atom_feat = pred_positions.sum(-2)
    #print(atom_feat.shape)

    return atom_feat

class Structure(nn.Module):
    # Alg 20
    def __init__(self, config):
        super().__init__()


        self.num_layers = config.num_ipa_layers
        self.num_rigid_groups = config.num_rigid_groups

        seq_config = IPAConfig.from_dict(config.seq_ipa_config)
        smiles_config = IPAConfig.from_dict(config.smiles_ipa_config)

        self.initial_norm_seq = nn.LayerNorm(seq_config.bert_config['hidden_size'])
        self.initial_norm_smiles = nn.LayerNorm(smiles_config.bert_config['hidden_size'])

        self.receptor_self = torch.nn.ModuleList([PlacementIteration(seq_config, aniso=True, linear_mem=config.linear_mem_attn)
            for _ in range(self.num_layers)])
        self.ligand_self = torch.nn.ModuleList([PlacementIteration(smiles_config, aniso=False, linear_mem=config.linear_mem_attn)
            for _ in range(self.num_layers)])

        self.sidechain_self = torch.nn.ModuleList([SidechainRotation(config) for _ in range(self.num_layers)])

        self.gradient_checkpointing = False

        # for linear mem point attention
        self.query_chunk_size_receptor = seq_config.query_chunk_size
        self.key_chunk_size_receptor = seq_config.key_chunk_size

        self.query_chunk_size_ligand = smiles_config.query_chunk_size
        self.key_chunk_size_ligand = smiles_config.key_chunk_size

    def freeze_protein(self):
        for param in self.initial_norm_seq.parameters():
            param.requires_grad = False
        for param in self.receptor_self.parameters():
            param.requires_grad = False
        for param in self.sidechain_rotation.parameters():
            param.requires_grad = False

    def freeze_ligand(self):
        for param in self.initial_norm_smiles.parameters():
            param.requires_grad = False
        for param in self.ligand_self.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states_1,
        hidden_states_2,
        attention_mask_1,
        attention_mask_2,
        pair_representation_seq,
        pair_representation_smiles
    ):

        hidden_seq = self.initial_norm_seq(hidden_states_1)
        hidden_smiles = self.initial_norm_smiles(hidden_states_2)

        # "black-hole" initialization
        translations_receptor = torch.zeros(hidden_seq.size()[:2]+(3, ),
            device=hidden_seq.device, dtype=hidden_seq.dtype)
        rotations_receptor = torch.eye(3,
            device=hidden_seq.device,
            dtype=hidden_seq.dtype).repeat(hidden_seq.size()[0], hidden_seq.size()[1], 1, 1)

        translations_ligand = torch.zeros(hidden_smiles.size()[:2]+(3, ),
            device=hidden_smiles.device,dtype=hidden_smiles.dtype)
        rotations_ligand = torch.eye(3,
            device=hidden_smiles.device,
            dtype=hidden_smiles.dtype).repeat(hidden_smiles.size()[0], hidden_smiles.size()[1], 1, 1)

        # side chain rotations
        rotation_angles = torch.zeros(hidden_seq.size()[:2]+(self.num_rigid_groups-1,), device=hidden_seq.device, dtype=hidden_seq.dtype)

        # self interactions
        for (ligand_update, receptor_update, sidechain_rotation) in zip(self.ligand_self, self.receptor_self, self.sidechain_self):
            # receptor
            if self.gradient_checkpointing:
                hidden_seq, translations_receptor, rotations_receptor = checkpoint(receptor_update,
                    hidden_seq,
                    attention_mask_1,
                    pair_representation_seq,
                    rotations_receptor,
                    translations_receptor,
                    self.query_chunk_size_receptor,
                    self.key_chunk_size_receptor,
                )
            else:
                hidden_seq, translations_receptor, rotations_receptor = receptor_update(
                    hidden_states=hidden_seq,
                    attention_mask=attention_mask_1,
                    pair_representation=pair_representation_seq,
                    rigid_rotations=rotations_receptor,
                    rigid_translations=translations_receptor,
                    query_chunk_size=self.query_chunk_size_receptor,
                    key_chunk_size=self.key_chunk_size_receptor,
                )

            # update internal coordinates for protein
            rotation_angles = sidechain_rotation(hidden_seq, rotation_angles)
            # ligand
            if self.gradient_checkpointing:
               hidden_smiles, translations_ligand, rotations_ligand = checkpoint(ligand_update,
                    hidden_smiles,
                    attention_mask_2,
                    pair_representation_smiles,
                    rotations_ligand,
                    translations_ligand,
                    self.query_chunk_size_ligand,
                    self.key_chunk_size_ligand,
                )
            else:
                hidden_smiles, translations_ligand, rotations_ligand = ligand_update(
                    hidden_states=hidden_smiles,
                    attention_mask=attention_mask_2,
                    pair_representation=pair_representation_smiles,
                    rigid_rotations=rotations_ligand,
                    rigid_translations=translations_ligand,
                    query_chunk_size=self.query_chunk_size_ligand,
                    key_chunk_size=self.key_chunk_size_ligand,
                )

        return hidden_smiles, hidden_seq, translations_ligand, rotations_ligand, translations_receptor, rotations_receptor, rotation_angles

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

class CrossStructure(nn.Module):
    # Alg 20
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_ipa_layers
        self.num_rigid_groups = config.num_rigid_groups

        seq_config = IPAConfig.from_dict(config.seq_ipa_config)
        smiles_config = IPAConfig.from_dict(config.smiles_ipa_config)

        self.initial_norm_seq = nn.LayerNorm(seq_config.bert_config['hidden_size'])
        self.initial_norm_smiles = nn.LayerNorm(smiles_config.bert_config['hidden_size'])

        self.receptor_cross = torch.nn.ModuleList([PlacementIteration(seq_config, smiles_config, aniso=True, linear_mem=config.linear_mem_attn)
            for _ in range(self.num_layers)])
        self.ligand_cross = torch.nn.ModuleList([PlacementIteration(smiles_config, seq_config, aniso=False, linear_mem=config.linear_mem_attn)
            for _ in range(self.num_layers)])

        self.sidechain_cross = torch.nn.ModuleList([SidechainRotation(config) for _ in range(self.num_layers)])

        self.gradient_checkpointing = False

        # for linear mem point attention
        self.query_chunk_size_receptor = seq_config.query_chunk_size
        self.key_chunk_size_receptor = seq_config.key_chunk_size

        self.query_chunk_size_ligand = smiles_config.query_chunk_size
        self.key_chunk_size_ligand = smiles_config.key_chunk_size

    def forward(
        self,
        hidden_states_1,
        hidden_states_2,
        attention_mask_1,
        attention_mask_2,
        rotations_receptor,
        translations_receptor,
        rotations_ligand,
        translations_ligand,
        pair_representation_cross,
        rotation_angles,
    ):

        hidden_seq = self.initial_norm_seq(hidden_states_1)
        hidden_smiles = self.initial_norm_smiles(hidden_states_2)

        def cross(update_1,
            update_2,
            hidden_states_1,
            hidden_states_2,
            attention_mask_1,
            attention_mask_2,
            pair_representation_cross,
            rigid_rotations_1,
            rigid_translations_1,
            rigid_rotations_2,
            rigid_translations_2,
            query_chunk_size_1,
            key_chunk_size_1,
            query_chunk_size_2,
            key_chunk_size_2,
            sidechain_rotation,
            rotation_angles):
            output_1 = update_1(
                hidden_states=hidden_states_1,
                attention_mask=attention_mask_1,
                pair_representation=pair_representation_cross,
                rigid_rotations=rigid_rotations_1,
                rigid_translations=rigid_translations_1,
                query_chunk_size=query_chunk_size_1,
                key_chunk_size=key_chunk_size_1,
                encoder_hidden_states=hidden_states_2,
                encoder_attention_mask=attention_mask_2,
                encoder_rigid_rotations=rigid_rotations_2,
                encoder_rigid_translations=rigid_translations_2,
            )

            # update internal coordinates for protein
            rotation_angles = sidechain_rotation(output_1[0], rotation_angles)

            output_2 = update_2(
                hidden_states=hidden_states_2,
                attention_mask=attention_mask_2,
                pair_representation=pair_representation_cross.transpose(-2,-1),
                rigid_rotations=rigid_rotations_2,
                rigid_translations=rigid_translations_2,
                query_chunk_size=query_chunk_size_2,
                key_chunk_size=key_chunk_size_2,
                encoder_hidden_states=hidden_states_1,
                encoder_attention_mask=attention_mask_1,
                encoder_rigid_rotations=rigid_rotations_1,
                encoder_rigid_translations=rigid_translations_1,
            )
            return output_1 + output_2 + (rotation_angles, )

        for (ligand_update, receptor_update, sidechain_rotation) in zip(self.ligand_cross, self.receptor_cross, self.sidechain_cross):
            if self.gradient_checkpointing:
                cross_output = checkpoint(
                    cross,
                    receptor_update,
                    ligand_update,
                    hidden_seq,
                    hidden_smiles,
                    attention_mask_1,
                    attention_mask_2,
                    pair_representation_cross,
                    rotations_receptor,
                    translations_receptor,
                    rotations_ligand,
                    translations_ligand,
                    self.query_chunk_size_receptor,
                    self.key_chunk_size_receptor,
                    self.query_chunk_size_ligand,
                    self.key_chunk_size_ligand,
                    sidechain_rotation,
                    rotation_angles
                )
            else:
                cross_output = cross(
                    receptor_update,
                    ligand_update,
                    hidden_seq,
                    hidden_smiles,
                    attention_mask_1,
                    attention_mask_2,
                    pair_representation_cross,
                    rotations_receptor,
                    translations_receptor,
                    rotations_ligand,
                    translations_ligand,
                    self.query_chunk_size_receptor,
                    self.key_chunk_size_receptor,
                    self.query_chunk_size_ligand,
                    self.key_chunk_size_ligand,
                    sidechain_rotation,
                    rotation_angles
                )
            hidden_seq, translations_receptor, rotations_receptor, \
                hidden_smiles, translations_ligand, rotations_ligand, \
                rotation_angles = cross_output

        return hidden_smiles, hidden_seq, translations_ligand, rotations_ligand, translations_receptor, rotations_receptor, rotation_angles

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
