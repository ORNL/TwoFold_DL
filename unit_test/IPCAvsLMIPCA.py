
import torch
import torch.nn as nn
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../contact_pred/'))
from IPCA import InvariantPointCrossAttention, LinearMemInvariantPointCrossAttention
from structure import IPAConfig

torch.manual_seed(42)

if __name__ == "__main__":
    seq_config = IPAConfig()
    smiles_config = IPAConfig()

    npts = 256

    query = torch.rand(1, npts, seq_config.bert_config['hidden_size'])
    key = value = query

    pair = torch.rand(1, 
                      seq_config.bert_config['num_attention_heads'] + smiles_config.bert_config['num_attention_heads'],
                      npts, 
                      npts)

    initial_receptor_xyz = torch.rand(query.size()[:2]+(3, ))#, device=hidden_seq.device, dtype=hidden_seq.dtype)
    initial_receptor_rot = torch.rand(3, 3).unsqueeze(0).unsqueeze(0).repeat(query.size()[0], query.size()[1], 1, 1)

    translations_receptor = initial_receptor_xyz
    rotations_receptor = initial_receptor_rot

    translations_ligand = torch.rand(query.size()[:2]+(3, ))
    rotations_ligand = torch.rand(3, 3).unsqueeze(0).unsqueeze(0).repeat(query.size()[0], query.size()[1], 1, 1)

    IPCA = InvariantPointCrossAttention(seq_config, smiles_config)
    LMIPCA = LinearMemInvariantPointCrossAttention(seq_config, smiles_config)

    # use the same weights for both layers

    LMIPCA.query = IPCA.query
    LMIPCA.key = IPCA.key
    LMIPCA.value = IPCA.value
    LMIPCA.query_point = IPCA.query_point	
    LMIPCA.key_point = IPCA.key_point	
    LMIPCA.value_point = IPCA.value_point	
    LMIPCA.head_weight = IPCA.head_weight	
    LMIPCA.pair_attention = IPCA.pair_attention
    LMIPCA.output_layer = IPCA.output_layer

    out_IPCA = IPCA(hidden_states = query, encoder_hidden_states = key, pair_representation = pair,
                    rigid_rotations = rotations_receptor, rigid_translations = translations_receptor,
                    encoder_rigid_rotations = rotations_ligand, encoder_rigid_translations = translations_receptor)
    print('Output from original IPCA')
    print(out_IPCA, out_IPCA.shape)
    print()
    out_LMIPCA = LMIPCA(hidden_states = query, encoder_hidden_states = key, pair_representation = pair,
                    rigid_rotations = rotations_receptor, rigid_translations = translations_receptor,
                    encoder_rigid_rotations = rotations_ligand, encoder_rigid_translations = translations_receptor,
                    query_chunk_size = 16, key_chunk_size = 64) # Set chunk size to < npts to make sure chunking actually happened
    print('Output from linear memory IPCA')
    print(out_LMIPCA, out_LMIPCA.shape)
