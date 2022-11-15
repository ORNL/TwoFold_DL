
import torch
import torch.nn as nn
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../contact_pred/'))
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention
from transformers import PreTrainedModel, PretrainedConfig
from linear_mem_attn import attention
from structure import IPAConfig
from modules import LinearMemAttentionWithScoreOutput

torch.manual_seed(42)

if __name__ == "__main__":
    
    # The attention_probs_dropout_prob defaults to 0.1 in BertConfig
    # but in linear memory attention there is no dropout layer 
    # (i.e. attention_probs_dropout_prob is always 0)
    # So we set that to 0 to compare the results
    seq_config = BertConfig(attention_probs_dropout_prob=0)
    
    npts=256
 
    query = torch.rand(1, npts, seq_config.hidden_size)
    key = value = query
    print(query.shape) 

    # Regular memory attention
    ATTN = BertSelfAttention(seq_config)
    out_ATTN = ATTN(hidden_states = query, encoder_hidden_states = key)
    print('Output from original attention')
    print(out_ATTN, out_ATTN[0].shape) 
    print()

    # Linear memory attention
    LMATTN = LinearMemAttentionWithScoreOutput(seq_config)
    LMATTN.query = ATTN.query
    LMATTN.key = ATTN.key
    LMATTN.value = ATTN.value
   
    # Choose a smaller chunk size to test if chunking actually works 
    out_LMATTN = LMATTN(hidden_states = query, encoder_hidden_states = key, query_chunk_size=16, key_chunk_size=64)
    print('Output from linear memory attention')
    print(out_LMATTN, out_LMATTN[0].shape)
