# adapted to torch from: https://arxiv.org/abs/2112.05682
import math
import torch
from torch.utils import checkpoint
from typing import Tuple, Optional

from .linear_mem_attn_utils import dynamic_length_slice, dynamic_slice, torch_map, torch_scan

def query_chunk_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    key_chunk_size: int = 4096,
) -> torch.Tensor:
    """Multi-head dot product attention with a limited number of queries."""
    device, dtype = query.device, query.dtype
    batch, num_kv, num_heads, k_features = key.shape
    v_features = value.shape[-1]
    query_chunk = query.shape[1]  # b n h d
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / k_features ** 0.5

    # @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        attn_scores = torch.einsum("bqhd,bkhd->bqhk", query, key)
        attn_weights = attn_scores.clone()

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            max_neg = -torch.finfo(attn_weights.dtype).max
            mask = mask.bool()
            attn_weights.masked_fill_(~mask, max_neg)

        max_score = torch.amax(attn_weights, dim=-1, keepdim=True).detach()
        exp_weights = torch.exp(attn_weights - max_score)

        exp_values = torch.einsum("bvhf,bqhv->bqhf", value, exp_weights)
        # (b q h f), (b q h), (b q h 1)

        attn_scores_pad = torch.nn.functional.pad(attn_scores, (0, key_chunk_size-key.shape[1]))

        return exp_values, exp_weights.sum(dim=-1), max_score.squeeze(dim=-1), attn_scores_pad.permute(3,0,1,2)

    def chunk_scanner(
        chunk_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        key_chunk = dynamic_length_slice(key, chunk_idx, key_chunk_size)
        value_chunk = dynamic_length_slice(value, chunk_idx, key_chunk_size)

        mask_chunk = None
        if mask is not None:
            mask_chunk = dynamic_length_slice(mask, chunk_idx, key_chunk_size)

        return checkpoint.checkpoint(
            summarize_chunk, query, key_chunk, value_chunk, mask_chunk
        )

    num_chunks = math.ceil(num_kv / key_chunk_size)
    chunk_values = torch.zeros(
        num_chunks,
        batch,
        query_chunk,
        num_heads,
        v_features,
        dtype=dtype,
        device=device,
    )
    chunk_exp_weights = torch.zeros(
        num_chunks,
        batch,
        query_chunk,
        num_heads,
        dtype=dtype,
        device=device,
    )
    chunk_max = torch.zeros(
        num_chunks,
        batch,
        query_chunk,
        num_heads,
        dtype=dtype,
        device=device,
    )
    chunk_weights = torch.zeros(
        num_chunks,
        key_chunk_size,
        batch,
        query_chunk,
        num_heads,
        dtype=dtype,
        device=device,
    )

    for i in range(num_chunks):
        chunk_values[i], chunk_exp_weights[i], chunk_max[i], chunk_weights[i] = chunk_scanner(
            i * key_chunk_size
        )

    max_diffs = torch.exp(chunk_max - chunk_max.amax(dim=0))

    all_values = (max_diffs.unsqueeze(dim=-1) * chunk_values).sum(dim=0)
    all_exp_weights = (max_diffs * chunk_exp_weights).sum(dim=0).unsqueeze(dim=-1)

    all_scores = chunk_weights.view(-1,batch,query_chunk,num_heads).permute(1,2,0,3)
    all_scores = all_scores[:,:,:num_kv]

    return all_values / all_exp_weights, all_scores


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 4096,
) -> torch.Tensor:
    """Memory-efficient multi-head dot product attention.
    Inputs:
    * q, k, v: (b n h d) torch tensors
    * mask: (b n)
    * query_chunk_size: int.
    * key_chunk_size: int.
    Outputs: (b n h d) torch tensor (qk-weighted sum of v),
         and (b n m h) attention scores
    """
    batch, num_q, num_heads, q_features = query.shape

    def chunk_scanner(chunk_idx: int, _):
        query_chunk = dynamic_length_slice(query, chunk_idx, query_chunk_size)

        return (
            chunk_idx + query_chunk_size,
            query_chunk_attention(
                query_chunk, key, value, mask, key_chunk_size=key_chunk_size
            ),
        )

    _, res = torch_scan(
        chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size)
    )

    return res
