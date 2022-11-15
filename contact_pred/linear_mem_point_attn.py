# adapted to torch from: https://arxiv.org/abs/2112.05682
# inspired by https://github.com/CHARM-Tx/linear_mem_attention_pytorch

import math
import torch
from torch.utils import checkpoint
from typing import Tuple, Optional

from .linear_mem_attn_utils import dynamic_length_slice, dynamic_slice, torch_map, torch_scan

def query_chunk_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pair: torch.Tensor,
    pair_value: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    encoder_rotations: torch.Tensor,
    encoder_translations: torch.Tensor,
    points_query: torch.Tensor,
    points_key: torch.Tensor,
    points_value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    key_chunk_size: int = 4096,
    weight_kv = None,
    weight_points = None,
    gamma: torch.Tensor = None,
):
    """Multi-head dot product attention with a limited number of queries."""
    device, dtype = query.device, query.dtype
    batch, num_kv, num_heads, k_features = key.shape
    v_features = value.shape[-1]
    query_chunk = query.shape[1]  # b n h d
    key_chunk_size = min(key_chunk_size, num_kv)

    gamma = gamma.repeat(batch,num_kv,1)

    # @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        pair: torch.Tensor, pair_value: torch.Tensor,
        mask: torch.Tensor,
        translations: torch.Tensor, encoder_translations: torch.Tensor,
        rotations: torch.Tensor, encoder_rotations: torch.Tensor,
        points_query: torch.Tensor, points_key: torch.Tensor, points_value: torch.Tensor,
        gamma: torch.Tensor
    ):

        # sequences
        attn_weights = torch.einsum("bqhd,bkhd->bqhk", query, key)
        attn_weights = attn_weights * weight_kv

        # pair representation
        pair = pair.transpose(3,1)
        attn_weights = attn_weights + pair

        # point invariant
        a = torch.einsum('bnij,bnhpj->bnhpi', rotations, points_query)
        b = torch.einsum('bnij,bnhpj->bnhpi', encoder_rotations, points_key)
        a = a + translations[:,:,None,None]
        b = b + encoder_translations[:,:,None,None]

        a_sq = torch.sum(a**2,dim=[-2,-1])
        b_sq = torch.sum(b**2,dim=[-2,-1])
        b_sq = b_sq.transpose(2,1)
        invariant = a_sq[:,:,:,None] + b_sq[:,None] - 2*torch.einsum('bnhpi,bmhpi->bnhm',a,b)

        gamma = gamma.transpose(-2,-1)
        attn_weights = attn_weights - 0.5*weight_points*gamma[:,None,:,:] * invariant

        # overall scaling
        attn_weights = attn_weights / math.sqrt(3)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            max_neg = -torch.finfo(attn_weights.dtype).max
            mask = mask.bool()
            attn_weights.masked_fill_(~mask, max_neg)

        max_score = torch.amax(attn_weights, dim=-1, keepdim=True).detach()
        exp_weights = torch.exp(attn_weights - max_score)

        # context
        exp_values = torch.einsum("bvhf,bqhv->bqhf", value, exp_weights)

        # pair output
        pair_value = pair_value.transpose(3,1)
        exp_pair = torch.einsum('bqcv,bqhv->bqhc', pair_value, exp_weights)

        # point output
        c = torch.einsum('bnij,bnhpj->bnhpi', encoder_rotations, points_value)
        c = c + encoder_translations[:,:,None,None]
        exp_value_points = torch.einsum("bvhpi,bqhv->bqhpi", c, exp_weights)

        # ((b q h f), (b q h c), (b q h p i)), (b q h k), (b q h 1)
        return exp_values, exp_pair, exp_value_points, exp_weights.sum(dim=-1), max_score.squeeze(dim=-1)

    def chunk_scanner(
        chunk_idx: int,
    ):
        key_chunk = dynamic_length_slice(key, chunk_idx, key_chunk_size)
        value_chunk = dynamic_length_slice(value, chunk_idx, key_chunk_size)

        pair_chunk = dynamic_length_slice(pair.transpose(3,1), chunk_idx, key_chunk_size)
        pair_value_chunk = dynamic_length_slice(pair_value.transpose(3,1), chunk_idx, key_chunk_size)

        encoder_rotations_chunk = dynamic_length_slice(encoder_rotations, chunk_idx, key_chunk_size)
        encoder_translations_chunk = dynamic_length_slice(encoder_translations, chunk_idx, key_chunk_size)

        points_key_chunk = dynamic_length_slice(points_key, chunk_idx, key_chunk_size)
        points_value_chunk = dynamic_length_slice(points_value, chunk_idx, key_chunk_size)

        mask_chunk = None
        if mask is not None:
            mask_chunk = dynamic_length_slice(mask, chunk_idx, key_chunk_size)

        gamma_chunk = dynamic_length_slice(gamma, chunk_idx, key_chunk_size)

        return checkpoint.checkpoint(
            summarize_chunk, query, key_chunk, value_chunk,
            pair_chunk, pair_value_chunk,
            mask_chunk,
            translations, encoder_translations_chunk,
            rotations, encoder_rotations_chunk,
            points_query, points_key_chunk, points_value_chunk,
            gamma_chunk
        )

    num_chunks = int(math.ceil(num_kv / key_chunk_size))
    chunk_values = torch.zeros(
        num_chunks,
        batch,
        query_chunk,
        num_heads,
        v_features,
        dtype=dtype,
        device=device,
    )

    pair_channels = pair_value.shape[2]
    chunk_pair = torch.zeros(
        num_chunks,
        batch,
        query_chunk,
        num_heads,
        pair_channels,
        dtype=dtype,
        device=device,
    )

    num_value_points = points_value.shape[-2]
    point_dim = points_value.shape[-1]
    chunk_points = torch.zeros(
        num_chunks,
        batch,
        query_chunk,
        num_heads,
        num_value_points,
        point_dim,
        dtype=dtype,
        device=device,
    )

    chunk_weights = torch.zeros(
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

    for i in range(num_chunks):
        chunk_values[i], chunk_pair[i], chunk_points[i], chunk_weights[i], chunk_max[i] = chunk_scanner(
            i * key_chunk_size
        )

    max_diffs = torch.exp(chunk_max - chunk_max.amax(dim=0))

    all_weights = (max_diffs * chunk_weights).sum(dim=0).unsqueeze(dim=-1)

    all_values = (max_diffs.unsqueeze(dim=-1) * chunk_values).sum(dim=0) / all_weights
    all_pair = (max_diffs.unsqueeze(dim=-1) * chunk_pair).sum(dim=0) / all_weights
    all_points = (max_diffs.unsqueeze(dim=-1).unsqueeze(dim=-1) * chunk_points).sum(dim=0) / all_weights.unsqueeze(dim=-1)

    # transpose of an orthogonal matrix == inverse
    all_points = all_points - translations[:,:,None,None]
    all_points = torch.einsum('bnji,bnhpj->bnhpi', rotations, all_points)

    return all_values, all_pair, all_points

def point_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pair: torch.Tensor,
    pair_value: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    encoder_rotations: torch.Tensor,
    encoder_translations: torch.Tensor,
    points_query: torch.Tensor,
    points_key: torch.Tensor,
    points_value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    weight_kv = None,
    weight_points = None,
    gamma: torch.Tensor = None,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 4096,
):
    """Memory-efficient multi-head dot product attention.
    Inputs:
    * q, k, v: (b n h d) torch tensors
    * pair, pair_value: (b n h m), (b n c m)
    * rotations, translations, encoder_rotations, encoder_translations:
    * points_query, points_key, points_value:
    * mask: (b n)
    * query_chunk_size: int.
    * key_chunk_size: int.
    Outputs: (b n h d) torch tensor (qk-weighted sum of v)
    """
    batch, num_q, num_heads, q_features = query.shape

    def chunk_scanner(chunk_idx: int, _):
        query_chunk = dynamic_length_slice(query, chunk_idx, query_chunk_size)
        pair_chunk = dynamic_length_slice(pair, chunk_idx, query_chunk_size)
        pair_value_chunk = dynamic_length_slice(pair_value, chunk_idx, query_chunk_size)
        translations_chunk = dynamic_length_slice(translations, chunk_idx, query_chunk_size)
        rotations_chunk = dynamic_length_slice(rotations, chunk_idx, query_chunk_size)
        points_chunk = dynamic_length_slice(points_query, chunk_idx, query_chunk_size)

        return (
            chunk_idx + query_chunk_size,
            query_chunk_attention(
                query_chunk,
                key,
                value,
                pair_chunk,
                pair_value_chunk,
                rotations_chunk,
                translations_chunk,
                encoder_rotations,
                encoder_translations,
                points_chunk,
                points_key,
                points_value,
                mask,
                key_chunk_size,
                weight_kv,
                weight_points,
                gamma,
            ),
        )

    _, res = torch_scan(
        chunk_scanner, init=0, xs=None, length=int(math.ceil(num_q / query_chunk_size))
    )

    (context, pair_output, points_output) = res
    context = context.reshape(batch, num_q, num_heads, value.shape[-1])
    pair_output = pair_output.reshape(batch, num_q, num_heads, pair_value.shape[2])
    points_output = points_output.reshape(batch, num_q, num_heads, points_value.shape[-2], points_value.shape[-1])

    return context, pair_output, points_output
