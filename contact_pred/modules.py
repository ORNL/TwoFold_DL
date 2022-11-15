from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertIntermediate, BertSelfOutput, BertOutput
from transformers.modeling_utils import apply_chunking_to_forward

from .structure import Structure, compute_weighted_RMSD, IPAConfig

from .utils import get_extended_attention_mask

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import math

from .linear_mem_attn import attention

class AttentionWithScoreOutput(nn.Module):
    def __init__(self, config, other_config=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if other_config is None:
            other_config = config

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(other_config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(other_config.hidden_size, self.all_head_size, bias=False)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))

            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # apply mask
        if attention_mask is not None:
            if attention_mask.dtype != torch.float32 and attention_mask.dtype != torch.float16:
                inv_attention_mask = get_extended_attention_mask(
                    attention_mask,
                    hidden_states.shape[:-1],
                    hidden_states.device,
                    hidden_states.dtype
                    )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores + inv_attention_mask, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_scores

class LinearMemAttentionWithScoreOutput(nn.Module):
    def __init__(self, config, other_config=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if other_config is None:
            other_config = config

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(other_config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(other_config.hidden_size, self.all_head_size, bias=False)

    def view_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(new_x_shape)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        query_chunk_size=1024,
        key_chunk_size=4096,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.view_for_scores(self.key(encoder_hidden_states))
            value_layer = self.view_for_scores(self.value(encoder_hidden_states))

            attention_mask = encoder_attention_mask
        else:
            key_layer = self.view_for_scores(self.key(hidden_states))
            value_layer = self.view_for_scores(self.value(hidden_states))

        query_layer = self.view_for_scores(mixed_query_layer)

        context, attention_scores = attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            query_chunk_size,
            key_chunk_size,
        )

        context_layer_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(context_layer_shape)

        attention_scores = attention_scores.permute(0,3,1,2)

        return context, attention_scores

class AttentionBlock(nn.Module):
    def __init__(
            self,
            config,
            other_config=None,
            linear_mem=False,
        ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.is_cross_attention = other_config is not None

        if linear_mem:
            if not self.is_cross_attention:
                self.attention = LinearMemAttentionWithScoreOutput(config)
            else:
                self.crossattention = LinearMemAttentionWithScoreOutput(config, other_config)
        else:
            if not self.is_cross_attention:
                self.attention = AttentionWithScoreOutput(config)
            else:
                self.crossattention = AttentionWithScoreOutput(config,other_config)

        if not self.is_cross_attention:
            self.self_output = BertSelfOutput(config)
        else:
            self.cross_output = BertSelfOutput(config)

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_chunk_size=1024,
        key_chunk_size=4096,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if not self.is_cross_attention:
            attention_outputs = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                query_chunk_size=query_chunk_size,
                key_chunk_size=key_chunk_size,
            )
            attention_output = attention_outputs[0]
            score_outputs = attention_outputs[1:]  # add cross attentions if we output attention weights

            hidden_states = self.self_output(attention_output, hidden_states)
        else:
            cross_attention_outputs = self.crossattention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                query_chunk_size=query_chunk_size,
                key_chunk_size=key_chunk_size,
            )
            attention_output = cross_attention_outputs[0]
            score_outputs = cross_attention_outputs[1:]  # add cross attentions if we output attention weights

            hidden_states = self.cross_output(attention_output, hidden_states)

        # add cross-attn cache to positions 3,4 of present_key_value tuple
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, hidden_states
        )
        outputs = (layer_output,) + score_outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class EnsembleEmbedding(torch.nn.Module):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__()

        self.config = config

        self.gradient_checkpointing = False

        self.seq_model = BertModel(
            BertConfig.from_dict(config.seq_config),
            add_pooling_layer=add_pooling_layer,
            )

        self.smiles_model = BertModel(
            BertConfig.from_dict(config.smiles_config),
            add_pooling_layer=add_pooling_layer,
        )

        # use the configuration of the model with the larger hidden dimensions
        self.hidden_size = self.seq_model.config.hidden_size + self.smiles_model.config.hidden_size

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

        # checkpoint gradients for models that are not frozen
        if any([p.requires_grad for p in self.seq_model.parameters()]):
            self.seq_model.gradient_checkpointing_enable()

        if any([p.requires_grad for p in self.smiles_model.parameters()]):
            self.smiles_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.seq_model.gradient_checkpointing_disable()
        self.smiles_model.gradient_checkpointing_disable()

    def load_pretrained(self, seq_model_name, smiles_model_name, add_pooling_layer=False):
        self.seq_model = BertModel.from_pretrained(seq_model_name,
            add_pooling_layer=add_pooling_layer,
            config=BertConfig.from_dict(self.config.seq_config),
        )
        self.smiles_model = BertModel.from_pretrained(smiles_model_name,
            add_pooling_layer=add_pooling_layer,
            config=BertConfig.from_dict(self.config.smiles_config),
        )

    def freeze_protein(self):
        for param in self.seq_model.parameters():
            param.requires_grad = False

    def freeze_ligand(self):
        for param in self.smiles_model.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_ids_1=None,
            inputs_embeds_1=None,
            attention_mask_1=None,
            input_ids_2=None,
            inputs_embeds_2=None,
            attention_mask_2=None,
    ):
        # embed amino acids, sharing the same model
        encoder_outputs = self.seq_model(
            input_ids=input_ids_1,
            inputs_embeds=inputs_embeds_1,
            attention_mask=attention_mask_1,
        )
        hidden_seq = encoder_outputs.last_hidden_state

        # encode SMILES
        encoder_outputs = self.smiles_model(
            input_ids=input_ids_2,
            inputs_embeds=inputs_embeds_2,
            attention_mask=attention_mask_2,
        )
        hidden_smiles = encoder_outputs.last_hidden_state

        # concatenate the outputs
        return hidden_seq, hidden_smiles

class PairRepresentation(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = EnsembleEmbedding(config, add_pooling_layer=False)

        self.attn_seq = torch.nn.ModuleList([
            AttentionBlock(self.embedding.seq_model.config, linear_mem=config.linear_mem_attn)
            for i in range(config.n_cross_attention)])

        self.attn_smiles = torch.nn.ModuleList([
            AttentionBlock(self.embedding.smiles_model.config, linear_mem=config.linear_mem_attn)
            for i in range(config.n_cross_attention)])

        self.query_chunk_size_seq = config.query_chunk_size_seq
        self.key_chunk_size_seq = config.key_chunk_size_seq

        self.query_chunk_size_smiles = config.query_chunk_size_smiles
        self.key_chunk_size_smiles = config.key_chunk_size_smiles

        self.initial_norm_seq = nn.LayerNorm(self.embedding.seq_model.config.hidden_size)
        self.initial_norm_smiles = nn.LayerNorm(self.embedding.smiles_model.config.hidden_size)

        self.gradient_checkpointing=False

    def forward(
            self,
            input_ids_1=None,
            inputs_embeds_1=None,
            attention_mask_1=None,
            input_ids_2=None,
            inputs_embeds_2=None,
            attention_mask_2=None,
            return_dict=False,
            **kwargs,
    ):
        embedding = self.embedding(
            input_ids_1=input_ids_1,
            inputs_embeds_1=inputs_embeds_1,
            attention_mask_1=attention_mask_1,
            input_ids_2=input_ids_2,
            inputs_embeds_2=inputs_embeds_2,
            attention_mask_2=attention_mask_2,
        )

        hidden_seq, hidden_smiles = embedding

        hidden_seq = self.initial_norm_seq(hidden_seq)
        hidden_smiles = self.initial_norm_smiles(hidden_smiles)

        for attn_1, attn_2 in zip(self.attn_seq, self.attn_smiles):
            # receptor
            if self.gradient_checkpointing:
                hidden_seq, attention_score_1 = checkpoint(attn_1,
                    hidden_seq,
                    attention_mask_1,
                    self.query_chunk_size_seq,
                    self.key_chunk_size_seq,
                 )
            else:
                hidden_seq, attention_score_1 = attn_1(
                    hidden_states=hidden_seq,
                    attention_mask=attention_mask_1,
                    query_chunk_size=self.query_chunk_size_seq,
                    key_chunk_size=self.key_chunk_size_seq,
                 )

            # ligand
            if self.gradient_checkpointing:
                hidden_smiles, attention_score_2 = checkpoint(attn_2,
                    hidden_smiles,
                    attention_mask_2,
                    self.query_chunk_size_smiles,
                    self.key_chunk_size_smiles,
                 )
            else:
                hidden_smiles, attention_score_2 = attn_2(
                    hidden_states = hidden_smiles,
                    attention_mask = attention_mask_2,
                    query_chunk_size = self.query_chunk_size_smiles,
                    key_chunk_size = self.key_chunk_size_smiles,
                 )

        pair_representation_seq = attention_score_1
        pair_representation_smiles = attention_score_2

        outputs = dict()

        outputs['pair_representation_seq'] = pair_representation_seq
        outputs['pair_representation_smiles'] = pair_representation_smiles

        outputs['hidden_seq'] = hidden_seq
        outputs['hidden_smiles'] = hidden_smiles

        if not return_dict:
            outputs = tuple(outputs.values())

        return outputs

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        self.embedding.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.embedding.gradient_checkpointing_disable()

    def freeze_protein(self):
        self.embedding.freeze_protein()
        for param in self.attn_seq.parameters():
            param.requires_grad = False
        for param in self.initial_norm_seq.parameters():
            param.requires_grad = False

    def freeze_ligand(self):
        self.embedding.freeze_ligand()
        for param in self.attn_smiles.parameters():
            param.requires_grad = False
        for param in self.initial_norm_smiles.parameters():
            param.requires_grad = False

class CrossPairRepresentation(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_seq = torch.nn.ModuleList([
            AttentionBlock(BertConfig.from_dict(config.seq_config),
                           BertConfig.from_dict(config.smiles_config),
                           linear_mem=config.linear_mem_attn)
            for i in range(config.n_cross_attention)])

        self.attn_smiles = torch.nn.ModuleList([
            AttentionBlock(BertConfig.from_dict(config.smiles_config),
                           BertConfig.from_dict(config.seq_config),
                           linear_mem=config.linear_mem_attn)
            for i in range(config.n_cross_attention)])

        self.query_chunk_size_seq = config.query_chunk_size_seq
        self.key_chunk_size_seq = config.key_chunk_size_seq

        self.query_chunk_size_smiles = config.query_chunk_size_smiles
        self.key_chunk_size_smiles = config.key_chunk_size_smiles

        self.initial_norm_seq = nn.LayerNorm(config.seq_config['hidden_size'])
        self.initial_norm_smiles = nn.LayerNorm(config.smiles_config['hidden_size'])

        self.gradient_checkpointing=False

    def forward(
            self,
            hidden_states_1,
            hidden_states_2,
            attention_mask_1,
            attention_mask_2,
            return_dict=False,
            **kwargs,
    ):
        hidden_seq = self.initial_norm_seq(hidden_states_1)
        hidden_smiles = self.initial_norm_smiles(hidden_states_2)

        def cross(attn_1, attn_2, hidden_states_1, hidden_states_2, attention_mask_1, attention_mask_2,
            query_chunk_size_1, key_chunk_size_1,
            query_chunk_size_2, key_chunk_size_2
            ):
            attention_output_1 = attn_1(
                hidden_states = hidden_states_1,
                attention_mask = attention_mask_1,
                query_chunk_size = query_chunk_size_1,
                key_chunk_size = key_chunk_size_1,
                encoder_hidden_states = hidden_states_2,
                encoder_attention_mask = attention_mask_2,
                )
            attention_output_2 = attn_2(
                hidden_states = hidden_states_2,
                attention_mask = attention_mask_2,
                query_chunk_size = query_chunk_size_2,
                key_chunk_size = key_chunk_size_2,
                encoder_hidden_states = hidden_states_1,
                encoder_attention_mask = attention_mask_1,
                )

            # torch.utils.checkpoint does not support nested structures, concatenate the outputs
            output = attention_output_1 + attention_output_2

            return output

        for attn_1, attn_2 in zip(self.attn_seq, self.attn_smiles):
            if self.gradient_checkpointing:
                 (hidden_seq,
                 attention_score_1,
                 hidden_smiles,
                 attention_score_2) = checkpoint(
                    cross,
                    attn_1,
                    attn_2,
                    hidden_seq,
                    hidden_smiles,
                    attention_mask_1,
                    attention_mask_2,
                    self.query_chunk_size_seq,
                    self.key_chunk_size_seq,
                    self.query_chunk_size_smiles,
                    self.key_chunk_size_smiles,
                )
            else:
                (hidden_seq,
                 attention_score_1,
                 hidden_smiles,
                 attention_score_2) = cross(
                    attn_1,
                    attn_2,
                    hidden_seq,
                    hidden_smiles,
                    attention_mask_1,
                    attention_mask_2,
                    self.query_chunk_size_seq,
                    self.key_chunk_size_seq,
                    self.query_chunk_size_smiles,
                    self.key_chunk_size_smiles,
                )

        # concatenate attention heads
        pair_representation_cross = torch.cat((attention_score_1, torch.transpose(attention_score_2, 2, 3)), dim=1)

        outputs = dict()
        outputs['pair_representation_cross'] = pair_representation_cross
        outputs['hidden_seq'] = hidden_seq
        outputs['hidden_smiles'] = hidden_smiles

        if not return_dict:
            outputs = tuple(outputs.values())

        return outputs

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        self.embedding.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.embedding.gradient_checkpointing_disable()
