import torch
import torch.nn as nn
import math

from .utils import get_extended_attention_mask

from .linear_mem_point_attn import point_attention

class InvariantPointCrossAttention(nn.Module):
    def __init__(self, config, other_config, is_cross_attention=True):
        super().__init__()
        if config.bert_config['hidden_size'] % config.num_ipa_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.bert_config['hidden_size']}) is not a multiple of the number of IPA attention "
                f"heads ({config.num_ipa_heads})"
            )

        self.num_ipa_heads = config.num_ipa_heads
        self.attention_head_size = int(config.bert_config['hidden_size'] / self.num_ipa_heads)
        self.all_head_size = self.num_ipa_heads * self.attention_head_size

        self.query = nn.Linear(config.bert_config['hidden_size'], self.all_head_size, bias=False)
        self.key = nn.Linear(other_config.bert_config['hidden_size'], self.all_head_size, bias=False)
        self.value = nn.Linear(other_config.bert_config['hidden_size'], self.all_head_size, bias=False)

        self.num_query_points = config.num_points
        self.num_value_points = other_config.num_points

        # points in R3
        self.query_point = nn.Linear(config.bert_config['hidden_size'],
            self.num_ipa_heads*self.num_query_points*3, bias=False)
        self.key_point = nn.Linear(other_config.bert_config['hidden_size'],
            self.num_ipa_heads*self.num_query_points*3, bias=False)
        self.value_point = nn.Linear(other_config.bert_config['hidden_size'],
            self.num_ipa_heads*self.num_value_points*3, bias=False)

        self.head_weight = torch.nn.Parameter(torch.zeros(config.num_ipa_heads))
        torch.nn.init.normal_(self.head_weight)

        # scalar self attention weights
        self.n_pair_channels = config.bert_config['num_attention_heads']
        if is_cross_attention:
            self.n_pair_channels += other_config.bert_config['num_attention_heads']

        self.pair_attention = nn.Linear(self.n_pair_channels, self.num_ipa_heads, bias=False)

        self.output_layer = nn.Linear(self.num_ipa_heads * (self.n_pair_channels +
            self.attention_head_size + self.num_value_points*(3+1)), config.bert_config['hidden_size'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_ipa_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_points_for_scores(self, x, num_points):
        new_x_shape = x.size()[:-1] + (self.num_ipa_heads, num_points, 3)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3, 4)

    def transpose_pair_representation(self, x):
        return x.permute(0, 2, 3, 1)

    def transpose_pair_attention(self, x):
        return x.permute(0, 3, 1, 2)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        pair_representation=None,
        rigid_rotations=None,
        rigid_translations=None,
        encoder_rigid_rotations=None,
        encoder_rigid_translations=None,
        **kwargs
    ):
        is_cross_attention = encoder_hidden_states is not None

        if not is_cross_attention:
            encoder_hidden_states = hidden_states
            encoder_rigid_translations = rigid_translations
            encoder_rigid_rotations = rigid_rotations
            encoder_attention_mask = attention_mask

        inv_attention_mask = None
        if encoder_attention_mask is not None:
            if encoder_attention_mask.dtype != torch.float32 and encoder_attention_mask.dtype != torch.float16:
                inv_attention_mask = get_extended_attention_mask(
                    encoder_attention_mask,
                    encoder_hidden_states.shape[:-1],
                    encoder_hidden_states.device,
                    encoder_hidden_states.dtype
                    )

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # pair representation contribution
        attention_scores = attention_scores + self.transpose_pair_attention(self.pair_attention(self.transpose_pair_representation(pair_representation)))

        # point contribution
        query_points = self.transpose_points_for_scores(self.query_point(hidden_states), self.num_query_points)
        key_points = self.transpose_points_for_scores(self.key_point(encoder_hidden_states), self.num_query_points)
        value_points = self.transpose_points_for_scores(self.value_point(encoder_hidden_states), self.num_value_points)

        # rigid update
        a = torch.einsum('bnij,bhnpj->bhnpi', rigid_rotations, query_points)
        b = torch.einsum('bnij,bhnpj->bhnpi', encoder_rigid_rotations, key_points)
        a = a + rigid_translations[:,None,:,None]
        b = b + encoder_rigid_translations[:,None,:,None]

        weight_points = math.sqrt(2/(9*self.num_query_points))
        gamma = torch.nn.functional.softplus(self.head_weight)

#        invariant = torch.sum((a[:,:,:,None,:,:] - b[:,:,None,:,:,:])**2,dim=[-2,-1])
        a_sq = torch.sum(a**2,dim=[-2,-1])
        b_sq = torch.sum(b**2,dim=[-2,-1])
        invariant = a_sq[:,:,:,None] + b_sq[:,:,None,:] - 2*torch.einsum('bhnpi,bhmpi->bhnm',a,b)

        attention_scores = attention_scores - 0.5*weight_points*gamma[None,:,None,None] * invariant

        # overall scaling
        attention_scores = attention_scores / math.sqrt(3)

        if inv_attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + inv_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        pair_output = torch.einsum('bhnm,bcnm->bnhc', attention_probs, pair_representation)

        # rigid update on output [Eq. 10]
        c = torch.einsum('bnij,bhnpj->bhnpi', encoder_rigid_rotations, value_points)
        c = c + encoder_rigid_translations[:,None,:,None]
        point_output = torch.einsum('bhij,bhjpk->bhipk', attention_probs, c)

        # transpose of an orthogonal matrix == inverse
        point_output = point_output - rigid_translations[:,None,:,None]
        point_output = torch.einsum('bnji,bhnpj->bhnpi', rigid_rotations, point_output)

        context_layer = torch.matmul(attention_probs, value_layer)

        new_pair_shape = pair_output.size()[:2] + (self.n_pair_channels*self.num_ipa_heads,)
        pair_output = pair_output.view(new_pair_shape)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        point_output = point_output.permute(0, 2, 1, 3, 4).contiguous()
        # the vector norm is a separate channel [in Eq. 11]
        point_output_sq = torch.sqrt(torch.sum(point_output*point_output,-1))

        new_point_shape = point_output.size()[:-3] + (3*self.num_value_points*self.num_ipa_heads,)
        point_output = point_output.view(new_point_shape)

        new_point_sq_shape = point_output_sq.size()[:-2] + (self.num_value_points*self.num_ipa_heads,)
        point_output_sq = point_output_sq.view(new_point_sq_shape)

        return self.output_layer(torch.cat([pair_output, context_layer, point_output, point_output_sq], dim=-1))

class LinearMemInvariantPointCrossAttention(nn.Module):
    def __init__(self, config, other_config, is_cross_attention=True):
        super().__init__()
        if config.bert_config['hidden_size'] % config.num_ipa_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.bert_config['hidden_size']}) is not a multiple of the number of IPA attention "
                f"heads ({config.num_ipa_heads})"
            )

        self.num_ipa_heads = config.num_ipa_heads
        self.attention_head_size = int(config.bert_config['hidden_size'] / self.num_ipa_heads)
        self.all_head_size = self.num_ipa_heads * self.attention_head_size

        self.query = nn.Linear(config.bert_config['hidden_size'], self.all_head_size, bias=False)
        self.key = nn.Linear(other_config.bert_config['hidden_size'], self.all_head_size, bias=False)
        self.value = nn.Linear(other_config.bert_config['hidden_size'], self.all_head_size, bias=False)

        self.num_query_points = config.num_points
        self.num_value_points = other_config.num_points

        # points in R3
        self.query_point = nn.Linear(config.bert_config['hidden_size'],
            self.num_ipa_heads*self.num_query_points*3, bias=False)
        self.key_point = nn.Linear(other_config.bert_config['hidden_size'],
            self.num_ipa_heads*self.num_query_points*3, bias=False)
        self.value_point = nn.Linear(other_config.bert_config['hidden_size'],
            self.num_ipa_heads*self.num_value_points*3, bias=False)

        self.head_weight = torch.nn.Parameter(torch.zeros(config.num_ipa_heads))
        torch.nn.init.normal_(self.head_weight)

        # scalar self attention weights
        self.n_pair_channels = config.bert_config['num_attention_heads']
        if is_cross_attention:
            self.n_pair_channels += other_config.bert_config['num_attention_heads']

        self.pair_attention = nn.Linear(self.n_pair_channels, self.num_ipa_heads, bias=False)

        self.output_layer = nn.Linear(self.num_ipa_heads * (self.n_pair_channels +
            self.attention_head_size + self.num_value_points*(3+1)), config.bert_config['hidden_size'])

    def view_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_ipa_heads, self.attention_head_size)
        return x.view(new_x_shape)

    def view_points_for_scores(self, x, num_points):
        new_x_shape = x.size()[:-1] + (self.num_ipa_heads, num_points, 3)
        return x.view(new_x_shape)

    def transpose_pair_representation(self, x):
        return x.permute(0, 2, 3, 1)

    def transpose_pair_representation_for_value(self, x):
        return x.permute(0, 2, 1, 3)

    def transpose_pair_attention(self, x):
        return x.permute(0, 1, 3, 2)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        pair_representation=None,
        rigid_rotations=None,
        rigid_translations=None,
        encoder_rigid_rotations=None,
        encoder_rigid_translations=None,
        query_chunk_size = 1024,
        key_chunk_size = 4096,
    ):
        is_cross_attention = encoder_hidden_states is not None

        if not is_cross_attention:
            encoder_hidden_states = hidden_states
            encoder_rigid_translations = rigid_translations
            encoder_rigid_rotations = rigid_rotations
            encoder_attention_mask = attention_mask

        key_layer = self.view_for_scores(self.key(encoder_hidden_states))
        value_layer = self.view_for_scores(self.value(encoder_hidden_states))
        key_points = self.view_points_for_scores(self.key_point(encoder_hidden_states), self.num_query_points)
        value_points = self.view_points_for_scores(self.value_point(encoder_hidden_states), self.num_value_points)
        value_layer = self.view_for_scores(self.value(encoder_hidden_states))

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.view_for_scores(mixed_query_layer)

        # pair representation
        pair = self.transpose_pair_attention(self.pair_attention(self.transpose_pair_representation(pair_representation)))
        pair_value = self.transpose_pair_representation_for_value(pair_representation)

        # points
        query_points = self.view_points_for_scores(self.query_point(hidden_states), self.num_query_points)

        weight_kv = 1 / math.sqrt(self.attention_head_size)
        weight_points = math.sqrt(2/(9*self.num_query_points))
        gamma = torch.nn.functional.softplus(self.head_weight)
        point_attention_output = point_attention(
            query=query_layer,
            key=key_layer,
            value=value_layer,
            pair=pair,
            pair_value=pair_value,
            rotations=rigid_rotations,
            translations=rigid_translations,
            encoder_rotations=encoder_rigid_rotations,
            encoder_translations=encoder_rigid_translations,
            points_query=query_points,
            points_key=key_points,
            points_value=value_points,
            mask=encoder_attention_mask,
            weight_kv=weight_kv,
            weight_points=weight_points,
            gamma=gamma,
            query_chunk_size=query_chunk_size,
            key_chunk_size=key_chunk_size,
        )

        context, pair_output, points_output = point_attention_output

        new_context_layer_shape = context.size()[:-2] + (self.all_head_size,)
        context_layer = context.view(new_context_layer_shape)

        new_pair_shape = pair_output.size()[:2] + (self.n_pair_channels*self.num_ipa_heads,)
        pair_output = pair_output.view(new_pair_shape)

        # the vector norm is a separate channel [in Eq. 11]
        points_output_sq = torch.sqrt(torch.sum(points_output*points_output,-1))
        new_points_sq_shape = points_output_sq.size()[:-2] + (self.num_value_points*self.num_ipa_heads,)
        points_output_sq = points_output_sq.view(new_points_sq_shape)

        new_point_shape = points_output.size()[:-3] + (3*self.num_value_points*self.num_ipa_heads,)
        points_output = points_output.view(new_point_shape)

        return self.output_layer(torch.cat([pair_output, context_layer, points_output, points_output_sq], dim=-1))
