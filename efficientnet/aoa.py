# Implementation for paper 'Attention on Attention for Image Captioning'
# https://arxiv.org/abs/1908.06954

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(
            att_feats, att_masks.data.long().sum(1)
        )
        return pad_unsort_packed_sequence(
            PackedSequence(module(packed[0]), packed[1]), inv_ix
        )
    else:
        return module(att_feats)


class MultiHeadedDotAttention(nn.Module):
    def __init__(
        self,
        h,
        d_model,
        dropout=0.1,
        scale=1,
        project_k_v=1,
        use_output_layer=1,
        do_aoa=0,
        norm_q=0,
        dropout_aoa=0.3,
    ):
        super(MultiHeadedDotAttention, self).__init__()
        assert d_model * scale % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model * scale // h
        self.h = h

        # Do we need to do linear projections on K and V?
        self.project_k_v = project_k_v

        # normalize the query?
        if norm_q:
            self.norm = LayerNorm(d_model)
        else:
            self.norm = lambda x: x
        self.linears = clones(nn.Linear(d_model, d_model * scale), 1 + 2 * project_k_v)

        # output linear layer after the multi-head attention?
        self.output_layer = nn.Linear(d_model * scale, d_model)

        # apply aoa after attention?
        self.use_aoa = do_aoa
        if self.use_aoa:
            self.aoa_layer = nn.Sequential(
                nn.Linear((1 + scale) * d_model, 2 * d_model), nn.GLU()
            )
            # dropout to the input of AoA layer
            if dropout_aoa > 0:
                self.dropout_aoa = nn.Dropout(p=dropout_aoa)
            else:
                self.dropout_aoa = lambda x: x

        if self.use_aoa or not use_output_layer:
            # AoA doesn't need the output linear layer
            del self.output_layer
            self.output_layer = lambda x: x

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, value, key, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        single_query = 0
        if len(query.size()) == 2:
            single_query = 1
            query = query.unsqueeze(1)

        nbatches = query.size(0)

        query = self.norm(query)

        # Do all the linear projections in batch from d_model => h x d_k
        if self.project_k_v == 0:
            query_ = (
                self.linears[0](query)
                .view(nbatches, -1, self.h, self.d_k)
                .transpose(1, 2)
            )
            key_ = key.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value_ = value.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_, key_, value_ = [
                l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for l, x in zip(self.linears, (query, key, value))
            ]

        # Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query_, key_, value_, mask=mask, dropout=self.dropout)

        # "Concat" using a view
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        if self.use_aoa:
            # Apply AoA
            x = self.aoa_layer(
                self.dropout_aoa(
                    torch.cat([x, query.view(nbatches, -1, self.h * self.d_k)], -1)
                )
            )
        x = self.output_layer(x)

        if single_query:
            query = query.squeeze(1)
            x = x.squeeze(1)
        return x


class AoA_Refiner_Layer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(AoA_Refiner_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.use_ff = 0
        if self.feed_forward is not None:
            self.use_ff = 1
        self.sublayer = clones(SublayerConnection(size, dropout), 1 + self.use_ff)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[-1](x, self.feed_forward) if self.use_ff else x


class AoA_Refiner_Core(nn.Module):
    def __init__(self, num_heads, enc_size, use_ff):
        super(AoA_Refiner_Core, self).__init__()
        attn = MultiHeadedDotAttention(
            num_heads,
            enc_size,
            project_k_v=1,
            scale=1,
            do_aoa=1,
            norm_q=0,
            dropout_aoa=0.3,
        )
        layer = AoA_Refiner_Layer(
            enc_size,
            attn,
            PositionwiseFeedForward(opt.enc_size, 2048, 0.1) if use_ff else None,
            0.1,
        )
        self.layers = clones(layer, 6)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# class AoA_Decoder_Core(nn.Module):
#     def __init__(self, opt):
#         super(AoA_Decoder_Core, self).__init__()
#         self.drop_prob_lm = opt.drop_prob_lm
#         self.d_model = opt.rnn_size
#         self.multi_head_scale = opt.multi_head_scale
#         self.use_ctx_drop = getattr(opt, "ctx_drop", 0)
#         self.out_res = getattr(opt, "out_res", 0)
#         self.decoder_type = getattr(opt, "decoder_type", "AoA")
#         self.att_lstm = nn.LSTMCell(
#             opt.input_encoding_size + opt.rnn_size, opt.rnn_size
#         )  # we, fc, h^2_t-1
#         self.out_drop = nn.Dropout(self.drop_prob_lm)

#         if self.decoder_type == "AoA":
#             # AoA layer
#             self.att2ctx = nn.Sequential(
#                 nn.Linear(
#                     self.d_model * opt.multi_head_scale + opt.rnn_size, 2 * opt.rnn_size
#                 ),
#                 nn.GLU(),
#             )
#         elif self.decoder_type == "LSTM":
#             # LSTM layer
#             self.att2ctx = nn.LSTMCell(
#                 self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size
#             )
#         else:
#             # Base linear layer
#             self.att2ctx = nn.Sequential(
#                 nn.Linear(
#                     self.d_model * opt.multi_head_scale + opt.rnn_size, opt.rnn_size
#                 ),
#                 nn.ReLU(),
#             )

#         self.attention = MultiHeadedDotAttention(
#             opt.num_heads,
#             opt.rnn_size,
#             project_k_v=0,
#             scale=opt.multi_head_scale,
#             use_output_layer=0,
#             do_aoa=0,
#             norm_q=1,
#         )

#         if self.use_ctx_drop:
#             self.ctx_drop = nn.Dropout(self.drop_prob_lm)
#         else:
#             self.ctx_drop = lambda x: x

#     def forward(self, xt, mean_feats, att_feats, p_att_feats, state, att_masks=None):
#         # state[0][1] is the context vector at the last step
#         h_att, c_att = self.att_lstm(
#             torch.cat([xt, mean_feats + self.ctx_drop(state[0][1])], 1),
#             (state[0][0], state[1][0]),
#         )

#         att = self.attention(
#             h_att,
#             p_att_feats.narrow(2, 0, self.multi_head_scale * self.d_model),
#             p_att_feats.narrow(
#                 2,
#                 self.multi_head_scale * self.d_model,
#                 self.multi_head_scale * self.d_model,
#             ),
#             att_masks,
#         )

#         ctx_input = torch.cat([att, h_att], 1)
#         if self.decoder_type == "LSTM":
#             output, c_logic = self.att2ctx(ctx_input, (state[0][1], state[1][1]))
#             state = (torch.stack((h_att, output)), torch.stack((c_att, c_logic)))
#         else:
#             output = self.att2ctx(ctx_input)
#             # save the context vector to state[0][1]
#             state = (torch.stack((h_att, output)), torch.stack((c_att, state[1][1])))

#         if self.out_res:
#             # add residual connection
#             output = output + h_att

#         output = self.out_drop(output)
#         return output, state


# class AoAModel(nn.Module):
#     def __init__(self, opt):
#         super(AoAModel, self).__init__()
#         self.num_layers = 2
#         # mean pooling
#         self.ctx2att = nn.Linear(opt.rnn_size, 2 * opt.multi_head_scale * opt.rnn_size)

#         self.refiner = AoA_Refiner_Core(opt)
#         self.core = AoA_Decoder_Core(opt)

#         self.vocab_size = opt.vocab_size
#         self.input_encoding_size = opt.input_encoding_size
#         # self.rnn_type = opt.rnn_type
#         self.rnn_size = opt.rnn_size
#         self.num_layers = opt.num_layers
#         self.drop_prob_lm = opt.drop_prob_lm
#         self.seq_length = (
#             getattr(opt, "max_len", 275) or opt.seq_length
#         )  # maximum sample length
#         self.att_feat_size = opt.att_feat_size

#         self.use_bn = getattr(opt, "use_bn", 0)

#         self.ss_prob = 0.0  # Schedule sampling probability
#         self.embed = nn.Sequential(
#             nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
#             nn.ReLU(),
#             nn.Dropout(self.drop_prob_lm),
#         )
#         self.att_embed = nn.Sequential(
#             *(
#                 ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())
#                 + (
#                     nn.Linear(self.att_feat_size, self.rnn_size),
#                     nn.ReLU(),
#                     nn.Dropout(self.drop_prob_lm),
#                 )
#                 + ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())
#             )
#         )

#     def clip_att(self, att_feats, att_masks):
#         # Clip the length of att_masks and att_feats to the maximum length
#         if att_masks is not None:
#             max_len = att_masks.data.long().sum(1).max()
#             att_feats = att_feats[:, :max_len].contiguous()
#             att_masks = att_masks[:, :max_len].contiguous()
#         return att_feats, att_masks

#     def init_hidden(self, bsz):
#         weight = next(self.parameters())
#         return (
#             weight.new_zeros(self.num_layers, bsz, self.rnn_size),
#             weight.new_zeros(self.num_layers, bsz, self.rnn_size),
#         )

#     def forward(self, fc_feats, att_feats, seq, att_masks=None):
#         batch_size = fc_feats.size(0)
#         state = self.init_hidden(batch_size)

#         outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)

#         # Prepare the features
#         p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self.prepare_feature(
#             fc_feats, att_feats, att_masks
#         )
#         # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

#         for i in range(seq.size(1) - 1):
#             if (
#                 self.training and i >= 1 and self.ss_prob > 0.0
#             ):  # otherwiste no need to sample
#                 sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
#                 sample_mask = sample_prob < self.ss_prob
#                 if sample_mask.sum() == 0:
#                     it = seq[:, i].clone()
#                 else:
#                     sample_ind = sample_mask.nonzero().view(-1)
#                     it = seq[:, i].data.clone()
#                     # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
#                     # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
#                     # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
#                     prob_prev = torch.exp(
#                         outputs[:, i - 1].detach()
#                     )  # fetch prev distribution: shape Nx(M+1)
#                     it.index_copy_(
#                         0,
#                         sample_ind,
#                         torch.multinomial(prob_prev, 1)
#                         .view(-1)
#                         .index_select(0, sample_ind),
#                     )
#             else:
#                 it = seq[:, i].clone()
#             # break if all the sequences end
#             if i >= 1 and seq[:, i].sum() == 0:
#                 break

#             output, state = self.get_logprobs_state(
#                 it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state
#             )
#             outputs[:, i] = output

#         return outputs

#     def get_logprobs_state(
#         self, it, fc_feats, att_feats, p_att_feats, att_masks, state
#     ):
#         # 'it' contains a word index
#         xt = self.embed(it)

#         output, state = self.core(
#             xt, fc_feats, att_feats, p_att_feats, state, att_masks
#         )
#         logprobs = F.log_softmax(self.logit(output), dim=1)

#         return logprobs, state
