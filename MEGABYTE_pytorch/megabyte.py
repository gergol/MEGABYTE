import math
import functools
from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

# from beartype import beartype
# from beartype.typing import Tuple, Union, List
from typing import Tuple, Union, List

from MEGABYTE_pytorch.attend import Attend

from tqdm import tqdm

# helpers


def inspect_shapes(prefix, **tensors):
    if False:
        shapes = [f"{k}={tuple(x for x in v.shape)}" for k, v in tensors.items()]
        print(f"inspect_shapes {prefix}: {shapes}")


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def reduce_mult(nums):
    return functools.reduce(lambda x, y: x * y, nums, 1)


# tensor helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


# token shift, from Peng et al of RWKV


def token_shift(t):
    t, t_shift = t.chunk(2, dim=-1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim=-1)


# rotary positional embedding


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


# norm


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# helper classes


def FeedForward(*, dim, mult=4, dropout=0.0):
    return nn.Sequential(
        RMSNorm(dim), nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim)
    )


class Attention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, dropout=0.0, flash=False, is_cross_attention=False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.is_cross_attention = is_cross_attention
        inner_dim = dim_head * heads

        self.attend = Attend(causal=not is_cross_attention, flash=flash, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, rotary_emb=None, encoder_hidden_states=None):
        assert self.is_cross_attention == (encoder_hidden_states is not None)
        h, device = self.heads, x.device
        x = self.norm(x)
        if self.is_cross_attention:
            q, k, v = (self.to_q(x), *self.to_kv(encoder_hidden_states).chunk(2, dim=-1))
        else:
            q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        # inspect_shapes("before attend: ", q=q, k=k, v=v)
        out = self.attend(q, k, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        layers,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        rel_pos=True,
        flash_attn=False,
        has_cross_attention=False,
    ):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim_head) if rel_pos else None
        self.layers = nn.ModuleList([])

        for _ in range(layers):
            ll: List[nn.Module] = [
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn),
            ]
            if has_cross_attention:
                ll.append(
                    Attention(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        dropout=attn_dropout,
                        flash=flash_attn,
                        is_cross_attention=True,
                    ),
                )
            ll.append(FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout))
            self.layers.append(nn.ModuleList(ll))

        self.norm = RMSNorm(dim)
        self.has_cross_attention = has_cross_attention

    def forward(self, x, encoder_hidden_states=None):
        inspect_shapes("Transformer", x=x)
        n = x.shape[-2]
        rotary_emb = self.rotary_emb(n) if exists(self.rotary_emb) else None

        if self.has_cross_attention:
            for attn, cross_attn, ff in self.layers:
                x = attn(token_shift(x), rotary_emb=rotary_emb) + x
                inspect_shapes("Transformer post attn", x=x)
                x = cross_attn(token_shift(x), rotary_emb=rotary_emb, encoder_hidden_states=encoder_hidden_states) + x
                inspect_shapes("Transformer post cross attn", x=x)
                x = ff(token_shift(x)) + x
                inspect_shapes("Transformer post ff", x=x)
        else:
            for attn, ff in self.layers:
                x = attn(token_shift(x), rotary_emb=rotary_emb) + x
                inspect_shapes("Transformer post attn", x=x)
                x = ff(token_shift(x)) + x
                inspect_shapes("Transformer post ff", x=x)

        return self.norm(x)


# main class


class MEGABYTE(nn.Module):

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_sizes: List[int],
        num_hidden_layers: List[int],
        max_sequence_lengths: List[int],
        dim_head: int = 64,
        num_heads: int = 8,
        attention_dropout_prob: float = 0.1,
        feed_forward_scaleup: int = 4,
        feed_forward_dropout_prob: float = 0.0,
        pad_token_id: int = 0,
        rel_pos: bool = False,
        pos_emb: bool = False,
        flash_attn: bool = False,
        add_cross_attention: bool = False,
    ):
        super().__init__()

        # simplified configuration for each stage of the hierarchy
        # depth = (2, 2, 4) would translate to depth 2 at first stage, depth 2 second stage, depth 4 third
        # max_seq_len = (16, 8, 4) would translate to max sequence length of 16 at first stage, length of 8 at second stage, length of 4 for last

        # assert isinstance(num_hidden_layers, tuple) and isinstance(max_sequence_lengths, tuple)
        assert len(num_hidden_layers) == len(max_sequence_lengths)

        self.stages = len(num_hidden_layers)
        hidden_sizes = cast_tuple(hidden_sizes, self.stages)

        assert len(hidden_sizes) == self.stages

        coarsest_dim, *_, fine_dim = hidden_sizes

        self.max_seq_len = max_sequence_lengths
        self.add_cross_attention = add_cross_attention

        self.start_tokens = nn.ParameterList(
            [nn.Parameter(torch.randn(h_dim)) for h_dim, seq_len in zip(hidden_sizes, max_sequence_lengths)]
        )
        self.pos_embs = (
            nn.ModuleList([nn.Embedding(seq_len, h_dim) for h_dim, seq_len in zip(hidden_sizes, max_sequence_lengths)])
            if pos_emb
            else None
        )

        self.token_embs = nn.ModuleList([])

        patch_size = 1
        self.token_embs.append(nn.Embedding(vocab_size, fine_dim))

        for dim_out, seq_len in zip(reversed(hidden_sizes[:-1]), reversed(max_sequence_lengths[1:])):
            patch_size *= seq_len

            self.token_embs.append(
                nn.Sequential(
                    nn.Embedding(vocab_size, fine_dim),
                    Rearrange("... r d -> ... (r d)"),
                    nn.LayerNorm(patch_size * fine_dim),
                    nn.Linear(patch_size * fine_dim, dim_out),
                    nn.LayerNorm(dim_out),
                )
            )

        self.transformers = nn.ModuleList([])
        self.to_next_transformer_projections = nn.ModuleList([])

        first_layer = True
        for h_dim, next_h_dim, stage_depth, next_seq_len in zip_longest(
            hidden_sizes, hidden_sizes[1:], num_hidden_layers, max_sequence_lengths[1:]
        ):
            self.transformers.append(
                Transformer(
                    dim=h_dim,
                    layers=stage_depth,
                    dim_head=dim_head,
                    heads=num_heads,
                    attn_dropout=attention_dropout_prob,
                    ff_dropout=feed_forward_dropout_prob,
                    ff_mult=feed_forward_scaleup,
                    rel_pos=rel_pos,
                    flash_attn=flash_attn,
                    has_cross_attention=self.add_cross_attention and first_layer,
                )
            )

            proj = nn.Identity()

            if exists(next_h_dim) and next_h_dim != hidden_sizes:
                proj = nn.Sequential(
                    Rearrange("b ... d -> b (...) d"),
                    nn.Linear(h_dim, next_h_dim * next_seq_len),
                    Rearrange("b m (n d) -> (b m) n d", n=next_seq_len),
                )

            self.to_next_transformer_projections.append(proj)
            first_layer = False

        self.lm_head = nn.Linear(fine_dim, vocab_size)
        self.pad_id = pad_token_id

    def generate(self, prime=None, filter_thres=0.9, temperature=1.0, default_batch_size=1, encoder_hidden_states=None):
        total_seq_len = reduce_mult(self.max_seq_len)
        device = next(self.parameters()).device

        if prime is None:
            prime = torch.empty((default_batch_size, 0), dtype=torch.long, device=device)

        seq = prime
        batch = seq.shape[0]

        for _ in tqdm(range(total_seq_len - seq.shape[-1])):
            logits = self.forward(seq, encoder_hidden_states=encoder_hidden_states)[:, -1]
            logits = top_k(logits, thres=filter_thres)
            sampled = gumbel_sample(logits, dim=-1, temperature=temperature)
            seq = torch.cat((seq, rearrange(sampled, "b -> b 1")), dim=-1)

        return seq.reshape(batch, *self.max_seq_len)

    def forward_empty(self, batch_size, encoder_hidden_states=None):
        # take care of special case
        # where you sample from input of 0 (start token only)

        prev_stage_tokens_repr = None
        first_stage = True
        for stage_start_tokens, transformer, proj in zip(
            self.start_tokens, self.transformers, self.to_next_transformer_projections
        ):
            tokens = repeat(stage_start_tokens, "d -> b 1 d", b=batch_size)

            if prev_stage_tokens_repr is not None:
                tokens = tokens + prev_stage_tokens_repr[..., : tokens.shape[-2], :]

            if first_stage and self.add_cross_attention:
                tokens = transformer(tokens, encoder_hidden_states=encoder_hidden_states)
            else:
                tokens = transformer(tokens)
            prev_stage_tokens_repr = proj(tokens)
            first_stage = False

        return self.lm_head(tokens)

    def forward(self, ids, return_loss=False, encoder_hidden_states=None, **kwargs):
        batch = ids.shape[0]

        inspect_shapes("MEGABYTE", ids=ids)
        assert ids.ndim in {2, self.stages + 1}
        assert self.add_cross_attention == (
            encoder_hidden_states is not None
        ), "encoder_hidden_states are expected if and only if self.add_cross_attention == True"

        flattened_dims = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0], encoder_hidden_states=encoder_hidden_states)

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_seq_len[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value=self.pad_id)
            ids = ids.reshape(batch, -1, *self.max_seq_len[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        # check some dimensions

        assert (
            prec_dims[0] <= self.max_seq_len[0]
        ), "the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_seq_len (like any autoregressive transformer)"
        assert tuple(prec_dims[1:]) == tuple(self.max_seq_len[1:]), "all subsequent dimensions must match exactly"

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        pos_embs = default(self.pos_embs, (None,))

        for ind, pos_emb, token_emb in zip_longest(range(len(prec_dims)), pos_embs, self.token_embs):
            is_first = ind == 0

            tokens = token_emb(ids)

            if exists(pos_emb):
                positions = pos_emb(torch.arange(tokens.shape[-2], device=device))
                tokens = tokens + positions

            tokens_at_stages.insert(0, tokens)

            if is_first:
                continue

            ids = rearrange(ids, "... m n -> ... (m n)")

        # the un-pixelshuffled representations of the previous hierarchy, starts with None

        prev_stage_tokens_repr = None

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions
        first_stage = True
        for stage_start_tokens, stage_tokens, transformer, proj in zip(
            self.start_tokens, tokens_at_stages, self.transformers, self.to_next_transformer_projections
        ):
            stage_tokens, ps = pack_one(stage_tokens, "* n d")
            stage_start_tokens = repeat(stage_start_tokens, "f -> b 1 f", b=stage_tokens.shape[0])

            # concat start token

            stage_tokens = torch.cat(
                (
                    stage_start_tokens,
                    stage_tokens,
                ),
                dim=-2,
            )

            # sum the previous hierarchy's representation

            if exists(prev_stage_tokens_repr):
                prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value=0.0)
                stage_tokens = stage_tokens + prev_stage_tokens_repr

            if first_stage and self.add_cross_attention:
                attended = transformer(stage_tokens, encoder_hidden_states=encoder_hidden_states)
            else:
                attended = transformer(stage_tokens)

            attended = unpack_one(attended, ps, "* n d")

            # project for next stage in the hierarchy

            inspect_shapes("MEGABYTE pre proj", x=attended)
            # TODO: the projection output of the last stage is not being used
            # check if this is intended or not? If it is, remover this unnecessary calculation
            prev_stage_tokens_repr = proj(attended[..., :-1, :])
            inspect_shapes("MEGABYTE post proj", x=prev_stage_tokens_repr)
            first_stage = False

        # project to logits

        logits = self.lm_head(attended)
        inspect_shapes("MEGABYTE logits", logits=logits)

        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]
        start_tokens = rearrange(start_tokens, "b d -> b 1 d")

        logits = logits[..., 1:, :]
        inspect_shapes("MEGABYTE logits before rearrange", logits=logits)
        if not return_loss:

            if flattened_dims:
                logits = rearrange(logits, "b ... c -> b (...) c")
                logits = logits[:, :seq_len]

            inspect_shapes("MEGABYTE output logits", logits=logits)
            return logits

        logits = rearrange(logits, "b ... c -> b (...) c")
        inspect_shapes("Pre add start tokens: ", logits=logits)
        inspect_shapes("Start tokens: ", start_tokens=start_tokens)
        # logits = torch.cat((start_tokens, logits), dim=-2)
        inspect_shapes("Post add start tokens: ", logits=logits)

        preds = rearrange(logits, "b n c -> b c n")
        labels = rearrange(ids, "b ... -> b (...)")
        inspect_shapes("MEGABYTE pre loss", preds=preds, labels=labels)
        loss = F.cross_entropy(preds, labels, ignore_index=self.pad_id)
        # loss = F.cross_entropy(preds[..., 1:], labels[..., :-1], ignore_index=self.pad_id)

        # This part of the code comes from the GPT2 implementation in transformers
        # I added this because I wanted to check if their way of shaping labels and preds is different
        # labels = ids.to(logits.device)
        # # Shift so that tokens < n predict n
        # shift_logits = logits.contiguous()
        # shift_labels = labels.contiguous()
        # # shift_logits = logits[..., :-1, :].contiguous()
        # # shift_labels = labels[..., 1:].contiguous()
        # # Flatten the tokens
        # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_id)
        # loss2 = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return logits, loss  # , loss2
