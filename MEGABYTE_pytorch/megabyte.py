import functools
import math
import pprint
import time
from collections.abc import Callable
from itertools import zip_longest

import torch
import torch.nn.functional as F
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple, Union
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from torch import einsum, nn
from tqdm import tqdm

from MEGABYTE_pytorch.attend import Attend

# helpers

DEBUG = False


def dprint(*args, **kwargs):
    if not DEBUG:
        return
    print(*args, **kwargs)


def inspect_shapes(prefix, *, suppress=False, print_values=False, select=lambda x: x, **tensors):
    if not DEBUG:
        return
    if suppress:
        return
    shapes = [f"{k}={tuple(x for x in v.shape)}" for k, v in tensors.items() if v is not None]
    import numpy as np

    np.set_printoptions(precision=2)
    print(f"{prefix}: {shapes}")
    # vals = [f"{k}={tuple(x for x in v.cpu().numpy())}" for k, v in tensors.items()]
    if print_values:
        vals = {k: select(v.cpu().numpy()) for k, v in tensors.items() if v is not None}
    else:
        vals = {k: str(hash(str(select(v.cpu().numpy()))))[-5:] for k, v in tensors.items() if v is not None}
    # vals = []
    # import numpy as np
    # for k, v in tensors.items():
    #     val = ""
    #     for x in v.cpu().numpy():
    #         val += f"{x}"
    #     vals.append(f"{k}={val}")
    pprint.pp((f"values: {prefix}:", vals))
    pass


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


# def token_shift(t, bypass=False, use_cache=False, cache=None):
#     inspect_shapes("TOKENSHIFR INPUT", t=t)
#     if bypass:
#         return t
#     t, t_shift = t.chunk(2, dim=-1)
#     t_shift = F.pad(t_shift, (0, 0, 1, -1))
#     result = torch.cat((t, t_shift), dim=-1)
#     # print(result)
#     if use_cache:
#         return result, None
#     return result


def token_shift(t, use_cache=False, cache=None):
    if not use_cache:
        t, t_shift = t.chunk(2, dim=-1)
        t_shift = F.pad(t_shift, (0, 0, 1, -1))
        return torch.cat((t, t_shift), dim=-1)

    # TODO: the current cache implementation only saves the
    # data of the previous invocation, however in order to fully
    # restore the state at any past time we would need to store the
    # full history. Maybe we want to reconsider this at a later point.
    t, t_shift = t.chunk(2, dim=-1)
    if cache is None:
        cache = t_shift[:, -1:, :].clone()
        t_shift = F.pad(t_shift, (0, 0, 1, -1))
    else:
        t_shift, cache = torch.cat((cache, t_shift[:, :-1, :]), dim=-2), t_shift[:, -1:, :].clone()
    ret = torch.cat((t, t_shift), dim=-1)
    return ret, cache


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


def get_cache(cache: Optional[Dict], key: Union[int, str], init=False):
    if cache is None:
        return None
    if key not in cache:
        if not init:
            return None
        cache[key] = {}
    return cache[key]


def set_cache(
    cache: Optional[Dict], key: Optional[Union[int, str]] = None, value: Optional[torch.Tensor] = None, **entries
):
    if isinstance(cache, dict):
        if key is not None:
            assert value is not None
            cache[key] = value.clone().float()
        for k, v in entries.items():
            cache[k] = v.clone().float()
    return cache


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

    def forward(self, x, rotary_emb=None, encoder_hidden_states=None, use_cache=False, cache=None):
        assert self.is_cross_attention == (encoder_hidden_states is not None)
        h = self.heads
        x = self.norm(x)
        if self.is_cross_attention:
            q, k, v = (self.to_q(x), *self.to_kv(encoder_hidden_states).chunk(2, dim=-1))
        else:
            q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if use_cache:
            assert cache is not None, "You must provide an empty dict as cache for the inital run"
            k_cache = get_cache(cache, "k")
            v_cache = get_cache(cache, "v")
            if k_cache is not None and v_cache is not None:  # and in_cache is not None:
                k = torch.cat((k_cache, k), dim=2)
                v = torch.cat((v_cache, v), dim=2)
            set_cache(cache, k=k, v=v)
            dprint("kv cache: ", cache["v"].shape)
        if exists(rotary_emb):
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))

        out = self.attend(q.float(), k.float(), v.float())
        inspect_shapes("attend direct out", suppress=True, print_values=True, out=out)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), cache


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
        use_old_layout=False,  # this is temporarily necessary because in previous versions the ordering of the cross attention and ff layers in the transformer was different. This is only an issue when trying to load an old checkpoint preceeding this change
    ):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim_head) if rel_pos else None
        self.layers = nn.ModuleList([])
        self.use_old_layout = use_old_layout

        for _ in range(layers):
            ll: List[nn.Module] = [
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn),
            ]
            if use_old_layout and has_cross_attention:
                # in the old layout, the cross attention layers were located at this position
                # so we need to keep this option if we want to load an old layout checkpoint
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
            if not use_old_layout and has_cross_attention:
                # now, the cross attention is here, so that we can initialize the model from checkpoints
                # that do not have cross attention layers at all...
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
            self.layers.append(nn.ModuleList(ll))

        self.norm = RMSNorm(dim)
        self.has_cross_attention = has_cross_attention

    def forward(self, x, encoder_hidden_states=None, use_cache=False, cache=None, debug=False):
        assert not use_cache
        assert cache is None
        inspect_shapes("Transformer", suppress=(not debug), print_values=True, x=x)
        n = x.shape[-2]
        rotary_emb = self.rotary_emb(n) if exists(self.rotary_emb) else None  # type: ignore

        if use_cache and cache is None:
            # cache = [{}] * len(self.layers)
            cache = {}
        if self.has_cross_attention:

            for layer_idx, (attn, ff, cross_attn) in enumerate(self.layers):  # type: ignore
                if self.use_old_layout:
                    cross_attn, ff = ff, cross_attn  # swap the variable names to match the old layout if needed

                layer_cache = get_cache(cache, layer_idx, init=True)
                shifted, shift_cache = token_shift(x, use_cache=True, cache=get_cache(layer_cache, "tok_shift_0"))
                set_cache(layer_cache, "tok_shift_0", shift_cache)
                attended, layer_cache = attn(
                    shifted,
                    rotary_emb=rotary_emb,
                    use_cache=use_cache,
                    cache=layer_cache,
                    # x, rotary_emb=rotary_emb, use_cache=use_cache, cache=layer_cache
                )
                inspect_shapes("attn out", suppress=(not debug), print_values=True, attended=attended)
                x = attended + x

                shifted, shift_cache = token_shift(x, use_cache=True, cache=get_cache(layer_cache, "tok_shift_1"))
                set_cache(layer_cache, "tok_shift_1", shift_cache)

                x = (
                    cross_attn(
                        shifted,
                        rotary_emb=rotary_emb,
                        encoder_hidden_states=encoder_hidden_states,
                    )[0]
                    + x
                )
                shifted, shift_cache = token_shift(x, use_cache=True, cache=get_cache(layer_cache, "tok_shift_2"))
                set_cache(layer_cache, "tok_shift_2", shift_cache)
                x = ff(shifted) + x
                if cache is not None:
                    cache[layer_idx] = layer_cache
        else:
            for layer_idx, (attn, ff) in enumerate(self.layers):  # type: ignore
                layer_cache = get_cache(cache, layer_idx, init=True)
                shifted, shift_cache = token_shift(x, use_cache=True, cache=get_cache(layer_cache, "tok_shift_0"))
                set_cache(layer_cache, "tok_shift_0", shift_cache)
                attended, layer_cache = attn(
                    shifted,
                    rotary_emb=rotary_emb,
                    use_cache=use_cache,
                    cache=layer_cache,
                    # x, rotary_emb=rotary_emb, use_cache=use_cache, cache=layer_cache
                )
                inspect_shapes("attn out", suppress=(not debug), print_values=True, attended=attended)
                x = attended + x

                shifted, shift_cache = token_shift(x, use_cache=True, cache=get_cache(layer_cache, "tok_shift_1"))
                set_cache(layer_cache, "tok_shift_1", shift_cache)
                x = ff(shifted) + x
                if cache is not None:
                    cache[layer_idx] = layer_cache
        inspect_shapes("transofrmer out pre norm", suppress=(not debug), print_values=True, x=x)
        pre_norm_history = get_cache(cache, "pre_norm", init=False)
        orig_shape = x.shape
        if pre_norm_history is not None:
            norm_in = torch.cat((pre_norm_history, x), dim=-2)
        else:
            norm_in = x
        x = self.norm(norm_in)
        inspect_shapes("norm", suppress=(not debug), print_values=True, norm_in=norm_in, norm_out=x)
        set_cache(cache, pre_norm=norm_in)
        # if x.shape != orig_shape:
        #     keep = orig_shape[-2]
        #     x = x[:, -keep:, :]
        inspect_shapes("transofrmer out post norm", suppress=(not debug), print_values=True, x=x)
        return x, cache


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
        eos_token_id: int = None,
        bos_token_id: int = None,
        rel_pos: bool = False,
        pos_emb: bool = False,
        flash_attn: bool = False,
        add_cross_attention: bool = False,
        use_old_layout: bool = False,  # this is temporarily necessary because in previous versions the ordering of the cross attention and ff layers in the transformer was different. This is only an issue when trying to load an old checkpoint preceeding this change
        criterion: Callable[[torch.Tensor, torch.Tensor, ...], float] = F.cross_entropy,
    ):
        super().__init__()

        # simplified configuration for each stage of the hierarchy
        # depth = (2, 2, 4) would translate to depth 2 at first stage, depth 2 second stage, depth 4 third
        # max_sequence_lengths = (16, 8, 4) would translate to max sequence length of 16 at first stage, length of 8 at second stage, length of 4 for last

        assert len(num_hidden_layers) == len(max_sequence_lengths)

        self.stages = len(num_hidden_layers)
        hidden_sizes = cast_tuple(hidden_sizes, self.stages)
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) == self.stages

        coarsest_dim, *_, fine_dim = hidden_sizes

        self.max_sequence_lengths = max_sequence_lengths
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
                    use_old_layout=use_old_layout,
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

        self.to_logits = nn.Linear(fine_dim, vocab_size)
        self.pad_token_id = pad_token_id
        self.criterion = criterion

    def generate(self, prime=None, filter_thres=0.9, temperature=1.0, default_batch_size=1):
        total_seq_len = reduce_mult(self.max_sequence_lengths)
        device = next(self.parameters()).device

        if prime is None:
            prime = torch.empty((default_batch_size, 0), dtype=torch.long, device=device)

        seq = prime
        batch = seq.shape[0]

        for _ in tqdm(range(total_seq_len - seq.shape[-1])):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres=filter_thres)
            sampled = gumbel_sample(logits, dim=-1, temperature=temperature)
            seq = torch.cat((seq, rearrange(sampled, "b -> b 1")), dim=-1)

        return seq.reshape(batch, *self.max_sequence_lengths)

    @property
    def depth(self):
        return len(self.max_sequence_lengths)

    def forward_empty(self, batch_size, encoder_hidden_states=None, use_cache=False, cache=None, use_kv_cache=None):
        # take care of special case
        # where you sample from input of 0 (start token only)
        if use_kv_cache is None:
            use_kv_cache = use_cache
        prev_stage_tokens_repr = None

        is_first_stage = True
        if use_cache:
            assert cache is not None
            assert len(cache["kv"]) == len(self.transformers), "kv cache has incorrect size"
            assert cache["hidden_states"] is not None

        for stage_idx, stage_start_tokens, transformer, proj in zip(
            range(self.depth), self.start_tokens, self.transformers, self.to_next_transformer_projections
        ):
            tokens = repeat(stage_start_tokens, "d -> b 1 d", b=batch_size)

            if prev_stage_tokens_repr is not None:
                tokens = tokens + prev_stage_tokens_repr[..., : tokens.shape[-2], :]

            stage_cache = cache["kv"][stage_idx] if cache else None
            inspect_shapes(f"forward_empty TF input stage {stage_idx}", print_values=True, tokens=tokens)
            if use_kv_cache:
                if is_first_stage:
                    tokens, stage_cache = transformer(
                        tokens, encoder_hidden_states=encoder_hidden_states, use_cache=use_cache, cache=stage_cache
                    )
                    is_first_stage = False
                else:
                    tokens, stage_cache = transformer(tokens, use_cache=use_cache, cache=stage_cache)
            else:
                if is_first_stage:
                    tokens, _ = transformer(
                        tokens, encoder_hidden_states=encoder_hidden_states, use_cache=False, cache=None
                    )
                    is_first_stage = False
                else:
                    tokens, _ = transformer(tokens, use_cache=False, cache=None)
            prev_stage_tokens_repr = proj(tokens)
            if use_kv_cache:
                inspect_shapes(f"out (forward_empty) KV of stage {stage_idx}", k=stage_cache[0]["k"])
                cache["kv"][stage_idx] = stage_cache
            if use_cache and stage_idx < self.depth - 1:
                cache["hidden_states"][stage_idx] = prev_stage_tokens_repr.float()

        if use_cache:
            return self.to_logits(tokens), cache  # type: ignore
        return self.to_logits(tokens)  # type: ignore

    def forward(
        self,
        ids,
        return_loss=False,
        encoder_hidden_states=None,
        return_preds_and_labels=False,
        use_cache=False,
        cache: Optional[Dict] = None,
        profile: bool = False,
    ):
        dprint("input ids: ", ids)
        batch = ids.shape[0]
        N = ids.shape[1]

        assert use_cache or cache is None, "You must not provide a cache when use_cache=False"
        # print("\n\nCALLING MEGABYTE FORWARD", ids.shape)
        # inspect_shapes("MEGABYTE", ids=ids)
        assert ids.ndim in {2, self.stages + 1}
        assert self.add_cross_attention == (
            encoder_hidden_states is not None
        ), "encoder_hidden_states are expected if and only if self.add_cross_attention == True"

        assert not use_cache or self.depth == 2, "cache is only implemented for two-layer MEGABYTE models"
        assert (
            not use_cache or len(ids.shape) == 2 and batch == 1
        ), "caching is curently only supported for batch size == 1"

        flattened_dims = ids.ndim == 2

        if use_cache and cache is None:
            cache = {}
            cache["kv"] = [None] * len(self.transformers)
            cache["hidden_states"] = [None] * (self.depth - 1)
            if profile:
                cache["profile"] = [[] for _ in range(self.depth)]

        do_profile = cache is not None and "profile" in cache
        if ids.numel() == 0:
            return self.forward_empty(
                ids.shape[0], encoder_hidden_states=encoder_hidden_states, use_cache=use_cache, cache=cache
            )

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_sequence_lengths[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value=self.pad_token_id)
            ids = ids.reshape(batch, -1, *self.max_sequence_lengths[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        # check some dimensions

        assert (
            prec_dims[0] <= self.max_sequence_lengths[0]
        ), "the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_sequence_lengths (like any autoregressive transformer)"
        assert tuple(prec_dims[1:]) == tuple(
            self.max_sequence_lengths[1:]
        ), "all subsequent dimensions must match exactly"

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        pos_embs = default(self.pos_embs, (None,))

        for stage_idx, pos_emb, token_emb in zip_longest(range(len(prec_dims)), pos_embs, self.token_embs):
            is_first = stage_idx == 0
            # inspect_shapes(f"stage {stage_idx} pre token emb", tokens=ids)

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
        for stage_idx, stage_start_tokens, stage_tokens, transformer, proj in zip(
            range(self.depth),
            self.start_tokens,
            tokens_at_stages,
            self.transformers,
            self.to_next_transformer_projections,
        ):
            inspect_shapes(
                f"STAGE {stage_idx}", print_values=False, stage_tokens=stage_tokens, start_tokens=stage_start_tokens
            )
            # print(stage_tokens)
            if do_profile:
                start_time = time.time()
            if use_cache and stage_idx < len(cache["hidden_states"]):
                hs = cache["hidden_states"][stage_idx]
                # for networks with higer depth, we need to change this by the product of the subsequent layers
                scale_factor = self.max_sequence_lengths[1]
                if hs is not None and hs.shape[0] * scale_factor >= ids.shape[-1]:
                    # we have cached values for the current step, so we can skip that forward pass
                    prev_stage_tokens_repr = hs
                    if do_profile:
                        cache["profile"][stage_idx].append(time.time() - start_time)
                    continue
                else:
                    dprint("NOT USING HS CACHE")
                # if hs is not None:

            stage_tokens, ps = pack_one(stage_tokens, "* n d")
            stage_start_tokens = repeat(stage_start_tokens, "f -> b 1 f", b=stage_tokens.shape[0])
            inspect_shapes(
                "after pack",
                print_values=True,
                select=lambda x: x[-1],
                stage_tokens=stage_tokens,
                start_tokens=stage_start_tokens,
                prev_stage_tokens_repr=prev_stage_tokens_repr,
            )

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

            stage_cache = cache["kv"][stage_idx] if cache else None
            inspect_shapes(f"stage tokens to transormer {stage_idx}", stage_tokens=stage_tokens[-1:])
            # if DO_HACK:
            #     # TODO this only works for batch size 1 and only during inference without prompt
            #     stage_tokens = stage_tokens[-1].unsqueeze(0)
            if first_stage and self.add_cross_attention:
                attended, stage_cache = transformer(
                    stage_tokens, encoder_hidden_states=encoder_hidden_states, use_cache=use_cache, cache=stage_cache
                )
            else:
                attended, stage_cache = transformer(stage_tokens, use_cache=use_cache, cache=stage_cache, debug=False)
            # inspect_shapes(f"attention output stage {stage_idx}", attended=attended)
            # print("before unpacking: ps: ", ps)
            attended = unpack_one(attended, ps, "* n d")
            # inspect_shapes(f"attention UNPACKED output stage {stage_idx}", attended_unpacked=attended)

            # project for next stage in the hierarchy

            inspect_shapes("to_next_layer_proj", print_values=True, to_next_layer=attended[..., :-1, :])
            prev_stage_tokens_repr = proj(attended[..., :-1, :])

            # if not self.training:
            #     print("BAMMM")
            #     # TODO: this is for testing only as it will break handling longer input prompts
            #     # it will onlhy work for token by token inference
            #     prev_stage_tokens_repr = prev_stage_tokens_repr[-1]

            # inspect_shapes(f"attention PROJECTED output stage {stage_idx}", proj=prev_stage_tokens_repr)
            if use_cache:
                cache["kv"][stage_idx] = stage_cache
            if use_cache and stage_idx < self.depth - 1:
                cache["hidden_states"][stage_idx] = prev_stage_tokens_repr.detach().clone()
            first_stage = False
            if do_profile:
                cache["profile"][stage_idx].append(time.time() - start_time)

        # project to logits

        logits = self.to_logits(attended)

        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]
        start_tokens = rearrange(start_tokens, "b d -> b 1 d")

        logits = logits[..., 1:, :]

        if not return_loss:

            if flattened_dims:
                logits = rearrange(logits, "b ... c -> b (...) c")
                logits = logits[:, :seq_len]

            inspect_shapes("output", logits=logits.round(decimals=2))
            if use_cache:
                return logits, cache
            return logits

        logits = rearrange(logits, "b ... c -> b (...) c")
        logits = torch.cat((start_tokens, logits), dim=-2)

        preds = rearrange(logits, "b n c -> b c n")
        labels = rearrange(ids, "b ... -> b (...)")

        loss = self.criterion(preds[..., :-1], labels, ignore_index=self.pad_token_id)

        if return_preds_and_labels:
            return loss, preds, labels
        return loss

    def forward_inference_prompt(
        self,
        ids,
        return_loss=False,
        encoder_hidden_states=None,
        return_preds_and_labels=False,
        use_cache=False,
        cache: Optional[Dict] = None,
        profile: bool = False,
        streaming=True,
    ):
        batch = ids.shape[0]
        N = ids.shape[1]

        assert use_cache or cache is None, "You must not provide a cache when use_cache=False"
        # print("\n\nCALLING MEGABYTE FORWARD", ids.shape)
        # inspect_shapes("MEGABYTE", ids=ids)
        assert ids.ndim in {2, self.stages + 1}
        assert self.add_cross_attention == (
            encoder_hidden_states is not None
        ), "encoder_hidden_states are expected if and only if self.add_cross_attention == True"

        assert not use_cache or self.depth == 2, "cache is only implemented for two-layer MEGABYTE models"
        assert (
            not use_cache or len(ids.shape) == 2 and batch == 1
        ), "caching is curently only supported for batch size == 1"

        flattened_dims = ids.ndim == 2

        # if we are streaming and it's not the first token / run
        is_streaming_continued = streaming and cache is not None

        if use_cache and cache is None:
            cache = {}
            cache["kv"] = [None] * len(self.transformers)
            cache["hidden_states"] = [None] * (self.depth - 1)
            if profile:
                cache["profile"] = [[] for _ in range(self.depth)]

        do_profile = cache is not None and "profile" in cache
        if ids.numel() == 0:
            return self.forward_empty(
                ids.shape[0], encoder_hidden_states=encoder_hidden_states, use_cache=use_cache, cache=cache
            )

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_sequence_lengths[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value=self.pad_token_id)
            ids = ids.reshape(batch, -1, *self.max_sequence_lengths[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        # check some dimensions

        assert (
            prec_dims[0] <= self.max_sequence_lengths[0]
        ), "the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_sequence_lengths (like any autoregressive transformer)"
        assert tuple(prec_dims[1:]) == tuple(
            self.max_sequence_lengths[1:]
        ), "all subsequent dimensions must match exactly"

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        pos_embs = default(self.pos_embs, (None,))

        for stage_idx, pos_emb, token_emb in zip_longest(range(len(prec_dims)), pos_embs, self.token_embs):
            is_first = stage_idx == 0

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
        for stage_idx, stage_start_tokens, stage_tokens, transformer, proj in zip(
            range(self.depth),
            self.start_tokens,
            tokens_at_stages,
            self.transformers,
            self.to_next_transformer_projections,
        ):
            if do_profile:
                start_time = time.time()
            # if use_cache and stage_idx < len(cache["hidden_states"]):
            #     hs = cache["hidden_states"][stage_idx]
            #     # for networks with higer depth, we need to change this by the product of the subsequent layers
            #     scale_factor = self.max_sequence_lengths[stage_idx + 1]
            #     # if hs is not None and hs.shape[0] * scale_factor >= ids.shape[-1]:
            #     if hs is not None and (tok_idx_in_seq + 1) % scale_factor == 0:
            #         # we have cached values for the current step, so we can skip that forward pass
            #         prev_stage_tokens_repr = hs
            #         if do_profile:
            #             cache["profile"][stage_idx].append(time.time() - start_time)
            #         continue
            #     print("NOT USING HS CACHE")
            #     # if hs is not None:

            stage_tokens, ps = pack_one(stage_tokens, "* n d")

            # if not is_streaming_continued:
            stage_start_tokens = repeat(stage_start_tokens, "f -> b 1 f", b=stage_tokens.shape[0])

            # concat start token
            stage_tokens = torch.cat(
                (
                    stage_start_tokens,
                    stage_tokens,
                ),
                dim=-2,
            )
            dprint("stage_tokens", stage_tokens)

            # sum the previous hierarchy's representation
            if exists(prev_stage_tokens_repr):
                prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value=0.0)
                stage_tokens = stage_tokens + prev_stage_tokens_repr

            stage_cache = cache["kv"][stage_idx] if cache else None
            # inspect_shapes(f"stage tokens to transormer {stage_idx}", stage_tokens=stage_tokens)
            # if DO_HACK:
            #     # TODO this only works for batch size 1 and only during inference without prompt
            #     stage_tokens = stage_tokens[-1].unsqueeze(0)

            if first_stage and self.add_cross_attention:
                attended, stage_cache = transformer(
                    stage_tokens,
                    encoder_hidden_states=encoder_hidden_states,
                    use_cache=False,
                )
            else:
                attended, stage_cache = transformer(
                    stage_tokens,
                    use_cache=False,
                )
            # inspect_shapes(f"attention output stage {stage_idx}", attended=attended)
            # print("before unpacking: ps: ", ps)
            attended = unpack_one(attended, ps, "* n d")
            # inspect_shapes(f"attention UNPACKED output stage {stage_idx}", attended_unpacked=attended)

            # project for next stage in the hierarchy

            prev_stage_tokens_repr = proj(attended[..., :-1, :])

            # if not self.training:
            #     print("BAMMM")
            #     # TODO: this is for testing only as it will break handling longer input prompts
            #     # it will onlhy work for token by token inference
            #     prev_stage_tokens_repr = prev_stage_tokens_repr[-1]

            # inspect_shapes(f"attention PROJECTED output stage {stage_idx}", proj=prev_stage_tokens_repr)
            if use_cache and first_stage:
                # inspect_shapes(f"out KV of stage {stage_idx}", k=stage_cache[0]["k"])
                cache["kv"][stage_idx] = stage_cache
            if use_cache and stage_idx < self.depth - 1:
                cache["hidden_states"][stage_idx] = prev_stage_tokens_repr.detach().clone()
                # inspect_shapes(f"out HS of stage {stage_idx}", hs=cache["hidden_states"][stage_idx])
            first_stage = False
            if do_profile:
                cache["profile"][stage_idx].append(time.time() - start_time)

        # project to logits

        logits = self.to_logits(attended)

        start_tokens = logits[(slice(None), *((0,) * (logits.ndim - 2)), slice(None))]
        start_tokens = rearrange(start_tokens, "b d -> b 1 d")

        logits = logits[..., 1:, :]

        if not return_loss:

            if flattened_dims:
                logits = rearrange(logits, "b ... c -> b (...) c")
                logits = logits[:, :seq_len]

            if use_cache:
                return logits, cache
            return logits

        logits = rearrange(logits, "b ... c -> b (...) c")
        logits = torch.cat((start_tokens, logits), dim=-2)

        preds = rearrange(logits, "b n c -> b c n")
        labels = rearrange(ids, "b ... -> b (...)")

        loss = self.criterion(preds[..., :-1], labels, ignore_index=self.pad_token_id)

        if return_preds_and_labels:
            return loss, preds, labels
        return loss

    def embed_tokens(self, ids, prompt=False):

        batch = ids.shape[0]
        flattened_dims = ids.ndim == 2

        if flattened_dims:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            multiple_of = reduce_mult(self.max_sequence_lengths[1:])
            padding = remainder_to_mult(seq_len, multiple_of)
            ids = F.pad(ids, (0, padding), value=self.pad_token_id)
            ids = ids.reshape(batch, -1, *self.max_sequence_lengths[1:])

        b, *prec_dims, device = *ids.shape, ids.device

        assert (
            prec_dims[0] <= self.max_sequence_lengths[0]
        ), "the first dimension of your axial autoregressive transformer must be less than the first tuple element of max_sequence_lengths (like any autoregressive transformer)"

        assert tuple(prec_dims[1:]) == tuple(
            self.max_sequence_lengths[1:]
        ), "all subsequent dimensions must match exactly"

        # get tokens for all hierarchical stages, reducing by appropriate dimensions
        # and adding the absolute positional embeddings

        tokens_at_stages = []
        pos_embs = default(self.pos_embs, (None,))

        for stage_idx_from_back, pos_emb, token_emb in zip_longest(range(len(prec_dims)), pos_embs, self.token_embs):  # type: ignore
            # Do tho the structure of the algorithm, stage indexes here are actually
            # reversed (see below, where the tokens_at_stages.insert(0, tokens) inserts everything in the reverse order)

            is_last_stage = stage_idx_from_back == 0
            stage_idx = self.depth - stage_idx_from_back - 1

            stage_ids = ids

            if prompt:
                # If we're processing a prompt we need to keep all COMPLETE patches for the
                # initial stage, but only the last one for the fine (second) stage.
                # This is only implemented for depth == 2 yet.
                assert flattened_dims, "Prompts need to be passed in 2 dimensions: (B, N)"
                assert self.depth == 2, "Currently only two-stage models are supported"
                last_frame_is_incomplete = seq_len % self.max_sequence_lengths[-1] != 0  # type: ignore
                if last_frame_is_incomplete and stage_idx == 0:
                    # print("last frame is incomplete")
                    # cut the incomplete final patch out for the first layer
                    stage_ids = stage_ids[:, :-1, :]
                if stage_idx != 0:
                    # for the second stage we're only interested in the last frame,
                    # as all the others are already in the past
                    stage_ids = stage_ids[:, -1:, :]
                    # if not last_frame_is_incomplete:
                    #     stage_ids = torch.full_like(stage_ids, fill_value=self.pad_token_id)
            inspect_shapes(f"stage {stage_idx} pre token emb", print_values=True, tokens=ids, stage_ids=stage_ids)

            tokens = token_emb(stage_ids)

            if exists(pos_emb):
                positions = pos_emb(torch.arange(tokens.shape[-2], device=device))
                tokens = tokens + positions

            tokens_at_stages.insert(0, tokens)

            if is_last_stage:
                continue

            ids = rearrange(ids, "... m n -> ... (m n)")
        return tokens_at_stages

    def forward_inference(
        self,
        ids,
        encoder_hidden_states=None,
        use_cache=False,
        cache: Optional[Dict] = None,
        profile: bool = False,
    ):
        dprint("input ids: ", ids)
        batch = ids.shape[0]
        seq_len = ids.shape[-1]

        run_prompt = seq_len > 0 and cache is None
        logits, cache = self._forward_inference_impl(
            ids=ids, encoder_hidden_states=encoder_hidden_states, use_cache=use_cache, cache=cache, profile=profile
        )
        if run_prompt and seq_len % self.max_sequence_lengths[-1] == 0:

            logits, cache = self._forward_inference_impl(
                ids=ids, encoder_hidden_states=encoder_hidden_states, use_cache=use_cache, cache=cache, profile=profile
            )

        return logits, cache

    def _forward_inference_impl(
        self,
        ids,
        encoder_hidden_states=None,
        use_cache=False,
        cache: Optional[Dict] = None,
        profile: bool = False,
    ):
        dprint("input ids: ", ids)
        batch = ids.shape[0]
        seq_len = ids.shape[-1]

        # print("\n\nCALLING MEGABYTE FORWARD", ids.shape)
        # inspect_shapes("MEGABYTE", ids=ids)
        assert ids.ndim in {2, self.stages + 1}
        assert self.add_cross_attention == (
            encoder_hidden_states is not None
        ), "encoder_hidden_states are expected if and only if self.add_cross_attention == True"

        assert not use_cache or self.depth == 2, "cache is only implemented for two-layer MEGABYTE models"
        # assert (
        #     not use_cache or len(ids.shape) == 2 and batch == 1
        # ), "caching is curently only supported for batch size == 1"
        assert self.pos_embs is None, "not yet implemented for models with positional embeddings"
        assert use_cache or cache is None, "You must not provide a cache when use_cache=False"

        flattened_dims = ids.ndim == 2

        run_prompt = seq_len > 0 and cache is None

        if use_cache and cache is None:
            cache = {}
            cache["kv"] = [None] * self.depth
            cache["hidden_states"] = [None] * (self.depth - 1)
            if profile:
                cache["profile"] = [[] for _ in range(self.depth)]

        if ids.numel() == 0:
            return self.forward_empty(
                ids.shape[0],
                encoder_hidden_states=encoder_hidden_states,
                use_cache=use_cache,
                cache=cache,
                use_kv_cache=False,
            )

        tok_idx_in_seq = seq_len
        # assert batch == 1, "currnelyt only batch size 1 supported"
        assert cache is not None

        embedded_tokens = self.embed_tokens(ids, prompt=run_prompt)

        inspect_shapes("embedded tokens", print_values=True, stage_0=embedded_tokens[0], stage_1=embedded_tokens[1])

        period_0 = self.max_sequence_lengths[-1]
        if run_prompt or tok_idx_in_seq % period_0 == 0:
            dprint("RUNNING INFERENCE ON LAYER 0")
            stage_tokens = embedded_tokens[0]
            _, cache = self.forward_stage(
                0,
                stage_tokens=stage_tokens,
                tok_idx_in_seq=tok_idx_in_seq,
                cache=cache,
                profile=profile,
                run_full_sequence_attention=True,
                encoder_hidden_states=encoder_hidden_states,
                is_complete_window_prompt_on_stage0=run_prompt and seq_len % self.max_sequence_lengths[-1] == 0,
            )

        prev_stage_tokens_repr = get_cache(cache, "prev_stage_tokens_repr", init=False)
        if prev_stage_tokens_repr is None:
            prev_stage_tokens_repr = cache["hidden_states"][0]

        dprint("RUNNING INFERENCE ON LAYER 1")
        stage_tokens = embedded_tokens[1]
        # these will contain the whole sequence windowed into B, N, n, D where n is the stage context size and N is the number of
        # windows of size n the sequence is splitted in.
        # During token-by-token inference we are only interested in the newest window
        stage_tokens = stage_tokens[:, -1, :, :]
        attended, cache = self.forward_stage(1, stage_tokens, prev_stage_tokens_repr=prev_stage_tokens_repr, profile=profile, cache=cache, run_full_sequence_attention=True)  # type: ignore[]

        set_cache(cache, "prev_stage_tokens_repr", cache["hidden_states"][0])

        logits = self.to_logits(attended)

        logits_idx = ((tok_idx_in_seq - 1) % self.max_sequence_lengths[-1]) + 1
        inspect_shapes("output raw", print_values=False, logits=logits.round(decimals=2), attended=attended)
        logits = logits[..., logits_idx : logits_idx + 1, :]
        if flattened_dims:
            logits = rearrange(logits, "b ... c -> b (...) c")
            logits = logits[:, :seq_len]
        inspect_shapes("output", logits=logits.round(decimals=2))

        return logits, cache

    def forward_stage(
        self,
        stage_idx,
        stage_tokens,
        encoder_hidden_states=None,
        cache: Optional[Dict] = None,
        tok_idx_in_seq: int = 0,
        prev_stage_tokens_repr=None,
        profile=False,
        run_full_sequence_attention=False,
        is_complete_window_prompt_on_stage0=False,
    ):
        assert cache is not None
        use_cache = not run_full_sequence_attention
        # assert stage_tokens.shape[0] == 1, "we shuld only have batch size 1 here"
        assert stage_tokens.ndim == 3
        if profile:
            start_time = time.time()

        context_size = reduce_mult(self.max_sequence_lengths[stage_idx:])
        # The active token is the one we want to process, i.e. the last incoming token
        # that we want to attend to. Earlier token attention kv's will come from cache
        active_token_idx = tok_idx_in_seq % context_size
        dprint(f"stage {stage_idx} context size: {context_size}, active token: {active_token_idx}")
        inspect_shapes(f"STAGE INPUT TOKOENS {stage_idx}", stage_tokens=stage_tokens)

        if active_token_idx == 0 or run_full_sequence_attention:
            # if we start a new context window, we need to prepend the start tokens
            stage_start_tokens = self.start_tokens[stage_idx]
            stage_start_tokens = repeat(stage_start_tokens, "f -> b 1 f", b=stage_tokens.shape[0])
            inspect_shapes(
                f"STAGE {stage_idx}", print_values=False, stage_tokens=stage_tokens, start_tokens=stage_start_tokens
            )
            inspect_shapes(
                "after pack",
                print_values=True,
                stage_tokens=stage_tokens,
                start_tokens=stage_start_tokens,
                prev_stage_tokens_repr=prev_stage_tokens_repr,
            )
            # concat start token
            stage_tokens = torch.cat(
                (
                    stage_start_tokens,
                    stage_tokens,
                ),
                dim=-2,
            )

            # also, if we start a new context window, we need to clear the kv cache
            cache["kv"][stage_idx] = None

        kv_cache = cache["kv"][stage_idx]
        if prev_stage_tokens_repr is not None:
            prev_stage_tokens_repr = F.pad(prev_stage_tokens_repr, (0, 0, 1, 0), value=0.0)
            stage_tokens = stage_tokens + prev_stage_tokens_repr

        # assert stage_tokens.shape[0] == 1, "we shuld only have batch size 1 here"

        if active_token_idx == 0 or run_full_sequence_attention:
            new_tokens = stage_tokens
        else:
            # select only the most recent token (the other ones are padding)
            # keeping the dims
            select_idx = active_token_idx if stage_idx != 0 else -1
            inspect_shapes(
                f"selecting active token index {select_idx}",
                suppress=True,
                print_values=True,
                stage_tokens=stage_tokens,
            )
            new_tokens = stage_tokens[..., [select_idx], :]
        transformer = self.transformers[stage_idx]
        if stage_idx == 0 and self.add_cross_attention:
            attended, kv_cache = transformer(
                new_tokens,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=use_cache,
                cache=kv_cache,
            )
        else:
            attended, kv_cache = transformer(
                new_tokens,
                use_cache=use_cache,
                cache=kv_cache if use_cache else None,
                debug=False,
            )
        # project for next stage in the hierarchy

        proj = self.to_next_transformer_projections[stage_idx]
        # to_next_layer = attended[..., :-1, :] if attended.shape[-2] > 1 else attended
        to_next_layer = attended[..., -1, :]  # if attended.shape[-2] > 1 else attended
        if is_complete_window_prompt_on_stage0:
            to_next_layer = attended[..., -2, :]  # if attended.shape[-2] > 1 else attended
        inspect_shapes("to_next_layer_proj", print_values=True, to_next_layer=to_next_layer)
        prev_stage_tokens_repr = proj(to_next_layer)

        # update cache
        # inspect_shapes(f"output cache {stage_idx}", v=kv_cache[0]["v"])
        cache["kv"][stage_idx] = kv_cache  # type: ignore
        if cache is not None and stage_idx < self.depth - 1:
            # inspect_shapes(
            #     "updating hs cache",
            #     print_values=True,
            #     current=cache["hidden_states"][stage_idx],
            #     new_line=prev_stage_tokens_repr.detach().clone(),
            # )
            cache["hidden_states"][stage_idx] = prev_stage_tokens_repr.detach().clone().float()  # type: ignore
        if profile:
            cache["profile"][stage_idx].append(time.time() - start_time)  # type: ignore

        # attended = attended[:, active_token_idx : active_token_idx + 1, :]
        return attended, cache


if __name__ == "__main__":

    print("\n*************************************************************************************")
    print("STARTING SCRIPT", time.time())
    print("*************************************************************************************\n")

    import lightning as L

    # L.seed_everything(43894)
    L.seed_everything(43895)

    def generate_few(
        model,
        prime=None,
        filter_thres=0.9,
        temperature=0.0,
        default_batch_size=1,
    ):

        model.eval()

        start_time = time.time()

        with torch.inference_mode():

            # total_seq_len = reduce_mult(model.max_sequence_lengths)
            device = "cuda"
            # print(device)

            if not exists(prime):
                prime = torch.empty((default_batch_size, 0), dtype=torch.long, device=device)
            prime = prime.to(device)

            seq = prime
            batch = seq.shape[0]

            seq_len = seq.shape[-1]
            # cache = {'profile': [[], []], 'hidden_states': [None, None], 'kv': [[],[]]}
            cache = None
            x = prime

            seq2 = prime
            # cache = {'profile': [[], []], 'hidden_states': [None, None], 'kv': [[],[]]}
            seq_len2 = seq.shape[-1]
            cache2 = None
            x2 = prime
            manual = False
            use_same_sequence_for_both = True
            N = 32 - prime.numel()
            for tok_idx in range(N):
                tok_idx += prime.numel()
                non_manual_logits = None
                print("\n", "*" * 90)
                if True:

                    use_old_algo = True
                    print("GENRATING  NON MANUAL", tok_idx, time.time(), "\n")
                    logits = model.forward(ids=x2, use_cache=False, cache=None, profile=False)
                    # logits, cache = model.forward_old(ids=x, use_cache=True, cache=cache, profile=False)
                    # inspect_shapes("CACHE_RESULT", cache=cache['hidden_states'][0])
                    logits = logits[:, -1]
                    inspect_shapes("ORIGINAL ALGO", print_values=True, logits=logits)
                    non_manual_logits = logits
                    logits = top_k(logits, thres=filter_thres)
                    sampled = gumbel_sample(logits, dim=-1, temperature=temperature)
                    seq_len2 += 1
                    x2 = rearrange(sampled, "b -> b 1")
                    seq2 = torch.cat((seq2, x2), dim=-1)
                    x2 = seq2
                if True:
                    print("\nGENERATING  INFERENCE ", tok_idx, time.time(), "\n")
                    logits, cache = model.forward_inference(
                        ids=x,
                        use_cache=True,
                        cache=cache,
                        profile=False,
                    )
                    # inspect_shapes("CACHE_RESULT", cache=cache['hidden_states'][0])
                    logits = logits[:, -1]
                    inspect_shapes("INFERENCE LOGITS", print_values=True, logits=logits)
                    if torch.any(logits.round(decimals=3) != non_manual_logits.round(decimals=3)):  # type: ignore
                        assert False, "Results are not equal"
                    logits = top_k(logits, thres=filter_thres)
                    sampled = gumbel_sample(logits, dim=-1, temperature=temperature)
                    seq_len += 1
                    x = rearrange(sampled, "b -> b 1")
                    seq = torch.cat((seq, x), dim=-1)
                    if use_same_sequence_for_both:
                        x = seq2.clone()
                    else:
                        x = seq

            dur = time.time() - start_time
            print(f"Generated {seq_len} tokens in {dur:.2f} seconds ({seq_len/dur:.1f} tokens/s)")
            print("Non manual seq:", seq2)
            print("Manual seq    :", seq)
            return seq.reshape(batch, seq_len).flatten(-1), cache

    prime = None
    # prime = torch.ones((1, 4), dtype=torch.long, device="cuda")
    # prime = (torch.arange(4, dtype=torch.long, device="cuda").unsqueeze(0)) % 5 + 1
    model = MEGABYTE(
        vocab_size=6,
        hidden_sizes=(3, 2),
        num_hidden_layers=(1, 1),
        max_sequence_lengths=(384, 4),
        dim_head=2,
        num_heads=2,
    )

    model = model.cuda()
    model.eval()
    Y, cache = generate_few(model, prime=prime, temperature=0.5, default_batch_size=1)
    print(Y)
