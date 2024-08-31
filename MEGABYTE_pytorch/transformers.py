import os

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin, AutoModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithCrossAttentions

from dataclasses import dataclass

from typing import Optional, List, Tuple, Union

from .megabyte import MEGABYTE


class MegabyteConfig(PretrainedConfig):
    model_type = "megabyte"

    def __init__(
        self,
        *,
        vocab_size: int = 256,
        hidden_sizes: List[int] = [768, 256],
        num_hidden_layers: List[int] = [12, 8],
        max_sequence_lengths: List[int] = [1024, 8],
        dim_head: int = 64,
        num_heads: int = 8,
        attention_dropout_prob: float = 0.1,
        feed_forward_scaleup: int = 4,
        feed_forward_dropout_prob: float = 0.1,
        rel_pos: bool = False,
        pos_emb: bool = False,
        flash_attn: bool = False,
        add_cross_attention: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: Optional[int] = 2,
        **kwargs
    ):

        super().__init__(**kwargs, bos_token_id=bos_token_id, eos_token_id=eos_token_id)

        self.vocab_size: int = vocab_size
        self.hidden_sizes: List[int] = hidden_sizes
        self.num_hidden_layers: List[int] = num_hidden_layers
        self.max_sequence_lengths: List[int] = max_sequence_lengths
        self.dim_head: int = dim_head
        self.num_heads: int = num_heads
        self.attention_dropout_prob: float = attention_dropout_prob
        self.feed_forward_scaleup: int = feed_forward_scaleup
        self.feed_forward_dropout_prob: float = feed_forward_dropout_prob
        self.rel_pos: bool = rel_pos
        self.pos_emb: bool = pos_emb
        self.flash_attn: bool = flash_attn
        self.add_cross_attention: bool = add_cross_attention
        self.pad_token_id: int = pad_token_id
        self.bos_token_id: int = bos_token_id
        self.eos_token_id: int = eos_token_id or bos_token_id
        self.is_encoder_decoder = False

    @property
    def hidden_size(self):
        return self.hidden_sizes[0]


class MegabyteLMHeadModel(PreTrainedModel, GenerationMixin):
    config_class = MegabyteConfig

    def __init__(self, config: MegabyteConfig):
        super().__init__(config)
        self.model = MEGABYTE(
            vocab_size=config.vocab_size,
            hidden_sizes=config.hidden_sizes,
            num_hidden_layers=config.num_hidden_layers,
            max_sequence_lengths=config.max_sequence_lengths,
            dim_head=config.dim_head,
            num_heads=config.num_heads,
            attention_dropout_prob=config.attention_dropout_prob,
            feed_forward_scaleup=config.feed_forward_scaleup,
            feed_forward_dropout_prob=config.feed_forward_dropout_prob,
            rel_pos=config.rel_pos,
            pos_emb=config.pos_emb,
            flash_attn=config.flash_attn,
            add_cross_attention=config.add_cross_attention,
            pad_token_id=config.pad_token_id,
        )
        self.config = config

    def forward(self, input_ids, encoder_hidden_states: Optional[torch.Tensor] = None, return_dict=False, **kwargs):

        logits, loss = self.model(input_ids, encoder_hidden_states=encoder_hidden_states, return_loss=True, **kwargs)
        if not return_dict:
            return logits, loss
        if self.config.add_cross_attention:
            # TODO fill out the remaining constructor args
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                hidden_states=None,
                attentions=None,
                past_key_values=None,
                cross_attentions=None,
            )
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=None, attentions=None)


AutoConfig.register("megabyte", MegabyteConfig)
AutoModel.register(MegabyteConfig, MegabyteLMHeadModel)

# class MegabyteTokenizer:
#     def __init__(self, pad_token_id=0, bos_token_id=1, eos_token_id=2):
#         super().__init__()
#         self.pad_token_id = pad_token_id
#         self.bos_token_id = bos_token_id
#         self.eos_token_id = eos_token_id
#
#     def __call__(self, text, return_tensors="pt"):
#         tokens = torch.frombuffer(bytearray(text.encode("utf-8")), dtype=torch.uint8).to(torch.int64)
#         tokens = tokens.reshape(1, tokens.numel())
#         return {"input_ids": tokens}
#
#     def decode(self, ids):
#         texts = []
#         for id_list in ids.tolist():
#             line_ids = filter(lambda x: 0 <= x and x < 256, id_list)
#             text = bytearray(list(line_ids)).decode("utf-8")
#             texts.append(text)
#
#         return texts
