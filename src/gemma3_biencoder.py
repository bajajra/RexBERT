# gemma3_biencoder.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import MaskedLMOutput
from transformers.utils import logging

# Import Gemma-3 internals (ships with 🤗 Transformers 4.46+)
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3PreTrainedModel,
    Gemma3TextModel,
)

logger = logging.get_logger(__name__)

def _padding_only_4d_mask(attention_mask: Optional[Tensor],
                          dtype: torch.dtype,
                          tgt_len: Optional[int] = None) -> Optional[Tensor]:
    """
    Build a 4D additive attention mask that ONLY masks pads.
    Shape: (batch, 1, tgt_len, src_len) with 0.0 allowed, -inf disallowed.
    No causal triangle → fully bidirectional.
    """
    if attention_mask is None:
        return None
    # attention_mask: [bsz, src_len] with 1 for tokens, 0 for pads
    bsz, src_len = attention_mask.shape
    tgt_len = int(tgt_len or src_len)
    # 1 -> keep (0.0), 0 -> mask (-inf)
    keep = attention_mask[:, None, None, :]  # [bsz, 1, 1, src_len]
    mask = (1.0 - keep.to(dtype=dtype)) * torch.finfo(dtype).min
    # expand target length
    if mask.shape[2] != tgt_len:
        mask = mask.expand(bsz, 1, tgt_len, src_len)
    return mask

class Gemma3EncoderForMaskedLM(Gemma3PreTrainedModel):
    """
    An encoder-only, bidirectional model reusing Gemma-3 blocks,
    trained with Masked Language Modeling.  This **removes causality**
    by supplying a padding-only mask to all self-attention layers.
    """
    config_class = Gemma3TextConfig
    base_model_prefix = "encoder_model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        # Reuse the text stack from Gemma-3; we will feed a padding-only mask.
        self.encoder = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()  # weight tying, init hooks

    # --- Embedding tying helpers so save/load stay standard ---
    def get_input_embeddings(self):
        # Gemma3TextModel keeps token embeddings inside .embed_tokens
        return self.encoder.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.encoder.embed_tokens = new_embeddings

    def tie_weights(self):
        # Standard tie: lm_head.weight <- input embedding matrix
        if getattr(self.config, "tie_word_embeddings", True):
            self._tie_or_clone_weights(self.lm_head, self.get_input_embeddings())

    # --- Forward (pad-only mask = fully bidirectional) ---
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[MaskedLMOutput, Tuple[Tensor, ...]]:

        # Build pad-only (non-causal) 4D mask in the model’s dtype/device
        # Gemma-3 uses bfloat16/float16 typically; ask encoder for dtype
        dtype = self.encoder._get_input_dtype(input_ids, inputs_embeds)
        # Infer query length for expansion
        if input_ids is not None:
            tgt_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            tgt_len = inputs_embeds.shape[1]
        else:
            tgt_len = None

        bidir_mask = _padding_only_4d_mask(attention_mask, dtype=dtype, tgt_len=tgt_len)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,           # still pass 2D mask (for kv caching size, norms)
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # Crucial: we route our own non-causal mask via kwargs name expected by blocks.
            # Gemma3TextModel forwards this to each layer as `attention_mask=...`
            # so we override with our pad-only 4D mask:
            attention_bias=bidir_mask,               # many HF models accept attention_bias/attention_mask
            use_cache=False,                         # encoder: no kv cache
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Standard MLM loss: ignore_index = -100
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))

        if not return_dict:
            out = (logits, hidden_states)
            if output_hidden_states:
                out += (outputs.hidden_states,)
            if output_attentions:
                out += (outputs.attentions,)
            if loss is not None:
                out = (loss,) + out
            return out

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )