# --- begin patched Gemma3EncoderForMaskedLM ---
from __future__ import annotations
import inspect
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3PreTrainedModel,
    Gemma3TextModel,
)

def _padding_only_4d_mask(attention_mask: Optional[Tensor],
                          dtype: torch.dtype,
                          tgt_len: Optional[int] = None) -> Optional[Tensor]:
    if attention_mask is None:
        return None
    # attention_mask 2D: [bsz, src_len] with 1 for keep, 0 for pad
    bsz, src_len = attention_mask.shape
    tgt_len = int(tgt_len or src_len)
    keep = attention_mask[:, None, None, :].to(dtype=dtype)  # [bsz,1,1,src_len]
    # 0.0 = allowed, -inf = masked out
    bias = (1.0 - keep) * torch.finfo(dtype).min
    if bias.shape[2] != tgt_len:
        bias = bias.expand(bsz, 1, tgt_len, src_len)
    return bias

class Gemma3EncoderForMaskedLM(Gemma3PreTrainedModel):
    """
    Encoder-only Gemma-3 with bidirectional self-attention (padding-only mask) and an MLM head.
    """
    config_class = Gemma3TextConfig
    base_model_prefix = "encoder_model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.encoder = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()  # enable tying etc.

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.encoder.embed_tokens = new_embeddings

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", True):
            self._tie_or_clone_weights(self.lm_head, self.get_input_embeddings())

    @staticmethod
    def _infer_dtype(encoder: Gemma3TextModel,
                     input_ids: Optional[Tensor],
                     inputs_embeds: Optional[Tensor]) -> torch.dtype:
        if inputs_embeds is not None:
            return inputs_embeds.dtype
        # fall back to embedding weight dtype (reliable for bf16/fp16/fp32 choice)
        return encoder.embed_tokens.weight.dtype

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,   # 2D pad mask from dataset
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[MaskedLMOutput, Tuple[Tensor, ...]]:

        # Determine target length and floating dtype
        if input_ids is not None:
            tgt_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            tgt_len = inputs_embeds.shape[1]
        else:
            tgt_len = None
        dtype = self._infer_dtype(self.encoder, input_ids, inputs_embeds)

        # Build a fully bidirectional (padding-only) additive bias
        bias4d = _padding_only_4d_mask(attention_mask, dtype=dtype, tgt_len=tgt_len)

        # Figure out how to feed the 4D mask depending on the installed Transformers version
        enc_sig = inspect.signature(self.encoder.forward)
        params = set(enc_sig.parameters.keys())
        call_kwargs = dict(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )
        if "attention_bias" in params:
            # Pass both: normal 2D attention_mask (pads) + 4D additive bias (non-causal)
            call_kwargs["attention_mask"] = attention_mask
            call_kwargs["attention_bias"]  = bias4d
        else:
            # Fallback: supply 4D bias via attention_mask directly
            # (omit 2D attention_mask to avoid shape confusion)
            call_kwargs["attention_mask"] = bias4d if bias4d is not None else attention_mask

        outputs = self.encoder(**call_kwargs)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
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