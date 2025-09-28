#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLM training for the converted Gemma3 encoder (bidirectional) on pre-tokenized Ecom-niverse,
with optional FlashAttention (FA2 or SDPA flash) and gradient checkpointing.

Usage:
  pip install -U "transformers>=4.46" datasets torch
  # Optional for FA2 (Ampere+ GPUs):
  pip install "flash-attn>=2.5.7" --no-build-isolation

  python train_mlm_gemma3_encoder.py \
    --model-dir ./gemma3-270m-encoder-bidir \
    --dataset-path ./data/ecom_prepared \
    --output-dir ./models/gemma3-270m-ecom-mlm-fa2-gc \
    --batch-size 8 --epochs 3 --lr 1e-4 --mlm-prob 0.20 \
    --attn-impl flash_attention_2 \
    --grad-checkpointing \
    --bf16
"""

from __future__ import annotations
import argparse, os
from typing import Optional
import importlib

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Your custom encoder wrapper (patched version that supports padding-only 4D masks)
from gemma3_biencoder import Gemma3EncoderForMaskedLM


def parse_args():
    ap = argparse.ArgumentParser(description="MLM train Gemma3 encoder on Ecom-niverse (pre-tokenized).")
    ap.add_argument("--model-dir", required=True, help="Path to converted encoder (e.g., ./gemma3-270m-encoder-bidir)")
    ap.add_argument("--dataset-path", required=True, help="Path to pre-tokenized dataset (save_to_disk dir)")
    ap.add_argument("--output-dir", required=True, help="Where to save checkpoints")

    ap.add_argument("--mlm-prob", type=float, default=0.15, help="Masking probability")
    ap.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    ap.add_argument("--epochs", type=int, default=2, help="Num epochs")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    ap.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Grad accumulation")
    ap.add_argument("--save-steps", type=int, default=2000, help="Checkpoint save frequency (steps)")
    ap.add_argument("--logging-steps", type=int, default=100, help="Logging frequency (steps)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--shuffle-buffer", type=int, default=0, help="Optional in-epoch shuffle (0=off)")
    ap.add_argument("--pad-to-multiple-of", type=int, default=8, help="Pad to multiple of N (tensor cores)")

    # Precision
    ap.add_argument("--bf16", action="store_true", help="Use bf16 (Ampere+)")
    ap.add_argument("--fp16", action="store_true", help="Use fp16")

    # NEW: attention implementation + grad checkpointing
    ap.add_argument("--attn-impl", type=str, default="auto",
                    choices=["auto", "flash_attention_2", "sdpa", "eager"],
                    help="Attention backend: try FA2, SDPA, or eager.")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="Enable gradient checkpointing (use_cache will be disabled).")
    return ap.parse_args()


def _fa2_available() -> bool:
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except Exception:
        return False


def _set_attn_impl(model: Gemma3EncoderForMaskedLM, impl: str):
    """
    Try to set attention implementation on the encoder/model/config, depending on what
    this Transformers version exposes.
    """
    # Preferred path: encoder.set_attn_implementation if present
    if hasattr(model, "encoder") and hasattr(model.encoder, "set_attn_implementation"):
        model.encoder.set_attn_implementation(impl)
        print(f"[INFO] Set encoder attention implementation -> {impl}")
        return
    # Sometimes exists on the top-level module
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(impl)
        print(f"[INFO] Set model attention implementation -> {impl}")
        return
    # Fallback: stash on config (many models read this)
    try:
        model.config.attn_implementation = impl
    except Exception:
        setattr(model.config, "_attn_implementation", impl)
    print(f"[INFO] Recorded attention implementation on config -> {impl} (model may read at layer build time)")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Slight perf win on Ampere+ even with SDPA/eager
    torch.set_float32_matmul_precision("high")

    # 1) Tokenizer (+ ensure MLM mask token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.mask_token is None:
        print("[INFO] Adding <mask> token to tokenizer and resizing embeddings.")
        tokenizer.add_special_tokens({"mask_token": "<mask>"})

    # 2) Model load with desired dtype
    dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if (args.fp16 and torch.cuda.is_available()) else torch.float32
    )
    model = Gemma3EncoderForMaskedLM.from_pretrained(args.model_dir, torch_dtype=dtype)

    # Resize if tokenizer changed
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    # 3) Attention implementation selection
    chosen_impl = args.attn_impl
    if args.attn_impl == "auto":
        if _fa2_available() and torch.cuda.is_available():
            chosen_impl = "flash_attention_2"
        else:
            # SDPA uses PyTorch's scaled_dot_product_attention; will select flash kernel when available
            chosen_impl = "sdpa"

    # Apply the selection
    _set_attn_impl(model, chosen_impl)

    # If using SDPA, you can force flash kernel where supported:
    # (optional; PyTorch will usually pick it automatically)
    if chosen_impl == "sdpa" and torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            print("[INFO] Enabled SDPA backends (flash/mem_efficient/math).")
        except Exception:
            pass

    # 4) Gradient checkpointing
    if args.grad_checkpointing:
        # Encoder-only model: make sure we don't use KV cache anywhere
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        # Recommended on PT2.1+ with HF: use_reentrant=False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            # Older HF fallback
            model.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled.")

    # 5) Load pre-tokenized dataset
    ds = load_from_disk(args.dataset_path)
    if args.shuffle_buffer and args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed)
    ds = ds.with_format(type="torch", columns=["input_ids", "attention_mask"])

    # 6) MLM collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
        pad_to_multiple_of=args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else None,
    )

    # 7) TrainingArguments
    use_bf16 = bool(dtype == torch.bfloat16)
    use_fp16 = bool(dtype == torch.float16)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print(f"[INFO] Starting training with attn_impl='{chosen_impl}', "
          f"grad_checkpointing={'on' if args.grad_checkpointing else 'off'}, "
          f"dtype={dtype}")
    trainer.train()
    print("[INFO] Training complete.")

    # Save model + tokenizer; also save encoder path
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    enc_dir = os.path.join(args.output_dir, "encoder")
    os.makedirs(enc_dir, exist_ok=True)
    model.save_pretrained(enc_dir)
    tokenizer.save_pretrained(enc_dir)
    print(f"[INFO] Saved encoder to: {enc_dir}")


if __name__ == "__main__":
    main()