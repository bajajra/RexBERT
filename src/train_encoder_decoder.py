#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MLM training for the converted Gemma3 encoder (bidirectional) using pre-tokenized Ecom-niverse.

- Expects a directory produced by prepare_ecom_data.py with columns:
  input_ids (List[int]), attention_mask (List[int]) — already padded/truncated.

- Uses Hugging Face Trainer + DataCollatorForLanguageModeling (BERT-style MLM).
- Ensures a mask token exists; if not, adds "<mask>" and resizes embeddings.

Usage example:
    python train_mlm_gemma3_encoder.py \
      --model-dir ./gemma3-270m-encoder-bidir \
      --dataset-path ./data/ecom_prepared \
      --output-dir ./models/gemma3-270m-ecom-mlm \
      --batch-size 8 \
      --epochs 3 \
      --lr 1e-4 \
      --mlm-prob 0.20
"""

from __future__ import annotations
import argparse, os, math, random
from typing import Optional

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Import your custom encoder class used during conversion
from gemma3_biencoder import Gemma3EncoderForMaskedLM


def parse_args():
    ap = argparse.ArgumentParser(description="MLM train Gemma3 encoder on Ecom-niverse (pre-tokenized).")
    ap.add_argument("--model-dir", required=True, help="Path to converted encoder (e.g., ./gemma3-270m-encoder-bidir)")
    ap.add_argument("--dataset-path", required=True, help="Path to pre-tokenized dataset (save_to_disk dir)")
    ap.add_argument("--output-dir", required=True, help="Where to save checkpoints")
    ap.add_argument("--mlm-prob", type=float, default=0.15, help="Masking probability (default 0.15)")
    ap.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    ap.add_argument("--epochs", type=int, default=2, help="Num epochs")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    ap.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Grad accumulation")
    ap.add_argument("--save-steps", type=int, default=2000, help="Checkpoint save frequency (steps)")
    ap.add_argument("--logging-steps", type=int, default=100, help="Logging frequency (steps)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--shuffle-buffer", type=int, default=0, help="Optional in-epoch shuffling buffer (0=off)")
    ap.add_argument("--bf16", action="store_true", help="Force bf16 if available")
    ap.add_argument("--fp16", action="store_true", help="Force fp16 if available")
    ap.add_argument("--pad-to-multiple-of", type=int, default=8, help="Pad to multiple of N (for tensor cores)")
    ap.add_argument("--max-train-samples", type=int, default=0, help="Optional cap for quick runs")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Load tokenizer and ensure there is a mask token for MLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.mask_token is None:
        print("[INFO] Adding <mask> token to tokenizer and resizing embeddings.")
        tokenizer.add_special_tokens({"mask_token": "<mask>"})

    # 2) Load model (your converted encoder) and resize if tokenizer changed
    dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if (args.fp16 and torch.cuda.is_available()) else torch.float32
    )
    model = Gemma3EncoderForMaskedLM.from_pretrained(args.model_dir, torch_dtype=dtype)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    # 3) Load pre-tokenized dataset (arrow on disk)
    ds = load_from_disk(args.dataset_path)
    # Optional small run
    if args.max_train_samples and args.max_train_samples < len(ds):
        ds = ds.select(range(args.max_train_samples))

    # Optional lightweight shuffle for better mixing across shards
    if args.shuffle_buffer and args.shuffle_buffer > 0:
        # Use Dataset.shuffle for on-disk datasets (seeded)
        ds = ds.shuffle(seed=args.seed)

    # Set format for faster collation
    ds = ds.with_format(type="torch", columns=["input_ids", "attention_mask"])

    # 4) MLM data collator (BERT-style) acting on already-tokenized inputs
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_prob,
        pad_to_multiple_of=args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else None,
    )

    # 5) TrainingArguments
    # mixed precision flags: prefer bf16 on Ampere+ if not manually overridden
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        if args.bf16 and torch.cuda.is_bf16_supported():
            use_bf16 = True
        elif args.fp16:
            use_fp16 = True

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
        report_to=[],                     # no wandb by default
        remove_unused_columns=False,      # keep input_ids/attention_mask for our custom model
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("[INFO] Starting training…")
    trainer.train()
    print("[INFO] Training complete.")

    # Save full model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Also save just the encoder again (same format, convenient path)
    encoder_dir = os.path.join(args.output_dir, "encoder")
    os.makedirs(encoder_dir, exist_ok=True)
    model.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(encoder_dir)
    print(f"[INFO] Saved encoder to: {encoder_dir}")


if __name__ == "__main__":
    main()