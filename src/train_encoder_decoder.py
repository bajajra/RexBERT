#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLM training for your Gemma3 bidirectional encoder with:
  - Sequence packing (concatenate trimmed examples into fixed-length blocks)
  - FlashAttention-2 (if available) / bf16 / tf32
  - Gradient checkpointing + length grouping
  - Adjustable sliding-window for local attention (ModernBERT-like efficiency)

Assumes:
  * You already converted -> ./gemma3-270m-encoder-bidir (or similar)
  * You prepared pretokenized data -> ./data/ecom_prepared with columns:
      input_ids (List[int]) and attention_mask (List[int])
"""

from __future__ import annotations
import argparse, os, math, random
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Your custom encoder (bidirectional, MLM head)
from gemma3_biencoder import Gemma3EncoderForMaskedLM


# ----------------- Custom Trainer for MLM Accuracy Logging -----------------

class MLMAccuracyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Subclass to compute and log MLM accuracy on training batches.
        """
        # The `compute_loss` from `Trainer` will return the loss and model outputs
        # if `return_outputs=True`. The outputs contain the logits.
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # Log MLM accuracy on the main process
        if self.state.is_local_process_zero and "labels" in inputs:
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            if logits is not None and labels is not None:
                predictions = torch.argmax(logits, dim=-1)
                
                # Mask for only the tokens that were actually masked for MLM
                masked_tokens_mask = labels != -100
                
                # If there are any masked tokens in the batch...
                if torch.any(masked_tokens_mask):
                    # Calculate accuracy only on the masked tokens
                    correct_predictions = (predictions[masked_tokens_mask] == labels[masked_tokens_mask]).sum()
                    total_masked = masked_tokens_mask.sum()
                    mlm_accuracy = correct_predictions / total_masked
                    
                    # Log the metric
                    self.log({"mlm_accuracy": mlm_accuracy.item()})

        return (loss, outputs) if return_outputs else loss


# ------------------------- Packing utilities -------------------------

def iter_trimmed_tokens(ds, eos_id: int) -> Iterator[int]:
    """
    Stream all NON-PAD tokens from the dataset, inserting EOS between examples
    (so masking/reconstruction doesn't bleed sentences together too much).
    """
    for rec in ds:
        ids = rec["input_ids"]
        mask = rec["attention_mask"]
        # trim pads
        L = int(np.sum(mask))
        if L <= 0:
            continue
        yield from ids[:L]
        if eos_id is not None:
            yield eos_id

def pack_stream_to_blocks(ds, block_len: int, eos_id: int, drop_last: bool = True) -> Iterator[Dict[str, List[int]]]:
    """
    Concatenate trimmed token streams into fixed-length blocks of size block_len.
    Builds attention_mask=1 for all tokens in each block.
    """
    buf: List[int] = []
    for tok in iter_trimmed_tokens(ds, eos_id):
        buf.append(int(tok))
        while len(buf) >= block_len:
            chunk = buf[:block_len]
            buf = buf[block_len:]
            yield {"input_ids": chunk, "attention_mask": [1]*block_len}
    if not drop_last and len(buf) > 0:
        # pad the tail to block_len (rarely used; packed training prefers drop_last)
        pad_len = block_len - len(buf)
        yield {
            "input_ids": buf + [0]*pad_len,
            "attention_mask": [1]*len(buf) + [0]*pad_len,
        }

def make_packed_dataset(dataset_path: str, tokenizer, pack_len: int, out_cap: int | None = None, num_proc: int = 16) -> Dataset:
    """
    Build a packed Dataset from an on-disk pretokenized dataset.
    Uses `ds.map` for parallel processing.
    """
    base = load_from_disk(dataset_path)
    eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    def pack_batch(batch: Dict[str, List]) -> Dict[str, List]:
        """Concatenate and chunk a batch of examples."""
        # 1. Concatenate all tokens in the batch, separated by EOS
        all_tokens = []
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
            L = int(np.sum(mask))
            if L > 0:
                all_tokens.extend(ids[:L])
                if eos_id is not None:
                    all_tokens.append(eos_id)

        # 2. Chunk the concatenated tokens into fixed-length blocks
        packed_ids = []
        for i in range(0, len(all_tokens), pack_len):
            chunk = all_tokens[i : i + pack_len]
            if len(chunk) == pack_len:
                packed_ids.append(chunk)
        
        # 3. Create attention masks for the packed blocks
        return {
            "input_ids": packed_ids,
            "attention_mask": [[1] * pack_len for _ in packed_ids],
        }

    # Use map to parallelize the packing process.
    # batched=True sends chunks of the dataset to pack_batch.
    # remove_columns is important to drop the original, variable-length columns.
    packed = base.map(
        pack_batch,
        batched=True,
        batch_size=1000, # Adjust batch_size based on RAM
        num_proc=num_proc,
        remove_columns=base.column_names,
    )

    if out_cap and out_cap > 0:
        packed = packed.select(range(min(out_cap, len(packed))))

    return packed


# ------------------------- Training script -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Path to converted encoder (e.g., ./gemma3-270m-encoder-bidir)")
    ap.add_argument("--dataset-path", required=True, help="Path to pretokenized dataset (save_to_disk dir)")
    ap.add_argument("--output-dir", required=True, help="Where to save checkpoints")
    # Efficiency / ModernBERT-like toggles
    ap.add_argument("--pack-seq-len", type=int, default=2048, help="Packed block length (tokens)")
    ap.add_argument("--pack-cap", type=int, default=0, help="Optional cap (#packed blocks) for quick runs")
    ap.add_argument("--group-by-length", action="store_true", help="Enable Trainer length-based packing (minor win if not pre-packed)")
    ap.add_argument("--sliding-window", type=int, default=128, help="Local attention window for Gemma3 (smaller => less memory)")
    ap.add_argument("--flash-attn2", action="store_true", help="Try to force FlashAttention-2 attention implementation")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    ap.add_argument("--compile", action="store_true", help="torch.compile for a bit more speed (PyTorch 2.3+)")
    # Standard training knobs
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--save-steps", type=int, default=2000)
    ap.add_argument("--logging-steps", type=int, default=100)
    ap.add_argument("--pad-to-multiple-of", type=int, default=8)
    ap.add_argument("--max-train-pct", type=float, default=1.0, help="Use first pct of packed dataset (0<..<=1.0)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # ---- Load tokenizer & model ----
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "<mask>"})

    # dtype selection
    dtype = torch.float32
    if torch.cuda.is_available():
        if args.bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif args.fp16:
            dtype = torch.float16
        # allow tf32 for matmul on Ampere+
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    model = Gemma3EncoderForMaskedLM.from_pretrained(args.model_dir, torch_dtype=dtype)
    if len(tok) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tok))

    # ModernBERT-like: shrink local window; Gemma3 already alternates local/global.
    # This reduces memory while keeping periodic global attention.
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = int(args.sliding_window)

    # Try to engage FlashAttention-2
    # Different HF versions gate this under _attn_implementation or attn_implementation
    if args.flash_attn2:
        for key in ["_attn_implementation", "attn_implementation"]:
            if hasattr(model.config, key):
                setattr(model.config, key, "flash_attention_2")

    # Gradient checkpointing (big memory saver)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # torch.compile (optional)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    # ---- PACK the pretokenized dataset into fixed-length blocks ----
    print(f"[INFO] Building packed dataset with block length {args.pack_seq_len} ...")
    packed = make_packed_dataset(
        dataset_path=args.dataset_path,
        tokenizer=tok,
        pack_len=args.pack_seq_len,
        out_cap=(args.pack_cap if args.pack_cap > 0 else None),
    )
    # (Optional) subselect a percentage for quick runs
    if args.max_train_pct < 1.0:
        n = len(packed)
        m = max(1, int(n * args.max_train_pct))
        packed = packed.select(range(m))

    # Add a length column for grouped sampling (doesn't matter much when everything is equal length)
    packed = packed.map(lambda ex: {"length": int(sum(ex["attention_mask"]))})
    packed = packed.with_format(type="torch", columns=["input_ids", "attention_mask"])

    # ---- Collator: dynamic BERT-style masking on packed blocks ----
    collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=True,
        mlm_probability=args.mlm_prob,
        pad_to_multiple_of=(args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else None),
    )

    # ---- Training args ----
    use_bf16 = (dtype == torch.bfloat16)
    use_fp16 = (dtype == torch.float16)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to=[],
        remove_unused_columns=False,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        group_by_length=args.group_by_length,
        length_column_name="length",
    )

    trainer = MLMAccuracyTrainer(
        model=model,
        args=targs,
        train_dataset=packed,
        data_collator=collator,
        tokenizer=tok,
    )

    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training complete.")

    # Save
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    enc_dir = os.path.join(args.output_dir, "encoder")
    os.makedirs(enc_dir, exist_ok=True)
    model.save_pretrained(enc_dir)
    tok.save_pretrained(enc_dir)
    print(f"[INFO] Saved encoder to: {enc_dir}")


if __name__ == "__main__":
    main()