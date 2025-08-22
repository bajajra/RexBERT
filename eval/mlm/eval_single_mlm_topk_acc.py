#!/usr/bin/env python
# eval_mlm_topk_single.py
import argparse
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    set_seed,
)
from contextlib import nullcontext
from tqdm.auto import tqdm


def tokenize_function(examples, tokenizer, text_column):
    return tokenizer(examples[text_column], return_special_tokens_mask=True)


def group_texts(examples, block_size: int):
    # Keep only known tokenization keys to avoid string columns slipping in
    keys = [k for k in ("input_ids", "attention_mask", "special_tokens_mask", "token_type_ids") if k in examples]
    concatenated = {k: sum(examples[k], []) for k in keys}
    if len(concatenated["input_ids"]) == 0:
        return {k: [] for k in keys}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    return {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }


@torch.inference_mode()
def evaluate_model(
    model_name: str,
    dataset,
    text_column: str,
    block_size: int,
    batch_size: int,
    mlm_probability: float,
    device: torch.device,
    max_samples: int = None,
    fp16: bool = False,
) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.mask_token is None:
        raise ValueError(f"Tokenizer for '{model_name}' has no mask token (not an MLM checkpoint).")

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device).eval()

    # Drop all original columns so group_texts only sees token lists
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer, text_column),
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing for {model_name}",
    )
    lm_dataset = tokenized.map(
        lambda x: group_texts(x, block_size),
        batched=True,
        desc=f"Grouping texts for {model_name}",
    )

    if max_samples is not None:
        lm_dataset = lm_dataset.select(range(min(max_samples, len(lm_dataset))))

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )
    dataloader = DataLoader(lm_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    autocast_ctx = torch.cuda.amp.autocast if (fp16 and device.type == "cuda") else nullcontext

    total_masked = 0
    correct_at = {1: 0, 3: 0, 5: 0}

    for batch in tqdm(dataloader, desc=f"Evaluating {model_name}", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx():
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            ).logits  # [B, T, V]

        labels = batch["labels"]            # [B, T], -100 where not masked
        mask = labels.ne(-100)              # [B, T] bool
        if mask.sum().item() == 0:
            continue

        masked_labels = labels[mask]        # [N]
        masked_logits = logits[mask]        # [N, V]
        topk = torch.topk(masked_logits, k=5, dim=-1).indices  # [N, 5]

        for k in (1, 3, 5):
            correct = (topk[:, :k] == masked_labels.unsqueeze(-1)).any(dim=-1).float().sum().item()
            correct_at[k] += correct

        total_masked += mask.sum().item()

    denom = max(total_masked, 1)  # avoid div-by-zero
    return {
        "model": model_name,
        "acc@1": correct_at[1] / denom,
        "acc@3": correct_at[3] / denom,
        "acc@5": correct_at[5] / denom,
        "num_masked_tokens": total_masked,
    }


def combine_fields_map(fields: List[str]):
    # Concatenate multiple JSONL fields into one "__text__" column
    def _map(example):
        parts = []
        for f in fields:
            v = example.get(f, "")
            parts.append("" if v is None else str(v))
        return {"__text__": " ".join(p for p in parts if p)}
    return _map


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single MLM on token recovery accuracy @1,3,5.")
    parser.add_argument("--model", required=True, help="HF model id or local path (must be an MLM)")

    # === Data sources ===
    parser.add_argument("--jsonl_file", default=None, help="Path to a JSONL file (one JSON object per line)")
    parser.add_argument("--jsonl_text_field", default="text", help="Field name for text in JSONL")
    parser.add_argument("--jsonl_concat_fields", nargs="*", default=None,
                        help="Concatenate these fields into one text column (overrides --jsonl_text_field)")

    # Fallback: Hugging Face dataset
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text_column", default="text", help="Text column for HF datasets")

    # === Eval options ===
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of chunks AFTER grouping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # === Load data ===
    if args.jsonl_file:
        if not os.path.exists(args.jsonl_file):
            raise FileNotFoundError(args.jsonl_file)
        raw = load_dataset("json", data_files={"eval": args.jsonl_file}, split="eval")

        if args.jsonl_concat_fields:
            raw = raw.map(combine_fields_map(args.jsonl_concat_fields), desc="Concatenating JSONL fields")
            text_column = "__text__"
        else:
            text_column = args.jsonl_text_field
            if text_column not in raw.column_names:
                raise ValueError(f"JSONL text field '{text_column}' not found. Available columns: {raw.column_names}")
    else:
        raw = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
        text_column = args.text_column
        if text_column not in raw.column_names:
            raise ValueError(f"Column '{text_column}' not in dataset columns: {raw.column_names}")

    print(f"\nDevice: {device}")
    print(f"Examples (rows) in source: {len(raw)}")
    print(f"Using text column: {text_column}\n")

    # === Evaluate ===
    res = evaluate_model(
        model_name=args.model,
        dataset=raw,
        text_column=text_column,
        block_size=args.block_size,
        batch_size=args.batch_size,
        mlm_probability=args.mlm_probability,
        device=device,
        max_samples=args.max_samples,
        fp16=args.fp16,
    )

    # === Report ===
    def fmt(v): return f"{v:.4f}" if isinstance(v, float) else v
    print("=== Results ===")
    print(f"Model: {res['model']}")
    print({k: fmt(v) for k, v in res.items() if k != 'model'})


if __name__ == "__main__":
    main()