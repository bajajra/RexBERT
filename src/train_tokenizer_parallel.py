#!/usr/bin/env python3
"""
Parallel Gemma-style SentencePiece (Unigram) tokenizer for e-commerce.

Accelerations:
- HF datasets .map(..., num_proc=CPU) for BOTH the counting pass and preprocessing pass
- SentencePiece --num_threads for model training

Usage examples at bottom or run -h.

Requires: sentencepiece>=0.1.99, transformers>=4.41, datasets>=2.20, pandas>=2.0
"""

import os
import re
import json
import argparse
from collections import Counter
from typing import Dict, List, Iterable, Tuple

import sentencepiece as spm
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, IterableDataset
from transformers import GemmaTokenizerFast
from functools import lru_cache


# ---------------------------
# CLI
# ---------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train a fast, Gemma-style Unigram tokenizer with e-commerce specials.")
    p.add_argument("--hf_dataset", type=str, required=True,
                   help="HF dataset id or local path, e.g. 'user/ecom-corpus' or '/data/corpus'.")
    p.add_argument("--splits", type=str, default="train",
                   help="Comma-separated splits to use (default: 'train').")
    p.add_argument("--text_columns", type=str, default="text,title,description",
                   help="Comma-separated text columns to concatenate if present.")
    p.add_argument("--output_dir", type=str, default="./tok_ecom_gemma3_262k_fast",
                   help="Where to write model + HF tokenizer.")
    p.add_argument("--vocab_size", type=int, default=262_144,
                   help="Target vocab size (200–300k suggested).")
    p.add_argument("--character_coverage", type=float, default=1.0,
                   help="SPM character coverage.")
    p.add_argument("--threads", type=int, default=max(2, os.cpu_count() or 8),
                   help="Threads for SPM training (default: all cores).")
    p.add_argument("--num_proc", type=int, default=max(2, (os.cpu_count() or 8) // 2),
                   help="Parallel workers for HF .map(). Best is 1–N cores.")
    p.add_argument("--streaming", action="store_true",
                   help="Use HF streaming mode (limits parallel .map()).")

    # Alias sources
    p.add_argument("--brand_alias_csv", type=str, default=None,
                   help="CSV with columns: alias,canonical (case-insensitive match).")
    p.add_argument("--type_alias_csv", type=str, default=None,
                   help="CSV with columns: alias,canonical (case-insensitive match).")
    p.add_argument("--min_alias_len", type=int, default=2,
                   help="Ignore aliases shorter than this length when compiling regex.")

    # Budgets
    p.add_argument("--max_brand_specials", type=int, default=15000, help="Max <BRAND=…> specials.")
    p.add_argument("--max_type_specials", type=int, default=5000, help="Max <TYPE=…> specials.")

    # Domain placeholders + digit policy
    p.add_argument("--add_domain_placeholders", action="store_true",
                   help="Add <PRICE>, <PERCENT>, <SIZE>, <DIMENSION>, <MODEL>, <SKU>.")
    p.add_argument("--split_digits", action="store_true",
                   help="SentencePiece --split_digits=true (recommended for SKUs).")

    # Demo
    p.add_argument("--sample_encode", action="store_true",
                   help="After training, print a few sampled encodes (subword regularization).")

    return p.parse_args()


# ---------------------------
# Common helpers
# ---------------------------

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def join_text_fields_row(row: dict, text_cols: List[str]) -> str:
    parts = []
    for c in text_cols:
        if c in row and row[c] is not None:
            v = row[c]
            if isinstance(v, (list, tuple)):
                v = " ".join(map(str, v))
            parts.append(str(v))
    return normalize_space(" ".join(parts)) if parts else ""


def load_aliases_from_csv(path: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    df = df.dropna(subset=["alias", "canonical"])
    df["alias"] = df["alias"].astype(str).str.strip()
    df["canonical"] = df["canonical"].astype(str).str.strip()
    alias_map = {}
    for _, r in df.iterrows():
        a, c = r["alias"], r["canonical"]
        if a and c:
            alias_map[a] = c
    return alias_map


def default_brand_aliases() -> Dict[str, str]:
    return {
        "nike": "NIKE", "adidas": "ADIDAS", "apple": "APPLE", "samsung": "SAMSUNG",
        "h&m": "H&M", "h and m": "H&M", "uniqlo": "UNIQLO",
    }

def default_type_aliases() -> Dict[str, str]:
    return {
        "running shoes": "RUNNING_SHOES", "sneakers": "SNEAKERS",
        "smartphone": "SMARTPHONE", "phone": "SMARTPHONE",
        "t-shirt": "T_SHIRT", "tee": "T_SHIRT", "jeans": "JEANS", "hoodie": "HOODIE",
    }


# Placeholders (optional)
RE_PRICE = re.compile(r"(?<!\w)(?:\$|USD\s*)?\d{1,6}(?:[.,]\d{2})?(?!\w)")
RE_PERCENT = re.compile(r"\b\d{1,3}(?:[.,]\d+)?\s*%\b")
RE_SIZE = re.compile(r"\b(?:XS|S|M|L|XL|XXL|XXXL|XXXXL|\d{1,2}(?:[./-]\d{1,2})?(?:\s?(?:US|EU|UK))?)\b", re.IGNORECASE)
RE_DIM = re.compile(r"\b\d{1,4}\s?[x×]\s?\d{1,4}(?:\s?[x×]\s?\d{1,4})?\s?(?:cm|mm|in|inch|inches|ft)?\b", re.IGNORECASE)
RE_MODEL = re.compile(r"\b(?:[A-Z]{1,3}\d{2,5}[A-Z]?(?:-\d{2,5})?)\b")
RE_SKU = re.compile(r"\bSKU[:\s]*[A-Z0-9\-]{4,}\b", re.IGNORECASE)

def apply_placeholders(text: str) -> str:
    text = RE_PRICE.sub("<PRICE>", text)
    text = RE_PERCENT.sub("<PERCENT>", text)
    text = RE_SIZE.sub("<SIZE>", text)
    text = RE_DIM.sub("<DIMENSION>", text)
    text = RE_MODEL.sub("<MODEL>", text)
    text = RE_SKU.sub("<SKU>", text)
    return text


def compile_alias_regex(alias_to_canonical: Dict[str, str], min_len=2) -> Tuple[str, Dict[str, str]]:
    """
    Returns (pattern_string, normalized_alias_map). We return a **pattern string** (pickle-safe).
    """
    cleaned = {}
    for a, c in alias_to_canonical.items():
        a_clean = normalize_space(a)
        c_clean = normalize_space(c)
        if len(a_clean) >= min_len:
            cleaned[a_clean] = c_clean
    aliases_sorted = sorted(cleaned.keys(), key=len, reverse=True)
    escaped = [re.escape(x) for x in aliases_sorted] or ["_NOOP_"]
    pattern = r"(?:^|(?<=\W))(" + "|".join(escaped) + r")(?=\W|$)"
    return pattern, {k.lower(): v for k, v in cleaned.items()}


# Lazy per-process compiles (works with HF multiprocessing)
@lru_cache(maxsize=None)
def _get_brand_re(pattern: str):
    return re.compile(pattern, flags=re.IGNORECASE)

@lru_cache(maxsize=None)
def _get_type_re(pattern: str):
    return re.compile(pattern, flags=re.IGNORECASE)


# ---------------------------
# Parallel pass 1: COUNT hits
# ---------------------------

def count_batch(examples, text_cols: List[str], brand_pat: str, brand_map: Dict[str, str],
                type_pat: str, type_map: Dict[str, str]):
    """
    Returns lists of matched canonical brands/types per example.
    We aggregate them later to choose top-K specials.
    """
    brand_re = _get_brand_re(brand_pat)
    type_re = _get_type_re(type_pat)
    out_brands, out_types = [], []
    for i in range(len(examples[text_cols[0]]) if text_cols and text_cols[0] in examples else len(next(iter(examples.values())))):
        row = {c: examples[c][i] for c in examples if isinstance(examples[c], list)}
        text = join_text_fields_row(row, text_cols)
        brands, types = [], []
        if text:
            for m in brand_re.finditer(text):
                alias = m.group(1).lower()
                brands.append(brand_map.get(alias, alias))
            for m in type_re.finditer(text):
                alias = m.group(1).lower()
                types.append(type_map.get(alias, alias))
        out_brands.append(brands)
        out_types.append(types)
    return {"_brands": out_brands, "_types": out_types}


# ---------------------------
# Parallel pass 2: PREPROCESS to specials
# ---------------------------

def preprocess_batch(examples, text_cols: List[str], brand_pat: str, brand_map: Dict[str, str],
                     type_pat: str, type_map: Dict[str, str],
                     allowed_brand_tokens: set, allowed_type_tokens: set,
                     add_placeholders: bool):
    brand_re = _get_brand_re(brand_pat)
    type_re = _get_type_re(type_pat)

    def _b(m):
        alias = m.group(1).lower()
        canon = brand_map.get(alias)
        tok = f"<BRAND={canon}>" if canon and f"<BRAND={canon}>" in allowed_brand_tokens else "<BRAND=OTHER>"
        return tok

    def _t(m):
        alias = m.group(1).lower()
        canon = type_map.get(alias)
        tok = f"<TYPE={canon}>" if canon and f"<TYPE={canon}>" in allowed_type_tokens else "<TYPE=OTHER>"
        return tok

    processed = []
    n_rows = len(examples[text_cols[0]]) if text_cols and text_cols[0] in examples else len(next(iter(examples.values())))
    for i in range(n_rows):
        row = {c: examples[c][i] for c in examples if isinstance(examples[c], list)}
        text = join_text_fields_row(row, text_cols)
        if not text:
            processed.append("")
            continue
        if add_placeholders:
            text = apply_placeholders(text)
        text = brand_re.sub(_b, text)
        text = type_re.sub(_t, text)
        processed.append(text)
    return {"processed_text": processed}


# ---------------------------
# Train & Save
# ---------------------------

def train_sentencepiece_gemma(inputs: List[str], out_prefix: str, vocab_size: int,
                              user_defined_symbols: List[str], character_coverage: float,
                              threads: int, split_digits: bool):
    """
    Train Gemma-like SPM Unigram. You can pass **multiple input shards** for better I/O.
    """
    args = [
        f"--input={','.join(inputs)}",
        f"--model_prefix={out_prefix}",
        "--model_type=unigram",
        f"--vocab_size={vocab_size}",
        f"--character_coverage={character_coverage}",
        "--byte_fallback=true",
        "--add_dummy_prefix=true",
        "--normalization_rule_name=nmt_nfkc",
        "--remove_extra_whitespaces=false",
        "--unk_id=0", "--unk_piece=<unk>",
        "--bos_id=1", "--bos_piece=<bos>",
        "--eos_id=2", "--eos_piece=<eos>",
        "--pad_id=3", "--pad_piece=<pad>",
        f"--num_threads={threads}",
        "--train_extremely_large_corpus=true",
    ]
    if split_digits:
        args.append("--split_digits=true")
    if user_defined_symbols:
        # Deduplicate while preserving order
        uds, seen = [], set()
        for s in user_defined_symbols:
            if s not in seen:
                seen.add(s); uds.append(s)
        args.append("--user_defined_symbols=" + ",".join(uds))

    spm.SentencePieceTrainer.Train(" ".join(args))


def save_gemma_tokenizer_wrapper(sp_model_path: str, out_dir: str,
                                 additional_specials: List[str], model_max_length: int = 8192):
    os.makedirs(out_dir, exist_ok=True)
    special_map = {
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "additional_special_tokens": additional_specials,
    }
    with open(os.path.join(out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_map, f, ensure_ascii=False, indent=2)
    cfg = {
        "tokenizer_class": "GemmaTokenizerFast",
        "model_max_length": model_max_length,
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "add_bos_token": True,
        "add_eos_token": False,
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # Copy .model/.vocab for convenience
    for ext in [".model", ".vocab"]:
        src = sp_model_path.replace(".model", ext)
        dst = os.path.join(out_dir, "spm" + ext)
        with open(src, "rb") as r, open(dst, "wb") as w:
            w.write(r.read())

    tok = GemmaTokenizerFast(vocab_file=os.path.join(out_dir, "spm.model"))
    tok.add_special_tokens({"additional_special_tokens": additional_specials})
    tok.save_pretrained(out_dir)
    print(f"[OK] Gemma-style HF tokenizer saved to: {out_dir}")


# ---------------------------
# Main
# ---------------------------

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    text_cols = [c.strip() for c in args.text_columns.split(",") if c.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print("[1/8] Loading dataset …")
    ds = load_dataset(args.hf_dataset, split=None, streaming=args.streaming)
    if isinstance(ds, IterableDataset):
        # Streaming: wrap as dict with a single split if needed
        ds = DatasetDict({splits[0]: ds})
    elif isinstance(ds, dict):
        pass
    else:
        ds = DatasetDict({splits[0]: ds})

    # Concatenate requested splits into a single Dataset when NOT streaming (enables true parallel map)
    can_parallel = not args.streaming
    if can_parallel:
        parts = [ds[s] for s in splits if s in ds]
        base: Dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    else:
        # For streaming, just chain them as an IterableDataset
        from datasets import interleave_datasets
        parts = [ds[s] for s in splits if s in ds]
        base = interleave_datasets(parts) if len(parts) > 1 else parts[0]

    # Alias maps
    print("[2/8] Preparing alias dictionaries …")
    brand_alias_map = load_aliases_from_csv(args.brand_alias_csv) if args.brand_alias_csv else default_brand_aliases()
    type_alias_map  = load_aliases_from_csv(args.type_alias_csv)  if args.type_alias_csv  else default_type_aliases()

    brand_pat, brand_norm = compile_alias_regex(brand_alias_map, min_len=args.min_alias_len)
    type_pat,  type_norm  = compile_alias_regex(type_alias_map,  min_len=args.min_alias_len)

    # PASS 1: Count hits in parallel to pick top-K specials
    print("[3/8] Scanning corpus (parallel) to choose top brand/type specials …")
    if can_parallel:
        counted = base.map(
            count_batch,
            batched=True,
            batch_size=1000,
            num_proc=args.num_proc,
            fn_kwargs=dict(
                text_cols=text_cols,
                brand_pat=brand_pat, brand_map=brand_norm,
                type_pat=type_pat,  type_map=type_norm,
            ),
            remove_columns=base.column_names
        )
    else:
        # Streaming: single-process (HF limitation), still batched for speed
        counted = base.map(
            count_batch,
            batched=True,
            batch_size=1000,
            fn_kwargs=dict(
                text_cols=text_cols,
                brand_pat=brand_pat, brand_map=brand_norm,
                type_pat=type_pat,  type_map=type_norm,
            ),
            remove_columns=base.column_names
        )

    brand_ctr, type_ctr = Counter(), Counter()
    for b_list in counted["_brands"]:
        brand_ctr.update(b_list)
    for t_list in counted["_types"]:
        type_ctr.update(t_list)
    # Free memory
    counted = None

    def build_top_specials(ctr: Counter, max_k: int, prefix: str):
        toks = [f"{prefix}{c}>" for c, _ in ctr.most_common(max_k)]
        other = f"{prefix}OTHER>"
        if other not in toks:
            toks.append(other)
        return toks

    brand_specials = build_top_specials(brand_ctr, args.max_brand_specials, "<BRAND=")
    type_specials  = build_top_specials(type_ctr,  args.max_type_specials,  "<TYPE=")

    domain_specials = ["<PRICE>", "<PERCENT>", "<SIZE>", "<DIMENSION>", "<MODEL>", "<SKU>"] \
                      if args.add_domain_placeholders else []
    user_defined_symbols = brand_specials + type_specials + domain_specials

    total_specials = len(user_defined_symbols) + 4  # unk/bos/eos/pad
    if total_specials >= args.vocab_size:
        raise ValueError(f"Vocab too small for specials ({total_specials} >= {args.vocab_size}). "
                         f"Increase --vocab_size or reduce specials budgets.")

    print(f"  • Brand specials: {len(brand_specials)} | Type specials: {len(type_specials)} | Domain: {len(domain_specials)}")

    # PASS 2: Preprocess to specials in parallel
    print("[4/8] Preprocessing corpus to specials (parallel) …")
    allowed_brand_tokens = set(brand_specials)
    allowed_type_tokens = set(type_specials)

    processed = base.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,
        num_proc=(args.num_proc if can_parallel else None),
        fn_kwargs=dict(
            text_cols=text_cols,
            brand_pat=brand_pat, brand_map=brand_norm,
            type_pat=type_pat,  type_map=type_norm,
            allowed_brand_tokens=allowed_brand_tokens,
            allowed_type_tokens=allowed_type_tokens,
            add_placeholders=args.add_domain_placeholders
        ),
        remove_columns=base.column_names
    )

    # Write processed text to shards (SPM can train from multiple files)
    print("[5/8] Writing corpus shards …")
    shard_count = min(max(1, args.num_proc), 64) if can_parallel else 1
    shard_paths = []
    if shard_count == 1:
        shard_path = os.path.join(args.output_dir, "corpus.txt")
        with open(shard_path, "w", encoding="utf-8") as wf:
            for s in processed["processed_text"]:
                wf.write(s + "\n")
        shard_paths.append(shard_path)
    else:
        for i in range(shard_count):
            shard = processed.shard(num_shards=shard_count, index=i)
            shard_path = os.path.join(args.output_dir, f"corpus_shard_{i:04d}.txt")
            with open(shard_path, "w", encoding="utf-8") as wf:
                for s in shard["processed_text"]:
                    wf.write(s + "\n")
            shard_paths.append(shard_path)

    # Train SentencePiece Unigram (Gemma-style)
    print("[6/8] Training SentencePiece (Gemma-style Unigram) …")
    out_prefix = os.path.join(args.output_dir, "ecom_sp_unigram")
    train_sentencepiece_gemma(
        inputs=shard_paths,
        out_prefix=out_prefix,
        vocab_size=args.vocab_size,
        user_defined_symbols=user_defined_symbols,
        character_coverage=args.character_coverage,
        threads=args.threads,
        split_digits=args.split_digits
    )
    print(f"  • Model written: {out_prefix}.model")

    # HF wrapper (GemmaTokenizerFast)
    print("[7/8] Creating Hugging Face tokenizer wrapper …")
    save_gemma_tokenizer_wrapper(
        sp_model_path=f"{out_prefix}.model",
        out_dir=args.output_dir,
        additional_specials=user_defined_symbols,
        model_max_length=8192
    )

    # Optional demo
    if args.sample_encode:
        print("[8/8] Demo (subword-regularization sampling):")
        sp = spm.SentencePieceProcessor(model_file=f"{out_prefix}.model")
        demo = "Nike running shoes, 10×12 in, $19.99 — SKU: AB1234"
        if args.add_domain_placeholders:
            demo = apply_placeholders(demo)
        # Replace with specials using allowed sets (quick path)
        brand_re = _get_brand_re(brand_pat); type_re = _get_type_re(type_pat)
        def _bs(m):
            alias = m.group(1).lower()
            canon = brand_norm.get(alias)
            return f"<BRAND={canon}>" if canon and f"<BRAND={canon}>" in allowed_brand_tokens else "<BRAND=OTHER>"
        def _ts(m):
            alias = m.group(1).lower()
            canon = type_norm.get(alias)
            return f"<TYPE={canon}>" if canon and f"<TYPE={canon}>" in allowed_type_tokens else "<TYPE=OTHER>"
        demo = brand_re.sub(_bs, demo); demo = type_re.sub(_ts, demo)

        for i in range(3):
            ids = sp.encode(demo, out_type=int, enable_sampling=True, nbest_size=-1, alpha=0.1)
            print(f"  Sample {i+1}: {ids}")

    print("Done ✅")


if __name__ == "__main__":
    main()