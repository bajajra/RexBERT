#!/usr/bin/env python3
"""
Build a Gemma-3–style SentencePiece (Unigram) tokenizer for e-commerce:
- Large vocab (default 262_144)
- Brand / Product Type specials as never-split tokens
- Optional domain placeholders (PRICE, PERCENT, SIZE, DIMENSION, MODEL, SKU)
- Gemma-style SPM flags and HF wrapper (GemmaTokenizerFast)

Usage example at the bottom or run with -h.
"""

import os
import re
import json
import argparse
from collections import Counter
from typing import Dict, List, Iterable, Tuple

import sentencepiece as spm
import pandas as pd
from datasets import load_dataset, IterableDataset, DatasetDict
from transformers import GemmaTokenizerFast


# ---------------------------
# CLI
# ---------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Train a Gemma-style Unigram SentencePiece tokenizer for e-commerce."
    )
    p.add_argument("--hf_dataset", type=str, required=True,
                   help="HF dataset id or local path, e.g. 'user/ecom-corpus' or '/data/corpus'.")
    p.add_argument("--splits", type=str, default="train",
                   help="Comma-separated splits to use (default: 'train').")
    p.add_argument("--text_columns", type=str, default="text,title,description",
                   help="Comma-separated text columns to concatenate if present.")
    p.add_argument("--output_dir", type=str, default="./tok_ecom_gemma3_262k",
                   help="Where to write model + HF tokenizer.")
    p.add_argument("--vocab_size", type=int, default=262_144,
                   help="Target vocab size (200–300k suggested).")
    p.add_argument("--character_coverage", type=float, default=1.0,
                   help="SPM character coverage (1.0 for full Unicode).")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) // 2),
                   help="Threads for SPM training. Default ~half cores.")
    p.add_argument("--streaming", action="store_true",
                   help="Use HF streaming mode (good for very large corpora).")

    # Brand / Type alias sources
    p.add_argument("--brand_alias_csv", type=str, default=None,
                   help="CSV with columns: alias,canonical (case-insensitive match).")
    p.add_argument("--type_alias_csv", type=str, default=None,
                   help="CSV with columns: alias,canonical (case-insensitive match).")
    p.add_argument("--min_alias_len", type=int, default=2,
                   help="Ignore aliases shorter than this length when compiling regex.")

    # Budgets
    p.add_argument("--max_brand_specials", type=int, default=15000,
                   help="Max <BRAND=…> specials to inject.")
    p.add_argument("--max_type_specials", type=int, default=5000,
                   help="Max <TYPE=…> specials to inject.")

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
# Helpers
# ---------------------------

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def join_text_fields(ex: dict, text_cols: List[str]) -> str:
    parts = []
    for c in text_cols:
        if c in ex and ex[c] is not None:
            v = ex[c]
            if isinstance(v, (list, tuple)):
                v = " ".join(map(str, v))
            parts.append(str(v))
    return normalize_space(" ".join(parts)) if parts else ""


def load_aliases_from_csv(path: str) -> Dict[str, str]:
    """CSV with columns: alias,canonical."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["alias", "canonical"])
    df["alias"] = df["alias"].astype(str).str.strip()
    df["canonical"] = df["canonical"].astype(str).str.strip()
    # Keep last occurrence to allow overrides
    alias_map = {}
    for _, row in df.iterrows():
        a, c = row["alias"], row["canonical"]
        if a and c:
            alias_map[a] = c
    return alias_map


def default_brand_aliases() -> Dict[str, str]:
    return {
        "nike": "NIKE",
        "adidas": "ADIDAS",
        "apple": "APPLE",
        "samsung": "SAMSUNG",
        "h&m": "H&M",
        "h and m": "H&M",
        "uniqlo": "UNIQLO",
    }

def default_type_aliases() -> Dict[str, str]:
    return {
        "running shoes": "RUNNING_SHOES",
        "sneakers": "SNEAKERS",
        "smartphone": "SMARTPHONE",
        "phone": "SMARTPHONE",
        "t-shirt": "T_SHIRT",
        "tee": "T_SHIRT",
        "jeans": "JEANS",
        "hoodie": "HOODIE",
    }


# Placeholders (optional but helpful for normalization)
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


def compile_alias_regex(alias_to_canonical: Dict[str, str], min_len=2) -> Tuple[re.Pattern, Dict[str, str]]:
    cleaned = {}
    for a, c in alias_to_canonical.items():
        a_clean = normalize_space(a)
        c_clean = normalize_space(c)
        if len(a_clean) >= min_len:
            cleaned[a_clean] = c_clean
    aliases_sorted = sorted(cleaned.keys(), key=len, reverse=True)
    escaped = [re.escape(x) for x in aliases_sorted] or ["_NOOP_"]  # avoid empty pattern
    pattern = r"(?:^|(?<=\W))(" + "|".join(escaped) + r")(?=\W|$)"
    return re.compile(pattern, flags=re.IGNORECASE), {k.lower(): v for k, v in cleaned.items()}


def count_alias_hits(text_iter: Iterable[str],
                     brand_re: re.Pattern, brand_map: Dict[str, str],
                     type_re: re.Pattern, type_map: Dict[str, str]) -> Tuple[Counter, Counter]:
    brand_ctr, type_ctr = Counter(), Counter()
    for t in text_iter:
        if not t:
            continue
        for m in brand_re.finditer(t):
            alias = m.group(1).lower()
            brand_ctr[brand_map.get(alias, alias)] += 1
        for m in type_re.finditer(t):
            alias = m.group(1).lower()
            type_ctr[type_map.get(alias, alias)] += 1
    return brand_ctr, type_ctr


def build_top_specials(brand_ctr: Counter, type_ctr: Counter,
                       max_brand: int, max_type: int) -> Tuple[List[str], List[str]]:
    top_brands = [f"<BRAND={c}>" for c, _ in brand_ctr.most_common(max_brand)]
    top_types = [f"<TYPE={c}>" for c, _ in type_ctr.most_common(max_type)]
    if "<BRAND=OTHER>" not in top_brands:
        top_brands.append("<BRAND=OTHER>")
    if "<TYPE=OTHER>" not in top_types:
        top_types.append("<TYPE=OTHER>")
    return top_brands, top_types


def replace_with_specials(text: str,
                          brand_re: re.Pattern, brand_map: Dict[str, str], allowed_brands: set,
                          type_re: re.Pattern, type_map: Dict[str, str], allowed_types: set) -> str:
    def _b(m):
        alias = m.group(1).lower()
        canon = brand_map.get(alias)
        tok = f"<BRAND={canon}>" if canon and f"<BRAND={canon}>" in allowed_brands else "<BRAND=OTHER>"
        return tok
    def _t(m):
        alias = m.group(1).lower()
        canon = type_map.get(alias)
        tok = f"<TYPE={canon}>" if canon and f"<TYPE={canon}>" in allowed_types else "<TYPE=OTHER>"
        return tok
    text = brand_re.sub(_b, text)
    text = type_re.sub(_t, text)
    return text


def iter_text(dataset: DatasetDict, splits: List[str], text_cols: List[str]) -> Iterable[str]:
    for sp in splits:
        if sp not in dataset:
            continue
        for ex in dataset[sp]:
            yield join_text_fields(ex, text_cols)


def train_sentencepiece_gemma(corpus_path: str, out_prefix: str, vocab_size: int,
                              user_defined_symbols: List[str], character_coverage: float,
                              threads: int, split_digits: bool):
    """
    Train Gemma-like SPM Unigram:
    - BOS/EOS/PAD/UNK pieces, preserved whitespace, NFKC, byte_fallback, split_digits
    """
    args = [
        f"--input={corpus_path}",
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
        # Ensure uniqueness and stable order
        uds = []
        seen = set()
        for s in user_defined_symbols:
            if s not in seen:
                seen.add(s)
                uds.append(s)
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
        "add_bos_token": True,    # typical Gemma chat templates
        "add_eos_token": False
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

    print("[1/7] Loading dataset …")
    ds = load_dataset(args.hf_dataset, split=None, streaming=args.streaming)
    if isinstance(ds, IterableDataset):
        ds = DatasetDict({splits[0]: ds})
    elif isinstance(ds, dict):
        pass
    else:
        ds = DatasetDict({splits[0]: ds})

    # Alias maps
    print("[2/7] Preparing alias dictionaries …")
    brand_alias_map = load_aliases_from_csv(args.brand_alias_csv) if args.brand_alias_csv else default_brand_aliases()
    type_alias_map = load_aliases_from_csv(args.type_alias_csv) if args.type_alias_csv else default_type_aliases()

    brand_re, brand_norm = compile_alias_regex(brand_alias_map, min_len=args.min_alias_len)
    type_re, type_norm = compile_alias_regex(type_alias_map, min_len=args.min_alias_len)

    # Pass 1: frequency scan (to choose top-K specials)
    print("[3/7] Scanning corpus to select top brand/type specials …")
    brand_ctr, type_ctr = count_alias_hits(
        (join_text_fields(ex, text_cols) for sp in splits for ex in ds[sp]),
        brand_re, brand_norm, type_re, type_norm
    )
    brand_specials, type_specials = build_top_specials(brand_ctr, type_ctr,
                                                       args.max_brand_specials, args.max_type_specials)

    domain_specials = ["<PRICE>", "<PERCENT>", "<SIZE>", "<DIMENSION>", "<MODEL>", "<SKU>"] \
                      if args.add_domain_placeholders else []
    user_defined_symbols = brand_specials + type_specials + domain_specials

    print(f"  • Selected {len(brand_specials)} brand specials and {len(type_specials)} type specials.")
    if domain_specials:
        print(f"  • Added domain placeholders: {domain_specials}")
    total_specials = len(user_defined_symbols) + 4  # +4 for unk/bos/eos/pad
    if total_specials >= args.vocab_size:
        raise ValueError(f"Vocab too small for specials ({total_specials} >= {args.vocab_size}). "
                         f"Increase --vocab_size or reduce specials budgets.")

    # Pass 2: write processed training corpus
    print("[4/7] Writing processed training corpus …")
    corpus_path = os.path.join(args.output_dir, "corpus.txt")
    allowed_brand_tokens = set(brand_specials)
    allowed_type_tokens = set(type_specials)

    with open(corpus_path, "w", encoding="utf-8") as wf:
        for sp in splits:
            for ex in ds[sp]:
                t = join_text_fields(ex, text_cols)
                if not t:
                    continue
                if args.add_domain_placeholders:
                    t = apply_placeholders(t)
                t = replace_with_specials(
                    t, brand_re, brand_norm, allowed_brand_tokens,
                    type_re, type_norm, allowed_type_tokens
                )
                wf.write(t + "\n")

    # Train SPM Unigram with Gemma-ish flags
    print("[5/7] Training SentencePiece (Unigram, Gemma-style) …")
    out_prefix = os.path.join(args.output_dir, "ecom_sp_unigram")
    train_sentencepiece_gemma(
        corpus_path=corpus_path,
        out_prefix=out_prefix,
        vocab_size=args.vocab_size,
        user_defined_symbols=user_defined_symbols,
        character_coverage=args.character_coverage,
        threads=args.threads,
        split_digits=args.split_digits
    )
    print(f"  • Model written: {out_prefix}.model")

    # HF wrapper (GemmaTokenizerFast)
    print("[6/7] Creating Hugging Face tokenizer wrapper …")
    save_gemma_tokenizer_wrapper(
        sp_model_path=f"{out_prefix}.model",
        out_dir=args.output_dir,
        additional_specials=user_defined_symbols,
        model_max_length=8192
    )

    # Demo: subword regularization sampling
    if args.sample_encode:
        print("[7/7] Demo (subword-regularization sampling):")
        sp = spm.SentencePieceProcessor(model_file=f"{out_prefix}.model")
        demo = "Nike running shoes, 10×12 in, $19.99 — SKU: AB1234"
        if args.add_domain_placeholders:
            demo = apply_placeholders(demo)
        demo = replace_with_specials(demo, brand_re, brand_norm, allowed_brand_tokens,
                                     type_re, type_norm, allowed_type_tokens)
        for i in range(3):
            ids = sp.encode(demo, out_type=int, enable_sampling=True, nbest_size=-1, alpha=0.1)
            print(f"  Sample {i+1}: {ids}")

    print("Done ✅")


if __name__ == "__main__":
    main()
