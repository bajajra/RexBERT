# convert_and_save_encoder.py
# -*- coding: utf-8 -*-
import argparse, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gemma3_biencoder import Gemma3EncoderForMaskedLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="google/gemma-3-270m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--add-mask-token", action="store_true",
                    help="Add <mask> to tokenizer vocab if missing (recommended for MLM).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) Load original Gemma-3 270M (decoder-only)
    print(f"Loading source model: {args.src}")
    dec = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(args.src, use_fast=True)

    # 2) Create encoder-only model with same config
    enc = Gemma3EncoderForMaskedLM(dec.config)

    # 3) Copy weights (token embeddings + all transformer layers)
    print("Copying transformer weights into encoder…")
    enc.encoder.load_state_dict(dec.model.state_dict(), strict=True)

    # 4) Tie lm_head to embeddings & (optionally) copy the original lm_head
    #    (not required when tying, but harmless)
    enc.lm_head.weight.data = dec.lm_head.weight.data.clone()

    # 5) Optional: add a <mask> token so MLM works out of the box
    if args.add_mask_token and tok.mask_token is None:
        print("Adding <mask> token to tokenizer…")
        tok.add_special_tokens({"mask_token": "<mask>"})
        # Resize embeddings in the encoder and (tied) head
        enc.resize_token_embeddings(len(tok))

    # 6) Save encoder + tokenizer
    print(f"Saving encoder checkpoint to: {args.out}")
    enc.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Done.")

if __name__ == "__main__":
    main()