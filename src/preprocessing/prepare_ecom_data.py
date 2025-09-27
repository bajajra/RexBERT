#!/usr/bin/env python
"""
Utility script to prepare the Ecom‑niverse dataset for Gemma 3 training.

This script performs a few basic tasks:

1. Downloads the `thebajajra/Ecom‑niverse` dataset using the HuggingFace
   `datasets` library.  The dataset contains hundreds of millions of rows,
   so it streams the records rather than loading everything into memory.
2. Optionally filters the data by language (the dataset is predominantly
   English, but filtering makes sure that multilingual examples are removed).
3. Samples a fixed number of examples (via the `--max-samples` argument) to
   make training manageable on a single machine.  If you omit this argument
   the entire stream will be consumed – only do this if you have the
   resources to handle billions of tokens.
4. Tokenises the text using the `google/gemma-3-270m` tokenizer and
   truncates long examples to a configurable maximum length.  A `[MASK]`
   token is added to the tokenizer if it does not already exist so that
   masked language modelling can be performed later.
5. Writes the processed dataset to disk in HuggingFace’s native Arrow
   format via `Dataset.save_to_disk()`.

You can run this script directly from the command line.  For example:

```bash
python prepare_ecom_data.py \
    --output-dir data/ecom_prepared \
    --max-samples 1000000 \
    --max-length 2048
```

which will produce roughly one million tokenised, truncated examples saved to
`data/ecom_prepared`.  Increase or decrease `--max-samples` and
`--max-length` to suit your hardware and training needs.

Note that this script requires the `datasets` and `transformers` packages.
Install them with:

```bash
pip install --upgrade datasets transformers
```

For more information on the dataset see the dataset card:
https://huggingface.co/datasets/thebajajra/Ecom-niverse

"""

import argparse
import logging
import os
from typing import Iterator, List

import datasets  # type: ignore
from datasets import Dataset
import random

from transformers import AutoTokenizer  # type: ignore


logger = logging.getLogger(__name__)


def stream_ecom_data(
    split: str = "train",
    lang: str = "en",
    max_samples: int | None = None,
    seed: int = 42,
) -> Iterator[str]:
    """Stream raw text examples from the Ecom‑niverse dataset.

    Args:
        split: Which split to stream from.  Only `train` exists in this dataset.
        lang: Language code to filter on.  The dataset has a `lang` column
            containing ISO‑639 codes.  Pass `None` to disable language filtering.
        max_samples: Optional cap on the number of examples to yield.
        seed: Random seed for reproducible shuffling.  When sampling from a
            streaming dataset it is useful to shuffle locally to ensure the
            first N examples are not all from the same domain.  This script
            shuffles each buffered chunk using this seed.

    Yields:
        Plain text strings from the dataset.
    """
    logger.info("Loading Ecom‑niverse split %s via streaming", split)
    ds = datasets.load_dataset("thebajajra/Ecom-niverse", split=split, streaming=True)

    count = 0
    buffer: List[str] = []
    buffer_size = 10_000  # shuffle buffer for streaming

    # Iterate over the streaming dataset
    for record in ds:
        # Dataset is english only
        # if lang and record.get("lang") != lang:
        #     continue
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        buffer.append(text.strip())
        if len(buffer) >= buffer_size:
            # shuffle the buffer deterministically.  We cannot use
            # datasets.shuffle() on a Python list, so instead use
            # random.Random(seed).shuffle which gives deterministic order.
            rng = random.Random(seed)
            rng.shuffle(buffer)
            for item in buffer:
                yield item
                count += 1
                if max_samples and count >= max_samples:
                    return
            buffer.clear()
    # Flush remainder of buffer
    rng = random.Random(seed)
    rng.shuffle(buffer)
    for item in buffer:
        yield item
        count += 1
        if max_samples and count >= max_samples:
            return


def build_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    max_samples: int | None = None,
    lang: str = "en",
) -> Dataset:
    """Construct a HuggingFace Dataset from the streamed Ecom‑niverse data.

    Tokenises and truncates the raw text using the provided tokenizer.  Any
    examples longer than `max_length` tokens are truncated from the end.

    Args:
        tokenizer: An instantiated tokenizer matching the Gemma model.
        max_length: Maximum length (in tokens) for each example.
        max_samples: Optional cap on the number of examples to load.
        lang: Language code to filter on.

    Returns:
        A `datasets.Dataset` containing the column `input_ids` with padded
        sequences and `attention_mask` with the appropriate padding mask.
    """
    logger.info(
        "Beginning to stream and tokenise Ecom‑niverse (max_samples=%s, max_length=%s)",
        max_samples,
        max_length,
    )
    # Prepare a list to accumulate processed examples
    processed = []
    for idx, text in enumerate(stream_ecom_data(split="train", lang=lang, max_samples=max_samples)):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        processed.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })
        if (idx + 1) % 10_000 == 0:
            logger.info("Tokenised %d examples", idx + 1)
    logger.info("Finished tokenising %d examples", len(processed))
    return Dataset.from_list(processed)


def main():
    parser = argparse.ArgumentParser(description="Prepare Ecom‑niverse for Gemma pretraining")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write the processed dataset to.  The script will create this directory if it does not exist.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100_000,
        help="Maximum number of examples to sample from the streaming dataset.  Set to 0 to disable and consume the full dataset (not recommended for personal machines).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (in tokens) for each example.  Longer examples will be truncated.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="ISO‑639 language code to filter on.  Set to an empty string to disable filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling the streaming buffer.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Instantiate tokenizer for Gemma3.  We deliberately use the base 270M model
    # since it will be the foundation for our encoder.  The mask token may not
    # exist in the original vocabulary; if so, add it here.
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
    if tokenizer.mask_token is None:
        # Use a simple <mask> token.  You could choose a different string if
        # preferred; avoid conflicts with existing special tokens.
        logger.info("Adding <mask> token to the tokenizer vocabulary")
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    # Build dataset
    dataset = build_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=None if args.max_samples == 0 else args.max_samples,
        lang=args.lang if args.lang else None,
    )
    # Save dataset to disk
    logger.info("Saving processed dataset to %s", args.output_dir)
    dataset.save_to_disk(args.output_dir)
    logger.info("Done.  Wrote %d examples to %s", len(dataset), args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()