#!/usr/bin/env python
"""
Train an encoder–decoder model initialised from Gemma3 on an e‑commerce
dataset using a masked language modelling objective.

This script implements the first stages of the EmbeddingGemma recipe:

* We initialise an `EncoderDecoderModel` from the pretrained
  `google/gemma-3-270m` weights.  This step follows the adaptation
  procedure described in the EmbeddingGemma paper: we take a decoder‑only
  model and convert it into an encoder–decoder architecture by reusing
  the same weights for both the encoder and decoder.  The resulting
  model is capable of ingesting a fully observed input (via the encoder)
  and generating a target sequence (via the decoder).

* We then further adapt this model via a masked language modelling (MLM)
  objective reminiscent of BERT, but implemented within a seq2seq
  framework.  For each example we produce a masked version of the
  original input (some tokens are replaced with a `[MASK]` token) and
  compute a reconstruction loss only for those masked positions.  The
  decoder learns to predict the original tokens, and the encoder
  consequently learns bidirectional contextual representations.  This
  objective aligns with the goal of producing a strong encoder as
  described in the EmbeddingGemma paper【622905097956174†L110-L117】.

After training completes the script saves both the full
encoder–decoder model and the standalone encoder for downstream use.

Example usage:

```bash
python train_gemma_encoder.py \
    --dataset-path data/ecom_prepared \
    --output-dir models/gemma3-ecom-encoder \
    --per-device-batch-size 4 \
    --num-epochs 1
```

This script requires the `transformers`, `datasets` and `torch`
packages.  Install them via:

```bash
pip install --upgrade transformers datasets torch
```
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch  # type: ignore
from datasets import Dataset, load_from_disk  # type: ignore
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


logger = logging.getLogger(__name__)


def mask_tokens(
    input_ids: List[int],
    mask_token_id: int,
    special_token_ids: List[int],
    vocab_size: int,
    mlm_probability: float = 0.15,
) -> Tuple[List[int], List[int]]:
    """Perform BERT‑style random masking on a single sequence.

    Returns a pair of lists `(masked_input_ids, labels)` where
    `labels` contains the original token ids for masked positions and
    `-100` elsewhere (to ignore those positions in the loss).

    The masking procedure follows the common heuristic used in BERT and
    other MLM setups: for each token (excluding special tokens) a
    masking decision is sampled with probability `mlm_probability`.
    * 80% of masked tokens are replaced by `[MASK]`.
    * 10% are replaced by a random token.
    * 10% are left unchanged.

    Args:
        input_ids: A list of token ids representing the input sequence.
        mask_token_id: The token id used for masking.
        special_token_ids: A list of special token ids that should never
            be masked (e.g. `pad_token_id`, `eos_token_id`, etc.).
        mlm_probability: Probability of masking each token.

    Returns:
        Tuple of (masked_input_ids, labels).
    """
    labels = input_ids.copy()
    masked_input = input_ids.copy()

    for i, token_id in enumerate(input_ids):
        if token_id in special_token_ids:
            labels[i] = -100
            continue
        if np.random.rand() < mlm_probability:
            # This token will be masked
            # 80% replace with mask token
            if np.random.rand() < 0.8:
                masked_input[i] = mask_token_id
            # 10% replace with random token
            elif np.random.rand() < 0.5:
                masked_input[i] = int(np.random.randint(0, vocab_size))
            # 10% leave unchanged
            # labels[i] remains original token id
        else:
            # Not masked; label should be ignored
            labels[i] = -100
    return masked_input, labels


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    mlm_probability: float = 0.15,
) -> Dataset:
    """Apply masking to the entire dataset.

    The input dataset must contain `input_ids` and `attention_mask` columns.
    The returned dataset will contain `input_ids` (masked), `attention_mask`
    and `labels` with `-100` in positions that are not masked.

    Args:
        dataset: A `datasets.Dataset` with tokenised examples.
        tokenizer: The tokenizer used for encoding the data.  The mask
            token must already exist in the tokenizer's vocabulary.
        mlm_probability: Masking probability for each token.

    Returns:
        A new `datasets.Dataset` with masked inputs and labels.
    """
    mask_token_id = tokenizer.mask_token_id
    special_ids = set(tokenizer.all_special_ids)
    vocab_size = len(tokenizer)

    def transform(batch: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        masked_inputs = []
        labels_list = []
        for input_ids in batch["input_ids"]:
            masked, labels = mask_tokens(
                input_ids,
                mask_token_id=mask_token_id,
                special_token_ids=special_ids,
                vocab_size=vocab_size,
                mlm_probability=mlm_probability,
            )
            masked_inputs.append(masked)
            labels_list.append(labels)
        return {"input_ids": masked_inputs, "labels": labels_list}

    # Use batched mapping for efficiency; remove original attention_mask
    processed = dataset.map(
        transform,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in {"input_ids", "attention_mask"}],
    )
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gemma3 encoder via MLM on e‑commerce data")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the tokenised Ecom‑niverse dataset (created by prepare_ecom_data.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the trained model and tokenizer",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="google/gemma-3-270m",
        help="Name or path of the pretrained Gemma model to adapt",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=4,
        help="Batch size per device during training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for the AdamW optimiser",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="Probability of masking each token (BERT style)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps for the learning rate scheduler",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=10_000,
        help="Interval (in training steps) at which to save checkpoints",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Interval of logging",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional maximum number of training examples to use.  Useful for quick experiments",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Load the preprocessed dataset
    logger.info("Loading dataset from %s", args.dataset_path)
    dataset = load_from_disk(args.dataset_path)
    # Optionally subsample for debugging
    if args.max_train_samples:
        dataset = dataset.select(range(args.max_train_samples))

    # Load tokenizer and ensure mask token exists
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.mask_token is None:
        logger.info("Tokenizer has no mask token; adding <mask> as the mask token")
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    # Preprocess dataset to create masked inputs and labels
    logger.info("Applying masking to dataset (mlm_probability=%.2f)", args.mlm_probability)
    with torch.no_grad():
        processed_dataset = preprocess_dataset(dataset, tokenizer, mlm_probability=args.mlm_probability)

    # Initialise encoder–decoder model from a pretrained decoder
    logger.info("Initialising EncoderDecoderModel from %s", args.pretrained_model)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        args.pretrained_model,
        args.pretrained_model,
    )
    # Resize token embeddings if special tokens were added
    model.resize_token_embeddings(len(tokenizer))
    # Set special token ids required by seq2seq models
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    # Reduce sequence lengths if needed (Gemma 270M can handle long contexts)

    # Data collator pads inputs and creates decoder_input_ids from labels
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # Prepare training arguments
    total_steps = None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        gradient_accumulation_steps=1,
        bf16=True,
        report_to=[],
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    # Train
    logger.info("Starting training")
    trainer.train()
    logger.info("Training complete")

    # Save the full model and tokenizer
    logger.info("Saving full encoder–decoder model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Extract and save just the encoder for downstream use
    encoder_output_dir = os.path.join(args.output_dir, "encoder")
    logger.info("Saving encoder to %s", encoder_output_dir)
    os.makedirs(encoder_output_dir, exist_ok=True)
    model.get_encoder().save_pretrained(encoder_output_dir)
    # Also save tokenizer here for convenience
    tokenizer.save_pretrained(encoder_output_dir)
    logger.info("All done.  The encoder can now be loaded with `AutoModel.from_pretrained('%s')`.", encoder_output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()