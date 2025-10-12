from datasets import load_dataset
from sentence_transformers.losses import CoSENTLoss
from datasets import load_dataset, load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from transformers import TrainerCallback
import torch
import os

model_name = "thebajajra/RexBERT-base"
dataset_path = "thebajajra/amazon-esci-english-small"

# Load a model to train/finetune
model = SentenceTransformer(model_name, device="cuda", model_kwargs={
    "attn_implementation": "flash_attention_2"
})


# Initialize the CoSENTLoss
# This loss requires pairs of text and a float similarity score as a label
loss = CoSENTLoss(model)

# Load an example training dataset that works with our loss function:
train_dataset = load_dataset(dataset_path, "default", split="train")
test_dataset = load_dataset(dataset_path, "default", split="test")

def label_to_score(row):
    """
    Convert the esci_label to a float score.
    The CoSENTLoss requires a float score as label.
    """
    label = row["esci_label"]
    if label == "E":
        score = 1.0
    elif label == "S":
        score = 0.66
    elif label == "C":
        score = 0.33
    elif label == "I":
        score = 0.0
    row["score"] = score
    return row


def process_hf_dset(hf_dataset):
    """
    Process a Hugging Face dataset to the format required by Sentence Transformers.
    """
    hf_dataset = hf_dataset.rename_column("query", "sentence1")
    hf_dataset = hf_dataset.rename_column("product_title", "sentence2")
    hf_dataset = hf_dataset.remove_columns(
        ['example_id', 'query_id', 'product_id', 'product_locale', 'esci_label', 'small_version', 'large_version', 'split', 'product_description', 'product_bullet_point', 'product_brand', 'product_color', '__index_level_0__']
    )
    return hf_dataset

train_dataset = train_dataset.map(label_to_score, num_proc=8)
test_dataset = test_dataset.map(label_to_score, num_proc=8)

train_dataset = process_hf_dset(train_dataset)
test_dataset = process_hf_dset(test_dataset)

dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
    batch_size=128,
)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/{}-amazon-esci-english-small-cosent".format(model_name.split("/")[-1]),
    # Optional training parameters:
    num_train_epochs=3,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=128,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    logging_steps=10,
    run_name="{}-amz-esci-small-cosent".format(model_name.split("/")[-1]), # Will be used in W&B if `wandb` is installed
    metric_for_best_model="sts-dev_pearson_cosine",
    greater_is_better=True,
    gradient_checkpointing=True,
    bf16_full_eval=True,
    report_to=["wandb"],  # or "tensorboard"
    # eval_on_start=True,
    # load_best_model_at_end=True,
)

LOG_DIR = "profiles/run"
prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],  # drop CUDA if CPU only
    schedule=schedule(skip_first=10, wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=tensorboard_trace_handler(LOG_DIR),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,     # get Python stack traces
    with_flops=True      # rough FLOPs estimates
)

class TorchProfilerCallback(TrainerCallback):
    def __init__(self, prof): self.prof, self._ctx = prof, None
    def on_train_begin(self, args, state, control, **kw):
        self._ctx = self.prof.__enter__()  # start profiler
    def on_step_end(self, args, state, control, **kw):
        self.prof.step()                   # advance profiler schedule each step
    def on_train_end(self, args, state, control, **kw):
        self.prof.__exit__(None, None, None)  # flush/close

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=loss,
    evaluator=dev_evaluator,
    # callbacks=[TorchProfilerCallback(prof)]
)

trainer.train()

