

## Run Training
```bash
python train_encoder_decoder.py \
  --model-dir ./gemma3-270M_encoder \
  --dataset-path preprocessing/data/ecom_prepared \
  --output-dir ./models/gemma3-270m-ecom-mlm \
  --batch-size 64 \
  --epochs 3 \
  --lr 2e-5 \
  --mlm-prob 0.15 \
  --pad-to-multiple-of 0 \
  --bf16 \
  --group-by-length \
  --flash-attn2 \
  --gradient-checkpointing \
  --compile
```