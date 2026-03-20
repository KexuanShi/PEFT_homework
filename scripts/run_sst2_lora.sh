#!/usr/bin/env bash

set -e

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python train_lora_qwen_sst2.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir outputs/qwen2_5_1b_sst2_oft \
  --batch_size 16 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --num_train_samples 2000 \
  --use_4bit \
  --gradient_accumulation_steps 4 \
  --max_source_length 128 \
  --max_target_length 8 \
  --logging_steps 10 \
  --eval_steps 200 \
  --save_steps 200

