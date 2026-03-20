## Parameter-efficient Finetuning Mini Project
This mini-project demonstrates **parameter-efficient finetuning (PEFT)** using **OFT** on a relatively small LLM (**Qwen2.5 1B** or similar) for a simple NLP task (SST-2 sentiment classification reformulated as instruction-style text).

### Environment Setup

1. Create and activate a virtual environment (optional):

```bash
cd peft-mini-project
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> By default, dependencies rely on Hugging Face `transformers`, `datasets`, `accelerate`, `peft`, etc.

### Training Script Overview

- **Core script**: `train_lora_qwen_sst2.py`
  Uses OFT to fine-tune a Qwen2.5 1B-class causal language model on the `sst2` dataset in an instruction-style format.
  Before training, it evaluates the base model on the validation set (`loss / perplexity / accuracy`), then evaluates again after fine-tuning and saves the results as:
  `base_model_metrics.json` and `finetuned_metrics.json`.

- **Example run script**: `scripts/run_sst2_lora.sh`
  Wraps typical training hyperparameters for easy execution.

### Running Training

From the project root:

```bash
bash scripts/run_sst2_lora.sh
```

The script will:

- Automatically download the `sst2` dataset;
- Download the pretrained Qwen2.5 1B model (or the model specified in the run script);
- Train only the small set of parameters introduced by OFT;
- Save outputs to `outputs/qwen2_5_1b_sst2_oft/`.

### Inference Example

After training, you can run simple inference with:

```bash
python infer_lora_qwen.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --lora_path outputs/qwen2_5_1b_sst2_oft \
  --text "The movie is boring and too long."
```

(See `infer_lora_qwen.py` in the project code; adjust as needed.)

### Project Structure

```text
peft-mini-project/
  README.md
  requirements.txt
  train_lora_qwen_sst2.py     # Training script (OFT + Qwen2.5 1B + SST-2)
  infer_lora_qwen.py          # Inference script (base model + adapter)
  scripts/
    run_sst2_lora.sh          # One-click training script
  outputs/                    # Training outputs (created automatically)
```

You can extend this project further by comparing different PEFT methods, adding more NLP tasks (e.g., AG News, QNLI), or including visualization and reporting.

