import argparse
import csv
import json
import os
from typing import Optional, Dict, Any, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, set_seed
from peft import OFTConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OFT (Orthogonal Finetuning) of Qwen2.5-1B (or similar) on SST-2 as instruction-style task"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base causal LM model name or path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen2_5_1b_sst2_oft",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument(
        "--oft_r",
        type=int,
        default=8,
        help="OFT rank (number of OFT blocks per injected layer).",
    )
    parser.add_argument(
        "--oft_module_dropout",
        type=float,
        default=0.0,
        help="Module dropout probability for OFT blocks.",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization for base model weights (QLoRA-style).",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=2000,
        help="For quick experiments you can subsample training data.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--eval_accuracy_max_samples",
        type=int,
        default=872,
        help="Maximum number of validation samples used for accuracy evaluation.",
    )
    return parser.parse_args()


def build_prompt(sentence: str, label: Optional[int] = None) -> str:
    """
    将 SST-2 分类任务转换成指令式文本：
    输入：电影评论
    输出：positive / negative
    """
    label_text = ""
    if label is not None:
        label_text = "positive" if label == 1 else "negative"
    # 简单 prompt，你可以根据需要改成对话格式
    if label is None:
        return f"Review: {sentence}\nSentiment:"
    else:
        return f"Review: {sentence}\nSentiment: {label_text}"


def preprocess_dataset(tokenizer, max_source_length: int, max_target_length: int, num_train_samples: int):
    raw_datasets = load_dataset("glue", "sst2")

    if num_train_samples is not None and num_train_samples > 0:
        raw_datasets["train"] = raw_datasets["train"].select(range(min(num_train_samples, len(raw_datasets["train"]))))

    def tokenize_example(example):
        sentence = example["sentence"]
        label = example["label"]

        source_text = build_prompt(sentence, None)
        target_text = "positive" if label == 1 else "negative"

        source_ids = tokenizer(
            source_text,
            truncation=True,
            max_length=max_source_length,
            add_special_tokens=True,
        )["input_ids"]
        target_ids = tokenizer(
            target_text,
            truncation=True,
            max_length=max_target_length,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

        if len(input_ids) > max_source_length + max_target_length + 1:
            input_ids = input_ids[: max_source_length + max_target_length + 1]
            labels = labels[: max_source_length + max_target_length + 1]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    tokenized_train = raw_datasets["train"].map(
        tokenize_example,
        remove_columns=raw_datasets["train"].column_names,
    )
    tokenized_validation = raw_datasets["validation"].map(
        tokenize_example,
        remove_columns=raw_datasets["validation"].column_names,
    )

    return tokenized_train, tokenized_validation, raw_datasets["validation"]


def build_data_collator(tokenizer):
    def data_collator(features):
        batch = {}
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for ids, lbl in zip(input_ids, labels):
            padding_len = max_len - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * padding_len)
            padded_labels.append(lbl + [-100] * padding_len)
            attention_masks.append([1] * len(ids) + [0] * padding_len)

        batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
        return batch

    return data_collator


def evaluate_loss(model, tokenizer, eval_dataset, output_dir: str, batch_size: int):
    eval_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_dataset,
        data_collator=build_data_collator(tokenizer),
    )
    metrics = trainer.evaluate()
    eval_loss = float(metrics["eval_loss"])
    perplexity = float(torch.exp(torch.tensor(eval_loss)).item())
    return {
        "eval_loss": eval_loss,
        "perplexity": perplexity,
    }


def _sequence_logprob(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, label_ids: List[int]) -> float:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)

        prompt_len = input_ids.shape[1] - len(label_ids)
        total = 0.0
        for i, token_id in enumerate(label_ids):
            total += float(log_probs[0, prompt_len - 1 + i, token_id].item())
        return total


def evaluate_accuracy(model, tokenizer, raw_eval_dataset, max_samples: int) -> float:
    model.eval()
    device = next(model.parameters()).device
    label_options: Dict[int, str] = {
        0: "negative",
        1: "positive",
    }

    total = min(max_samples, len(raw_eval_dataset)) if max_samples and max_samples > 0 else len(raw_eval_dataset)
    correct = 0

    for idx in range(total):
        example = raw_eval_dataset[idx]
        prompt = build_prompt(example["sentence"], None)
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_ids = {k: v.to(device) for k, v in prompt_ids.items()}

        scores: Dict[int, float] = {}
        for label_id, label_text in label_options.items():
            label_ids = tokenizer(label_text, add_special_tokens=False)["input_ids"]
            full_input_ids = torch.cat(
                [prompt_ids["input_ids"], torch.tensor([label_ids], dtype=torch.long, device=device)],
                dim=1,
            )
            full_attention_mask = torch.ones_like(full_input_ids, device=device)
            scores[label_id] = _sequence_logprob(model, full_input_ids, full_attention_mask, label_ids)

        pred = max(scores.items(), key=lambda item: item[1])[0]
        if pred == int(example["label"]):
            correct += 1

    return correct / total if total > 0 else 0.0


def create_model_and_tokenizer(
    base_model: str,
    use_4bit: bool,
    oft_r: int,
    oft_module_dropout: float,
):
    # tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("/home/shikexuan/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B",trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}

    if use_4bit:
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "auto"

    # model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained("/home/shikexuan/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B", **load_kwargs)

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # 对所有线性层使用 OFT（不包含最终输出头），也可以根据需要改为具体模块名列表
    oft_config = OFTConfig(
        r=oft_r,
        oft_block_size=0,
        target_modules="all-linear",
        module_dropout=oft_module_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, oft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = create_model_and_tokenizer(
        base_model=args.base_model,
        use_4bit=args.use_4bit,
        oft_r=args.oft_r,
        oft_module_dropout=args.oft_module_dropout,
    )

    train_dataset, eval_dataset, raw_eval_dataset = preprocess_dataset(
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_train_samples=args.num_train_samples,
    )

    base_eval_loss_metrics = evaluate_loss(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        output_dir=os.path.join(args.output_dir, "base_eval"),
        batch_size=args.batch_size,
    )
    base_eval_accuracy = evaluate_accuracy(
        model=model,
        tokenizer=tokenizer,
        raw_eval_dataset=raw_eval_dataset,
        max_samples=args.eval_accuracy_max_samples,
    )

    print(
        "Base model metrics: "
        f"loss={base_eval_loss_metrics['eval_loss']:.4f}, "
        f"ppl={base_eval_loss_metrics['perplexity']:.4f}, "
        f"accuracy={base_eval_accuracy:.4f}"
    )

    metrics_path = os.path.join(args.output_dir, "base_model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_eval_loss": base_eval_loss_metrics["eval_loss"],
                "base_eval_perplexity": base_eval_loss_metrics["perplexity"],
                "base_eval_accuracy": base_eval_accuracy,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
    )

    train_losses: List[float] = []

    class LossRecorderCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and "loss" in logs:
                train_losses.append(logs["loss"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=build_data_collator(tokenizer),
        callbacks=[LossRecorderCallback()],
    )

    trainer.train()
    finetuned_metrics = trainer.evaluate()
    finetuned_eval_loss = float(finetuned_metrics["eval_loss"])
    finetuned_perplexity = float(torch.exp(torch.tensor(finetuned_eval_loss)).item())
    finetuned_accuracy = evaluate_accuracy(
        model=trainer.model,
        tokenizer=tokenizer,
        raw_eval_dataset=raw_eval_dataset,
        max_samples=args.eval_accuracy_max_samples,
    )

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "finetuned_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "finetuned_eval_loss": finetuned_eval_loss,
                "finetuned_eval_perplexity": finetuned_perplexity,
                "finetuned_eval_accuracy": finetuned_accuracy,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 保存 loss 曲线和 CSV
    if train_losses:
        steps = list(range(1, len(train_losses) + 1))
        # 保存为 CSV
        csv_path = os.path.join(args.output_dir, "training_loss.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            for s, loss in zip(steps, train_losses):
                writer.writerow([s, loss])
        print(f"Training loss CSV saved to {csv_path}")
        # 画图
        plt.figure()
        plt.plot(steps, train_losses, label="training loss")
        plt.xlabel("logging step")
        plt.ylabel("loss")
        plt.title("Training Loss Curve (OFT)")
        plt.legend()
        curve_path = os.path.join(args.output_dir, "training_loss_curve.png")
        plt.savefig(curve_path, dpi=200, bbox_inches="tight")
        print(f"Training loss curve saved to {curve_path}")

    print(
        "Finetuned model metrics: "
        f"loss={finetuned_eval_loss:.4f}, "
        f"ppl={finetuned_perplexity:.4f}, "
        f"accuracy={finetuned_accuracy:.4f}"
    )
    print(f"Training finished. OFT adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()

