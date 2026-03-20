import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with LoRA-finetuned Qwen2.5 model.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA adapter (training output_dir).",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input review text.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.lora_path)
    model.to(device)
    model.eval()

    prompt = f"Review: {args.text}\nSentiment:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== Model Output ===")
    print(decoded)


if __name__ == "__main__":
    main()

