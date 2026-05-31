import os
import sys

# Keep ONLY the essential compiler bypass switches
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"

import torch
import model as model_utils
from unsloth import FastLanguageModel
import argparse
import data
import train
from config import Config
from transformers import pipeline


def main(args):
    print(f"...loading model: {args.model_name}")

    Config.MODEL_NAME = args.model_name
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.learning_rate
    Config.OUTPUT_DIR = args.output_dir
    Config.MAX_OUTPUT_TOKEN = args.max_output_token
    Config.TEMPERATURE = args.temperature
    Config.TOP_P = args.top_p
    Config.MAX_LENGTH = args.max_length

    print(f" --- Initializing Unsloth unified loader ({args.fine_tuning.upper()})...")
    model, tokenizer = model_utils.load_model_and_tokenizer(fine_tuning_type=args.fine_tuning)

    dataset = data.get_dataset()

    print(" --- Starting training...")
    train.train(model, tokenizer, dataset)

    if args.evaluate:
        print(" --- Evaluating model...")
        evaluator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("Evaluation pipeline created successfully")

    print("\n --- All done!")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Gemma with LoRA, QLoRA, rsLoRA, and DoRA")

    parser.add_argument(
        "--model_name",
        type=str,
        default=Config.MODEL_NAME,
        help="Unsloth model ID (default:unsloth/gemma-2-2b-it)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=Config.EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=Config.BATCH_SIZE,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=Config.LEARNING_RATE,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_output_token",
        type=int,
        default=Config.MAX_OUTPUT_TOKEN,
        help="Maximum number of new tokens to generate during evaluation"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=Config.TOP_P,
        help="Nucleus sampling probability threshold (controls diversity during generation)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=Config.TEMPERATURE,
        help="Sampling temperature (higher = more randomness in generation)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=Config.MAX_LENGTH,
        help="Maximum sequence length for tokenizing prompts and completions."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Config.OUTPUT_DIR,
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training"
    )
    parser.add_argument(
        "--fine_tuning",
        type=str,
        choices=["sft", "lora", "qlora", "rslora", "dora"], 
        default="lora",
        help="Type of fine-tuning to run: 'sft', 'lora', 'qlora', 'rslora', or 'dora'."
    )

    args = parser.parse_args()
    main(args)