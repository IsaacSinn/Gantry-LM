'''
To fine-tune efficiently with a low cost, we use PEFT library for Low-Rank Adaptation (LoRA) training
and bitsandbytes for 4bit quantization. 
We also use the SFTTrainer from TRL.

Use command:
accelerate launch train_starcoder.py --model_id "bigcode/starcoder2-3b" --dataset_name "json" --dataset_path "data/train_1000.jsonl" --dataset_text_field "completion" --prompt_field "prompt" --max_seq_length 1024 --max_steps 1000 --micro_batch_size 1 --gradient_accumulation_steps 4 --learning_rate 2e-4 --warmup_steps 100 --num_proc 8


'''


import argparse
import multiprocessing
import os

import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
from trl import SFTTrainer, SFTConfig


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="bigcode/starcoder2-3b")
    parser.add_argument("--dataset_name", type=str, default="json")
    parser.add_argument("--dataset_path", type=str, default="data/train_1000.jsonl")
    parser.add_argument("--dataset_text_field", type=str, default="completion")
    parser.add_argument("--prompt_field", type=str, default="prompt")

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_starcoder2")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    return parser.parse_args()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(args):
    # config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # load model and dataset
    token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
    )
    print_trainable_parameters(model)

    # Load training data
    train_data = load_dataset(
        args.dataset_name,
        data_files=args.dataset_path,
        split="train",
        token=token,
        num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
    )
    
    # Load development data (assuming you have a dev.jsonl file)
    # If your dev data is in a different format, adjust accordingly
    dev_data = load_dataset(
        args.dataset_name,
        data_files="data/dev_100.jsonl",  # Path to your dev data
        split="train",  # Using "train" split since we're loading from a file
        token=token,
        num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
    )

    # Function to format the data with prompt and completion
    def format_data(example):
        return {
            "text": f"Prompt: {example[args.prompt_field]}\nCompletion: {example[args.dataset_text_field]}"
        }
    
    # Apply the formatting to both datasets
    train_data = train_data.map(format_data, num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count())
    dev_data = dev_data.map(format_data, num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count())

    # setup the trainer with SFTConfig and evaluation
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=dev_data,  # Add the evaluation dataset
        args=SFTConfig(
            per_device_train_batch_size=args.micro_batch_size,
            per_device_eval_batch_size=args.micro_batch_size,  # Eval batch size
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            
            # Evaluation settings
            eval_strategy="steps",  # Evaluate during training
            eval_steps=10,  # Evaluate every 100 steps #TODO: change to 100
            save_strategy="steps",  # Save checkpoints
            save_steps=10,  # Save every 100 steps #TODO: change to 100
            save_total_limit=3,  # Keep only the 3 most recent checkpoints
            load_best_model_at_end=True,  # Load the best model at the end
            metric_for_best_model="eval_loss",  # Use eval loss to determine best model
            
            logging_strategy="steps",
            logging_steps=10,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            seed=args.seed,
            run_name=f"train-{args.model_id.split('/')[-1]}",
            report_to="wandb",
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",  # Use our formatted text field
            packing=False,
            dataset_num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
        ),
        peft_config=lora_config,
    )

    # launch
    print("Training...")
    trainer.train()

    print("Saving the last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    if args.push_to_hub:
        trainer.push_to_hub("Upload model")
    print("Training Done! 💥")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)