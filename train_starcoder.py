from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
import torch
from accelerate import Accelerator

# Model configuration
model_id = "bigcode/starcoder2-3b"
dataset_path = "data/train_sample.jsonl"  # Update with your full dataset path

# Quantization for efficient training
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Configure LoRA
peft_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_proj", "c_attn", "q_attn"],  # Typical target modules for StarCoder models
)

# Configure the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="prompt",  # Input field in your dataset
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        max_steps=1000,  # Adjust based on your dataset size
        learning_rate=2e-5,
        fp16=True,
        logging_steps=10,
        output_dir="output/gcode-starcoder2-3b",
        save_strategy="steps",
        save_steps=100,
    ),
)

# Start training
trainer.train()

# Save the model
trainer.model.save_pretrained("output/gcode-starcoder2-3b")