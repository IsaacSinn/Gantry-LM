import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import re
import difflib

# Path to your checkpoint
checkpoint_path = "starcoder2-3b-gcode/checkpoint-20"
eval_file_path = "data/dev_sample.jsonl"

# Load the configuration
config = PeftConfig.from_pretrained(checkpoint_path)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the LoRA model
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.eval()

# Function to generate text
def generate_response(prompt, max_new_tokens=512):
    # formatted_prompt = f"Prompt: {prompt}\nCompletion: "
    formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            # NOTE: 
            # input_ids=inputs.input_ids,
            # attention_mask=inputs.attention_mask,
            # max_new_tokens=max_new_tokens,
            # temperature=0.7,
            # top_p=0.9,
            # do_sample=False  # Set to False for deterministic output during evaluation\
            **inputs
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = generated_text.split("Completion: ")[-1].strip()
    
    return completion

# Load evaluation data
def load_eval_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Normalize G-code for comparison
def normalize_gcode(gcode):
    # Remove comments
    gcode = re.sub(r';.*$', '', gcode, flags=re.MULTILINE)
    # Remove empty lines
    gcode = re.sub(r'\n\s*\n', '\n', gcode)
    
    # Temporarily replace newlines with a special marker
    gcode = gcode.replace('\n', '<|NEWLINE|>')
    
    # Standardize whitespace (including \n)
    gcode = re.sub(r'\s+', ' ', gcode).strip()
    
    # Restore newlines
    gcode = gcode.replace('<|NEWLINE|>', '\n')

    # Remove the <|endoftext|>
    gcode = gcode.split('<|endoftext|>')[0]

    # Split into commands - now split by newlines and spaces
    commands = []
    for line in gcode.split('\n'):
        if line.strip():
            commands.append(line.strip())
    
    return commands

# Calculate command-level metrics
def calculate_command_metrics(pred_commands, true_commands):
    # Find common commands
    common = set(pred_commands).intersection(set(true_commands))
    
    # Calculate metrics
    precision = len(common) / len(pred_commands) if pred_commands else 0
    recall = len(common) / len(true_commands) if true_commands else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Calculate sequence similarity using difflib
def sequence_similarity(pred, true):
    matcher = difflib.SequenceMatcher(None, pred, true)
    return matcher.ratio()

# Main evaluation function
def evaluate_model():
    eval_data = load_eval_data(eval_file_path)
    
    results = {
        'command_precision': [],
        'command_recall': [],
        'command_f1': [],
        'sequence_similarity': [],
        'exact_match': []
    }
    
    print(f"Evaluating model on {len(eval_data)} examples...")
    
    for item in tqdm(eval_data):
        prompt = item['prompt']
        true_completion = item['completion']

        # Generate prediction
        pred_completion = generate_response(prompt)

        # Normalize G-code for comparison
        pred_commands = normalize_gcode(pred_completion)
        true_commands = normalize_gcode(true_completion)

        print(pred_commands)
        print("--------------------------------")
        print(true_commands)
        
        # Calculate command-level metrics
        precision, recall, f1 = calculate_command_metrics(pred_commands, true_commands)
        results['command_precision'].append(precision)
        results['command_recall'].append(recall)
        results['command_f1'].append(f1)
        
        # Calculate sequence similarity
        similarity = sequence_similarity(pred_completion, true_completion)
        results['sequence_similarity'].append(similarity)
        
        # Check for exact match
        exact_match = pred_completion.strip() == true_completion.strip()
        results['exact_match'].append(int(exact_match))
    
    # Calculate average metrics
    avg_results = {k: np.mean(v) for k, v in results.items()}
    
    # Print results
    print("\n===== Evaluation Results =====")
    print(f"Command Precision: {avg_results['command_precision']:.4f}")
    print(f"Command Recall: {avg_results['command_recall']:.4f}")
    print(f"Command F1 Score: {avg_results['command_f1']:.4f}")
    print(f"Sequence Similarity: {avg_results['sequence_similarity']:.4f}")
    print(f"Exact Match Rate: {avg_results['exact_match']:.4f}")
    
    return avg_results, results

# Save detailed results to file
def save_results(avg_results, detailed_results, output_file="model/starcoder-3b/evaluation_results.json"):
    with open(output_file, 'w') as f:
        json.dump({
            "average_metrics": avg_results,
            "detailed_metrics": {k: list(map(float, v)) for k, v in detailed_results.items()}
        }, f, indent=2)
    print(f"Detailed results saved to {output_file}")

# Run evaluation
if __name__ == "__main__":
    avg_results, detailed_results = evaluate_model()
    save_results(avg_results, detailed_results)
    
    # Optional: Print examples of predictions vs ground truth
    print("\n===== Sample Predictions =====")
    eval_data = load_eval_data(eval_file_path)
    for i in range(min(3, len(eval_data))):
        prompt = eval_data[i]['prompt']
        true_completion = eval_data[i]['completion']
        pred_completion = generate_response(prompt)
        
        print(f"\nExample {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"True: {true_completion[:100]}..." if len(true_completion) > 100 else f"True: {true_completion}")
        print(f"Pred: {pred_completion[:100]}..." if len(pred_completion) > 100 else f"Pred: {pred_completion}")
        print(f"Similarity: {sequence_similarity(pred_completion, true_completion):.4f}")