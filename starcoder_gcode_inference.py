import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Path to your checkpoint
checkpoint_path = "finetune_starcoder2/checkpoint-40"

# Load the configuration
config = PeftConfig.from_pretrained(checkpoint_path)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the LoRA model
model = PeftModel.from_pretrained(base_model, checkpoint_path)

# Optional: Merge LoRA weights with base model for faster inference
# model = model.merge_and_unload()

# Set to evaluation mode
model.eval()

# Function to generate text
def generate_response(prompt, max_new_tokens=256):
    # Format the prompt as it was during training
    formatted_prompt = f"Prompt: {prompt}\nCompletion: "
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the completion part
    completion = generated_text.split("Completion: ")[-1].strip()
    
    return completion

# Example usage
if __name__ == "__main__":
    # Test with a sample prompt
    test_prompt = "Write a function to calculate the Fibonacci sequence in Python"
    response = generate_response(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")
    
    # Interactive mode
    print("\nEnter prompts (type 'exit' to quit):")
    while True:
        user_input = input("\nPrompt: ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(user_input)
        print(f"Response: {response}")