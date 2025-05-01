# pip install -q transformers
# pip install bitsandbytes accelerate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuration
checkpoint = "bigcode/starcoder2-3b"


quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    quantization_config=quantization_config,
    device_map="cuda"
)

model.eval()

# Function to generate text
def generate_response(prompt, max_new_tokens=1024):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # max_new_tokens=max_new_tokens,
            # temperature=0.7,
            # top_p=0.9,
            # do_sample=True,
            # repetition_penalty=1.2,
            # no_repeat_ngram_size=3,
            # early_stopping=True
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

# Example usage
if __name__ == "__main__":
    # Test with a sample prompt
    test_prompt = "def print_hello_world():"
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
