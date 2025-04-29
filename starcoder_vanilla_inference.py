# pip install -q transformers
# pip install bitsandbytes accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# to use 4bit use `load_in_4bit=True` instead
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

checkpoint = "bigcode/starcoder2-3b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to("cuda")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
