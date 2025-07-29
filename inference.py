from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import torch

# Set adapter and base model paths
adapter_path = "./adapters/sports"  #change according to the adapter
base_model_name = "gpt2" 

# Load tokenizer and base model
tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # add pad_token to avoid warnings
base_model = GPT2LMHeadModel.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Inference function
def generate(prompt, max_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompt
prompt = "XXXX"
output = generate(prompt)
print("=== Prompt ===")
print(prompt)
print("\n=== Response ===")
print(output)