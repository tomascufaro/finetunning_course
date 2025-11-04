# merge_lora.py
# /// script
# dependencies = [
#     "transformers>=4.36.0",
#     "peft>=0.7.0",
#     "torch>=2.0.0"
# ]
# ///

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
base_model_name = "HuggingFaceTB/SmolLM3-3B-Base"
adapter_repo = "tomascufaro/SmolLM3-3B-LoRA-SFT"
merged_model_name = "tomascufaro/SmolLM3-3B-LoRA-SFT"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("Loading LoRA adapter...")
model_with_adapter = PeftModel.from_pretrained(base_model, adapter_repo)

print("Merging adapter with base model...")
merged_model = model_with_adapter.merge_and_unload()

print("Saving merged model locally...")
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

print("Pushing to Hugging Face Hub...")
merged_model.push_to_hub(merged_model_name)
tokenizer.push_to_hub(merged_model_name)

print(f"✓ Done! Merged model pushed to: https://huggingface.co/{merged_model_name}")

