# main.py
# /// script
# dependencies = [
#     "trl[sft]>=0.7.0",
#     "transformers>=4.36.0", 
#     "datasets>=2.14.0",
#     "accelerate>=0.24.0",
#     "peft>=0.7.0",
#     "tensorboard>=2.14.0",
#     "wandb>=0.17.0",
# ]
# ///

import torch
from itertools import islice
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from peft import LoraConfig
import wandb


wandb.login()
# ==================== CONFIGURATION ====================
# Dataset size configuration - adjust for testing vs production
TRAINING_SAMPLES = 10000  # Set to 100 for quick testing, 1000+ for actual training
# For production: 5000-10000 samples recommended for better quality

# ==================== MODEL & TOKENIZER ====================
model_name = "HuggingFaceTB/SmolLM3-3B-Base"
instruct_model_name = "HuggingFaceTB/SmolLM3-3B"  # Instruct model for chat template
new_model_name = "SmolLM3-3B-LoRA-SFT"

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency on modern GPUs
    device_map="auto"  # Automatically distribute model across available devices
)

# Load base model tokenizer for training
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Configure tokenizer for proper padding and generation
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Pad on right for training
model.config.pad_token_id = tokenizer.eos_token_id

# Load instruct model tokenizer for chat template formatting
print(f"Loading instruct tokenizer from: {instruct_model_name}")
instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)
print(f"✓ Instruct tokenizer loaded for chat template formatting")

print(f"✓ Model loaded with {model.num_parameters():,} parameters")

# ==================== DATASET LOADING ====================
print("\n=== PREPARING DATASET ===\n")

# Load SmolTalk2 dataset in streaming mode to avoid loading entire dataset
# This loads only the samples we need without downloading/loading everything
print(f"Loading {TRAINING_SAMPLES} samples from dataset...")
dataset = load_dataset(
    "HuggingFaceTB/smoltalk2", 
    "SFT",
    streaming=True  # Stream data instead of loading everything
)

# Take only TRAINING_SAMPLES from the streaming dataset
train_dataset = Dataset.from_dict({
    k: [item[k] for item in islice(
        dataset["smoltalk_everyday_convs_reasoning_Qwen3_32B_think"], 
        TRAINING_SAMPLES
    )]
    for k in next(iter(dataset["smoltalk_everyday_convs_reasoning_Qwen3_32B_think"])).keys()
})

print(f"✓ Loaded {len(train_dataset)} training examples (without loading full dataset)")
print(f"Example raw data: {train_dataset[0]}\n")

# ==================== APPLY CHAT TEMPLATE ====================
print("Formatting dataset with instruct model's chat template...")

def format_chat_template(example):
    """
    Format the messages using the chat template from the instruct model.
    Converts conversation messages into the model's expected format with special tokens.
    Uses instruct_tokenizer which has a properly tuned chat template.
    """
    if "messages" in example:
        # SmolTalk2 format - already has messages structure
        messages = example["messages"]
    else:
        # Fallback for custom format (e.g., instruction/response datasets)
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]}
        ]
    
    # Apply chat template using instruct tokenizer
    # This adds special tokens like <|im_start|>, <|im_end|> for proper formatting
    text = instruct_tokenizer.apply_chat_template(
        messages, 
        tokenize=False,  # Return string, not token IDs
        add_generation_prompt=False  # Don't add trailing assistant prompt (data is complete)
    )
    return {"text": text}

# Apply formatting using .map() - efficient and keeps Dataset structure
formatted_dataset = train_dataset.map(format_chat_template)

# Remove all columns except 'text' to avoid conflicts during training
formatted_dataset = formatted_dataset.remove_columns(
    [col for col in formatted_dataset.column_names if col != "text"]
)

print(f"✓ Formatted {len(formatted_dataset)} examples")
print("\n--- FORMATTED DATASET EXAMPLE ---")
print(formatted_dataset[0]['text'])
print("--- END FORMATTED EXAMPLE \n")

# Show tokenized version
tokenized_example = tokenizer(formatted_dataset[0]['text'], truncation=True, max_length=1024)
print("--- TOKENIZED EXAMPLE ---")
print(f"Token IDs (first 50): {tokenized_example['input_ids'][:50]}")
print(f"Total tokens: {len(tokenized_example['input_ids'])}")
print(f"Decoded back (first 50 tokens): {tokenizer.decode(tokenized_example['input_ids'][:50])}")
print("--- END TOKENIZED EXAMPLE \n")

# ==================== LORA CONFIGURATION ====================
peft_config = LoraConfig(
    r=8,  # LoRA rank - lower = fewer parameters (cheaper), 8 is good balance
    lora_alpha=16,  # LoRA scaling factor - typically 2x the rank
    lora_dropout=0.05,  # Dropout for regularization
    bias="none",  # Don't train bias terms
    task_type="CAUSAL_LM",  # Causal language modeling task
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Apply LoRA to attention layers
)
print("✓ LoRA configured (trains ~1-2% of parameters)")

# ==================== TRAINING CONFIGURATION ====================
training_config = SFTConfig(
    # ===== Model and data =====
    output_dir=f"./{new_model_name}",  # Where to save checkpoints
    dataset_text_field="text",  # Field name containing the formatted text
    max_length=1024,  # Max sequence length - shorter = faster & cheaper (1024 vs 2048)
    
    # ===== Training hyperparameters =====
    per_device_train_batch_size=2,  # Batch size per GPU - 2 for 3B model on 24GB GPU
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps = effective batch size of 8
    learning_rate=2e-4,  # Higher LR for LoRA (vs 5e-5 for full fine-tuning)
    num_train_epochs=2,  # Train for 2 epochs through data (1000 samples = ~125 steps/epoch)
    max_steps=-1,  # Use num_train_epochs instead (set to number like 250 to limit steps)
    
    # ===== Optimization =====
    warmup_steps=50,  # Gradually increase LR for first 50 steps - stabilizes training
    weight_decay=0.01,  # L2 regularization to prevent overfitting
    optim="adamw_torch",  # AdamW optimizer - standard for transformers
    lr_scheduler_type="cosine",  # Cosine learning rate decay - smooth decay
    max_grad_norm=1.0,  # Gradient clipping to prevent exploding gradients
    
    # ===== Logging and saving =====
    logging_steps=25,  # Log metrics every 25 steps
    save_steps=100,  # Save checkpoint every 100 steps (~1 per epoch)
    save_total_limit=2,  # Keep only 2 most recent checkpoints to save disk space
    
    # ===== Memory optimization =====
    bf16=True,  # Use bfloat16 precision - faster & less memory than float32
    gradient_checkpointing=True,  # Trade compute for memory - enables larger batch sizes
    dataloader_num_workers=0,  # Data loading workers - 0 for simple setup (avoid multiprocessing issues)
    group_by_length=True,  # Group similar length sequences - reduces padding waste
    
    # ===== Hugging Face Hub integration =====
    push_to_hub=True,  # Automatically push to HF Hub when training completes
    hub_model_id=f"tomascufaro/{new_model_name}",  # Your HF username/model-name
    hub_strategy="end",  # Push only at the end (vs "every_save" which pushes each checkpoint)
    
    # ===== Experiment tracking =====
    report_to=["tensorboard", "wandb"],  # Use TensorBoard for monitoring (logs saved in output_dir)
    run_name=f"{new_model_name}-training",  # Name for this training run
)

print("\n=== Training Configuration ===")
effective_batch_size = training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps
steps_per_epoch = TRAINING_SAMPLES // effective_batch_size
total_steps = steps_per_epoch * training_config.num_train_epochs
print(f"Effective batch size: {effective_batch_size}")
print(f"Total samples: {TRAINING_SAMPLES}")
print(f"Steps per epoch: ~{steps_per_epoch} ({TRAINING_SAMPLES} / {effective_batch_size})")
print(f"Total epochs: {training_config.num_train_epochs}")
print(f"Total training steps: ~{total_steps}")
print(f"Estimated training time: ~{total_steps * 0.12:.1f} minutes on L4")
print(f"Estimated cost: ~${total_steps * 0.006:.2f}")

# ==================== INITIALIZE TRAINER ====================
print("\nInitializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=formatted_dataset,  # Use formatted dataset with 'text' field
    peft_config=peft_config,  # Enable LoRA
)
print("✓ Trainer initialized")

# ==================== TRAIN ====================
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50 + "\n")

trainer.train()

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)

# ==================== SAVE MODEL ====================
print("\nSaving model and tokenizer...")
trainer.save_model()
tokenizer.save_pretrained(training_config.output_dir)
print(f"✓ Model saved to {training_config.output_dir}")

if training_config.push_to_hub:
    print(f"✓ Model pushed to: https://huggingface.co/{training_config.hub_model_id}")

print("\n🎉 All done! Check your model at https://huggingface.co/" + training_config.hub_model_id)