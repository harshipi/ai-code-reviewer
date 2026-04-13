# =============================================================
# AI Code Reviewer - Fine-Tuning Script
# Run this on Google Colab with T4 GPU
# =============================================================

# ---- STEP 1: Install dependencies (run this cell first in Colab) ----
# !pip install transformers peft trl bitsandbytes datasets accelerate -q

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# =============================================================
# SECTION 1: CONFIGURATION
# Think of this as your "settings panel"
# =============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # The base model we download
OUTPUT_DIR = "./code-reviewer-model"           # Where to save our fine-tuned model
DATASET_PATH = "code_review_dataset.jsonl"    # Your training data file

# LoRA settings — these control how much we adapt the model
# r = rank: how many "sticky notes" we add. Higher = smarter but slower
# alpha = how strongly the new training overrides the original model
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training settings
NUM_EPOCHS = 3          # How many times to go through all training data
BATCH_SIZE = 2          # How many examples to process at once (limited by GPU RAM)
LEARNING_RATE = 2e-4    # How fast the model learns (too high = unstable)
MAX_SEQ_LENGTH = 1024   # Maximum token length per example


# =============================================================
# SECTION 2: LOAD & PREPARE DATASET
# =============================================================

def load_dataset_from_jsonl(filepath: str) -> Dataset:
    """
    Load our JSONL file and convert to HuggingFace Dataset format.
    Each line in the JSONL is one training example.
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data.append(json.loads(line))
    
    print(f"✅ Loaded {len(data)} training examples")
    return Dataset.from_list(data)


def format_to_chatml(example: dict, tokenizer) -> dict:
    """
    Convert our messages format into a single string using ChatML template.
    
    ChatML format looks like:
    <|im_start|>system
    You are an expert...
    <|im_end|>
    <|im_start|>user
    Review this code...
    <|im_end|>
    <|im_start|>assistant
    ## Code Review...
    <|im_end|>
    
    This is the exact format Qwen2.5 was pre-trained on, so it understands it.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,           # Return string, not token IDs
        add_generation_prompt=False
    )
    return {"text": text}


# =============================================================
# SECTION 3: LOAD MODEL WITH 4-BIT QUANTIZATION
# =============================================================

def load_quantized_model(model_name: str):
    """
    Load the model in 4-bit precision instead of 32-bit.
    
    Normal: 1.5B params × 4 bytes = ~6GB RAM needed
    4-bit:  1.5B params × 0.5 bytes = ~0.75GB RAM needed
    
    This is what BitsAndBytes does — it squishes the numbers
    without losing too much quality.
    """
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                        # Use 4-bit precision
        bnb_4bit_quant_type="nf4",                # NF4 = best quality 4-bit format
        bnb_4bit_compute_dtype=torch.bfloat16,    # Use bfloat16 for calculations
        bnb_4bit_use_double_quant=True,           # Extra compression on quantization constants
    )
    
    print(f"📥 Loading model: {model_name}")
    
    # Load the tokenizer (converts text to numbers the model understands)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"   # Pad on the right for training stability
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the actual model weights in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically use GPU if available
        trust_remote_code=True,
    )
    
    print("✅ Model loaded successfully!")
    return model, tokenizer


# =============================================================
# SECTION 4: APPLY LoRA ADAPTERS
# =============================================================

def apply_lora(model):
    """
    Apply LoRA (Low-Rank Adaptation) to the model.
    
    Instead of training all 1.5B parameters, LoRA adds small
    "adapter" layers to specific parts of the model.
    
    target_modules = which layers to add adapters to.
    For Qwen2.5, these are the attention and MLP layers.
    """
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,    # We're doing text generation
        target_modules=[
            "q_proj",    # Query projection in attention
            "k_proj",    # Key projection in attention
            "v_proj",    # Value projection in attention
            "o_proj",    # Output projection in attention
            "gate_proj", # Gate in MLP layer
            "up_proj",   # Up projection in MLP
            "down_proj", # Down projection in MLP
        ]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print how many parameters we're actually training
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    
    print(f"📊 Trainable parameters: {trainable:,} / {total:,}")
    print(f"📊 That's only {100 * trainable / total:.2f}% of all parameters!")
    
    return model


# =============================================================
# SECTION 5: TRAINING
# =============================================================

def train(model, tokenizer, dataset):
    """
    The actual fine-tuning loop.
    SFTTrainer (Supervised Fine-Tuning Trainer) handles all the
    complexity of the training loop for us.
    """
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,    # Simulate larger batch size
        learning_rate=LEARNING_RATE,
        fp16=True,                         # Use 16-bit floats for speed
        logging_steps=10,                  # Print loss every 10 steps
        save_steps=50,                     # Save checkpoint every 50 steps
        save_total_limit=2,                # Keep only last 2 checkpoints
        warmup_ratio=0.03,                 # Slowly ramp up learning rate
        lr_scheduler_type="cosine",        # Learning rate decay schedule
        report_to="none",                  # Don't use wandb/tensorboard
        dataloader_pin_memory=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",         # Which field contains our formatted text
    )
    
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete!")
    
    return trainer


# =============================================================
# SECTION 6: SAVE THE MODEL
# =============================================================

def save_model(trainer, tokenizer):
    """
    Save only the LoRA adapter weights (not the full model).
    This is tiny — usually just a few MB!
    """
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Model saved to {OUTPUT_DIR}")
    print("📦 Tip: Zip this folder and download it to your Ubuntu machine")


# =============================================================
# MAIN — Run everything
# =============================================================

if __name__ == "__main__":
    # 1. Load dataset
    raw_dataset = load_dataset_from_jsonl(DATASET_PATH)
    
    # 2. Load model
    model, tokenizer = load_quantized_model(MODEL_NAME)
    
    # 3. Format dataset using ChatML template
    formatted_dataset = raw_dataset.map(
        lambda x: format_to_chatml(x, tokenizer)
    )
    
    # 4. Apply LoRA
    model = apply_lora(model)
    
    # 5. Train
    trainer = train(model, tokenizer, formatted_dataset)
    
    # 6. Save
    save_model(trainer, tokenizer)
