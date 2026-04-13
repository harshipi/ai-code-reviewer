# =============================================================
# inference.py — Load the fine-tuned model and generate reviews
# =============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# Path to your downloaded model
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./code-reviewer-model"   # The LoRA adapters you trained


class CodeReviewer:
    """
    Wrapper class that loads the fine-tuned model and generates code reviews.
    We load it once and reuse — loading takes ~30 seconds, inference is fast.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        print("⏳ Loading model... (this takes ~30 seconds)")
        
        # Same 4-bit config as training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True
        )
        
        # Load base model in 4-bit
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load our fine-tuned LoRA adapters ON TOP of the base model
        # This merges our "sticky notes" with the original textbook
        if os.path.exists(ADAPTER_PATH):
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            print("✅ Fine-tuned model loaded!")
        else:
            # Fallback: use base model without fine-tuning
            self.model = base_model
            print("⚠️  No fine-tuned adapters found. Using base model.")
        
        self.model.eval()   # Set to evaluation mode (disables dropout etc.)
    
    def review_code(
        self,
        code: str,
        language: str = "Python",
        max_new_tokens: int = 512
    ) -> str:
        """
        Generate a code review for the given code snippet.
        
        Args:
            code: The code to review
            language: Programming language (for context)
            max_new_tokens: Maximum length of the review
        
        Returns:
            The review as a markdown string
        """
        
        # Build the prompt in ChatML format
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert code reviewer. Analyze the given code and "
                    "provide structured feedback covering: bugs, security vulnerabilities, "
                    "performance issues, and best practices. Be specific, educational, and constructive."
                )
            },
            {
                "role": "user",
                "content": f"Review this {language} code:\n```{language.lower()}\n{code}\n```"
            }
        ]
        
        # Apply ChatML template and tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True   # This adds the <|im_start|>assistant token
                                         # to prompt the model to start generating
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate the review
        with torch.no_grad():   # Don't compute gradients — saves memory
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,        # Lower = more focused/deterministic
                do_sample=True,         # Enable sampling
                top_p=0.9,              # Nucleus sampling
                repetition_penalty=1.1, # Penalize repeating the same phrases
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the NEW tokens (not the input prompt)
        input_length = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_length:]
        review = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return review.strip()


# Global instance — created once, reused across Streamlit reruns
_reviewer_instance = None

def get_reviewer() -> CodeReviewer:
    """Get or create the singleton CodeReviewer instance."""
    global _reviewer_instance
    if _reviewer_instance is None:
        _reviewer_instance = CodeReviewer()
    return _reviewer_instance
