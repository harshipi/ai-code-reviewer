# AI Code Reviewer — Fine-Tuned LLM for Automated Code Review

Ever pushed code with a bug you didn't catch? Or missed a security vulnerability that only showed up in production? This project is my attempt to build a smart, domain-specific code review assistant using a fine-tuned language model — one that actually understands *why* code is problematic, not just *that* it is.

---

## What Is This?

This is a lightweight AI-powered code reviewer built by fine-tuning **Qwen2.5-1.5B-Instruct** — a 1.5 billion parameter language model — using a technique called **QLoRA** (Quantized Low-Rank Adaptation). Instead of training the entire model from scratch (which would require massive compute and weeks of time), QLoRA lets us:

- Compress the model to **4-bit precision** (so it fits in normal GPU memory)
- Train only **~2% of the total parameters** using LoRA adapters (like adding smart sticky notes to a textbook instead of rewriting it)
- Achieve domain-specific behavior with a **custom dataset** I curated myself

The result? A model that reviews Python code and gives structured, educational feedback — covering bugs, security holes, performance problems, and best practices — all running locally on your machine.

---

## What Can It Do?

Paste any code snippet into the web interface and the model will analyze it and return a structured review like:

- *Bugs** — logical errors, unhandled edge cases, undefined variables
- **Security Vulnerabilities** — SQL injection, hardcoded secrets, unsafe deserialization
- **Performance Issues** — O(n³) loops that should be O(n), loading entire files into memory, etc.
- **Best Practices** — missing type hints, bare excepts, not using context managers

Every review comes with a fixed version of the code and an explanation of *why* the original was problematic.

---

## How It Works — The Tech Stack

| Component | Technology |
|---|---|
| Base Model | Qwen2.5-1.5B-Instruct |
| Fine-tuning Method | QLoRA (4-bit + LoRA) |
| Training Framework | HuggingFace Transformers + PEFT + TRL |
| Quantization | BitsAndBytes (NF4 4-bit) |
| Training Hardware | Google Colab T4 GPU |
| Web Interface | Streamlit |
| Language | Python |

### Why Qwen2.5-1.5B?
It's small enough to run on a laptop GPU (4GB VRAM is enough) but smart enough to understand code structure, patterns, and vulnerabilities. The "Instruct" variant means it's already trained to follow instructions, so fine-tuning it for code review is a natural extension.

### Why QLoRA?
Training a full 1.5B parameter model normally requires ~24GB of GPU RAM and days of compute. QLoRA reduces this to under 4GB and a few minutes on a free Colab T4 GPU — by compressing the model to 4-bit precision and only training a tiny set of adapter layers. We went from 907 million trainable parameters down to just 18 million (2.04%).

---

## 📁 Project Structure

```
ai-code-reviewer/
│
├── data/
│   └── code_review_dataset.jsonl   # Custom training dataset in ChatML format
│
├── training/
│   └── train.py                     # Full fine-tuning pipeline (run on Google Colab)
│
├── app/
│   ├── app.py                       # Streamlit web interface
│   └── inference.py                 # Model loading and review generation
│
├── requirements.txt                 # All Python dependencies
└── README.md                        # You are here
```
## Training Details

The model was fine-tuned on a custom dataset I built from scratch — real-world code patterns covering the most common categories of code problems:

- SQL injection and other injection attacks
- Hardcoded credentials and API keys
- Unsafe deserialization (pickle)
- Unclosed file handles
- O(n²) and O(n³) algorithms with better alternatives
- Mutable default arguments
- Bare except clauses
- Division by zero and missing input validation
- Financial precision errors (float vs Decimal)

Each example is formatted in **ChatML format** — the same conversational format Qwen2.5 was pre-trained on — making fine-tuning more stable and effective.

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Training epochs | 3 |
| Batch size | 2 |
| Learning rate | 2e-4 |
| Max sequence length | 1024 |
| Total parameters | 907,081,216 |
| Trainable parameters | 18,464,768 (2.04%) |
| GPU | Google Colab T4 (16GB) |

---

## Running It Yourself
### Prerequisites
- Python 3.10+
- Ubuntu / Linux (recommended)
- NVIDIA GPU with 4GB+ VRAM (or CPU, but it'll be slow)

### 1. Clone the repository
```bash
git clone https://github.com/harshipi/ai-code-reviewer.git
cd ai-code-reviewer
```

### 2. Set up Python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Get the fine-tuned model

**Option A — Train it yourself (recommended):**
- Upload `data/code_review_dataset.jsonl` and `training/train.py` to [Google Colab](https://colab.research.google.com)
- Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
- Run `train.py` — training takes around 15-30 minutes
- Download the output `code-reviewer-model` folder
- Place it in the root of this project
**Option B — Skip fine-tuning:**
- The app will automatically fall back to the base `Qwen/Qwen2.5-1.5B-Instruct` model
- It downloads automatically on first run (~3GB)
- Reviews will be more generic without the fine-tuning

### 4. Launch the web app
```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501` and start reviewing code!

---
## Web Interface

The Streamlit app gives you:
- A code input panel with syntax-aware text area
- Language selector (Python, JavaScript, TypeScript, Java, C++, Go, Rust)
- Review length control
- Side-by-side rendered markdown and raw output views
- **Download button** to export the review as a `.md` file
- Review history for the current session
- Built-in sample code snippets to test with

---

## Example

**Input code:**
```python
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    return db.execute(query)
```

**AI Review output:**
CODE REVIEW

Security Vulnerability (Critical)
SQL Injection: An attacker can pass user_id = "1 OR 1=1" to dump the entire table.

Bug
If user_id is not a string, concatenation raises TypeError at runtime.

Fixed Code
def get_user(user_id: int):
query = "SELECT * FROM users WHERE id = ?"
return db.execute(query, (user_id,))

Best Practices
Always use parameterized queries — never concatenate user input into SQL.
Add type hints to parameters.
Consider using an ORM like SQLAlchemy for safer database access.
---

## What I'd Do Next

This was built as a learning project to understand LLM fine-tuning end to end. If I were to keep building on it:

- Expand the dataset to 500+ examples across more languages
- Add support for full file uploads instead of just snippets
- Push the fine-tuned adapters to HuggingFace Hub for easy sharing
- Add a severity scoring system (Critical / High / Medium / Low)
- Integrate with GitHub Actions for automated PR reviews

---

## What I Learned

Building this taught me more about LLMs than any tutorial could. Some things that surprised me:

- How little data you actually need for domain adaptation with QLoRA — 10-20 good examples can already shift a model's behavior noticeably
- How sensitive training is to small API changes — TRL, PEFT, and Transformers all changed their APIs between versions during this project
- How much of "fine-tuning" is actually just data curation — the quality of your examples matters far more than hyperparameter tuning

---

## Built With

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Streamlit](https://streamlit.io)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

---

*Built from scratch on Ubuntu with a dual-boot RTX 3050 laptop and a lot of debugging. Every error in this project taught me something.*
