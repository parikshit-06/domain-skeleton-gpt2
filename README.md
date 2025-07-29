# Domain-Adaptive Skeleton Model with GPT-2 & LoRA

A lightweight, modular framework for rapid domain adaptation of GPT-2 using Low-Rank Adapters (LoRA). Train small, domain-specific adapters for legal, medical, and sports data without fine-tuning the entire GPT-2 model.

---

## 🚀 Features

* **Skeleton Architecture:** Minimal GPT-2 backbone with hooks for injecting LoRA adapters.
* **Parameter-Efficient Fine-Tuning:** Only low-rank adapter weights are trained, drastically reducing compute and memory requirements.
* **Multi-Domain Support:** Easy training and switching between domains (legal, medical, sports) via simple CLI.
* **Evaluation & Logging:** Built-in TensorBoard logging and evaluation scripts for generation quality and perplexity.
* **MVP-Ready:** End-to-end pipeline from data preprocessing to inference in minutes.

---

## 📦 Repository Structure

```
├── adapters/
│   ├── legal/            # Trained LoRA adapter for legal domain
│   ├── medical/          # Trained LoRA adapter for medical domain
│   └── sports/           # Trained LoRA adapter for sports domain
├── scripts/
│   ├── train_LoRA_adapter.py  # Script to train domain adapters
│   └── inference.py           # Script to load adapters and generate text
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🔧 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/parikshit-06/domain-adaptive-skeleton-gpt2.git
   cd domain-adaptive-skeleton-gpt2
   ```
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\\Scripts\\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### 1. Training Domain Adapters

Use `scripts/train_LoRA_adapter.py` to train LoRA adapters for supported domains.

```bash
# Single domain
python scripts/train_LoRA_adapter.py --domain <legal|medical|sports>
# Train all domains sequentially
python scripts/train_LoRA_adapter.py --all
```

* **Configuration** lives in the script under `DOMAIN_CONFIGS`:

  * `dataset`: Hugging Face loader for each domain
  * `preprocess`: Domain-specific tokenization function
  * `output_dir`: Where adapters and tokenizers are saved

---

### 2. Inference

Generate text using a trained adapter with `scripts/inference.py`.

```bash
python scripts/inference.py --adapter_path ./adapters/sports --prompt "Real Madrid just won their"
```

This script:

1. Loads base GPT-2 + specified LoRA adapter
2. Runs generation with sampling (top-k, top-p, temperature)
3. Prints the output text

---

## ⚙️ Configuration

* **BASE\_MODEL:** GPT-2 preset in Hugging Face transformers
* **LoRA Settings:**

  * `r=8`, `lora_alpha=16`, `target_modules=['c_attn','c_proj']`, `lora_dropout=0.1`
* **Tokenization:** Maximum length = 512 tokens, padded/truncated to fixed length
* **TrainingArgs:** Batch size 4, 2 epochs, mixed precision if CUDA available, logging every 10 steps

---

## 🎯 Example

```bash
# Train sports adapter
python scripts/train_LoRA_adapter.py --domain sports
# Generate with sports adapter
python scripts/inference.py --adapter_path ./adapters/sports --prompt "The final match highlights:" 
```

---

## 📈 Evaluation

* Track losses and perplexity in TensorBoard under each adapter folder (`./adapters/<domain>/runs`).
* Sample generation outputs in `scripts/inference.py`.

---

## 📝 Contributing

1. Fork this repo
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add some feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a pull request

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---
