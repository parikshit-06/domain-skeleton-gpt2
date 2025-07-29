import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
)
from peft import get_peft_model, LoraConfig, TaskType
import os


BASE_MODEL = "gpt2"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_legal(example, tokenizer):
    text = example["text"]
    tokens = tokenizer(text[:MAX_LENGTH], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def preprocess_medical(example, tokenizer):
    instruction = example["instruction"]
    response = example["output"]
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    tokens = tokenizer(prompt[:MAX_LENGTH], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def preprocess_sports(example, tokenizer):
    context = example["context"].strip()
    question = example["question"].strip()
    try:
        answer = eval(example["answer"])["text"]
    except:
        answer = "N/A"
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    full_text = f"{prompt} {answer}"
    tokens = tokenizer(full_text[:MAX_LENGTH], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


DOMAIN_CONFIGS = {
    "legal": {
        "dataset": lambda: load_dataset("lex_glue", "eurlex")["train"].select(range(500)).remove_columns(["labels"]),
        "preprocess": preprocess_legal,
        "output_dir": "./adapters/legal"
    },
    "medical": {
        "dataset": lambda: load_dataset("knowrohit07/know_medical_dialogue_v2")["train"].select(range(500)),
        "preprocess": preprocess_medical,
        "output_dir": "./adapters/medical"
    },
    "sports": {
        "dataset": lambda: load_dataset("PedroCJardim/QASports", "all", split="train").select(range(500)),
        "preprocess": preprocess_sports,
        "output_dir": "./adapters/sports"
    }
}


def train_domain(domain):
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unsupported domain: {domain}")

    print(f"\n🚀 Training domain adapter for: {domain}\n")
    config = DOMAIN_CONFIGS[domain]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token  # gpt2 needs manual padding token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(DEVICE)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    # Load and preprocess dataset
    dataset = config["dataset"]()
    tokenized = dataset.map(lambda x: config["preprocess"](x, tokenizer), batched=False)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=4,
        num_train_epochs=2,
        save_total_limit=1,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
        label_names=["labels"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized
    )

    try:
        trainer.train()
        model.save_pretrained(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])
        print(f"✅ Finished training: {domain}")
    except Exception as e:
        print(f"\n❌ Training failed for {domain}: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Train all domains
    for domain_name in DOMAIN_CONFIGS.keys():
        train_domain(domain_name)