"""Milestone 2: train a real LoRA adapter using the 16 Northstar examples."""

import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/training/instruction_examples.jsonl"

SYSTEM_MESSAGE = (
    "You are Northstar support. Answer using only the provided policy evidence. "
    "Be concise, give the next action, and cite policy IDs in square brackets. "
    "If the evidence does not answer the question, say you cannot confirm it."
)


def make_prompt(tokenizer, context, question):
    """Use the same chat prompt later for baseline and LoRA inference."""
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"Policy evidence:\n{context}\n\nQuestion:\n{question}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def encode_example(tokenizer, record, max_length):
    """Train only on answer tokens, never on the prompt or padding."""
    prompt_ids = tokenizer(
        make_prompt(tokenizer, record["context"], record["question"]),
        add_special_tokens=False,
    )["input_ids"]
    answer_ids = tokenizer(record["answer"], add_special_tokens=False)["input_ids"]
    answer_ids.append(tokenizer.eos_token_id)

    tokens = prompt_ids + answer_ids
    if len(tokens) > max_length:
        raise ValueError(
            f"{record['id']}: {len(tokens)} tokens exceed MAX_SEQUENCE_LENGTH={max_length}. "
            "Increase the limit rather than silently truncating training answers."
        )

    pad = max_length - len(tokens)
    return {
        "input_ids": torch.tensor(tokens + [tokenizer.pad_token_id] * pad),
        "attention_mask": torch.tensor([1] * len(tokens) + [0] * pad),
        "labels": torch.tensor([-100] * len(prompt_ids) + answer_ids + [-100] * pad),
    }


def main():
    load_dotenv(ROOT / ".env")
    model_name = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
    revision = os.getenv("BASE_MODEL_REVISION") or None
    max_length = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))
    epochs = int(os.getenv("TRAIN_EPOCHS", "8"))
    learning_rate = float(os.getenv("LEARNING_RATE", "0.0002"))
    seed = int(os.getenv("RANDOM_SEED", "42"))
    output = ROOT / os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter")

    if max_length < 2 or epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid training settings in .env")

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    records = [json.loads(line) for line in DATA_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Device: {device} | Examples: {len(records)} | Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs an EOS token for supervised training")

    # Validate all sequence lengths BEFORE downloading the full base model.
    dataset = [encode_example(tokenizer, record, max_length) for record in records]
    base_model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    base_model.config.use_cache = False

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora).to(device)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError("No LoRA parameters are trainable")
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        order = list(range(len(dataset)))
        random.shuffle(order)
        epoch_loss = 0.0
        for i in order:
            batch = {key: value.unsqueeze(0).to(device) for key, value in dataset[i].items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, example {records[i]['id']}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = round(epoch_loss / len(dataset), 4)
        history.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} | loss: {avg_loss:.4f}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)  # Only the adapter weights, not the 135M-parameter base model
    tokenizer.save_pretrained(output)
    metadata = {
        "base_model": model_name,
        "requested_base_revision": revision,
        "resolved_base_revision": getattr(base_model.config, "_commit_hash", None),
        "training_file": str(DATA_FILE.relative_to(ROOT)),
        "training_data_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "training_examples": len(records),
        "training_ids": [record["id"] for record in records],
        "prompt_format": "tokenizer chat template; system + user (context/question) -> assistant answer",
        "loss_masking": "assistant answer tokens only (prompt and padding ignored)",
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "sequence_length": max_length,
        "batch_size": 1,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "random_seed": seed,
        "device": str(device),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "epoch_loss": history,
        "adapter_path": str(output.relative_to(ROOT)),
    }
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Adapter saved: {output}")


if __name__ == "__main__":
    main()
