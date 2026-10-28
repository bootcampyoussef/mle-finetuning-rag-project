
"""Milestone 3 smoke test: compare base RAG and LoRA RAG on identical evidence."""

import json
import os
import re
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieval import create_index, load_chunks, retrieve
from training import make_prompt

ROOT = Path(__file__).resolve().parents[1]


def format_evidence(hits):
    """Preserve the original policy IDs through generation."""
    return "\n".join(f"[{hit['source_id']}] {hit['source_text']}" for hit in hits)


def check_answer(answer, hits):
    """Validate citation *IDs*, not whether the claim is factually correct."""
    lower = answer.lower()
    refusal_phrases = ("cannot confirm", "can't confirm", "not enough information", "insufficient information")
    if any(phrase in lower for phrase in refusal_phrases):
        return "abstained", "model declined to answer"

    cited = set(re.findall(r"\[([a-z][a-z0-9-]*)\]", answer))
    permitted = {hit["source_id"] for hit in hits}
    if not cited:
        return "rejected", "missing source citation"
    if not cited.issubset(permitted):
        return "rejected", f"unknown citations: {sorted(cited - permitted)}"
    return "answered", "citations refer to retrieved sources"


def generate(model, tokenizer, prompt, device, max_new_tokens, adapter_enabled):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    options = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with torch.inference_mode():
        if adapter_enabled:
            output = model.generate(**inputs, **options)
        else:
            with model.disable_adapter():
                output = model.generate(**inputs, **options)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    load_dotenv(ROOT / ".env")
    base_name = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
    base_revision = os.getenv("BASE_MODEL_REVISION") or None
    adapter_path = ROOT / os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter")
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing LoRA adapter: {adapter_path}. Run src/training.py first.")

    metadata_path = adapter_path / "training_metadata.json"
    if metadata_path.is_file():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta["base_model"] != base_name:
            raise ValueError("BASE_MODEL differs from the model used for LoRA training")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_name, revision=base_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_name, revision=base_revision)
    model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
    model.eval()
    model.config.use_cache = True

    embedding_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_revision = os.getenv("EMBEDDING_REVISION") or None
    embedder = SentenceTransformer(embedding_name, revision=embedding_revision)
    client = create_index(load_chunks(), embedder)
    top_k = int(os.getenv("TOP_K", "3"))
    min_score = float(os.getenv("MIN_SCORE", "0.30"))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "80"))

    questions = [
        "How long does standard shipping take?",
        "What should support do if a customer lost their authenticator?",
        "Where can I book a meeting room in the office?",
    ]
    print(f"Model: {base_name} | Device: {device} | top_k={top_k} | threshold={min_score}")
    try:
        for question in questions:
            hits = retrieve(question, client, embedder, top_k, min_score)
            print(f"\nQuestion: {question}")
            print("Retrieved:", ", ".join(f"{hit['source_id']} ({hit['score']:.2f})" for hit in hits) or "none")
            if not hits:
                print("Baseline RAG: ABSTAIN (no evidence above threshold)")
                print("Hybrid RAG:   ABSTAIN (no evidence above threshold)")
                continue

            # Use precisely the same context and prompt for both systems.
            prompt = make_prompt(tokenizer, format_evidence(hits), question)
            for label, use_adapter in (("Baseline RAG", False), ("Hybrid RAG", True)):
                answer = generate(model, tokenizer, prompt, device, max_new_tokens, use_adapter)
                status, reason = check_answer(answer, hits)
                print(f"{label}: {status.upper()} ({reason})\n  {answer}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
