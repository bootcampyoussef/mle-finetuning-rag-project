"""Milestone 4: held-out comparison of the SAME RAG evidence with/without LoRA.

Run from repo root: uv run python src/evaluate_generation.py
Writes reports/evaluation_results.json for the decision report.
"""

import json
import os
import re
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from dotenv import load_dotenv
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation import check_answer, format_evidence, generate
from retrieval import ROOT, create_index, load_chunks, retrieve
from training import make_prompt


def contains_phrase(text, phrase):
    """Case-insensitive phrase match, NOT an arbitrary substring match."""
    text = " ".join(text.casefold().split())
    phrase = " ".join(phrase.casefold().split())
    return bool(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def score_answer(case, answer, status, hits):
    """Transparent proxies: these are NOT semantic entailment judgments."""
    required = case["required_terms"]
    coverage = (
        sum(contains_phrase(answer, term) for term in required) / len(required)
        if required
        else None
    )
    forbidden = [
        term for term in case["forbidden_terms"] if contains_phrase(answer, term)
    ]
    cited = sorted(set(re.findall(r"\[([a-z][a-z0-9-]*)\]", answer)))
    retrieved_ids = {hit["source_id"] for hit in hits}
    expected_ids = set(case["expected_sources"])

    # Faithfulness proxy: valid AND expected citation, full term coverage, no known bad claim.
    # This does NOT prove that every generated claim is supported by the cited passage.
    faithfulness_proxy = (
        (
            status == "answered"
            and bool(set(cited) & expected_ids)
            and set(cited) <= retrieved_ids
            and coverage == 1.0
            and not forbidden
        )
        if not case["should_abstain"]
        else None
    )

    # Style proxy only: succinct, one-paragraph, no salutation/bullets, cited.
    # "Answer first" and "next action" still need human review.
    style_proxy = (
        (
            status == "answered"
            and len(answer.split()) <= 80
            and "\n" not in answer
            and not re.match(r"\s*(hello|hi|dear|[-*•])\b", answer, re.I)
            and bool(cited)
        )
        if not case["should_abstain"]
        else None
    )

    expected_abstain = case["should_abstain"]
    decision_ok = (
        (status == "abstained") if expected_abstain else (status == "answered")
    )
    return {
        "status": status,
        "decision_ok": decision_ok,
        "required_term_coverage": coverage,
        "forbidden_matches": forbidden,
        "cited_sources": cited,
        "citation_faithfulness_proxy": faithfulness_proxy,
        "style_proxy": style_proxy,
    }


def summarize(rows, system, supported_count):
    items = [row[system] for row in rows]
    supported = [row[system] for row in rows if not row["should_abstain"]]
    unsupported = [row[system] for row in rows if row["should_abstain"]]
    return {
        "decision_accuracy": sum(item["decision_ok"] for item in items) / len(items),
        "unsupported_abstentions": sum(
            item["status"] == "abstained" for item in unsupported
        ),
        "unsupported_total": len(unsupported),
        "required_term_coverage_accepted": sum(
            (item["required_term_coverage"] or 0) if item["status"] == "answered" else 0
            for item in supported
        )
        / supported_count,
        "citation_faithfulness_proxy": sum(
            bool(item["citation_faithfulness_proxy"]) for item in supported
        )
        / supported_count,
        "style_proxy": sum(bool(item["style_proxy"]) for item in supported)
        / supported_count,
        "mean_generation_seconds_per_case": mean(
            item["generation_seconds"] for item in items
        ),
    }


def main():
    load_dotenv(ROOT / ".env")
    cases = json.loads(
        (ROOT / "data/evaluation/cases.json").read_text(encoding="utf-8")
    )
    base_name = os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
    base_revision = os.getenv("BASE_MODEL_REVISION") or None
    adapter_path = ROOT / os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter")
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Adapter missing at {adapter_path}. Run training.py first."
        )
    metadata_file = adapter_path / "training_metadata.json"
    training_metadata = (
        json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    )
    if training_metadata and training_metadata.get("base_model") != base_name:
        raise ValueError("Base model differs from the trained adapter")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_name, revision=base_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_name, revision=base_revision)
    model = PeftModel.from_pretrained(base, adapter_path).to(device).eval()
    model.config.use_cache = True

    embedding_name = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedder = SentenceTransformer(
        embedding_name, revision=os.getenv("EMBEDDING_REVISION") or None
    )
    client = create_index(load_chunks(), embedder)
    top_k = int(os.getenv("TOP_K", "3"))
    min_score = float(os.getenv("MIN_SCORE", "0.30"))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "80"))
    rows = []
    print(
        f"Evaluating {len(cases)} held-out cases | device={device} | top_k={top_k} | min_score={min_score}",
        flush=True,
    )

    try:
        for case in cases:
            start = perf_counter()
            hits = retrieve(case["question"], client, embedder, top_k, min_score)
            retrieval_seconds = perf_counter() - start
            sources = [h["source_id"] for h in hits]
            expected = set(case["expected_sources"])
            rank = next(
                (n for n, src in enumerate(sources, 1) if src in expected), None
            )
            row = {
                "id": case["id"],
                "question": case["question"],
                "should_abstain": case["should_abstain"],
                "expected_sources": case["expected_sources"],
                "required_terms": case["required_terms"],
                "forbidden_terms": case["forbidden_terms"],
                "retrieved": hits,
                "retrieval_seconds": round(retrieval_seconds, 3),
                "reciprocal_rank": 1 / rank if rank else 0,
            }
            prompt = (
                make_prompt(tokenizer, format_evidence(hits), case["question"])
                if hits
                else None
            )
            for key, adapter_enabled in (("baseline", False), ("hybrid", True)):
                if prompt is None:
                    answer, status, reason, seconds = (
                        "",
                        "abstained",
                        "no retrieved evidence",
                        0.0,
                    )
                else:
                    start = perf_counter()
                    answer = generate(
                        model,
                        tokenizer,
                        prompt,
                        device,
                        max_new_tokens,
                        adapter_enabled,
                    )
                    seconds = perf_counter() - start
                    status, reason = check_answer(answer, hits)
                row[key] = {
                    "answer": answer,
                    "reason": reason,
                    "generation_seconds": round(seconds, 3),
                    **score_answer(case, answer, status, hits),
                }
            rows.append(row)
            b, h = row["baseline"], row["hybrid"]
            terms_b = (
                "-"
                if b["required_term_coverage"] is None
                else f"{b['required_term_coverage']:.0%}"
            )
            terms_h = (
                "-"
                if h["required_term_coverage"] is None
                else f"{h['required_term_coverage']:.0%}"
            )
            print(
                f"{case['id']}: base={b['status']} ({terms_b}) | LoRA={h['status']} ({terms_h})",
                flush=True,
            )
    finally:
        client.close()

    supported = [row for row in rows if not row["should_abstain"]]
    results = {
        "configuration": {
            "base_model": base_name,
            "base_revision": base_revision,
            "embedding_model": embedding_name,
            "embedding_revision": os.getenv("EMBEDDING_REVISION") or None,
            "adapter_path": str(adapter_path.relative_to(ROOT)),
            "device": device,
            "top_k": top_k,
            "min_score": min_score,
            "max_new_tokens": max_new_tokens,
            "prompt_format": "training.make_prompt() for both systems; do_sample=False",
            "training_metadata": training_metadata,
        },
        "retrieval": {
            "recall_at_k": sum(row["reciprocal_rank"] > 0 for row in supported)
            / len(supported),
            "mrr_at_k": mean(row["reciprocal_rank"] for row in supported),
            "retriever_only_decision_accuracy": sum(
                bool(row["retrieved"]) != row["should_abstain"] for row in rows
            )
            / len(rows),
        },
        "baseline": summarize(rows, "baseline", len(supported)),
        "hybrid": summarize(rows, "hybrid", len(supported)),
        "cases": rows,
        "metric_notes": [
            "Required terms: exact case-insensitive whole phrases; rejected answers receive zero accepted coverage.",
            "Citation-faithfulness is a proxy: accepted answer, expected valid source, all required terms, no forbidden phrases; NOT semantic entailment.",
            "Style is a proxy: accepted, <=80 words, one paragraph, no salutation/list, citation. Human review still required.",
            "Unsupported answers are correct only if the model abstains, not merely if a bad answer is rejected.",
            "Latency is generation time only; retrieval is measured separately and shared.",
        ],
    }
    out = ROOT / "reports/evaluation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nMetric                         Baseline        Hybrid")
    for metric in (
        "decision_accuracy",
        "required_term_coverage_accepted",
        "citation_faithfulness_proxy",
        "style_proxy",
    ):
        print(
            f"{metric:30} {results['baseline'][metric]:7.1%}       {results['hybrid'][metric]:7.1%}"
        )
    for system in ("baseline", "hybrid"):
        value = results[system]
        print(
            f"{system} unsupported abstentions: {value['unsupported_abstentions']}/{value['unsupported_total']}"
        )
        print(
            f"{system} avg generation: {value['mean_generation_seconds_per_case']:.2f}s per case"
        )
    print(
        f"Retrieval recall@{top_k}: {results['retrieval']['recall_at_k']:.1%}, MRR: {results['retrieval']['mrr_at_k']:.3f}"
    )
    print(f"Detailed traces: {out}")


if __name__ == "__main__":
    main()
