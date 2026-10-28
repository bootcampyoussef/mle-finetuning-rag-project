"""Minimal deterministic test for the Northstar evaluation report generator."""

import json

from src import build_evaluation_report as report


def test_report_contains_actual_traces(tmp_path, monkeypatch):
    trace = {
        "id": "example-supported",
        "question": "What does the example policy say?",
        "should_abstain": False,
        "expected_sources": ["shipping"],
        "required_terms": ["three days"],
        "forbidden_terms": [],
        "retrieved": [
            {
                "source_id": "shipping",
                "score": 0.8,
                "chunk_position": 0,
                "source_text": "Shipping takes three days.",
            }
        ],
        "baseline": {
            "status": "rejected",
            "reason": "missing citation",
            "required_term_coverage": 1.0,
            "cited_sources": [],
            "forbidden_matches": [],
            "generation_seconds": 0.6,
            "answer": "Shipping takes three days.",
        },
        "hybrid": {
            "status": "answered",
            "reason": "valid citation",
            "required_term_coverage": 1.0,
            "cited_sources": ["shipping"],
            "forbidden_matches": [],
            "generation_seconds": 0.61,
            "answer": "Shipping takes three days. [shipping]",
        },
    }
    metrics = {
        "decision_accuracy": 0.5,
        "required_term_coverage_accepted": 0.5,
        "citation_faithfulness_proxy": 0.5,
        "style_proxy": 0.5,
        "unsupported_abstentions": 0,
        "unsupported_total": 0,
        "mean_generation_seconds_per_case": 0.6,
    }
    result = {
        "configuration": {
            "base_model": "example-model",
            "embedding_model": "example-embedder",
            "device": "cpu",
            "top_k": 3,
            "min_score": 0.3,
            "max_new_tokens": 80,
            "training_metadata": {},
        },
        "retrieval": {
            "recall_at_k": 1.0,
            "mrr_at_k": 1.0,
            "retriever_only_decision_accuracy": 1.0,
        },
        "baseline": metrics,
        "hybrid": metrics,
        "cases": [trace],
    }
    input_file = tmp_path / "evaluation_results.json"
    output_file = tmp_path / "evaluation_report.md"
    input_file.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(report, "INPUT", input_file)
    monkeypatch.setattr(report, "OUTPUT", output_file)

    report.main()
    body = output_file.read_text(encoding="utf-8")

    assert "example-supported" in body
    assert "Shipping takes three days. [shipping]" in body
    assert "score 0.800" in body
    assert "REVISE" in body
