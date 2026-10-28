"""Milestone 1: evaluate retrieval on held-out Northstar questions."""

import json
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from retrieval import ROOT, create_index, load_chunks, retrieve


def main():
    load_dotenv(ROOT / ".env")
    cases = json.loads((ROOT / "data/evaluation/cases.json").read_text(encoding="utf-8"))

    embedder = SentenceTransformer(
        os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        revision=os.getenv("EMBEDDING_REVISION") or None,
    )
    client = create_index(load_chunks(), embedder)
    top_k = int(os.getenv("TOP_K", "3"))
    min_score = float(os.getenv("MIN_SCORE", "0.30"))

    supported = 0
    recall_total = 0.0
    rr_total = 0.0
    correct_decisions = 0
    correct_abstentions = 0

    try:
        for case in cases:
            hits = retrieve(case["question"], client, embedder, top_k, min_score)
            found = [hit["source_id"] for hit in hits]
            expected = case["expected_sources"]
            should_abstain = case["should_abstain"]
            actual_abstain = not hits

            correct_decisions += (actual_abstain == should_abstain)
            if should_abstain:
                correct_abstentions += actual_abstain
            else:
                supported += 1
                recall_total += len(set(found) & set(expected)) / len(expected)
                rr_total += next((1 / rank for rank, src in enumerate(found, 1)
                                  if src in expected), 0.0)

            status = "ABSTAIN" if actual_abstain else ", ".join(found)
            marker = "OK" if actual_abstain == should_abstain else "CHECK"
            print(f"{case['id']}: {status} | {marker}")
            if hits:
                print("  Scores:", ", ".join(f"{h['source_id']}={h['score']:.3f}" for h in hits))

        print(f"\nRecall@{top_k}: {recall_total / supported:.2%}")
        print(f"MRR@{top_k}: {rr_total / supported:.3f}")
        print(f"Supported/unsupported decision accuracy: {correct_decisions}/{len(cases)}")
        print(f"Unsupported abstentions: {correct_abstentions}/{sum(c['should_abstain'] for c in cases)}")
        print(f"Threshold: {min_score:.2f}")
    finally:
        client.close()


if __name__ == "__main__":
    main()