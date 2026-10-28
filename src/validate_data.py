"""Milestone 1: validate the Northstar policy, training and evaluation data."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_records(path):
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def citations(text):
    return set(re.findall(r"\[([a-z][a-z0-9-]*)\]", text))


def validate():
    policies = list((DATA / "knowledge_base").glob("*.md"))
    sources = {path.stem for path in policies}
    check(sources, "No policy Markdown files found")
    check(all(path.read_text().strip() for path in policies), "Empty policy file")

    training = load_records(DATA / "training" / "instruction_examples.jsonl")
    evaluation = load_records(DATA / "evaluation" / "cases.json")
    check(isinstance(training, list) and training, "Training data must be a nonempty list")
    check(isinstance(evaluation, list) and evaluation, "Evaluation data must be a nonempty list")

    for item in training:
        check(all(isinstance(item.get(field), str) and item[field].strip()
                  for field in ("id", "context", "question", "answer")),
              f"Incomplete training example: {item.get('id')}")
        refs = item.get("reference_ids")
        check(isinstance(refs, list) and refs, f"Missing reference_ids: {item['id']}")
        check(set(refs) <= sources, f"Unknown training source: {item['id']}")
        check(citations(item["context"]) <= set(refs),
              f"Unknown context citation: {item['id']}")
        check(citations(item["answer"]) <= set(refs),
              f"Answer cites a source absent from its references: {item['id']}")

    for item in evaluation:
        check(isinstance(item.get("id"), str) and item["id"].strip(), "Missing evaluation ID")
        check(isinstance(item.get("question"), str) and item["question"].strip(),
              f"Missing evaluation question: {item['id']}")
        refs = item.get("expected_sources")
        check(isinstance(refs, list) and set(refs) <= sources,
              f"Unknown evaluation source: {item['id']}")
        check(isinstance(item.get("should_abstain"), bool),
              f"Missing abstention label: {item['id']}")
        check(all(isinstance(item.get(field), list)
                  for field in ("required_terms", "forbidden_terms")),
              f"Missing evaluation terms: {item['id']}")
        if item["should_abstain"]:
            check(not refs, f"Abstention case has expected sources: {item['id']}")

    train_ids = [item["id"] for item in training]
    eval_ids = [item["id"] for item in evaluation]
    check(len(set(train_ids)) == len(train_ids), "Duplicate training IDs")
    check(len(set(eval_ids)) == len(eval_ids), "Duplicate evaluation IDs")
    check(not (set(train_ids) & set(eval_ids)), "Train/evaluation ID overlap")
    train_questions = {item["question"].strip().casefold() for item in training}
    eval_questions = {item["question"].strip().casefold() for item in evaluation}
    check(len(train_questions) == len(training), "Duplicate training questions")
    check(len(eval_questions) == len(evaluation), "Duplicate evaluation questions")
    check(not (train_questions & eval_questions), "Train/evaluation question overlap")

    print(f"VALID: {len(sources)} policies | {len(training)} training examples | "
          f"{len(evaluation)} evaluation cases")
    print(f"Supported: {sum(not x['should_abstain'] for x in evaluation)} | "
          f"Unsupported: {sum(x['should_abstain'] for x in evaluation)}")


if __name__ == "__main__":
    validate()
