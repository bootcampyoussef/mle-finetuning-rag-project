"""Create the Northstar decision report from real held-out evaluation traces.

From the project root:
    uv run python src/build_evaluation_report.py

Reads reports/evaluation_results.json and writes reports/evaluation_report.md.
No network, model loading, or third-party packages required.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/evaluation_results.json"
OUTPUT = ROOT / "reports/evaluation_report.md"


def pct(number):
    return f"{100 * number:.1f}%" if number is not None else "n/a"


def code(text):
    """Display model output literally so it cannot become Markdown formatting."""
    value = str(text or "(empty)")
    fence = "```"
    while fence in value:
        fence += "`"
    return f"{fence}text\n{value}\n{fence}"


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def top_score(row):
    hits = row.get("retrieved", [])
    return f"{hits[0]['score']:.3f}" if hits else "none >= threshold"


def main():
    if not INPUT.is_file():
        raise SystemExit(f"Missing {INPUT}. Run src/evaluate_generation.py first.")

    result = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = result["cases"]
    config = result["configuration"]
    train = config.get("training_metadata") or {}
    retrieval = result["retrieval"]
    base = result["baseline"]
    hybrid = result["hybrid"]
    supported = sum(not r["should_abstain"] for r in rows)
    unsupported = len(rows) - supported

    lines = [
        "# Northstar: Baseline RAG vs. Hybrid RAG + LoRA",
        "",
        "> **Decision: REVISE.** The adapter improves response-format compliance on this small",
        "> held-out benchmark, but the hybrid is **not ready for autonomous support replies**:",
        "> unsupported questions can still receive answers, and accepted answers can omit",
        "> essential policy requirements. Treat these results as experimental evidence.",
        "",
        "## 1. Experiment and reproducibility",
        "",
        f"- Evaluation set: **{len(rows)}** held-out cases; **{supported}** supported, **{unsupported}** unsupported.",
        "- Comparison: same policy corpus, retriever, top-k, threshold, chat prompt,",
        "  generation parameters and test questions. Only the LoRA adapter state differs.",
        f"- Base model: `{config.get('base_model', 'unknown')}`; revision: `{config.get('base_revision') or train.get('resolved_base_revision') or 'not specified'}`.",
        f"- Embedding model: `{config.get('embedding_model', 'unknown')}`; revision: `{config.get('embedding_revision') or 'not specified'}`.",
        f"- Vector search: Qdrant local index, cosine similarity, top-k = **{config['top_k']}**, min score = **{config['min_score']}**.",
        f"- Inference: device `{config.get('device', 'unknown')}`, greedy (`do_sample=False`), `MAX_NEW_TOKENS={config.get('max_new_tokens', 'unknown')}`.",
        f"- Prompt: {config.get('prompt_format', 'same chat prompt for both paths')}.",
        f"- Adapter: `{config.get('adapter_path', 'unknown')}`; only training adapter weights were saved.",
        f"- Training data: `{train.get('training_file', 'data/training/instruction_examples.jsonl')}`; **{train.get('training_examples', 'unknown')}** examples.",
        f"- Adapter configuration: rank **{train.get('lora_rank', 'unknown')}**, alpha **{train.get('lora_alpha', 'unknown')}**, dropout **{train.get('lora_dropout', 'unknown')}**, targets **{', '.join(train.get('lora_target_modules', [])) or 'unknown'}**.",
        f"- Training: {train.get('epochs', 'unknown')} epochs, learning rate {train.get('learning_rate', 'unknown')}, max sequence length {train.get('sequence_length', 'unknown')}, batch size {train.get('batch_size', 'unknown')}, seed {train.get('random_seed', 'unknown')}.",
        f"- Parameters: **{train.get('trainable_parameters', 'unknown')} trainable / {train.get('total_parameters', 'unknown')} total**.",
        "- Training loss per epoch: "
        + (", ".join(str(x) for x in train.get("epoch_loss", [])) or "not recorded")
        + ".",
        "- This was a **real local PEFT optimization loop**, not prompt-only fine-tuning.",
        "",
        "## 2. Aggregate evaluation",
        "",
        "| Metric | Baseline RAG | Hybrid RAG + LoRA |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("decision_accuracy", "Answered/abstained decision accuracy"),
        (
            "required_term_coverage_accepted",
            "Required-term coverage on accepted answers",
        ),
        ("citation_faithfulness_proxy", "Citation/term faithfulness proxy"),
        ("style_proxy", "Response-style proxy"),
    ):
        lines.append(f"| {label} | {pct(base[key])} | {pct(hybrid[key])} |")
    lines += [
        f"| Unsupported abstentions | {base['unsupported_abstentions']}/{base['unsupported_total']} | {hybrid['unsupported_abstentions']}/{hybrid['unsupported_total']} |",
        f"| Mean generation time per case | {base['mean_generation_seconds_per_case']:.2f} s | {hybrid['mean_generation_seconds_per_case']:.2f} s |",
        "",
        f"Retrieval is shared: **recall@{config['top_k']} = {pct(retrieval['recall_at_k'])}**, **MRR@{config['top_k']} = {retrieval['mrr_at_k']:.3f}**, "
        f"retriever-only supported/unsupported accuracy = **{pct(retrieval.get('retriever_only_decision_accuracy'))}**.",
        "Retrieval recall and MRR use *supported cases*; overall decision accuracy uses all cases.",
        "",
        "### Metric definitions and limitations",
        "",
        "- **Decision accuracy:** supported cases count only if the generated answer passed the citation validator; unsupported cases count only if the system explicitly abstained. A rejected answer is **not** a correct abstention.",
        "- **Required-term coverage:** exact whole-phrase matches; rejected outputs contribute zero to aggregate accepted coverage. It is a relevance proxy, not semantic correctness.",
        "- **Citation-faithfulness proxy:** accepted answer, expected retrieved citation, 100% required-term coverage, and no matched forbidden phrase. It does **not** establish that every claim is entailed by evidence.",
        "- **Style proxy:** accepted, concise, single paragraph, no greeting/list, at least one citation. It does not fully validate next-action quality.",
        "- **Latency:** generation only; shared retrieval time, model cold start and training are excluded from aggregate generation time.",
        "- **Forbidden-term detection:** exact phrase matching does not detect semantic paraphrases of prohibited claims.",
        "",
        "## 3. Per-case comparison",
        "",
        "| Case | Expected | Top retrieved score | Baseline | Hybrid | Accepted term coverage (base / hybrid) |",
        "|---|---|---:|---|---|---:|",
    ]
    for r in rows:
        b, h = r["baseline"], r["hybrid"]

        def accepted_cov(x):
            return (
                pct(x.get("required_term_coverage"))
                if x["status"] == "answered"
                else "0% (rejected)"
                if not r["should_abstain"]
                else "n/a"
            )

        lines.append(
            f"| `{cell(r['id'])}` | {'ABSTAIN' if r['should_abstain'] else 'ANSWER'} | {top_score(r)} | "
            f"{cell(b['status'])} | {cell(h['status'])} | "
            f"{accepted_cov(b)} / {accepted_cov(h)} |"
        )

    lines += ["", "## 4. Complete per-case evidence and model outputs", ""]
    for r in rows:
        b, h = r["baseline"], r["hybrid"]
        lines += [
            f"### {r['id']}",
            "",
            f"**Question:** {r['question']}",
            "",
            f"**Expected decision:** {'ABSTAIN' if r['should_abstain'] else 'ANSWER'}; "
            f"**expected source IDs:** {', '.join(r.get('expected_sources', [])) or '(none)'}. "
            f"**Required terms:** {', '.join(r.get('required_terms', [])) or '(none)'}. "
            f"**Forbidden phrases:** {', '.join(r.get('forbidden_terms', [])) or '(none)'}. ",
            "",
            "**Retrieved policy passages:**",
            "",
        ]
        if not r.get("retrieved"):
            lines.append("- None above the configured similarity threshold.")
        else:
            for hit in r["retrieved"]:
                lines.append(
                    f"- `{hit['source_id']}` / chunk {hit.get('chunk_position', '?')} / score {hit['score']:.3f} "
                    f"— {cell(hit.get('source_text', ''))}"
                )
        lines += [""]
        for label, item in (("Baseline RAG", b), ("Hybrid RAG + LoRA", h)):
            lines += [
                f"**{label}:** `{item['status']}` — {cell(item.get('reason', ''))}; "
                f"terms {pct(item.get('required_term_coverage'))}; "
                f"cited `{', '.join(item.get('cited_sources', [])) or '(none)'}`; "
                f"forbidden matches `{', '.join(item.get('forbidden_matches', [])) or '(none)'}`; "
                f"generation {item.get('generation_seconds', 0):.3f} s.",
                "",
                code(item.get("answer", "")),
                "",
            ]

    by_id = {row["id"]: row for row in rows}

    def observation(case_id, label):
        r = by_id.get(case_id)
        if r is None:
            return "Case not present."
        b, h = r["baseline"], r["hybrid"]
        return (
            f"{label}: baseline `{b['status']}`, hybrid `{h['status']}`; "
            f"hybrid required-term coverage {pct(h.get('required_term_coverage'))}; "
            f"top retrieval score {top_score(r)}."
        )

    lines += [
        "## 5. Failure analysis",
        "",
        "### A. Citation success can hide incomplete policy answers",
        "",
        observation("eval-authenticator", "Lost-authenticator case"),
        "A valid `[account-security]` citation does not prove that the answer contains the",
        "required registered-email and secure-upload-portal instructions. These must be",
        "checked against the retrieved passage, not merely by matching citation IDs.",
        "",
        "### B. Retrieval relevance does not prove answerability",
        "",
        observation("eval-bank-chargeback", "External bank chargeback"),
        observation("eval-shipping-compensation", "Late-shipping compensation"),
        "Both prompts are out of scope even when related policies rank highly.",
        "A pure cosine threshold cannot distinguish a semantically related passage",
        "from a passage that actually contains the answer. A separate evidence sufficiency",
        "check is needed before accepting generated claims.",
        "",
        "### C. LoRA is not a universal improvement",
        "",
        observation("eval-outage-updates", "Incident updates"),
        observation("eval-refund-ticket", "Workshop refund window"),
        "Inspect exact model answers above to distinguish omitted citations, missing",
        "required policy content, and refusals. A lower loss alone cannot explain or",
        "justify the behavior of held-out responses.",
        "",
        "## 6. Retrieval-threshold decision",
        "",
        f"The initial threshold **{config['min_score']}** is retained as a documented",
        "baseline, **not** a calibrated safe answerability cutoff. On this corpus,",
        "the two difficult unsupported cases pass retrieval. For example:",
        "",
    ]
    for case_id in (
        "eval-bank-chargeback",
        "eval-shipping-compensation",
        "eval-duplicate-payment",
    ):
        r = by_id.get(case_id)
        if r:
            lines.append(
                f"- `{case_id}`: expected {'abstain' if r['should_abstain'] else 'answer'}, "
                f"top score {top_score(r)}."
            )
    lines += [
        "",
        "The overlap between supported and unsupported similarity scores means a single",
        "higher threshold cannot separate all these cases without losing some valid queries.",
        "Build a separate calibration set for score selection and add a source-grounded",
        "answerability check. Do not optimize hyperparameters against this held-out set",
        "and then claim unbiased final-test performance.",
        "",
        "## 7. Cost and deployment trade-offs",
        "",
        "- **Training:** on-device Apple MPS; no hosted training API or rented GPU was",
        "  required. Training still consumes laptop time, electricity, and engineering effort.",
        "  The training script did not record total wall-clock training time or energy use.",
        f"- **Weights:** only {train.get('trainable_parameters', 'unknown')} parameters were updated",
        "  rather than all base-model parameters. Adapter maintenance adds revision control,",
        "  compatibility testing, evaluation and deployment artifacts.",
        f"- **Inference:** observed means {base['mean_generation_seconds_per_case']:.2f} s (baseline) versus",
        f"  {hybrid['mean_generation_seconds_per_case']:.2f} s (hybrid) *for generation only*",
        "  on the tested machine; 14 cases are too few for a reliable production latency claim.",
        "- **Operational risk:** incorrect chargeback/compensation promises and missing",
        "  account-recovery steps outweigh minor local inference overhead.",
        "",
        "## 8. Decision and prioritized next experiment",
        "",
        "**Recommendation: REVISE the hybrid design; do not deploy it for unsupervised customer support.**",
        "Retain the adapter as an experiment because it helps follow the requested",
        "response format, but treat claim grounding and abstention as independent controls.",
        "",
        "1. **Highest priority:** build an evidence-sufficiency/claim-verification gate using",
        "   retrieved policy text, with explicit refusal on unsupported questions.",
        "   Evaluate against new supported and *near-domain unsupported* examples.",
        "2. Use a stronger baseline with deterministic citation formatting or a controlled",
        "   response template, so LoRA's value is not confused with a formatting fix.",
        "3. Add validation split and multiple random seeds for LoRA; compare fewer epochs,",
        "   longer sequences if warranted, and augment training with well-checked examples.",
        "4. Review all outputs with a human policy rubric before increasing traffic.",
        "   Require complete account-security steps and zero unsupported customer promises.",
        "5. Reserve a **new, untouched final test set** after calibration and retraining.",
        "",
        "**Key uncertainty to reduce next:** does LoRA still improve held-out policy",
        "compliance when the baseline receives the same deterministic citation formatter",
        "and both answer paths use a shared evidence-sufficiency gate?",
        "",
        "## 9. Reproduction",
        "",
        "```bash",
        "uv sync",
        "uv run python src/validate_data.py",
        "uv run python src/retrieval.py",
        "uv run python src/training.py",
        "uv run python src/evaluate_generation.py",
        "uv run python src/build_evaluation_report.py",
        "uv run pytest -v",
        "```",
        "",
        "All aggregate metrics, case answers, citations and evidence above are read",
        "directly from `reports/evaluation_results.json`; the report generator does not",
        "rerun or change model outputs.",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {OUTPUT} ({len(rows)} cases; {supported} supported, {unsupported} unsupported)"
    )


if __name__ == "__main__":
    main()
