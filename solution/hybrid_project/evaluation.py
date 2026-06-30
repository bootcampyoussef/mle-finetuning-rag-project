"""Shared benchmark metrics and decision-report generation."""

import json
from pathlib import Path
from time import perf_counter

from .config import Settings
from .datasets import load_evaluation_cases
from .models import (
    BenchmarkSummary,
    EvaluationCase,
    RetrievalRow,
    SystemEvaluationRow,
    SystemName,
    SystemSummary,
    TrainingSummary,
)
from .pipeline import ComparisonPipeline
from .retrieval import Retriever


SYSTEMS: tuple[SystemName, ...] = ("base-only", "rag-base", "rag-lora")


def _first_relevant_rank(retrieved: list[str], expected: list[str]) -> int | None:
    """Return the first one-based expected-source rank."""

    for rank, source_id in enumerate(retrieved, start=1):
        if source_id in expected:
            return rank
    return None


def evaluate_retrieval(
    retriever: Retriever,
    cases: list[EvaluationCase],
) -> tuple[float, float, list[RetrievalRow]]:
    """Measure retrieval once because both RAG systems share it."""

    rows: list[RetrievalRow] = []
    for case in cases:
        hits = retriever.search(case.question)
        retrieved = list(dict.fromkeys(hit.document_id for hit in hits))
        rows.append(
            RetrievalRow(
                id=case.id,
                expected_sources=case.expected_sources,
                retrieved_sources=retrieved,
                first_relevant_rank=_first_relevant_rank(
                    retrieved, case.expected_sources
                ),
                top_score=hits[0].score if hits else None,
                should_abstain=case.should_abstain,
            )
        )
    supported = [row for row in rows if not row.should_abstain]
    recall = sum(row.first_relevant_rank is not None for row in supported) / len(
        supported
    )
    mrr = sum(
        0.0 if row.first_relevant_rank is None else 1.0 / row.first_relevant_rank
        for row in supported
    ) / len(supported)
    return recall, mrr, rows


def _term_coverage(answer: str, required_terms: list[str]) -> float:
    """Calculate transparent required-term coverage."""

    if not required_terms:
        return 1.0
    lowered = answer.lower()
    return sum(term.lower() in lowered for term in required_terms) / len(required_terms)


def _style_compliance(answer: str, status: str) -> float:
    """Score concise, direct language without vague hedging."""

    if status == "abstained":
        return 1.0
    words = answer.split()
    first_sentence = answer.split(".", maxsplit=1)[0].split()
    checks = (
        len(words) <= 80,
        len(first_sentence) <= 30,
        not any(hedge in answer.lower() for hedge in ("maybe", "probably", "perhaps")),
    )
    return sum(checks) / len(checks)


def _faithful(
    case: EvaluationCase,
    status: str,
    answer: str,
    citations: list[str],
) -> bool:
    """Check abstention, expected citations, and forbidden claims."""

    if case.should_abstain:
        return status == "abstained"
    if status != "answered" or not citations:
        return False
    forbidden_present = any(
        term.lower() in answer.lower() for term in case.forbidden_terms
    )
    return set(citations).issubset(set(case.expected_sources)) and not forbidden_present


def evaluate_system(
    pipeline: ComparisonPipeline,
    cases: list[EvaluationCase],
    system: SystemName,
) -> SystemSummary:
    """Run one system over every held-out case with identical rubrics."""

    rows: list[SystemEvaluationRow] = []
    for case in cases:
        started = perf_counter()
        result = pipeline.answer(case.question, system)
        latency = perf_counter() - started
        decision_correct = (
            result.status == "abstained"
            if case.should_abstain
            else result.status == "answered"
        )
        relevance = (
            1.0
            if case.should_abstain and result.status == "abstained"
            else _term_coverage(result.answer, case.required_terms)
        )
        rows.append(
            SystemEvaluationRow(
                id=case.id,
                system=system,
                status=result.status,
                answer=result.answer,
                citations=result.citations,
                relevance=relevance,
                faithful=_faithful(
                    case,
                    result.status,
                    result.answer,
                    result.citations,
                ),
                style_compliance=_style_compliance(result.answer, result.status),
                decision_correct=decision_correct,
                latency_seconds=latency,
            )
        )
    return SystemSummary(
        system=system,
        answer_relevance=sum(row.relevance for row in rows) / len(rows),
        faithfulness=sum(row.faithful for row in rows) / len(rows),
        style_compliance=sum(row.style_compliance for row in rows) / len(rows),
        decision_accuracy=sum(row.decision_correct for row in rows) / len(rows),
        mean_latency_seconds=sum(row.latency_seconds for row in rows) / len(rows),
        rows=rows,
    )


def evaluate(pipeline: ComparisonPipeline) -> BenchmarkSummary:
    """Run retrieval and every answer system on the shared benchmark."""

    cases = load_evaluation_cases(pipeline.settings.evaluation_file)
    recall, mrr, retrieval_rows = evaluate_retrieval(pipeline.retriever, cases)
    systems = [evaluate_system(pipeline, cases, system) for system in SYSTEMS]
    return BenchmarkSummary(
        retrieval_recall_at_k=recall,
        mean_reciprocal_rank=mrr,
        retrieval_rows=retrieval_rows,
        systems=systems,
    )


def _system(summary: BenchmarkSummary, name: SystemName) -> SystemSummary:
    """Return one named system summary from the benchmark."""

    return next(item for item in summary.systems if item.system == name)


def render_report(
    settings: Settings,
    pipeline: ComparisonPipeline,
    summary: BenchmarkSummary,
) -> str:
    """Create a reviewable Markdown decision report."""

    rag_base = _system(summary, "rag-base")
    rag_lora = _system(summary, "rag-lora")
    relevance_delta = rag_lora.answer_relevance - rag_base.answer_relevance
    faithfulness_delta = rag_lora.faithfulness - rag_base.faithfulness
    style_delta = rag_lora.style_compliance - rag_base.style_compliance
    supported_scores = [
        row.top_score
        for row in summary.retrieval_rows
        if not row.should_abstain and row.top_score is not None
    ]
    unsupported_scores = [
        row.top_score
        for row in summary.retrieval_rows
        if row.should_abstain and row.top_score is not None
    ]
    weakest_supported = min(supported_scores) if supported_scores else 0.0
    strongest_unsupported = max(unsupported_scores) if unsupported_scores else 0.0
    training_summary_path = settings.adapter_path / "training_summary.json"
    training_summary = (
        TrainingSummary.model_validate_json(
            training_summary_path.read_text(encoding="utf-8")
        )
        if training_summary_path.exists()
        else None
    )
    try:
        adapter_display = settings.adapter_path.relative_to(settings.project_root)
    except ValueError:
        adapter_display = settings.adapter_path

    lines = [
        "# Reference Hybrid Evaluation",
        "",
        "## Configuration",
        "",
        f"- Base model: `{settings.base_model}`",
        f"- Adapter path: `{adapter_display}`",
        f"- Embedding model: `{settings.embedding_model}`",
        "- Vector database: `Qdrant` local mode with cosine similarity",
        f"- Retrieval: `top_k={settings.top_k}`, `min_score={settings.min_score:.2f}`",
        f"- Baseline generator: `{pipeline.base_generator.mode}`",
        f"- Hybrid generator: `{pipeline.lora_generator.mode}`",
    ]
    if training_summary is not None:
        lines.extend(
            [
                f"- LoRA targets: `{', '.join(training_summary.lora_target_modules)}`",
                f"- LoRA rank / alpha / dropout: `{training_summary.lora_rank}` / "
                f"`{training_summary.lora_alpha}` / "
                f"`{training_summary.lora_dropout:.2f}`",
                f"- Training: `{training_summary.epochs}` epochs, learning rate "
                f"`{training_summary.learning_rate}`, seed "
                f"`{training_summary.random_seed}`",
                f"- Trainable parameters: `{training_summary.trainable_parameters}` "
                f"of `{training_summary.total_parameters}` "
                f"(`{training_summary.trainable_percentage:.3f}%`)",
                f"- Training loss: `{training_summary.epoch_losses[0]:.3f}` to "
                f"`{training_summary.epoch_losses[-1]:.3f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Metric Rubric",
            "",
            "- `recall@k`: supported cases whose expected source appears in the shared retrieval results.",
            "- `MRR`: reciprocal rank of the first expected source for supported cases.",
            "- `answer relevance`: proportion of labeled required terms present in the answer.",
            "- `faithfulness`: valid expected citations with no labeled forbidden claim; unsupported cases must abstain.",
            "- `style compliance`: concise first sentence, concise total response, and no vague hedge.",
            "- `decision accuracy`: answer supported cases and abstain on unsupported cases.",
            "",
            "## Shared Retrieval Results",
            "",
            "| Metric | Score |",
            "| --- | ---: |",
            f"| Retrieval recall@{settings.top_k} | {summary.retrieval_recall_at_k:.3f} |",
            f"| Mean reciprocal rank | {summary.mean_reciprocal_rank:.3f} |",
            "",
            "## System Comparison",
            "",
            "| System | Relevance | Faithfulness | Style | Decision accuracy | Mean latency (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for system in summary.systems:
        lines.append(
            f"| {system.system} | {system.answer_relevance:.3f} | "
            f"{system.faithfulness:.3f} | {system.style_compliance:.3f} | "
            f"{system.decision_accuracy:.3f} | "
            f"{system.mean_latency_seconds:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Hybrid Delta Against Baseline RAG",
            "",
            f"- Answer relevance delta: `{relevance_delta:+.3f}`",
            f"- Faithfulness delta: `{faithfulness_delta:+.3f}`",
            f"- Style-compliance delta: `{style_delta:+.3f}`",
            "",
            "## Per-Case Comparison",
            "",
            "| ID | System | Status | Relevance | Faithful | Style | Citations |",
            "| --- | --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for system in summary.systems:
        for row in system.rows:
            citations = ", ".join(row.citations) if row.citations else "-"
            lines.append(
                f"| {row.id} | {row.system} | {row.status} | "
                f"{row.relevance:.2f} | {'yes' if row.faithful else 'no'} | "
                f"{row.style_compliance:.2f} | {citations} |"
            )
    lines.extend(["", "## Failure And Fragility Analysis", ""])
    failure_count = 0
    for system in summary.systems:
        system_failures = [
            row
            for row in system.rows
            if (not row.decision_correct or not row.faithful or row.relevance < 1.0)
        ]
        for row in system_failures[:3]:
            failure_count += 1
            lines.append(
                f"- `{row.system}` on `{row.id}`: status `{row.status}`, "
                f"relevance `{row.relevance:.2f}`, faithful "
                f"`{'yes' if row.faithful else 'no'}`."
            )
    if failure_count == 0:
        lines.append(
            "- No supplied case fails the deterministic rubric. The benchmark "
            "is small and does not establish production reliability."
        )
    lines.extend(
        [
            f"- Shared retrieval recall is "
            f"`{summary.retrieval_recall_at_k:.3f}`; failures after successful "
            "retrieval are generation, citation, or answer-coverage failures.",
            f"- The adapter changes faithfulness by "
            f"`{faithfulness_delta:+.3f}` and decision accuracy from "
            f"`{rag_base.decision_accuracy:.3f}` to "
            f"`{rag_lora.decision_accuracy:.3f}`. Remaining invalid citations "
            "still require rejection rather than automatic repair.",
        ]
    )
    lines.extend(
        [
            "",
            "## Threshold Decision",
            "",
            f"The shared threshold is `{settings.min_score:.2f}`. The weakest "
            f"supported top score is `{weakest_supported:.3f}` and the strongest "
            f"unsupported top score is `{strongest_unsupported:.3f}`. The same "
            "threshold is applied to baseline RAG and hybrid RAG so adapter "
            "quality cannot hide retrieval uncertainty.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if relevance_delta > 0 or faithfulness_delta > 0 or style_delta > 0:
        lines.append(
            "Keep the hybrid path as an experiment, not as an automatic "
            "replacement. It improves at least one held-out metric, but the "
            "failure rows and added adapter lifecycle still require review."
        )
    else:
        lines.append(
            "Do not promote the hybrid path yet. The adapter does not improve "
            "the held-out metrics enough to justify training and maintenance."
        )
    lines.extend(
        [
            "",
            "## Prioritized Next Experiments",
            "",
            "1. Expand held-out paraphrases and policy-change cases before tuning the adapter again.",
            "2. Compare a stronger small instruction model using the same retrieval and metric contracts.",
            "3. Review invalid-citation cases and adjust training data only after identifying a repeatable failure pattern.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    pipeline: ComparisonPipeline,
    output_path: Path,
) -> BenchmarkSummary:
    """Run the benchmark and write Markdown plus a regenerated JSON trace."""

    summary = evaluate(pipeline)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_report(pipeline.settings, pipeline, summary),
        encoding="utf-8",
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return summary
