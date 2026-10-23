"""Tests for the benchmark and required report sections."""

import pytest

from solution.hybrid_project.evaluation import (
    _style_compliance,
    evaluate,
    render_report,
)
from solution.hybrid_project.pipeline import ComparisonPipeline


def test_benchmark_compares_all_systems(
    smoke_pipeline: ComparisonPipeline,
) -> None:
    """One run should retain base-only, baseline RAG, and hybrid rows."""

    summary = evaluate(smoke_pipeline)

    assert [system.system for system in summary.systems] == [
        "base-only",
        "rag-base",
        "rag-lora",
    ]
    assert all(len(system.rows) == 12 for system in summary.systems)


def test_report_contains_decision_evidence(
    smoke_pipeline: ComparisonPipeline,
) -> None:
    """The generated report must support a baseline-versus-hybrid decision."""

    summary = evaluate(smoke_pipeline)
    report = render_report(
        smoke_pipeline.settings,
        smoke_pipeline,
        summary,
    )

    assert "## Shared Retrieval Results" in report
    assert "## Hybrid Delta Against Baseline RAG" in report
    assert "## Failure And Fragility Analysis" in report
    assert "## Threshold Decision" in report
    assert "## Recommendation" in report


def test_unsupported_answers_do_not_receive_full_relevance(
    smoke_pipeline: ComparisonPipeline,
) -> None:
    """Empty required-term labels must not reward an unsupported answer."""

    summary = evaluate(smoke_pipeline)
    unsupported_ids = {row.id for row in summary.retrieval_rows if row.should_abstain}
    assert unsupported_ids
    base_only = next(
        system for system in summary.systems if system.system == "base-only"
    )
    for row in base_only.rows:
        if row.id in unsupported_ids:
            assert row.status == "answered"
            assert row.relevance == 0.0


@pytest.mark.parametrize(
    ("status", "expected"),
    [("answered", 1.0), ("abstained", 1.0), ("invalid-citations", 0.0)],
)
def test_style_score_does_not_reward_a_rejection_template(
    status: str, expected: float
) -> None:
    """A short validation message is not a successful model response."""

    answer = (
        "The generated answer failed citation validation. Escalate for manual review."
    )
    assert _style_compliance(answer, status) == expected
