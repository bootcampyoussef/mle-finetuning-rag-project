"""Tests for the benchmark and required report sections."""

from solution.hybrid_project.evaluation import evaluate, render_report
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
