"""Tests for retrieval thresholds, citations, and comparable answer paths."""

from collections.abc import Sequence

from solution.hybrid_project.models import SearchHit
from solution.hybrid_project.pipeline import ComparisonPipeline


class InventedCitationGenerator:
    """Generator stub proving that post-generation validation is active."""

    @property
    def mode(self) -> str:
        """Return a trace label."""

        return "invented-citation-test"

    def generate(self, question: str, hits: Sequence[SearchHit]) -> str:
        """Return a source ID that was never retrieved."""

        del question, hits
        return "Support can make an exception. [executive-override]"


def test_supported_question_returns_retrieved_citation(
    smoke_pipeline: ComparisonPipeline,
) -> None:
    """A lexical shipping question should use the shipping source."""

    result = smoke_pipeline.answer(
        "What is the usual standard shipping time?",
        "rag-base",
    )

    assert result.status == "answered"
    assert result.citations == ["shipping"]


def test_unsupported_question_abstains(
    smoke_pipeline: ComparisonPipeline,
) -> None:
    """Nearest-neighbor search must not invent cafeteria policy."""

    result = smoke_pipeline.answer(
        "Where can I find next week's cafeteria menu?",
        "rag-lora",
    )

    assert result.status == "abstained"
    assert not result.citations


def test_invented_citation_is_rejected(
    smoke_pipeline: ComparisonPipeline,
) -> None:
    """Generated source IDs must be a subset of accepted evidence."""

    invalid_pipeline = ComparisonPipeline(
        smoke_pipeline.settings,
        smoke_pipeline.retriever,
        smoke_pipeline.base_generator,
        InventedCitationGenerator(),
    )
    result = invalid_pipeline.answer(
        "What is the usual standard shipping time?",
        "rag-lora",
    )

    assert result.status == "invalid-citations"
    assert not result.citations
