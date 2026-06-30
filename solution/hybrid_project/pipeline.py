"""Comparable base-only, baseline RAG, and hybrid RAG answer paths."""

from .config import Settings
from .generation import (
    Generator,
    citations_are_valid,
    citations_in,
)
from .models import Answer, SystemName
from .retrieval import Retriever


class ComparisonPipeline:
    """Apply one retriever and prompt contract across all compared systems."""

    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        base_generator: Generator,
        lora_generator: Generator,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.base_generator = base_generator
        self.lora_generator = lora_generator

    def answer(self, question: str, system: SystemName) -> Answer:
        """Generate with the selected system and enforce RAG citations."""

        if system == "base-only":
            answer_text = self.base_generator.generate(question, [])
            return Answer(
                system=system,
                question=question,
                answer=answer_text,
                status="answered",
                citations=citations_in(answer_text),
            )

        hits = self.retriever.search(question)
        retrieved_sources = list(dict.fromkeys(hit.document_id for hit in hits))
        top_score = hits[0].score if hits else None
        accepted_hits = [hit for hit in hits if hit.score >= self.settings.min_score]
        if not accepted_hits:
            return Answer(
                system=system,
                question=question,
                answer=(
                    "The approved material is insufficient. Escalate for manual review."
                ),
                status="abstained",
                retrieved_sources=retrieved_sources,
                top_score=top_score,
            )

        generator = self.base_generator if system == "rag-base" else self.lora_generator
        answer_text = generator.generate(question, accepted_hits)
        if not citations_are_valid(answer_text, accepted_hits):
            return Answer(
                system=system,
                question=question,
                answer=(
                    "The generated answer failed citation validation. "
                    "Escalate for manual review."
                ),
                status="invalid-citations",
                retrieved_sources=retrieved_sources,
                top_score=top_score,
            )
        return Answer(
            system=system,
            question=question,
            answer=answer_text,
            status="answered",
            citations=citations_in(answer_text),
            retrieved_sources=retrieved_sources,
            top_score=top_score,
        )
