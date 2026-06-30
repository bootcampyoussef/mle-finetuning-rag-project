"""Tests for project data contracts and prompt separation."""

from pathlib import Path

from solution.hybrid_project.config import Settings
from solution.hybrid_project.datasets import validate_data
from solution.hybrid_project.documents import chunk_documents, load_documents


def test_shared_data_is_valid_and_auditable() -> None:
    """All labels should resolve to supplied policy sources."""

    root = Path(__file__).parents[2]
    settings = Settings.from_env(project_root=root)
    summary = validate_data(
        (
            settings.knowledge_base_dir,
            settings.training_file,
            settings.evaluation_file,
        )
    )

    assert summary.source_count == 6
    assert summary.training_example_count == 16
    assert summary.evaluation_case_count == 12


def test_chunks_retain_source_identity_and_position() -> None:
    """Citation metadata must survive policy chunking."""

    root = Path(__file__).parents[2]
    chunks = chunk_documents(load_documents(root / "data" / "knowledge_base"))

    assert chunks
    assert all(chunk.document_id in chunk.chunk_id for chunk in chunks)
    assert all(chunk.position >= 0 for chunk in chunks)
