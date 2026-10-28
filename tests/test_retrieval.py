


"""Fast retrieval checks: no Hugging Face model download needed."""

import numpy as np

from src import retrieval


class TinyEmbedder:
    """Deterministic test vectors, so Qdrant can be tested offline."""

    def encode(self, texts, normalize_embeddings=True):
        def vector(text):
            if "shipping" in text.lower():
                return [1.0, 0.0, 0.0]
            if "security" in text.lower():
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]

        if isinstance(texts, str):
            return np.array(vector(texts), dtype=float)
        return np.array([vector(text) for text in texts], dtype=float)


def test_chunking_keeps_source_metadata(tmp_path, monkeypatch):
    (tmp_path / "shipping.md").write_text(
        "# Shipping Policy\n\nStandard shipping takes 3 days.\n\n"
        "Express shipping takes 1 day.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval, "POLICY_DIR", tmp_path)

    chunks = retrieval.load_chunks()

    assert len(chunks) == 2
    assert [item["chunk_position"] for item in chunks] == [0, 1]
    assert all(item["source_id"] == "shipping" for item in chunks)
    assert all(item["filename"] == "shipping.md" for item in chunks)
    assert all(item["title"] == "Shipping Policy" for item in chunks)
    assert "Standard shipping" in chunks[0]["source_text"]


def test_qdrant_retrieval_and_abstention():
    chunks = [
        {
            "source_id": "shipping",
            "filename": "shipping.md",
            "title": "Shipping Policy",
            "chunk_position": 0,
            "source_text": "Shipping usually takes three to five business days.",
        },
        {
            "source_id": "account-security",
            "filename": "account-security.md",
            "title": "Account Security Policy",
            "chunk_position": 0,
            "source_text": "Security recovery requires a secure portal.",
        },
    ]
    embedder = TinyEmbedder()
    client = retrieval.create_index(chunks, embedder)
    try:
        hits = retrieval.retrieve("shipping times", client, embedder, top_k=2, min_score=0.30)
        assert len(hits) == 1
        assert hits[0]["source_id"] == "shipping"
        assert hits[0]["source_text"] == chunks[0]["source_text"]
        assert hits[0]["score"] >= 0.99

        # A query unrelated to either policy must not get evidence.
        assert retrieval.retrieve("office cafeteria", client, embedder, top_k=2, min_score=0.30) == []
    finally:
        client.close()

