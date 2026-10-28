import os
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer



ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "knowledge_base"
COLLECTION = "northstar_policies"


def load_chunks():
    """Each policy paragraph becomes a chunk with a stable source ID."""
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        lines = markdown.splitlines()
        title = lines[0].lstrip("# ") if lines and lines[0].startswith("# ") else path.stem
        body = "\n".join(lines[1:] if lines and lines[0].startswith("# ") else lines)

        for position, paragraph in enumerate(re.split(r"\n\s*\n", body.strip())):
            text = " ".join(paragraph.split())
            if text:
                chunks.append({
                    "source_id": path.stem,
                    "filename": path.name,
                    "title": title,
                    "chunk_position": position,
                    "source_text": text,
                })
    return chunks




def create_index(chunks, embedder):
    storage = os.getenv("VECTOR_STORAGE_PATH", "").strip()
    if storage:
        path = Path(storage)
        if not path.is_absolute():
            path = ROOT / path
        client = QdrantClient(path=str(path))
    else:
        client = QdrantClient(":memory:")

    vectors = embedder.encode(
        [f"{c['title']}. {c['source_text']}" for c in chunks],
        normalize_embeddings=True,
    )
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(id=i, vector=vector.tolist(), payload=chunk)
            for i, (chunk, vector) in enumerate(zip(chunks, vectors), start=1)
        ],
    )
    return client


def retrieve(question, client, embedder, top_k=3, min_score=0.30):
    query_vector = embedder.encode(question, normalize_embeddings=True).tolist()
    result = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {**point.payload, "score": round(point.score, 4)}
        for point in result.points
        if point.score >= min_score
    ]


def main():
    load_dotenv(ROOT / ".env")
    chunks = load_chunks()
    if not chunks:
        raise ValueError(f"No Markdown policies found in {POLICY_DIR}")

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    revision = os.getenv("EMBEDDING_REVISION") or None
    embedder = SentenceTransformer(model_name, revision=revision)
    client = create_index(chunks, embedder)
    top_k = int(os.getenv("TOP_K", "3"))
    min_score = float(os.getenv("MIN_SCORE", "0.30"))

    print(f"Indexed {len(chunks)} chunks from {len(set(c['source_id'] for c in chunks))} policies")
    print(f"Retrieval settings: top_k={top_k}, min_score={min_score}\n")

    questions = [
        "How long does standard shipping take?",
        "What should support do if a customer lost their authenticator?",
        "Where can I book a meeting room in the office?",
    ]
    try:
        for question in questions:
            print(f"Question: {question}")
            hits = retrieve(question, client, embedder, top_k, min_score)
            if not hits:
                print("ABSTAIN: no result above the similarity threshold")
            else:
                for hit in hits:
                    print(f"  [{hit['source_id']}] chunk={hit['chunk_position']} "
                          f"score={hit['score']:.4f}: {hit['source_text'][:140]}")
            print()
    finally:
        client.close()



if __name__ == "__main__":
    main()