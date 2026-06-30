"""Environment-backed configuration for training and evaluation."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """All paths and controls needed by the local project workflow."""

    project_root: Path
    knowledge_base_dir: Path
    training_file: Path
    evaluation_file: Path
    adapter_path: Path
    storage_path: Path | None
    base_model: str
    embedding_model: str
    top_k: int
    min_score: float
    max_sequence_length: int
    max_new_tokens: int
    train_epochs: int
    learning_rate: float
    random_seed: int

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        """Load repository defaults and allow explicit environment overrides."""

        load_dotenv()
        root = project_root or Path(__file__).resolve().parents[2]
        adapter_value = os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter")
        storage_value = os.getenv("VECTOR_STORAGE_PATH", "")
        return cls(
            project_root=root,
            knowledge_base_dir=root / "data" / "knowledge_base",
            training_file=root / "data" / "training" / "instruction_examples.jsonl",
            evaluation_file=root / "data" / "evaluation" / "cases.json",
            adapter_path=root / adapter_value,
            storage_path=root / storage_value if storage_value else None,
            base_model=os.getenv("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            top_k=int(os.getenv("TOP_K", "3")),
            min_score=float(os.getenv("MIN_SCORE", "0.30")),
            max_sequence_length=int(os.getenv("MAX_SEQUENCE_LENGTH", "256")),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "80")),
            train_epochs=int(os.getenv("TRAIN_EPOCHS", "8")),
            learning_rate=float(os.getenv("LEARNING_RATE", "0.0002")),
            random_seed=int(os.getenv("RANDOM_SEED", "42")),
        )
