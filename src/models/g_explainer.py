from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled via fallback encoder
    SentenceTransformer = None


class _HashingSentenceEncoder:
    """Deterministic fallback encoder when sentence-transformers is unavailable."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, sentences: Iterable[str]) -> np.ndarray:
        sentences = list(sentences)
        vectors = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for row, sentence in enumerate(sentences):
            tokens = sentence.lower().split()
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, index] += sign
            vectors[row] /= max(1, len(tokens))
        return vectors


class ExplanationInducedModel:
    """g(E): sentence embedding + logistic model over {0, 1}."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device

        self._encoder = self._load_encoder()
        self.clf = LogisticRegression(max_iter=1000)
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Encoder utilities
    # ------------------------------------------------------------------
    def _load_encoder(self):
        if SentenceTransformer is None:
            warnings.warn(
                "sentence-transformers is not installed; falling back to hashing encoder.",
                RuntimeWarning,
            )
            self._using_sentence_transformer = False
            return _HashingSentenceEncoder(self.embedding_dim)

        try:
            encoder = SentenceTransformer(self.model_name, device=self.device)
            self._using_sentence_transformer = True
            return encoder
        except Exception as exc:  # pragma: no cover - exercised when download fails
            warnings.warn(
                f"Failed to load SentenceTransformer '{self.model_name}': {exc}. "
                "Falling back to hashing encoder.",
                RuntimeWarning,
            )
            self._using_sentence_transformer = False
            return _HashingSentenceEncoder(self.embedding_dim)

    def _encode(self, explanations: Sequence[str]) -> np.ndarray:
        if self._using_sentence_transformer:
            vectors = self._encoder.encode(
                list(explanations), show_progress_bar=False
            )
        else:
            vectors = self._encoder.encode(list(explanations))
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_dummy(self) -> None:
        """Train a tiny prior so g is well-posed before RL."""
        texts = [
            "evidence indicates outcome 0 due to low risk factors",
            "strong contributing features suggest outcome 1",
            "negative contribution explains prediction 0",
            "positive contribution explains prediction 1",
            "lack of activating signals keeps class near 0",
            "supporting evidence pushes confidence toward class 1",
            "risk mitigation justifies probability of class 0",
            "dominant signal increases likelihood of class 1",
        ]
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        self.fit(texts, y)

    def fit(self, explanations: Sequence[str], y: Sequence[int]) -> None:
        if len(explanations) == 0:
            raise ValueError("explanations must be a non-empty sequence")
        if len(explanations) != len(y):
            raise ValueError("explanations and y must have the same length")

        embeddings = self._encode(explanations)
        targets = np.asarray(y)
        self.clf.fit(embeddings, targets)
        self.is_fitted = True

    def predict_proba(self, explanations: Sequence[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ExplanationInducedModel must be fitted before prediction")
        embeddings = self._encode(explanations)
        return self.clf.predict_proba(embeddings)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "using_sentence_transformer": self._using_sentence_transformer,
            "clf": self.clf,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, path / "model_state.pkl")

        if self._using_sentence_transformer:
            encoder_dir = path / "encoder"
            encoder_dir.mkdir(parents=True, exist_ok=True)
            # sentence-transformers requires a directory to persist the encoder
            self._encoder.save(str(encoder_dir))

    @classmethod
    def load(cls, path: str | Path) -> "ExplanationInducedModel":
        path = Path(path)
        state = joblib.load(path / "model_state.pkl")

        obj = cls(
            model_name=state["model_name"],
            embedding_dim=state["embedding_dim"],
            device=state.get("device"),
        )

        using_sentence_transformer = state.get("using_sentence_transformer", False)
        if using_sentence_transformer and SentenceTransformer is not None:
            encoder_dir = path / "encoder"
            obj._encoder = SentenceTransformer(str(encoder_dir), device=obj.device)
            obj._using_sentence_transformer = True
        elif using_sentence_transformer and SentenceTransformer is None:
            warnings.warn(
                "sentence-transformers not available when loading; using hashing encoder.",
                RuntimeWarning,
            )
            obj._encoder = _HashingSentenceEncoder(obj.embedding_dim)
            obj._using_sentence_transformer = False
        else:
            obj._encoder = _HashingSentenceEncoder(obj.embedding_dim)
            obj._using_sentence_transformer = False

        obj.clf = state["clf"]
        obj.is_fitted = state.get("is_fitted", False)
        return obj
