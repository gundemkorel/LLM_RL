"""Minimal tabular dataset abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TabularDataset:
    """A simple wrapper around tabular features and labels.

    The dataset stores feature matrices ``X`` and corresponding labels ``y``
    while offering helpers to build from NumPy arrays or pandas ``DataFrame``
    objects.  It also exposes basic sampling and splitting utilities that are
    convenient for quick experiments and tests.
    """

    _X: np.ndarray
    _y: np.ndarray
    _feature_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate invariants after initialization."""
        if self._X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self._y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if self._X.shape[0] != self._y.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        if self._feature_names is not None and len(self._feature_names) != self._X.shape[1]:
            raise ValueError("feature_names must match the number of columns in X")

    @classmethod
    def from_arrays(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "TabularDataset":
        """Construct a dataset from raw NumPy arrays.

        Args:
            X: Feature matrix with shape ``(n_samples, n_features)``.
            y: Label vector with shape ``(n_samples,)``.
            feature_names: Optional iterable of feature names.

        Returns:
            A :class:`TabularDataset` instance.
        """

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        names = list(feature_names) if feature_names is not None else None
        return cls(X_arr, y_arr, names)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        label_col: str,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> "TabularDataset":
        """Construct a dataset from a pandas ``DataFrame``.

        Args:
            df: Input dataframe containing features and labels.
            label_col: Name of the column containing labels.
            feature_cols: Optional sequence specifying which columns to use as
                features.  When ``None`` all columns except ``label_col`` are
                treated as features.

        Returns:
            A :class:`TabularDataset` with data extracted from ``df``.
        """

        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not found in dataframe")

        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != label_col]
        else:
            missing = [col for col in feature_cols if col not in df.columns]
            if missing:
                raise KeyError(f"Feature columns not found in dataframe: {missing}")

        X = df.loc[:, feature_cols].to_numpy()
        y = df[label_col].to_numpy()
        return cls(X, y, list(feature_cols))

    def split(
        self,
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
        seed: int = 0,
    ) -> Tuple["TabularDataset", "TabularDataset", "TabularDataset"]:
        """Split the dataset into train/validation/test subsets.

        Fractions default to ``0.7/0.15/0.15`` and must sum to one (within a
        small tolerance).  Splits are created by shuffling indices with the
        provided ``seed``.
        """

        total = train + val + test
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError("Split fractions must sum to 1.0")

        n_samples = len(self)
        indices = np.arange(n_samples)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        train_end = int(round(train * n_samples))
        val_end = train_end + int(round(val * n_samples))
        val_end = min(val_end, n_samples)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return (
            self._subset(train_idx),
            self._subset(val_idx),
            self._subset(test_idx),
        )

    def sample(self, n: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Sample ``n`` feature rows without replacement."""

        if n < 0:
            raise ValueError("n must be non-negative")
        if n > len(self):
            raise ValueError("Cannot sample more rows than available")

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self), size=n, replace=False) if n else np.array([], dtype=int)
        return self._X[indices]

    def get_row(self, i: int) -> Tuple[np.ndarray, int]:
        """Return the ``i``-th feature row and label."""

        if i < 0 or i >= len(self):
            raise IndexError("Row index out of range")
        x = self._X[i]
        y = int(self._y[i])
        return x, y

    @property
    def X(self) -> np.ndarray:
        """Underlying feature matrix."""

        return self._X

    @property
    def y(self) -> np.ndarray:
        """Underlying label vector."""

        return self._y

    @property
    def n_features(self) -> int:
        """Number of feature columns."""

        return self._X.shape[1]

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Optional feature names associated with columns of ``X``."""

        return self._feature_names

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._X.shape[0]

    def _subset(self, indices: np.ndarray) -> "TabularDataset":
        """Create a dataset from a subset of indices."""

        if indices.size == 0:
            X = self._X[:0].copy()
            y = self._y[:0].copy()
        else:
            X = self._X[indices].copy()
            y = self._y[indices].copy()
        names = list(self._feature_names) if self._feature_names is not None else None
        return TabularDataset(X, y, names)
