from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.data.datasets import TabularDataset


class BlackBoxModel:
    """Opaque binary classifier f: X -> [0, 1]^2 with ``predict_proba`` only."""

    def __init__(
        self,
        estimator: Any,
        n_features: int,
        class_order: Sequence[int] | None = None,
    ) -> None:
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                "Estimator must expose predict_proba; calibrate or wrap the model first."
            )

        self._estimator = estimator
        self.n_features = int(n_features)
        self._class_order = list(class_order) if class_order is not None else [0, 1]
        self._class_indices = self._resolve_class_indices()

    @classmethod
    def from_sklearn(
        cls,
        estimator: Any,
        n_features: int,
        class_order: Sequence[int] | None = None,
    ) -> "BlackBoxModel":
        """Wrap a pre-trained sklearn-style estimator."""

        return cls(estimator=estimator, n_features=n_features, class_order=class_order)

    @classmethod
    def from_training(
        cls,
        dataset: TabularDataset,
        estimator_factory: Callable[[], Any],
        fit_kwargs: Mapping[str, Any] | None = None,
        class_order: Sequence[int] | None = None,
    ) -> "BlackBoxModel":
        """Train a new estimator on the provided ``TabularDataset`` and wrap it."""

        estimator = estimator_factory()
        if not hasattr(estimator, "fit"):
            raise ValueError("estimator_factory must create an object with a fit method")

        kwargs = dict(fit_kwargs or {})
        estimator.fit(dataset.X, dataset.y, **kwargs)
        return cls(
            estimator=estimator,
            n_features=dataset.n_features,
            class_order=class_order,
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return ``P(y|x)`` as ``[p0, p1]`` while accepting 1-D or batched input."""

        array = np.asarray(x)
        was_1d = array.ndim == 1
        batch = np.atleast_2d(array)
        if batch.shape[1] != self.n_features:
            raise ValueError(
                f"Expected input with {self.n_features} features, received {batch.shape[1]}"
            )

        probs = np.asarray(self._estimator.predict_proba(batch))
        if probs.ndim != 2:
            raise ValueError("predict_proba must return a 2D array")

        if probs.shape[1] < len(self._class_order):
            raise ValueError("predict_proba returned fewer classes than expected")

        if self._class_indices is not None:
            probs = probs[:, self._class_indices]

        return probs[0] if was_1d else probs

    def _resolve_class_indices(self) -> np.ndarray | None:
        classes = getattr(self._estimator, "classes_", None)
        if classes is None:
            return None

        class_list = list(classes)
        indices: list[int] = []
        for cls in self._class_order:
            if cls not in class_list:
                raise ValueError(
                    f"Estimator is missing class {cls}; available classes: {class_list}"
                )
            indices.append(class_list.index(cls))
        return np.asarray(indices, dtype=int)

    # The 'opaque' interface: no attributes of the estimator are exposed to the policy.
