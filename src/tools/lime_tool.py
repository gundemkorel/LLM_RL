from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

try:  # pragma: no cover - exercised via optional dependency
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except ImportError:  # pragma: no cover - deterministic fallback exercised in tests
    LimeTabularExplainer = None

PredictProbaFn = Callable[[np.ndarray], Sequence[Sequence[float]]]


def make_lime_tool(
    predict_proba: PredictProbaFn,
    feature_names: Sequence[str] | None = None,
    *,
    training_data: np.ndarray | None = None,
    n_samples: int = 500,
    kernel_width: float | None = None,
    random_state: int | None = None,
) -> Callable[[np.ndarray], Dict[str, float | Sequence[float] | int | str]]:
    """Create a LIME-style local explanation tool.

    Parameters
    ----------
    predict_proba:
        Callable that returns class probabilities for a batch of inputs.
    feature_names:
        Optional feature names to pass to the underlying explainer.
    training_data:
        Optional background data. When provided and the ``lime`` package is
        available, the wrapper will use :class:`LimeTabularExplainer`.
    n_samples:
        Number of perturbations to sample around the input instance.
    kernel_width:
        Width of the exponential kernel weighting perturbations; if ``None`` a
        heuristic based on the feature dimensionality is used.
    random_state:
        Optional integer seed ensuring deterministic behaviour.
    """

    background = _ensure_2d_array(training_data) if training_data is not None else None
    base_seed = int(random_state) if random_state is not None else 0

    def get_local_explanation(x: np.ndarray) -> Dict[str, float | Sequence[float] | int | str]:
        sample = _ensure_1d_array(x)
        probs = _call_predict(predict_proba, sample.reshape(1, -1))
        target_class = int(np.argmax(probs[0]))
        seed = (base_seed + _hash_array(sample)) % (2**32)

        if background is not None and LimeTabularExplainer is not None:
            explainer = LimeTabularExplainer(
                training_data=background,
                feature_names=list(feature_names) if feature_names is not None else None,
                class_names=None,
                discretize_continuous=False,
                sample_around_instance=True,
                random_state=seed,
            )
            explanation = explainer.explain_instance(
                sample,
                lambda data: _call_predict(predict_proba, data),
                num_features=sample.size,
                labels=[target_class],
                num_samples=n_samples,
            )
            coefficients = np.zeros(sample.size, dtype=float)
            for idx, weight in explanation.local_exp.get(target_class, []):
                coefficients[int(idx)] = float(weight)
            intercept = float(explanation.intercept[target_class])
            r2 = float(explanation.score)
            r2 = float(np.clip(r2, 0.0, 1.0))
        else:
            intercept, coefficients, r2 = _lime_via_local_linear(
                predict_proba,
                sample,
                target_class,
                seed,
                n_samples=n_samples,
                kernel_width=kernel_width,
            )

        return {
            "method": "LIME",
            "coefficients": coefficients.tolist(),
            "intercept": float(intercept),
            "r2": float(r2),
            "target_class": target_class,
        }

    return get_local_explanation


def _lime_via_local_linear(
    predict_proba: PredictProbaFn,
    sample: np.ndarray,
    target_class: int,
    seed: int,
    *,
    n_samples: int,
    kernel_width: float | None,
) -> tuple[float, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    dim = sample.size
    width = kernel_width if kernel_width is not None else np.sqrt(dim) * 0.75

    scale = np.maximum(np.abs(sample), 1.0)
    perturbations = rng.normal(loc=0.0, scale=scale, size=(n_samples, dim))
    candidates = sample + perturbations

    preds = _call_predict(predict_proba, candidates)[:, target_class]
    distances = np.linalg.norm(candidates - sample, axis=1)
    weights = np.exp(-(distances ** 2) / (width ** 2 + 1e-12))
    weights = np.clip(weights, 1e-9, None)

    design = np.hstack([np.ones((n_samples, 1)), candidates])
    weighted_design = design * weights[:, None]
    weighted_targets = preds * weights

    beta, *_ = np.linalg.lstsq(weighted_design, weighted_targets, rcond=None)
    intercept = float(beta[0])
    coefficients = beta[1:].astype(float, copy=False)

    fitted = design @ beta
    target_mean = np.average(preds, weights=weights)
    total_var = np.sum(weights * (preds - target_mean) ** 2)
    if total_var <= 1e-12:
        r2 = 0.0
    else:
        residual = preds - fitted
        r2 = 1.0 - (np.sum(weights * residual**2) / total_var)
    r2 = float(np.clip(r2, 0.0, 1.0))

    return intercept, coefficients, r2


def _call_predict(predict_proba: PredictProbaFn, data: np.ndarray) -> np.ndarray:
    arr = np.asarray(predict_proba(np.asarray(data, dtype=float)))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _ensure_1d_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError("Expected a 1-D feature vector")
    return arr


def _ensure_2d_array(x: np.ndarray | None) -> np.ndarray:
    if x is None:
        raise ValueError("training_data cannot be None when using this helper")
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected training_data to be a 2-D array")
    return arr


def _hash_array(arr: np.ndarray) -> int:
    scaled = np.round(arr * 1e6).astype(np.int64, copy=False)
    weights = np.arange(1, scaled.size + 1, dtype=np.int64)
    hashed = int(np.abs(np.dot(scaled, weights)) % (2**32))
    return hashed


__all__ = ["make_lime_tool"]
