from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import warnings


PredictProbaFn = Callable[[np.ndarray], Sequence[Sequence[float]]]


def make_shap_tool(
    predict_proba: PredictProbaFn,
    background_X: np.ndarray,
    *,
    model_kind: str = "auto",
    max_background: int = 50,
) -> Callable[[np.ndarray, int], List[Tuple[int, float]]]:
    """Create a SHAP-based feature-importance tool bound to ``predict_proba``.

    Parameters
    ----------
    predict_proba:
        Callable mapping a 2-D ``numpy.ndarray`` of shape ``(n_samples, n_features)`` to
        probabilities. Typically ``model.predict_proba``.
    background_X:
        Background data used to initialise the explainer or to compute fallbacks.
    model_kind:
        Optional hint describing the underlying estimator. One of
        ``{"auto", "tree", "linear", "kernel"}``. When ``"auto"`` the wrapper will
        attempt to infer the best SHAP explainer based on the bound estimator.
    max_background:
        Maximum number of background rows kept for explainers that do not scale
        well with dataset size.
    """

    background = _ensure_2d_array(background_X)
    if background.shape[0] == 0:
        raise ValueError("background_X must contain at least one row")

    if max_background < 1:
        raise ValueError("max_background must be at least 1")

    normalized_kind = (model_kind or "auto").lower()
    if normalized_kind not in {"auto", "tree", "linear", "kernel"}:
        raise ValueError("model_kind must be one of 'auto', 'tree', 'linear', or 'kernel'")

    background_subset = background[:max_background]

    try:  # pragma: no cover - optional dependency
        import shap as shap_module  # type: ignore
    except Exception:  # pragma: no cover - deterministic fallback exercised in tests
        shap_module = None

    model = getattr(predict_proba, "__self__", None)
    inferred_kind = normalized_kind if normalized_kind != "auto" else _infer_model_kind(model)

    shap_values_fn = _select_shap_implementation(
        predict_proba,
        background_subset,
        shap_module,
        inferred_kind,
    )

    def get_feature_importance(x: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        sample = _ensure_2d_array(x)
        if sample.shape[1] != background.shape[1]:
            raise ValueError("x dimensionality does not match background data")

        shap_values = shap_values_fn(sample)
        shap_values = np.asarray(shap_values, dtype=float)
        if shap_values.ndim != 2 or shap_values.shape[0] != sample.shape[0]:
            raise ValueError("Unexpected SHAP output shape")

        values = shap_values[0]
        order = np.argsort(-np.abs(values))[: min(top_k, values.shape[0])]
        return [(int(i), float(values[i])) for i in order]

    return get_feature_importance


def _select_shap_implementation(
    predict_proba: PredictProbaFn,
    background: np.ndarray,
    shap_module: object | None,
    model_kind: str,
) -> Callable[[np.ndarray], np.ndarray]:
    if shap_module is None:
        warnings.warn(
            "SHAP is not installed; falling back to a deterministic approximation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return lambda x: _approximate_shap(predict_proba, background, x)

    model = getattr(predict_proba, "__self__", None)

    if model_kind == "tree" and hasattr(shap_module, "TreeExplainer"):
        try:  # pragma: no cover - relies on optional dependency
            explainer = shap_module.TreeExplainer(model, data=background)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - fall back to approximation when SHAP fails
            warnings.warn(
                f"Tree SHAP explainer failed ({exc}); using deterministic fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            def tree_values(x: np.ndarray) -> np.ndarray:
                raw = explainer.shap_values(x)
                return _extract_shap_values(raw, predict_proba, x)

            return tree_values

    if model_kind == "linear" and hasattr(shap_module, "LinearExplainer"):
        try:  # pragma: no cover - relies on optional dependency
            explainer = shap_module.LinearExplainer(model, background, link="logit")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - older SHAP versions fall back to defaults
            try:
                explainer = shap_module.LinearExplainer(model, background)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - fall back to approximation
                warnings.warn(
                    f"Linear SHAP explainer failed ({exc}); using deterministic fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                def linear_values(x: np.ndarray) -> np.ndarray:
                    raw = explainer.shap_values(x)
                    return _extract_shap_values(raw, predict_proba, x)

                return linear_values
        else:
            def linear_values(x: np.ndarray) -> np.ndarray:
                raw = explainer.shap_values(x)
                return _extract_shap_values(raw, predict_proba, x)

            return linear_values

    if hasattr(shap_module, "KernelExplainer"):
        kernel_background = background[: min(20, background.shape[0])]
        try:  # pragma: no cover - relies on optional dependency
            kernel_explainer = shap_module.KernelExplainer(  # type: ignore[attr-defined]
                lambda data: _call_predict(predict_proba, data),
                kernel_background,
            )
        except Exception as exc:  # pragma: no cover - fall back to deterministic computation
            warnings.warn(
                f"Kernel SHAP explainer failed ({exc}); using deterministic fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            def kernel_values(x: np.ndarray) -> np.ndarray:
                raw = kernel_explainer.shap_values(x)
                return _extract_shap_values(raw, predict_proba, x)

            return kernel_values

    warnings.warn(
        "Unable to build a SHAP explainer; falling back to a deterministic approximation.",
        RuntimeWarning,
        stacklevel=2,
    )
    return lambda x: _approximate_shap(predict_proba, background, x)


def _approximate_shap(
    predict_proba: PredictProbaFn,
    background: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    sample = _ensure_2d_array(x)
    preds = _call_predict(predict_proba, sample)
    baseline_preds = _call_predict(predict_proba, background)

    target_class = int(np.argmax(preds[0]))
    baseline_prob = float(np.mean(baseline_preds[:, target_class]))
    target_prob = float(preds[0, target_class])

    diffs = np.zeros(sample.shape[1], dtype=float)
    for idx in range(sample.shape[1]):
        substituted = np.array(background, copy=True)
        substituted[:, idx] = sample[0, idx]
        substituted_preds = _call_predict(predict_proba, substituted)
        diffs[idx] = float(np.mean(substituted_preds[:, target_class]) - baseline_prob)

    total = diffs.sum()
    desired_total = target_prob - baseline_prob
    if np.isfinite(total) and abs(total) > 1e-9:
        diffs *= desired_total / total
    else:
        diffs[:] = desired_total / sample.shape[1]

    return diffs.reshape(1, -1)


def _extract_shap_values(raw_values: Iterable, predict_proba: PredictProbaFn, x: np.ndarray) -> np.ndarray:
    values = raw_values

    if hasattr(values, "values"):
        values = getattr(values, "values")  # shap.Explanation

    if isinstance(values, list):
        arrays = [np.asarray(v, dtype=float) for v in values]
        probs = _call_predict(predict_proba, x)
        target_class = int(np.argmax(probs[0]))
        selected = arrays[target_class]
        if selected.ndim == 1:
            return selected.reshape(1, -1)
        return np.asarray(selected, dtype=float)

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        probs = _call_predict(predict_proba, x)
        target_class = int(np.argmax(probs[0]))
        return array[:, target_class, :]

    raise ValueError("Unable to interpret SHAP output shape")


def _call_predict(predict_proba: PredictProbaFn, data: np.ndarray) -> np.ndarray:
    arr = np.asarray(predict_proba(np.asarray(data, dtype=float)))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _infer_model_kind(model: object | None) -> str:
    if model is None:
        return "unknown"
    module = getattr(model.__class__, "__module__", "").lower()
    name = getattr(model.__class__, "__name__", "").lower()

    tree_tokens = ("tree", "forest", "boost", "gbm", "gb", "catboost", "xgboost")
    linear_tokens = ("linear", "logistic", "ridge", "lasso", "sgd")

    if any(token in module or token in name for token in tree_tokens):
        return "tree"
    if any(token in module or token in name for token in linear_tokens):
        return "linear"
    return "unknown"


def _ensure_2d_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected a 1-D or 2-D array")
    return arr


__all__ = ["make_shap_tool"]
