from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import warnings

PredictProbaFn = Callable[[np.ndarray], np.ndarray]


def make_pdp_tool(
    predict_proba: PredictProbaFn,
    feature_grid: Dict[int, np.ndarray],
    *,
    background: np.ndarray | None = None,
) -> Callable[[np.ndarray, int, int | None], Dict[str, object]]:
    """Create a partial dependence (or ICE) tool bound to ``predict_proba``.

    Parameters
    ----------
    predict_proba:
        Callable returning probabilities for a batch of inputs.
    feature_grid:
        Mapping from feature indices to 1-D arrays defining the evaluation grid.
    background:
        Optional background dataset used when averaging PDP curves. When ``None``
        the PDP defaults to the ICE curve for the queried instance.
    """

    try:  # pragma: no cover - optional dependency
        from sklearn.inspection import partial_dependence as sklearn_partial_dependence  # type: ignore
    except Exception:  # pragma: no cover - deterministic fallback exercised in tests
        sklearn_partial_dependence = None

    processed_grid = {_ensure_int(idx): _ensure_grid(values) for idx, values in feature_grid.items()}
    background_array = _ensure_2d(background) if background is not None else None

    sklearn_estimator: object | None = None
    if sklearn_partial_dependence is not None and background_array is not None:
        try:  # pragma: no cover - optional dependency
            from sklearn.base import BaseEstimator, ClassifierMixin  # type: ignore
        except Exception:  # pragma: no cover - fallback when sklearn is partially available
            BaseEstimator = object  # type: ignore[misc]
            ClassifierMixin = object  # type: ignore[misc]

        classes = np.arange(_call_predict(predict_proba, background_array[:1]).shape[1], dtype=int)

        class _PredictProbaEstimator(BaseEstimator, ClassifierMixin):  # type: ignore[misc,valid-type]
            _estimator_type = "classifier"

            def __init__(self, predict_fn: PredictProbaFn, class_labels: np.ndarray) -> None:
                self._predict_fn = predict_fn
                self._classes = class_labels
                self._is_fitted = False

            def fit(self, X: np.ndarray, y: np.ndarray | None = None):  # noqa: D401
                array = np.asarray(X, dtype=float)
                self.n_features_in_ = array.shape[1]
                self.classes_ = self._classes
                self._is_fitted = True
                return self

            def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - thin wrapper
                if not self._is_fitted:
                    raise RuntimeError("Estimator must be fitted before calling predict_proba")
                return np.asarray(self._predict_fn(X), dtype=float)

            def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - thin wrapper
                probs = self.predict_proba(X)
                return np.asarray(np.argmax(probs, axis=1), dtype=int)

        sklearn_estimator = _PredictProbaEstimator(predict_proba, classes)
        sklearn_estimator.fit(background_array, None)
    else:
        message = (
            "scikit-learn's partial_dependence is unavailable; using deterministic PDP." if sklearn_partial_dependence is None
            else "Background data missing for PDP; using deterministic computation."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def get_partial_dependence(
        x: np.ndarray,
        feature_i: int,
        grid_points: int | None = None,
    ) -> Dict[str, object]:
        sample = _ensure_1d(x)
        feature_idx = _ensure_int(feature_i)
        if feature_idx not in processed_grid:
            raise KeyError(f"No grid specified for feature index {feature_idx}")

        grid_values = processed_grid[feature_idx]
        if grid_points is not None:
            if grid_points < 2:
                raise ValueError("grid_points must be at least 2")
            grid_values = np.linspace(grid_values.min(), grid_values.max(), int(grid_points), dtype=float)

        base_probs = _call_predict(predict_proba, sample.reshape(1, -1))
        target_class = int(np.argmax(base_probs[0]))
        n_classes = base_probs.shape[1]

        pdp_curve = None
        result_grid = grid_values
        if sklearn_estimator is not None and sklearn_partial_dependence is not None:
            try:  # pragma: no cover - relies on optional dependency
                pd_result = sklearn_partial_dependence(
                    sklearn_estimator,
                    X=background_array,
                    features=[feature_idx],
                    kind="average",
                    custom_values={feature_idx: grid_values},
                )
                result_grid = np.asarray(pd_result["grid_values"][0], dtype=float)
                averaged = np.asarray(pd_result["average"], dtype=float)
                if averaged.ndim == 1:
                    averaged = averaged.reshape(1, -1)
                try:
                    pdp_curve = _select_average_for_class(averaged, target_class, n_classes)
                except ValueError as exc:
                    warnings.warn(
                        f"Unable to align sklearn PDP output with class {target_class}: {exc}. Using deterministic computation.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    pdp_curve = None
            except Exception as exc:  # pragma: no cover - deterministic fallback when sklearn fails
                warnings.warn(
                    f"sklearn.partial_dependence failed ({exc}); using deterministic PDP.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                pdp_curve = None

        if pdp_curve is None:
            pdp_curve = _compute_manual_pdp(
                predict_proba,
                background_array,
                sample,
                feature_idx,
                grid_values,
                target_class,
            )
            result_grid = grid_values

        ice_points = np.repeat(sample.reshape(1, -1), result_grid.size, axis=0)
        ice_points[:, feature_idx] = result_grid
        ice_curve = _call_predict(predict_proba, ice_points)[:, target_class]

        return {
            "method": "PDP",
            "feature": feature_idx,
            "target_class": target_class,
            "grid": result_grid.tolist(),
            "pdp": np.asarray(pdp_curve, dtype=float).tolist(),
            "ice": np.asarray(ice_curve, dtype=float).tolist(),
        }

    return get_partial_dependence


def _compute_manual_pdp(
    predict_proba: PredictProbaFn,
    background: np.ndarray | None,
    sample: np.ndarray,
    feature_idx: int,
    grid_values: np.ndarray,
    target_class: int,
) -> np.ndarray:
    base = background if background is not None else sample.reshape(1, -1)
    pdp_curve = np.zeros(grid_values.size, dtype=float)
    for position, grid_value in enumerate(grid_values):
        evaluation = np.array(base, copy=True)
        evaluation[:, feature_idx] = grid_value
        probs = _call_predict(predict_proba, evaluation)[:, target_class]
        pdp_curve[position] = float(np.mean(probs))
    return pdp_curve


def _select_average_for_class(averaged: np.ndarray, target_class: int, n_classes: int) -> np.ndarray:
    if averaged.ndim != 2:
        raise ValueError("averaged PDP output must be 2-D")
    if averaged.shape[0] == n_classes:
        return averaged[target_class]
    if n_classes == 2 and averaged.shape[0] == 1:
        values = averaged[0]
        return values if target_class == 1 else 1.0 - values
    if averaged.shape[0] == 1:
        return averaged[0]
    raise ValueError("unexpected averaged PDP shape")


def _call_predict(predict_proba: PredictProbaFn, data: np.ndarray) -> np.ndarray:
    arr = np.asarray(predict_proba(np.asarray(data, dtype=float)))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _ensure_grid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Grid values must be a 1-D array")
    if arr.size == 0:
        raise ValueError("Grid for partial dependence cannot be empty")
    return arr


def _ensure_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr.reshape(-1)
    raise ValueError("x must be a 1-D feature vector")


def _ensure_2d(x: np.ndarray | None) -> np.ndarray:
    if x is None:
        raise ValueError("background cannot be None when using this helper")
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("background must be a 2-D array")
    return arr


def _ensure_int(value: int) -> int:
    if isinstance(value, (np.integer,)):
        return int(value)
    if not isinstance(value, (int,)):
        raise TypeError("Feature indices must be integers")
    return int(value)


__all__ = ["make_pdp_tool"]
