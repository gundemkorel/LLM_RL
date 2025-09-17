from __future__ import annotations

from typing import Callable, Dict

import numpy as np

PredictProbaFn = Callable[[np.ndarray], np.ndarray]


def make_pdp_tool(
    predict_proba: PredictProbaFn,
    feature_grid: Dict[int, np.ndarray],
    *,
    background: np.ndarray | None = None,
) -> Callable[[np.ndarray, int], Dict[str, object]]:
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

    processed_grid = {_ensure_int(idx): _ensure_grid(values) for idx, values in feature_grid.items()}
    background_array = _ensure_2d(background) if background is not None else None

    def get_partial_dependence(x: np.ndarray, feature_i: int) -> Dict[str, object]:
        sample = _ensure_1d(x)
        feature_idx = _ensure_int(feature_i)
        if feature_idx not in processed_grid:
            raise KeyError(f"No grid specified for feature index {feature_idx}")

        grid_values = processed_grid[feature_idx]
        base_probs = _call_predict(predict_proba, sample.reshape(1, -1))
        target_class = int(np.argmax(base_probs[0]))

        pdp_base = background_array if background_array is not None else sample.reshape(1, -1)

        pdp_curve = np.zeros(grid_values.size, dtype=float)
        for position, grid_value in enumerate(grid_values):
            evaluation = np.array(pdp_base, copy=True)
            evaluation[:, feature_idx] = grid_value
            probs = _call_predict(predict_proba, evaluation)[:, target_class]
            pdp_curve[position] = float(np.mean(probs))

        ice_points = np.repeat(sample.reshape(1, -1), grid_values.size, axis=0)
        ice_points[:, feature_idx] = grid_values
        ice_curve = _call_predict(predict_proba, ice_points)[:, target_class]

        return {
            "method": "PDP",
            "feature": feature_idx,
            "target_class": target_class,
            "grid": grid_values.tolist(),
            "pdp": pdp_curve.tolist(),
            "ice": ice_curve.tolist(),
        }

    return get_partial_dependence


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
