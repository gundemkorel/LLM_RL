from __future__ import annotations
import numpy as np

def get_partial_dependence(x: np.ndarray, feature_i: int, grid_points: int = 10):
    """Stub PDP: returns a monotone curve if feature value is large, else flat-ish."""
    xi = float(x[feature_i])
    grid = np.linspace(xi - 1.0, xi + 1.0, grid_points)
    curve = 1 / (1 + np.exp(-grid))  # logistic-shaped
    return {"feature": int(feature_i), "grid": grid.tolist(), "pdp": curve.tolist()}
