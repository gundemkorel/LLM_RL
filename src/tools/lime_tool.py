from __future__ import annotations
import numpy as np

def get_local_explanation(x: np.ndarray):
    """Stub: returns local surrogate coefficients and a fake R^2."""
    rng = np.random.default_rng(abs(int(np.sum(x)*1e6)) % (2**32))
    w = rng.normal(0, 0.5, size=x.shape[0])
    r2 = float(np.clip(rng.normal(0.7, 0.05), 0, 1))
    return {"coefficients": w.tolist(), "r2": r2}
