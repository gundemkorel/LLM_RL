from __future__ import annotations
import numpy as np

def get_feature_importance(x: np.ndarray, top_k: int = 3):
    """Stub: returns signed magnitudes that sum to zero-ish for interpretability demos."""
    rng = np.random.default_rng(abs(int(np.sum(x)*1e6)) % (2**32))
    vals = rng.normal(0, 1, size=x.shape[0])
    order = np.argsort(-np.abs(vals))[:top_k]
    return [(int(i), float(vals[i])) for i in order]
