from __future__ import annotations
import numpy as np

def safe_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """KL(p || q) for discrete distributions p, q with shape (2,)."""
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    q = np.clip(q, eps, 1.0)
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))

def neg_kl_reward(p: np.ndarray, q: np.ndarray) -> float:
    return -kl_divergence(p, q)
