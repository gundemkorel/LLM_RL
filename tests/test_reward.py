import numpy as np
from src.utils.reward import kl_divergence, neg_kl_reward

def test_kl_simple():
    p = np.array([0.3, 0.7])
    q = np.array([0.3, 0.7])
    assert abs(kl_divergence(p,q)) < 1e-9
    assert abs(neg_kl_reward(p,q)) < 1e-9
