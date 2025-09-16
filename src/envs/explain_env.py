from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any

from src.models.blackbox import BlackBoxModel
from src.models.g_explainer import ExplanationInducedModel
from src.utils.reward import neg_kl_reward

@dataclass
class StepOutput:
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class ExplainEnv:
    """A minimal environment:
    - reset() returns an x and f(x)
    - step(explanation) computes g(E) and returns reward = -KL(f(x)||g(E)).
    """
    def __init__(self, n_features: int = 10, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.f = BlackBoxModel(n_features=n_features, random_state=seed)
        self.g = ExplanationInducedModel()
        self.g.fit_dummy()
        self.current_x = None
        self.current_p = None
        self.t = 0

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self.current_x = self.rng.normal(0, 1, size=self.f.n_features)
        self.current_p = self.f.predict_proba(self.current_x)
        return {"x": self.current_x, "p": self.current_p}

    def step(self, explanation: str) -> StepOutput:
        q = self.g.predict_proba([explanation])[0]  # [p0, p1]
        reward = neg_kl_reward(self.current_p, q)
        self.t += 1
        done = True  # single-step episodes (x -> E)
        obs = {"x": self.current_x, "p": self.current_p, "explanation": explanation, "q": q}
        info = {}
        return StepOutput(observation=obs, reward=reward, done=done, info=info)
