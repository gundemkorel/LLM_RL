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
    - reset() returns an x and either f(x) or its label depending on the reveal mode
    - step(explanation, tool_call_count) computes g(E) and returns reward = -KL(f(x)||g(E))
      minus a penalty for tool usage.
    """

    def __init__(
        self,
        n_features: int = 10,
        seed: int = 0,
        *,
        reveal: str = "probs",
        tool_penalty: float = 0.0,
        dataset_size: int = 128,
    ):
        if reveal not in {"probs", "label"}:
            raise ValueError(f"Unsupported reveal mode: {reveal}")
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")

        self.rng = np.random.default_rng(seed)
        self.f = BlackBoxModel(n_features=n_features, random_state=seed)
        self.g = ExplanationInducedModel()
        self.g.fit_dummy()

        self.reveal = reveal
        self.tool_penalty = float(tool_penalty)
        self.dataset_size = int(dataset_size)

        self.dataset_x = self.rng.normal(0, 1, size=(self.dataset_size, self.f.n_features))

        self.current_x = None
        self.current_p = None
        self.t = 0

    def _sample_from_pool(self) -> np.ndarray:
        idx = int(self.rng.integers(0, self.dataset_size))
        return np.array(self.dataset_x[idx], copy=True)

    def _current_observation(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {"x": self.current_x}
        if self.reveal == "probs":
            obs["p"] = self.current_p
        else:
            obs["y"] = int(self.current_p[1] > 0.5)
        return obs

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self.current_x = self._sample_from_pool()
        self.current_p = self.f.predict_proba(self.current_x)
        return self._current_observation()

    def step(self, explanation: str, tool_call_count: int = 0) -> StepOutput:
        q = self.g.predict_proba([explanation])[0]  # [p0, p1]
        base_reward = neg_kl_reward(self.current_p, q)
        reward = base_reward - self.tool_penalty * tool_call_count
        self.t += 1
        done = True  # single-step episodes (x -> E)
        obs = self._current_observation()
        obs.update({"explanation": explanation, "q": q})
        info: Dict[str, Any] = {
            "tool_call_count": tool_call_count,
            "base_reward": base_reward,
        }
        return StepOutput(observation=obs, reward=reward, done=done, info=info)
