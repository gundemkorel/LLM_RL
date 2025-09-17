from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


from src.data.datasets import TabularDataset
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

        *,
        dataset: TabularDataset,
        blackbox: BlackBoxModel,
        reveal: str = "probs",
        tool_penalty: float = 0.0,
        seed: int = 0,
    ):
        if reveal not in {"probs", "label"}:
            raise ValueError(f"Unsupported reveal mode: {reveal}")


        self.dataset = dataset
        self.blackbox = blackbox

        self.g = ExplanationInducedModel()
        self.g.fit_dummy()

        self.reveal = reveal
        self.tool_penalty = float(tool_penalty)

oe816        self._next_seed = int(seed)

        self.current_x = None
        self.current_p = None
        self.t = 0


    def _sample_from_dataset(self) -> np.ndarray:
        sample_seed = self._next_seed
        self._next_seed += 1
        batch = self.dataset.sample(n=1, seed=sample_seed)
        if batch.shape[0] != 1:
            raise ValueError("Dataset sampling must return a single row for n=1")
        return np.array(batch[0], copy=True)


    def _current_observation(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {"x": self.current_x}
        if self.reveal == "probs":
            obs["p"] = self.current_p
        else:
            obs["y"] = int(self.current_p[1] > 0.5)
        return obs

    def reset(self) -> Dict[str, Any]:
        self.t = 0

        self.current_x = self._sample_from_dataset()
        self.current_p = self.blackbox.predict_proba(self.current_x)

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

    @property
    def feature_names(self) -> list[str] | None:
        return self.dataset.feature_names
