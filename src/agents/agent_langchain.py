from __future__ import annotations
from typing import Callable, Dict, Any
import numpy as np

# This stub imitates a tool-using policy. Replace with LangChain or your LLM of choice.
class ToolUsingPolicy:
    def __init__(self, tools: Dict[str, Callable], seed: int = 0):
        self.tools = tools
        self.rng = np.random.default_rng(seed)

    def generate_explanation(self, x: np.ndarray, p: Any) -> str:
        # Heuristic: call feature importance, then craft a sentence.
        feats = self.tools["get_feature_importance"](x)
        parts = []
        for idx, val in feats:
            direction = "increases" if val > 0 else "decreases"
            parts.append(f"feature[{idx}] {direction} the odds of class 1 by ~{abs(val):.2f}")
        template = (
            "Based on local attributions, {details}. Overall, the model probability for class 1 is {p1:.2f}."
        )
        return template.format(details="; ".join(parts), p1=float(p[1]) if hasattr(p, '__getitem__') else float(p))
