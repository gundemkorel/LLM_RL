"""Run a single ExplainEnv episode on the breast cancer dataset.

This script stitches together the dataset abstraction, model wrappers,
explanation tools, and the LangChain-based policy to produce a cited
explanation for a RandomForestClassifier trained on the classic
``sklearn`` breast cancer dataset.

The demo intentionally keeps the language model deterministic by using a
scripted ``BaseLanguageModel`` subclass.  Install ``langchain-core`` to
run the demo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from src.agents.agent_langchain import ToolUsingPolicy
from src.data.datasets import TabularDataset
from src.envs.explain_env import ExplainEnv
from src.models.blackbox import BlackBoxModel
from src.tools.lime_tool import make_lime_tool
from src.tools.pdp_tool import make_pdp_tool
from src.tools.shap_tool import make_shap_tool

try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.messages import AIMessage
except ImportError as exc:  # pragma: no cover - demo requires langchain_core at runtime
    raise SystemExit(
        "LangChain Core is required for this demo. Install `langchain-core` to continue."
    ) from exc


@dataclass
class _ScriptedResponse:
    """Container describing the JSON instructions returned by the scripted LLM."""

    payload: Dict[str, object]


class ScriptedLLM(BaseLanguageModel):
    """A deterministic LLM that drives the ReAct loop for the demo."""

    def __init__(self, responses: Sequence[Dict[str, object]]) -> None:
        self._responses: List[_ScriptedResponse] = [
            _ScriptedResponse(payload=response) for response in responses
        ]
        self._cursor = 0

    def invoke(self, _messages: Iterable[object], **_kwargs: object) -> AIMessage:
        if self._cursor >= len(self._responses):
            payload = self._responses[-1].payload
        else:
            payload = self._responses[self._cursor].payload
            self._cursor += 1
        return AIMessage(content=json.dumps(payload))


SCRIPTED_PLAN: Sequence[Dict[str, object]] = (
    {"thought": "Need SHAP attributions", "action": {"tool": "get_feature_importance", "args": {}}},
    {"thought": "Check local explanation", "action": {"tool": "get_local_explanation", "args": {}}},
    {"thought": "Inspect partial dependence", "action": {"tool": "get_partial_dependence", "args": {}}},
    {"thought": "Summarise findings", "final": {}},
)


def build_dataset() -> tuple[TabularDataset, TabularDataset, TabularDataset]:
    """Load and split the breast cancer dataset into train/val/test sets."""

    features, target = load_breast_cancer(return_X_y=True, as_frame=True)
    df = features.copy()
    label_col = "target"
    df[label_col] = target
    dataset = TabularDataset.from_dataframe(df, label_col=label_col)
    return dataset.split(train=0.7, val=0.15, test=0.15, seed=0)


def build_blackbox(train: TabularDataset) -> BlackBoxModel:
    """Train a RandomForestClassifier and wrap it as a ``BlackBoxModel``."""

    estimator = RandomForestClassifier(n_estimators=128, random_state=0)
    estimator.fit(train.X, train.y)
    return BlackBoxModel.from_sklearn(estimator, n_features=train.n_features)


def build_feature_grid(dataset: TabularDataset, points: int = 15) -> Dict[int, np.ndarray]:
    """Construct evenly spaced feature grids for PDP queries."""

    grids: Dict[int, np.ndarray] = {}
    X = dataset.X
    for idx in range(dataset.n_features):
        grids[idx] = np.linspace(X[:, idx].min(), X[:, idx].max(), num=points, dtype=float)
    return grids


def build_tools(
    blackbox: BlackBoxModel,
    train: TabularDataset,
) -> Dict[str, object]:
    """Bind SHAP, LIME, and PDP tools to the provided black-box model."""

    background_size = min(len(train), 128)
    background = train.sample(n=background_size, seed=1)
    feature_grid = build_feature_grid(train)

    shap_tool = make_shap_tool(blackbox.predict_proba, background, model_kind="tree")
    lime_tool = make_lime_tool(
        blackbox.predict_proba,
        train.feature_names,
        training_data=background,
        n_samples=256,
        random_state=42,
    )
    pdp_tool = make_pdp_tool(
        blackbox.predict_proba,
        feature_grid,
        background=background,
    )

    return {
        "get_feature_importance": shap_tool,
        "get_local_explanation": lime_tool,
        "get_partial_dependence": pdp_tool,
    }


def main() -> None:
    train, val, _test = build_dataset()
    blackbox = build_blackbox(train)
    tools = build_tools(blackbox, train)

    env = ExplainEnv(
        dataset=val,
        blackbox=blackbox,
        reveal="probs",
        tool_penalty=0.01,
        seed=123,
    )

    llm = ScriptedLLM(SCRIPTED_PLAN)
    policy = ToolUsingPolicy(tools=tools, llm=llm, max_tool_calls=3)

    observation = env.reset()
    x = np.asarray(observation["x"], dtype=float)
    p = np.asarray(observation["p"], dtype=float)

    explanation, call_count = policy.generate_explanation(
        x,
        p,
        feature_names=env.feature_names,
        include_tool_count=True,
    )

    print("Observation x shape:", x.shape)
    print("Predicted probabilities:", p)
    print("Generated explanation:\n", explanation)
    print("Tool calls used:", call_count)

    step_output = env.step(explanation, tool_call_count=call_count)
    print("Reward:", float(step_output.reward))
    print("q (g(E) probabilities):", step_output.observation["q"])


if __name__ == "__main__":
    main()
