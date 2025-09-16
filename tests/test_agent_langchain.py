import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import-time path setup
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_langchain_stubs(monkeypatch):
    langchain_core = types.ModuleType("langchain_core")

    language_models = types.ModuleType("langchain_core.language_models")

    class BaseLanguageModel:  # pragma: no cover - simple stub
        def invoke(self, *_args, **_kwargs):
            raise NotImplementedError

    language_models.BaseLanguageModel = BaseLanguageModel

    messages = types.ModuleType("langchain_core.messages")

    class _BaseMessage:  # pragma: no cover - simple stub
        def __init__(self, content: str):
            self.content = content

    class AIMessage(_BaseMessage):
        pass

    class HumanMessage(_BaseMessage):
        pass

    messages.AIMessage = AIMessage
    messages.HumanMessage = HumanMessage

    langchain_core.language_models = language_models
    langchain_core.messages = messages

    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.language_models", language_models)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages)

    return language_models, messages


def _module_path() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "agents" / "agent_langchain.py"


def test_policy_requires_langchain(monkeypatch):
    # Ensure LangChain is absent.
    for key in list(sys.modules):
        if key.startswith("langchain_core"):
            monkeypatch.delitem(sys.modules, key, raising=False)

    module_name = "agent_langchain_no_dependency"
    spec = importlib.util.spec_from_file_location(module_name, _module_path())
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    tools = {
        "get_feature_importance": lambda **_: [],
        "get_local_explanation": lambda **_: {},
        "get_partial_dependence": lambda **_: {},
    }

    with pytest.raises(ImportError):
        module.ToolUsingPolicy(tools=tools, llm=object())


def test_policy_generates_cited_summary(monkeypatch):
    language_models, messages = _install_langchain_stubs(monkeypatch)

    module = importlib.import_module("src.agents.agent_langchain")
    importlib.reload(module)

    class DummyLLM(language_models.BaseLanguageModel):
        def __init__(self, scripted_steps):
            self._scripted_steps = scripted_steps
            self._calls = 0

        def invoke(self, _messages, **_kwargs):
            if self._calls >= len(self._scripted_steps):
                raise RuntimeError("LLM received more invocations than scripted")
            payload = json.dumps(self._scripted_steps[self._calls])
            self._calls += 1
            return messages.AIMessage(payload)

    shap_calls = []
    local_calls = []
    pdp_calls = []

    def feature_tool(x):
        shap_calls.append(x)
        return [(0, 0.15), (2, -0.62), (1, 0.2)]

    def local_tool(x):
        local_calls.append(x)
        return {"method": "LIME", "fidelity": 0.72}

    def pdp_tool(x, feature_i):
        pdp_calls.append((x, feature_i))
        return {"trend": "monotone↑"}

    scripted = [
        {"thought": "Need feature attribution", "action": {"tool": "get_feature_importance", "args": {}}},
        {"thought": "Check local explanation", "action": {"tool": "get_local_explanation", "args": {}}},
        {"thought": "Study shape", "action": {"tool": "get_partial_dependence", "args": {"feature_i": 0}}},
        {"thought": "Ready to summarise", "final": {}},
    ]

    llm = DummyLLM(scripted)
    tools = {
        "get_feature_importance": feature_tool,
        "get_local_explanation": local_tool,
        "get_partial_dependence": pdp_tool,
    }

    policy = module.ToolUsingPolicy(tools=tools, llm=llm, max_tool_calls=3)

    x = np.array([0.1, -0.4, 1.2])
    p = np.array([0.2, 0.8])
    explanation = policy.generate_explanation(x, p)

    assert shap_calls == [x.tolist()]
    assert local_calls == [x.tolist()]
    assert pdp_calls == [(x.tolist(), 2)]  # top feature is index 2 from SHAP values

    assert "SHAP: f[2]" in explanation
    assert "LIME fidelity=0.72" in explanation
    assert "PDP@f2 monotone↑" in explanation
    assert len(explanation.split()) <= 150

