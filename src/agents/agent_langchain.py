from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import hints for type checking only
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.messages import AIMessage, HumanMessage


try:  # pragma: no cover - exercised via dedicated test
    from langchain_core.language_models import BaseLanguageModel as _BaseLanguageModel
    from langchain_core.messages import AIMessage as _AIMessage
    from langchain_core.messages import HumanMessage as _HumanMessage
except ImportError as exc:  # pragma: no cover - behaviour validated in tests
    _BaseLanguageModel = None  # type: ignore[assignment]
    _AIMessage = None  # type: ignore[assignment]
    _HumanMessage = None  # type: ignore[assignment]
    _LANGCHAIN_IMPORT_ERROR = exc
else:
    _LANGCHAIN_IMPORT_ERROR = None


@dataclass
class _AgentState:
    """Holds intermediate results collected by the agent."""

    feature_importance: Optional[List[Tuple[int, float]]] = None
    local_explanation: Any = None
    partial_dependence: Any = None
    top_feature_index: Optional[int] = None
    feature_names: Optional[Sequence[str]] = None


class ToolUsingPolicy:
    """LangChain-powered policy that orchestrates explanation tools via ReAct."""

    REQUIRED_TOOLS: Tuple[str, str, str] = (
        "get_feature_importance",
        "get_local_explanation",
        "get_partial_dependence",
    )

    def __init__(
        self,
        tools: Dict[str, Callable[..., Any]],
        llm: "BaseLanguageModel",
        *,
        max_tool_calls: int = 3,
    ) -> None:
        if _LANGCHAIN_IMPORT_ERROR is not None:
            raise ImportError(
                "LangChain is required for ToolUsingPolicy. Install `langchain` to enable this agent."
            ) from _LANGCHAIN_IMPORT_ERROR

        if _BaseLanguageModel is not None and not isinstance(llm, _BaseLanguageModel):
            raise TypeError("llm must inherit from langchain_core.language_models.BaseLanguageModel")

        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")

        missing = [name for name in self.REQUIRED_TOOLS if name not in tools]
        if missing:
            raise KeyError(f"Missing required tools: {', '.join(missing)}")

        self._tools = tools
        self._llm = llm
        self._max_tool_calls = max_tool_calls

    def generate_explanation(
        self,
        x: np.ndarray,
        p: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        *,
        include_tool_count: bool = False,
    ) -> str | Tuple[str, int]:
        """Run a ReAct-style loop to obtain a cited explanation for ``x``."""

        state = _AgentState()
        if feature_names is not None:
            state.feature_names = tuple(str(name) for name in feature_names)
        scratchpad: List[str] = []
        calls_used = 0
        x_list = _ensure_list(x)
        p_list = _ensure_list(p)

        # Allow an extra iteration for the final response.
        for iteration in range(self._max_tool_calls + 1):
            prompt = self._render_prompt(x_list, p_list, scratchpad, calls_used, iteration)
            messages = [_HumanMessage(content=prompt)] if _HumanMessage else [prompt]  # type: ignore[list-item]
            response = self._llm.invoke(messages)
            content = getattr(response, "content", response)
            step = self._parse_step(content)

            thought = step.get("thought", "")
            if thought:
                scratchpad.append(f"Thought {iteration + 1}: {thought}")

            if "action" in step and calls_used < self._max_tool_calls:
                action = step["action"]
                observation = self._execute_action(action, x_list, state)
                scratchpad.append(f"Action {calls_used + 1}: {self._format_action(action)}")
                scratchpad.append(f"Observation {calls_used + 1}: {observation}")
                calls_used += 1
                # Continue loop to obtain next instruction or final response.

            if "final" in step:
                break

            if calls_used >= self._max_tool_calls and "action" not in step:
                # Agent hit the tool budget without a final response; exit loop.
                break

        if state.feature_importance is None:
            raise RuntimeError("Agent did not gather feature importance before finishing.")

        if state.partial_dependence is None and state.top_feature_index is not None:
            # Ensure partial dependence is available as required.
            pd_result = self._tools["get_partial_dependence"](
                x=x_list,
                feature_i=int(state.top_feature_index),
            )
            state.partial_dependence = pd_result

        summary = self._build_summary(p_list, state)
        if include_tool_count:
            return summary, calls_used
        return summary

    def _render_prompt(
        self,
        x_list: List[Any],
        p_list: List[Any],
        scratchpad: Sequence[str],
        calls_used: int,
        iteration: int,
    ) -> str:
        scratchpad_text = "\n".join(scratchpad) if scratchpad else "(no tool calls yet)"
        remaining = self._max_tool_calls - calls_used
        return (
            "You are a LangChain ReAct agent that explains model predictions using tools.\n"
            "Available tools: get_feature_importance(x), get_local_explanation(x), get_partial_dependence(x, feature_i).\n"
            "Always reply with valid JSON containing a 'thought' key and either an 'action' object or a 'final' key.\n"
            "Actions must be JSON objects like {'tool': 'tool_name', 'args': {...}} using only JSON-compatible values.\n"
            "Call get_partial_dependence on the feature index with the largest absolute SHAP score.\n"
            f"Remaining tool calls: {remaining}.\n"
            f"Iteration: {iteration + 1}.\n"
            f"Feature vector x: {x_list}.\n"
            f"Prediction p: {p_list}.\n"
            "Scratchpad so far:\n"
            f"{scratchpad_text}\n"
            "Reply with JSON only."
        )

    def _parse_step(self, content: Any) -> Dict[str, Any]:
        if _AIMessage is not None and isinstance(content, _AIMessage):
            content = content.content
        elif _HumanMessage is not None and isinstance(content, _HumanMessage):
            content = content.content

        if not isinstance(content, str):
            raise ValueError(f"LLM response must be a JSON string, received {type(content)!r}")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:  # pragma: no cover - error surface guarded in tests
            raise ValueError("LLM response was not valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("LLM response must decode to a JSON object")
        if "thought" not in data:
            raise ValueError("LLM response is missing the required 'thought' field")
        return data

    def _execute_action(self, action: Dict[str, Any], x_list: List[Any], state: _AgentState) -> str:
        tool_name = action.get("tool")
        if not isinstance(tool_name, str):
            raise ValueError("Action must specify a tool name")

        if tool_name not in self._tools:
            raise KeyError(f"Unknown tool requested: {tool_name}")

        raw_args = action.get("args", {})
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, dict):
            raise ValueError("Action 'args' must be a JSON object")

        prepared_args = self._prepare_tool_args(tool_name, raw_args, x_list, state)
        result = self._tools[tool_name](**prepared_args)

        if tool_name == "get_feature_importance":
            state.feature_importance = self._normalize_feature_importance(result)
            if not state.feature_importance:
                raise ValueError("Feature importance tool returned no attributions")
            state.top_feature_index = int(state.feature_importance[0][0])
            return self._summarize_feature_importance(
                state.feature_importance, state.feature_names
            )

        if tool_name == "get_local_explanation":
            state.local_explanation = result
            return self._summarize_local_explanation(result)

        # Partial dependence must use the top feature index when available.
        if state.top_feature_index is None:
            raise RuntimeError("Partial dependence requested before feature importance was available")

        state.partial_dependence = result
        return self._summarize_partial_dependence(
            result, state.top_feature_index, state.feature_names
        )

    def _prepare_tool_args(
        self,
        tool_name: str,
        raw_args: Dict[str, Any],
        x_list: List[Any],
        state: _AgentState,
    ) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        for key, value in raw_args.items():
            if isinstance(value, np.ndarray):
                args[key] = value.tolist()
            else:
                args[key] = value

        args.setdefault("x", list(x_list))

        if tool_name == "get_partial_dependence":
            feature_idx: Optional[int]
            if state.top_feature_index is not None:
                feature_idx = int(state.top_feature_index)
            else:
                raw_idx = args.get("feature_i")
                feature_idx = None if raw_idx is None else int(raw_idx)

            if feature_idx is None:
                raise RuntimeError("Cannot compute partial dependence without a feature index")

            args["feature_i"] = feature_idx

        return args

    def _build_summary(self, p_list: List[Any], state: _AgentState) -> str:
        shap_values = state.feature_importance or []
        if not shap_values:
            raise RuntimeError("Missing feature importance values for summary")

        top_idx, top_value = shap_values[0]
        shap_label = self._feature_label(int(top_idx), state.feature_names, bracketed=True)
        shap_text = f"SHAP: {shap_label}={float(top_value):+0.2f}"

        local_text = self._format_local_explanation(state.local_explanation)
        pdp_text = self._format_partial_dependence(
            state.partial_dependence, int(top_idx), state.feature_names
        )

        citations = [shap_text]
        if local_text:
            citations.append(local_text)
        if pdp_text:
            citations.append(pdp_text)

        prob_text = ", ".join(f"{float(prob):0.2f}" for prob in _flatten_probs(p_list))
        explanation = f"{' ; '.join(citations)}. Predicted probs=[{prob_text}]."

        if len(explanation.split()) > 150:
            explanation = f"{' ; '.join(citations[:3])}."

        return explanation

    def _summarize_feature_importance(
        self, values: List[Tuple[int, float]], feature_names: Optional[Sequence[str]]
    ) -> str:
        highlights = ", ".join(
            f"{self._feature_label(int(idx), feature_names)}:{float(val):+0.2f}"
            for idx, val in values[:3]
        )
        return f"Top SHAP attributions -> {highlights}"

    def _summarize_local_explanation(self, result: Any) -> str:
        return f"Local explanation: {self._format_local_explanation(result)}"

    def _summarize_partial_dependence(
        self,
        result: Any,
        feature_idx: int,
        feature_names: Optional[Sequence[str]],
    ) -> str:
        detail = self._format_partial_dependence(result, feature_idx, feature_names)
        return f"Partial dependence insight: {detail}" if detail else "Partial dependence insight: (none)"

    def _normalize_feature_importance(
        self, raw: Any
    ) -> List[Tuple[int, float]]:
        if isinstance(raw, dict):
            items = list(raw.items())
        else:
            items = list(raw)

        normalized: List[Tuple[int, float]] = []
        for entry in items:
            if isinstance(entry, Iterable) and not isinstance(entry, (str, bytes)):
                entry_list = list(entry)
                if len(entry_list) < 2:
                    continue
                idx, value = entry_list[0], entry_list[1]
            else:
                raise ValueError("Feature importance entries must be iterable pairs")

            normalized.append((int(idx), float(value)))

        normalized.sort(key=lambda item: abs(item[1]), reverse=True)
        return normalized

    def _format_action(self, action: Dict[str, Any]) -> str:
        tool = action.get("tool", "?")
        args = action.get("args", {})
        return f"{tool}({args})"

    def _format_local_explanation(self, result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, dict):
            method = str(result.get("method") or result.get("name") or "local")
            metric_name, metric_value = self._pick_metric(result)
            if metric_name:
                return f"{method} {metric_name}={metric_value}"
            return f"{method} details={self._shorten(str(result))}"
        if isinstance(result, (list, tuple)):
            return self._shorten(", ".join(map(str, result)))
        return self._shorten(str(result))

    def _format_partial_dependence(
        self,
        result: Any,
        feature_idx: int,
        feature_names: Optional[Sequence[str]],
    ) -> str:
        if result is None:
            return ""
        detail: str
        if isinstance(result, dict):
            for key in ("trend", "summary", "shape", "direction"):
                if key in result:
                    detail = str(result[key])
                    break
            else:
                if "slope" in result:
                    value = result["slope"]
                    detail = f"slope={value:0.2f}" if isinstance(value, (int, float)) else str(value)
                else:
                    detail = self._shorten(str(result))
        elif isinstance(result, (list, tuple)):
            detail = self._shorten(", ".join(map(str, result[:5])))
        else:
            detail = self._shorten(str(result))

        label = self._feature_label(int(feature_idx), feature_names)
        return f"PDP@{label} {detail}"

    def _feature_label(
        self,
        index: int,
        feature_names: Optional[Sequence[str]],
        *,
        bracketed: bool = False,
        prefix: str = "f",
    ) -> str:
        if feature_names is not None and 0 <= index < len(feature_names):
            name = feature_names[index]
            if name:
                return str(name)
        if bracketed:
            return f"{prefix}[{index}]"
        return f"{prefix}{index}"

    def _pick_metric(self, data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        for key in ("score", "r2", "r_squared", "fidelity", "accuracy"):
            if key in data:
                value = data[key]
                if isinstance(value, (int, float)):
                    return key, f"{value:0.2f}"
                return key, self._shorten(str(value))
        return None, None

    def _shorten(self, text: str, max_len: int = 60) -> str:
        return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _flatten_probs(values: Sequence[Any]) -> List[float]:
    flat: List[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(float(v) for v in value)
        else:
            flat.append(float(value))
    return flat

