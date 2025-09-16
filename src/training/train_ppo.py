from __future__ import annotations
import numpy as np
from src.envs.explain_env import ExplainEnv
from src.agents.agent_langchain import ToolUsingPolicy
from src.tools.shap_tool import get_feature_importance
from src.tools.lime_tool import get_local_explanation
from src.tools.pdp_tool import get_partial_dependence

def run_random_policy(n_episodes: int = 5):
    env = ExplainEnv(n_features=10, seed=0)
    tools = {
        "get_feature_importance": get_feature_importance,
        "get_local_explanation": get_local_explanation,
        "get_partial_dependence": get_partial_dependence,
    }
    policy = ToolUsingPolicy(tools=tools, seed=0)

    for ep in range(n_episodes):
        obs = env.reset()
        x, p = obs["x"], obs["p"]
        explanation = policy.generate_explanation(x, p)
        step_out = env.step(explanation)
        print(f"Episode {ep}: R={step_out.reward:.4f} | q={step_out.observation['q']}\nE: {explanation}\n")

if __name__ == "__main__":
    run_random_policy()
