import numpy as np
import pytest

from src.envs.explain_env import ExplainEnv

def test_env_step_runs_with_prob_reveal():
    env = ExplainEnv(n_features=10, seed=42, dataset_size=4)
    obs = env.reset()
    assert "p" in obs and obs["x"].shape == (10,)
    e = "features suggest class 1 with strong signal"
    out = env.step(e, tool_call_count=2)
    assert hasattr(out, "reward")
    assert np.isfinite(out.reward)
    assert out.done is True
    assert out.info["tool_call_count"] == 2
    assert "p" in out.observation and "y" not in out.observation


def test_env_label_reveal_hides_probabilities():
    env = ExplainEnv(n_features=10, seed=123, reveal="label", dataset_size=5)
    obs = env.reset()
    assert "y" in obs and "p" not in obs
    out = env.step("short explanation", tool_call_count=1)
    assert "y" in out.observation and "p" not in out.observation
    assert out.info["tool_call_count"] == 1


def test_tool_penalty_is_applied_to_reward():
    env = ExplainEnv(tool_penalty=0.25)
    env.reset()
    result = env.step("penalty test", tool_call_count=3)
    expected = result.info["base_reward"] - 0.75
    assert result.reward == pytest.approx(expected)


def test_seeded_dataset_is_reproducible():
    env_a = ExplainEnv(seed=777, dataset_size=3)
    env_b = ExplainEnv(seed=777, dataset_size=3)

    seq_a = [env_a.reset()["x"] for _ in range(5)]
    seq_b = [env_b.reset()["x"] for _ in range(5)]

    for xa, xb in zip(seq_a, seq_b):
        np.testing.assert_allclose(xa, xb)
