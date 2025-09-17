import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.data.datasets import TabularDataset

from src.envs.explain_env import ExplainEnv
from src.models.blackbox import BlackBoxModel


def _make_dataset_and_model(
    *,
    seed: int,
    n_features: int = 10,
    dataset_size: int = 64,
) -> tuple[TabularDataset, BlackBoxModel]:
    X, y = make_classification(
        n_samples=dataset_size,
        n_features=n_features,
        n_informative=min(6, n_features),
        n_redundant=min(2, max(n_features - 6, 0)),
        n_classes=2,
        random_state=seed,
    )
    dataset = TabularDataset.from_arrays(X, y, feature_names=None)
    blackbox = BlackBoxModel.from_training(
        dataset=dataset,
        estimator_factory=lambda: LogisticRegression(max_iter=1000, random_state=seed),
    )
    return dataset, blackbox


def test_env_step_runs_with_prob_reveal():

    dataset, blackbox = _make_dataset_and_model(seed=42)
    env = ExplainEnv(dataset=dataset, blackbox=blackbox, reveal="probs", seed=42)
    obs = env.reset()
    assert "p" in obs and obs["x"].shape == (dataset.n_features,)

    e = "features suggest class 1 with strong signal"
    out = env.step(e, tool_call_count=2)
    assert hasattr(out, "reward")
    assert np.isfinite(out.reward)
    assert out.done is True
    assert out.info["tool_call_count"] == 2
    assert "p" in out.observation and "y" not in out.observation


def test_env_label_reveal_hides_probabilities():

    dataset, blackbox = _make_dataset_and_model(seed=123)
    env = ExplainEnv(dataset=dataset, blackbox=blackbox, reveal="label", seed=123)

    obs = env.reset()
    assert "y" in obs and "p" not in obs
    out = env.step("short explanation", tool_call_count=1)
    assert "y" in out.observation and "p" not in out.observation
    assert out.info["tool_call_count"] == 1


def test_tool_penalty_is_applied_to_reward():

    dataset, blackbox = _make_dataset_and_model(seed=2023)
    env = ExplainEnv(dataset=dataset, blackbox=blackbox, tool_penalty=0.25, seed=2023)

    env.reset()
    result = env.step("penalty test", tool_call_count=3)
    expected = result.info["base_reward"] - 0.75
    assert result.reward == pytest.approx(expected)


def test_seeded_dataset_is_reproducible():

    dataset, blackbox = _make_dataset_and_model(seed=777, dataset_size=32)
    env_a = ExplainEnv(dataset=dataset, blackbox=blackbox, seed=777)
    env_b = ExplainEnv(dataset=dataset, blackbox=blackbox, seed=777)

    seq_a = [env_a.reset()["x"] for _ in range(5)]
    seq_b = [env_b.reset()["x"] for _ in range(5)]

    for xa, xb in zip(seq_a, seq_b):
        np.testing.assert_allclose(xa, xb)
