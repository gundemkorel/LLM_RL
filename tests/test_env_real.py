import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from src.data.datasets import TabularDataset
from src.envs.explain_env import ExplainEnv
from src.models.blackbox import BlackBoxModel


def test_explain_env_real_dataset_smoke():
    data = load_breast_cancer()
    dataset = TabularDataset.from_arrays(
        data.data,
        data.target,
        list(data.feature_names),
    )
    model = RandomForestClassifier(random_state=0)
    model.fit(dataset.X, dataset.y)
    blackbox = BlackBoxModel.from_sklearn(model, n_features=dataset.n_features)

    env = ExplainEnv(dataset=dataset, blackbox=blackbox, reveal="probs", seed=0)
    obs = env.reset()

    assert set(obs.keys()) == {"x", "p"}
    assert obs["x"].shape == (dataset.n_features,)
    assert obs["p"].shape == (2,)
    assert env.feature_names == list(data.feature_names)

    result = env.step("dummy explanation", tool_call_count=0)
    assert result.done is True
    assert np.isfinite(result.reward)
    assert "base_reward" in result.info
