from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from src.data.datasets import TabularDataset
from src.models.blackbox import BlackBoxModel


def test_blackbox_from_sklearn_predicts_probabilities() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    estimator = RandomForestClassifier(n_estimators=16, random_state=0)
    estimator.fit(X, y)

    model = BlackBoxModel.from_sklearn(estimator=estimator, n_features=X.shape[1])

    proba = model.predict_proba(X[0])
    assert proba.shape == (2,)
    assert np.isfinite(proba).all()
    assert np.isclose(proba.sum(), 1.0, atol=1e-6)


def test_blackbox_from_training_wraps_estimator_factory() -> None:
    data = load_breast_cancer()
    dataset = TabularDataset.from_arrays(data.data, data.target, feature_names=list(data.feature_names))

    def factory() -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=8, random_state=0)

    model = BlackBoxModel.from_training(dataset=dataset, estimator_factory=factory)

    proba = model.predict_proba(dataset.X[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
