from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import-time configuration
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.lime_tool import make_lime_tool
from src.tools.pdp_tool import make_pdp_tool
from src.tools.shap_tool import make_shap_tool


@pytest.fixture(scope="module")
def logistic_setup():
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    model = LogisticRegression(max_iter=1000, solver="lbfgs").fit(X, y)
    feature_grid = {
        i: np.linspace(X[:, i].min(), X[:, i].max(), num=6, dtype=float)
        for i in range(X.shape[1])
    }
    return X, model, feature_grid


def test_shap_tool_returns_top_k(logistic_setup):
    X, model, _ = logistic_setup
    tool = make_shap_tool(model.predict_proba, X[:50])
    x = X[0]
    result = tool(x, top_k=3)
    assert isinstance(result, list)
    assert len(result) == 3
    magnitudes = []
    for idx, value in result:
        assert isinstance(idx, int)
        assert 0 <= idx < X.shape[1]
        assert np.isfinite(value)
        magnitudes.append(abs(value))
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_lime_tool_produces_coefficients(logistic_setup):
    X, model, _ = logistic_setup
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    tool = make_lime_tool(
        model.predict_proba,
        feature_names,
        training_data=X[:100],
        n_samples=256,
        random_state=123,
    )
    explanation = tool(X[1])
    assert explanation["method"] == "LIME"
    coeffs = np.asarray(explanation["coefficients"], dtype=float)
    assert coeffs.shape == (X.shape[1],)
    assert np.all(np.isfinite(coeffs))
    assert np.isfinite(float(explanation["intercept"]))
    assert 0.0 <= float(explanation["r2"]) <= 1.0
    assert isinstance(explanation["target_class"], int)


def test_pdp_tool_returns_curve(logistic_setup):
    X, model, feature_grid = logistic_setup
    tool = make_pdp_tool(model.predict_proba, feature_grid, background=X[:40])
    feature_idx = 1
    pdp = tool(X[2], feature_idx)
    grid = np.asarray(pdp["grid"], dtype=float)
    assert grid.shape[0] == feature_grid[feature_idx].shape[0]
    curve = np.asarray(pdp["pdp"], dtype=float)
    ice = np.asarray(pdp["ice"], dtype=float)
    assert curve.shape == grid.shape
    assert ice.shape == grid.shape
    assert np.all(np.isfinite(curve))
    assert np.all(np.isfinite(ice))
    assert pdp["feature"] == feature_idx
    assert isinstance(pdp["target_class"], int)

