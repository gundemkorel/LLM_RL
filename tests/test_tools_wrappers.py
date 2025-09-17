from __future__ import annotations

import numpy as np
import pytest

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import-time configuration
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.lime_tool import make_lime_tool
from src.tools.pdp_tool import make_pdp_tool
from src.tools.shap_tool import make_shap_tool


@pytest.fixture(scope="module")
def breast_cancer_setup():
    data = load_breast_cancer()
    X = data.data.astype(float)
    y = data.target.astype(int)
    model = RandomForestClassifier(n_estimators=32, random_state=0).fit(X, y)
    background = X[:128]
    feature_grid = {
        i: np.linspace(X[:, i].min(), X[:, i].max(), num=8, dtype=float)
        for i in range(X.shape[1])
    }
    feature_names = list(data.feature_names)
    return X, model, background, feature_grid, feature_names


def test_shap_tool_returns_top_k(breast_cancer_setup):
    X, model, background, _, _ = breast_cancer_setup
    tool = make_shap_tool(model.predict_proba, background, model_kind="tree")
    x = X[0]
    result = tool(x, top_k=5)
    assert isinstance(result, list)
    assert len(result) == 5
    magnitudes = []
    for idx, value in result:
        assert isinstance(idx, int)
        assert 0 <= idx < X.shape[1]
        assert np.isfinite(value)
        magnitudes.append(abs(value))
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_lime_tool_produces_coefficients(breast_cancer_setup):
    X, model, background, _, feature_names = breast_cancer_setup
    tool = make_lime_tool(
        model.predict_proba,
        feature_names,
        training_data=background,
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


def test_pdp_tool_returns_curve(breast_cancer_setup):
    X, model, background, feature_grid, _ = breast_cancer_setup
    tool = make_pdp_tool(model.predict_proba, feature_grid, background=background)
    feature_idx = 3
    pdp = tool(X[2], feature_idx, grid_points=7)
    grid = np.asarray(pdp["grid"], dtype=float)
    assert grid.shape[0] == 7
    curve = np.asarray(pdp["pdp"], dtype=float)
    ice = np.asarray(pdp["ice"], dtype=float)
    assert curve.shape == grid.shape
    assert ice.shape == grid.shape
    assert np.all(np.isfinite(curve))
    assert np.all(np.isfinite(ice))
    assert pdp["feature"] == feature_idx
    assert isinstance(pdp["target_class"], int)

