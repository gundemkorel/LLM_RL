import numpy as np

from src.models.g_explainer import ExplanationInducedModel


def test_explanation_induced_model_smoke(tmp_path):
    texts = [
        "evidence leans toward class 0 due to mitigating factors",
        "sharp increase in risk drives class 1 prediction",
        "protective signal keeps probability near class 0",
        "dominant influence lifts class 1 likelihood",
        "negative contribution anchors class 0",
        "positive contribution raises class 1",
        "calming effect implies outcome 0",
        "escalating trend implies outcome 1",
    ]
    y = [0, 1, 0, 1, 0, 1, 0, 1]

    model = ExplanationInducedModel()
    model.fit(texts, y)

    probs = model.predict_proba(texts[:3])
    assert probs.shape == (3, 2)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    save_path = tmp_path / "explainer"
    model.save(save_path)
    loaded = ExplanationInducedModel.load(save_path)
    loaded_probs = loaded.predict_proba(texts[:3])
    assert np.allclose(probs, loaded_probs)
