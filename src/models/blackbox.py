from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

class BlackBoxModel:
    """Opaque binary classifier f: X -> [0,1]^2 with predict_proba."""
    def __init__(self, n_features: int = 10, random_state: int = 0):
        X, y = make_classification(n_samples=500, n_features=n_features, n_informative=6,
                                   n_redundant=2, n_classes=2, random_state=random_state)
        self.model = LogisticRegression(max_iter=1000).fit(X, y)
        self.n_features = n_features

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return P(y|x) as [p0, p1]. Accepts shape (n_features,) or (batch, n_features)."""
        x = np.atleast_2d(x)
        prob = self.model.predict_proba(x)
        return prob.squeeze()

    # The 'opaque' interface: no attributes of the model are exposed to the policy.
