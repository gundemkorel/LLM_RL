from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class ExplanationInducedModel:
    """g(E): simple bag-of-words -> logistic over classes {0,1}."""
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(1,2), min_df=1)
        self.clf = LogisticRegression(max_iter=1000)

    def fit_dummy(self):
        """Train a tiny prior so g is well-posed before RL.
        We fit on templated data linking words to class hints."""
        texts = [
            "evidence indicates class 0 due to low risk",
            "features suggest class 1 with strong signal",
            "likely class 0 because negative contribution",
            "likely class 1 because positive contribution"
        ]
        y = np.array([0,1,0,1])
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, y)

    def predict_proba(self, explanations: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(explanations)
        proba1 = self.clf.predict_proba(X)  # [:, 1]
        return proba1  # shape (n, 2)
