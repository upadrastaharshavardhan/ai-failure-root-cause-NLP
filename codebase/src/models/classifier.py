"""
Root-cause classification head on top of embeddings.
Supports Logistic Regression and Random Forest.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class RootCauseClassifier:
    def __init__(
        self,
        classifier_type: str = "logistic",
        max_iter: int = 1000,
        class_weight: str = "balanced",
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.classifier_type = classifier_type
        self.label_encoder = LabelEncoder()
        self.model = self._build_model(
            classifier_type, max_iter, class_weight, n_estimators, random_state
        )

    def _build_model(self, clf_type, max_iter, class_weight, n_estimators, random_state):
        if clf_type == "logistic":
            return LogisticRegression(
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            )
        elif clf_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown classifier_type: {clf_type}")

    def fit(self, X: np.ndarray, y: List[str] | np.ndarray) -> "RootCauseClassifier":
        y_enc = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_enc)
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds).tolist()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_with_confidence(self, X: np.ndarray) -> List[dict]:
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        labels = self.label_encoder.inverse_transform(preds)
        confidences = probs.max(axis=1)
        return [
            {"label": lab, "confidence": float(conf)}
            for lab, conf in zip(labels, confidences)
        ]

    @property
    def classes_(self) -> List[str]:
        return self.label_encoder.classes_.tolist()

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": self.label_encoder,
                "classifier_type": self.classifier_type,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RootCauseClassifier":
        data = joblib.load(path)
        obj = cls(classifier_type=data["classifier_type"])
        obj.model = data["model"]
        obj.label_encoder = data["label_encoder"]
        return obj
