"""Basic smoke tests for the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data.generator import generate_dataset
from src.data.preprocessing import TextPreprocessor
from src.models.embeddings import EmbeddingModel
from src.models.classifier import RootCauseClassifier
from src.models.similarity import SimilarityIndex


def test_generator():
    df = generate_dataset(n_samples=50, seed=1)
    assert len(df) == 50
    assert "root_cause_category" in df.columns
    assert df["root_cause_category"].nunique() > 3


def test_preprocessor():
    pre = TextPreprocessor()
    text = "Error: NullPointerException requestId=12345 at 2024-01-01T12:00:00"
    cleaned = pre.clean(text)
    assert "requestId" not in cleaned
    assert "2024-01-01" not in cleaned


def test_end_to_end_tiny():
    df = generate_dataset(n_samples=120, seed=7)
    pre = TextPreprocessor()
    df = pre.transform_df(df)

    embedder = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    X = embedder.encode(df["cleaned_text"].tolist(), batch_size=32, show_progress=False)

    clf = RootCauseClassifier(classifier_type="logistic")
    clf.fit(X[:100], df["root_cause_category"].iloc[:100].tolist())
    preds = clf.predict(X[100:])
    assert len(preds) == 20

    sim = SimilarityIndex(top_k=3)
    sim.build(X[:100], df.iloc[:100][["failure_id", "service", "error_message", "root_cause_category"]])
    results = sim.search(X[100:101])
    assert len(results[0]) == 3


if __name__ == "__main__":
    test_generator()
    test_preprocessor()
    test_end_to_end_tiny()
    print("All smoke tests passed.")
