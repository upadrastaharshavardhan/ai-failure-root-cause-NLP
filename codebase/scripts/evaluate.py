#!/usr/bin/env python
"""Evaluate a trained RootCausePredictor on a held-out set or new CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from src.pipeline.predictor import RootCausePredictor
from src.data.preprocessing import TextPreprocessor
from src.utils.helpers import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None, help="CSV with full_text + root_cause_category")
    args = parser.parse_args()

    cfg = load_config(args.config)
    predictor = RootCausePredictor.load(args.artifacts, args.config)

    data_path = args.data or cfg["paths"]["raw_data"]
    df = pd.read_csv(data_path)

    # Use a small sample for quick eval if large
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    texts = df["full_text"].tolist()
    true_labels = df["root_cause_category"].tolist()

    results = predictor.predict_batch(texts, top_k_similar=1)
    preds = [r["predicted_root_cause"] for r in results]

    print("Accuracy:", accuracy_score(true_labels, preds))
    print("\nClassification Report:")
    print(classification_report(true_labels, preds, digits=3))


if __name__ == "__main__":
    main()
