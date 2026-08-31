#!/usr/bin/env python
"""
Train the full Root Cause Prediction pipeline:
1. Load / generate data
2. Preprocess
3. Create embeddings
4. Train classifier
5. Build similarity index
6. Save artifacts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.generator import generate_dataset
from src.data.preprocessing import TextPreprocessor
from src.data.dataset import FailureDataset
from src.models.embeddings import EmbeddingModel
from src.models.classifier import RootCauseClassifier
from src.models.similarity import SimilarityIndex
from src.utils.helpers import load_config, ensure_dirs, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None, help="Path to CSV (optional)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["random_seed"])
    ensure_dirs(cfg["paths"]["data_dir"], cfg["paths"]["artifacts_dir"])

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    data_path = Path(args.data or cfg["paths"]["raw_data"])
    if data_path.exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("No data found – generating synthetic dataset...")
        df = generate_dataset(
            n_samples=cfg["data"]["n_samples"],
            seed=cfg["data"]["random_seed"],
            categories=cfg.get("categories"),
        )
        df.to_csv(data_path, index=False)
        print(f"Saved synthetic data → {data_path}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    pre_cfg = cfg["preprocessing"]
    preprocessor = TextPreprocessor(
        max_text_length=pre_cfg["max_text_length"],
        remove_request_ids=pre_cfg["remove_request_ids"],
        remove_hex=pre_cfg["remove_hex"],
        remove_timestamps=pre_cfg["remove_timestamps"],
        extract_key_phrases=pre_cfg.get("extract_key_phrases", False),
        spacy_model=pre_cfg.get("spacy_model", "en_core_web_sm"),
    )
    df = preprocessor.transform_df(df, text_col="full_text")

    dataset = FailureDataset(df)
    train_df, test_df = dataset.train_test_split(
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_seed"],
    )
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # ------------------------------------------------------------------
    # 3. Embeddings
    # ------------------------------------------------------------------
    emb_cfg = cfg["embedding"]
    embedder = EmbeddingModel(
        model_name=emb_cfg["model_name"],
        device=emb_cfg.get("device"),
        normalize=emb_cfg.get("normalize", True),
    )

    print("Encoding train set...")
    X_train = embedder.encode(
        train_df["cleaned_text"].tolist(),
        batch_size=emb_cfg.get("batch_size", 64),
    )
    print("Encoding test set...")
    X_test = embedder.encode(
        test_df["cleaned_text"].tolist(),
        batch_size=emb_cfg.get("batch_size", 64),
        show_progress=True,
    )

    # Save full embeddings (train+test) for reference
    all_emb = embedder.encode(df["cleaned_text"].tolist(), batch_size=emb_cfg.get("batch_size", 64))
    embedder.save_embeddings(all_emb, cfg["paths"]["embeddings"])

    # ------------------------------------------------------------------
    # 4. Classifier
    # ------------------------------------------------------------------
    clf_cfg = cfg["classifier"]
    classifier = RootCauseClassifier(
        classifier_type=clf_cfg["type"],
        max_iter=clf_cfg.get("max_iter", 1000),
        class_weight=clf_cfg.get("class_weight", "balanced"),
        n_estimators=clf_cfg.get("n_estimators", 200),
        random_state=clf_cfg.get("random_state", 42),
    )
    print(f"Training {clf_cfg['type']} classifier...")
    classifier.fit(X_train, train_df["root_cause_category"].tolist())

    y_pred = classifier.predict(X_test)
    y_true = test_df["root_cause_category"].tolist()

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=classifier.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classifier.classes_,
        yticklabels=classifier.classes_,
        cmap="Blues",
    )
    plt.title("Confusion Matrix – Root Cause Prediction")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = Path(cfg["paths"]["artifacts_dir"]) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved → {cm_path}")

    classifier.save(cfg["paths"]["classifier"])

    # ------------------------------------------------------------------
    # 5. Similarity index (built on train set)
    # ------------------------------------------------------------------
    sim_cfg = cfg["similarity"]
    similarity = SimilarityIndex(metric=sim_cfg["metric"], top_k=sim_cfg["top_k"])
    meta_cols = ["failure_id", "service", "error_message", "root_cause_category"]
    similarity.build(X_train, train_df[meta_cols])

    faiss_path = Path(cfg["paths"]["artifacts_dir"]) / "faiss.index"
    meta_path = Path(cfg["paths"]["artifacts_dir"]) / "metadata.csv"
    similarity.save(faiss_path, meta_path)
    print(f"Similarity index saved → {faiss_path}")

    # Also save full metadata for reference
    df[["failure_id", "service", "error_message", "root_cause_category"]].to_csv(
        cfg["paths"]["metadata"], index=False
    )

    print("\n✅ Training complete. Artifacts ready in:", cfg["paths"]["artifacts_dir"])


if __name__ == "__main__":
    main()
