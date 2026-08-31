"""
End-to-end Root Cause Prediction pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from src.data.preprocessing import TextPreprocessor
from src.models.embeddings import EmbeddingModel
from src.models.classifier import RootCauseClassifier
from src.models.similarity import SimilarityIndex


class RootCausePredictor:
    def __init__(
        self,
        embedder: EmbeddingModel,
        classifier: RootCauseClassifier,
        similarity: SimilarityIndex,
        preprocessor: TextPreprocessor,
    ):
        self.embedder = embedder
        self.classifier = classifier
        self.similarity = similarity
        self.preprocessor = preprocessor

    def predict(
        self,
        error_text: str,
        top_k_similar: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict root cause for a single error / stacktrace text.

        Returns
        -------
        dict with keys:
            predicted_root_cause, confidence, similar_historical_failures
        """
        cleaned = self.preprocessor.clean(error_text)
        emb = self.embedder.encode([cleaned], show_progress=False)

        pred_info = self.classifier.predict_with_confidence(emb)[0]
        similar = self.similarity.search(emb, top_k=top_k_similar)[0]

        return {
            "predicted_root_cause": pred_info["label"],
            "confidence": pred_info["confidence"],
            "similar_historical_failures": similar,
            "cleaned_input_preview": cleaned[:300] + ("..." if len(cleaned) > 300 else ""),
        }

    def predict_batch(
        self,
        error_texts: List[str],
        top_k_similar: int = 3,
    ) -> List[Dict[str, Any]]:
        cleaned = self.preprocessor.transform(error_texts)
        embs = self.embedder.encode(cleaned, show_progress=True)
        pred_infos = self.classifier.predict_with_confidence(embs)
        similars = self.similarity.search(embs, top_k=top_k_similar)

        results = []
        for pred, sim in zip(pred_infos, similars):
            results.append(
                {
                    "predicted_root_cause": pred["label"],
                    "confidence": pred["confidence"],
                    "similar_historical_failures": sim,
                }
            )
        return results

    def save(self, artifacts_dir: Union[str, Path]) -> None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.classifier.save(artifacts_dir / "classifier.joblib")
        # similarity index + metadata already saved during training

    @classmethod
    def load(
        cls,
        artifacts_dir: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
    ) -> "RootCausePredictor":
        artifacts_dir = Path(artifacts_dir)

        # Load config if provided
        if config_path is None:
            config_path = Path("config/config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        emb_cfg = cfg.get("embedding", {})
        pre_cfg = cfg.get("preprocessing", {})
        sim_cfg = cfg.get("similarity", {})

        embedder = EmbeddingModel(
            model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=emb_cfg.get("device"),
            normalize=emb_cfg.get("normalize", True),
        )

        classifier = RootCauseClassifier.load(artifacts_dir / "classifier.joblib")

        similarity = SimilarityIndex(
            metric=sim_cfg.get("metric", "cosine"),
            top_k=sim_cfg.get("top_k", 5),
        )
        similarity.load(
            artifacts_dir / "faiss.index",
            artifacts_dir / "metadata.csv",
        )

        preprocessor = TextPreprocessor(
            max_text_length=pre_cfg.get("max_text_length", 2000),
            remove_request_ids=pre_cfg.get("remove_request_ids", True),
            remove_hex=pre_cfg.get("remove_hex", True),
            remove_timestamps=pre_cfg.get("remove_timestamps", True),
            extract_key_phrases=pre_cfg.get("extract_key_phrases", False),
            spacy_model=pre_cfg.get("spacy_model", "en_core_web_sm"),
        )

        return cls(embedder, classifier, similarity, preprocessor)
