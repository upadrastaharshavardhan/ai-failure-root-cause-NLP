"""
FAISS-backed similarity index for retrieving similar historical failures.
Falls back to sklearn NearestNeighbors if FAISS is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    from sklearn.neighbors import NearestNeighbors


class SimilarityIndex:
    def __init__(self, metric: str = "cosine", top_k: int = 5):
        self.metric = metric
        self.top_k = top_k
        self.index = None
        self.metadata: Optional[pd.DataFrame] = None
        self.dimension: Optional[int] = None
        self._use_faiss = FAISS_AVAILABLE

    def build(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> "SimilarityIndex":
        """
        Build the index from embeddings and corresponding metadata rows.
        metadata must be aligned with embeddings (same order / length).
        """
        assert len(embeddings) == len(metadata), "Embeddings and metadata length mismatch"
        self.metadata = metadata.reset_index(drop=True)
        self.dimension = embeddings.shape[1]

        # Ensure float32 and contiguous
        emb = np.ascontiguousarray(embeddings.astype(np.float32))

        if self._use_faiss:
            if self.metric == "cosine":
                # Vectors should already be normalized for cosine via inner product
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(emb)
        else:
            # sklearn fallback
            metric = "cosine" if self.metric == "cosine" else "euclidean"
            self.index = NearestNeighbors(n_neighbors=self.top_k, metric=metric)
            self.index.fit(emb)

        return self

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: Optional[int] = None,
    ) -> List[List[dict]]:
        """
        Return list of lists of similar items (one list per query).
        Each item: {similarity, failure_id, category, error_message, service}
        """
        k = top_k or self.top_k
        query = np.ascontiguousarray(query_embeddings.astype(np.float32))

        results = []

        if self._use_faiss:
            scores, indices = self.index.search(query, k)
            for score_row, idx_row in zip(scores, indices):
                row_results = []
                for score, idx in zip(score_row, idx_row):
                    if idx < 0:
                        continue
                    meta = self.metadata.iloc[idx]
                    sim = float(score) if self.metric == "cosine" else float(1.0 / (1.0 + score))
                    row_results.append(
                        {
                            "similarity": sim,
                            "failure_id": meta.get("failure_id", ""),
                            "category": meta.get("root_cause_category", ""),
                            "error_message": str(meta.get("error_message", ""))[:150],
                            "service": meta.get("service", ""),
                        }
                    )
                results.append(row_results)
        else:
            distances, indices = self.index.kneighbors(query, n_neighbors=k)
            for dist_row, idx_row in zip(distances, indices):
                row_results = []
                for dist, idx in zip(dist_row, idx_row):
                    meta = self.metadata.iloc[idx]
                    sim = float(1.0 - dist) if self.metric == "cosine" else float(1.0 / (1.0 + dist))
                    row_results.append(
                        {
                            "similarity": sim,
                            "failure_id": meta.get("failure_id", ""),
                            "category": meta.get("root_cause_category", ""),
                            "error_message": str(meta.get("error_message", ""))[:150],
                            "service": meta.get("service", ""),
                        }
                    )
                results.append(row_results)

        return results

    def save(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        if self._use_faiss:
            faiss.write_index(self.index, str(index_path))
        else:
            import joblib
            joblib.dump(self.index, str(index_path) + ".sklearn")
        self.metadata.to_csv(metadata_path, index=False)

    def load(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> "SimilarityIndex":
        self.metadata = pd.read_csv(metadata_path)
        if self._use_faiss and Path(index_path).exists():
            self.index = faiss.read_index(str(index_path))
            self.dimension = self.index.d
        else:
            import joblib
            sklearn_path = str(index_path) + ".sklearn"
            self.index = joblib.load(sklearn_path)
            self._use_faiss = False
        return self
