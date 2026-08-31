"""
Dataset utilities for loading, splitting, and saving failure data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class FailureDataset:
    def __init__(self, df: pd.DataFrame, text_col: str = "cleaned_text", label_col: str = "root_cause_category"):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "FailureDataset":
        df = pd.read_csv(path)
        return cls(df, **kwargs)

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        stratify_col = self.df[self.label_col] if stratify else None
        train_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)

    @property
    def texts(self):
        return self.df[self.text_col].tolist()

    @property
    def labels(self):
        return self.df[self.label_col].tolist()

    def __len__(self) -> int:
        return len(self.df)
