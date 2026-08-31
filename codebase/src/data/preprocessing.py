"""
Text preprocessing for failure logs and stack traces.
Removes noise, normalizes, and optionally extracts key phrases with spaCy.
"""

from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd


class TextPreprocessor:
    """Clean and normalize failure text for embedding / classification."""

    def __init__(
        self,
        max_text_length: int = 2000,
        remove_request_ids: bool = True,
        remove_hex: bool = True,
        remove_timestamps: bool = True,
        extract_key_phrases: bool = False,
        spacy_model: str = "en_core_web_sm",
    ):
        self.max_text_length = max_text_length
        self.remove_request_ids = remove_request_ids
        self.remove_hex = remove_hex
        self.remove_timestamps = remove_timestamps
        self.extract_key_phrases = extract_key_phrases
        self._nlp = None
        self.spacy_model = spacy_model

    def _load_spacy(self):
        if self._nlp is None and self.extract_key_phrases:
            try:
                import spacy
                self._nlp = spacy.load(self.spacy_model)
            except OSError:
                print(f"[WARN] spaCy model '{self.spacy_model}' not found. "
                      f"Run: python -m spacy download {self.spacy_model}")
                self.extract_key_phrases = False

    def clean(self, text: str) -> str:
        """Clean a single failure text string."""
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""

        text = str(text)

        if self.remove_request_ids:
            text = re.sub(r"(?i)(request[_-]?id|correlation[_-]?id|trace[_-]?id)\s*[=:]\s*\S+", "", text)
            text = re.sub(r"\breq-[a-z0-9\-]+\b", "", text, flags=re.IGNORECASE)

        if self.remove_hex:
            # long hex strings / UUIDs
            text = re.sub(r"\b[0-9a-f]{8,}\b", "", text, flags=re.IGNORECASE)
            text = re.sub(
                r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                "",
                text,
                flags=re.IGNORECASE,
            )

        if self.remove_timestamps:
            text = re.sub(
                r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
                "",
                text,
            )
            text = re.sub(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}", "", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Truncate
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length]

        return text

    def extract_phrases(self, text: str, max_phrases: int = 15) -> str:
        """Extract noun chunks as a lightweight key-phrase representation."""
        self._load_spacy()
        if self._nlp is None:
            return ""
        doc = self._nlp(text[:1000])
        phrases = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
        return " ".join(phrases[:max_phrases])

    def transform(self, texts: List[str] | pd.Series) -> List[str]:
        """Clean a batch of texts."""
        cleaned = [self.clean(t) for t in texts]
        if self.extract_key_phrases:
            cleaned = [c + " " + self.extract_phrases(c) for c in cleaned]
        return cleaned

    def transform_df(self, df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
        """Add a 'cleaned_text' column to a DataFrame."""
        df = df.copy()
        df["cleaned_text"] = self.transform(df[text_col].tolist())
        return df
