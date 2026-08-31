# Detailed Methodology Documentation

## 1. Data Generation Process

The synthetic data generator creates realistic failure records by:

1. Selecting a root-cause category uniformly at random.
2. Sampling an error message template from a curated list of 5 representative messages per category.
3. Selecting a service name from a pool of 9 microservices.
4. Injecting realistic noise (request IDs, correlation IDs) with controlled probability.
5. Generating a multi-line Java/Spring-style stack trace that references the chosen service.
6. Concatenating service, error, and stack trace into a single `full_text` field used by the model.

This process yields high lexical diversity while preserving strong semantic signals for each category.

## 2. Preprocessing Algorithms

### 2.1 Noise Removal Regex Patterns

- Request / Correlation / Trace IDs: `(?i)(request[_-]?id|correlation[_-]?id|trace[_-]?id)\s*[=:]\s*\S+`
- Long hex / UUID: `\b[0-9a-f]{8,}\b` and standard UUID pattern
- Timestamps: ISO-8601 and `dd/mm/yyyy hh:mm:ss` patterns

### 2.2 Optional Key-Phrase Extraction

spaCy `en_core_web_sm` noun chunks are extracted (limited to first 15) and appended. This provides additional signal for models that benefit from explicit noun phrases.

## 3. Embedding Strategy

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Output dimension: 384
- Normalization: L2 (enables efficient cosine via inner product)
- Batch size: 64 (GPU) / 16-32 (CPU)

Alternative models supported via configuration:
- `all-mpnet-base-v2` (higher quality, slower)
- `BAAI/bge-base-en-v1.5` / `BAAI/bge-large-en-v1.5`

## 4. Classification

Logistic Regression with:
- `max_iter=1000`
- `class_weight='balanced'`
- Multi-class: one-vs-rest (scikit-learn default)

Random Forest alternative uses 200 estimators.

## 5. Similarity Search

FAISS `IndexFlatIP` on L2-normalized vectors. Fallback to scikit-learn `NearestNeighbors` (cosine) when FAISS is unavailable.

## 6. Evaluation Metrics Definitions

- **Accuracy**: fraction of correctly predicted categories
- **Macro F1**: unweighted mean of per-class F1 scores
- **Weighted F1**: support-weighted mean of per-class F1 scores
- **MRR@k**: mean reciprocal rank of the first relevant (same-category) neighbor within top-k
- **Recall@k**: fraction of queries for which at least one same-category neighbor appears in top-k
- **Precision@k**: average fraction of top-k neighbors that share the query category

## 7. Reproducibility Guarantees

- Fixed random seed (42) for data generation, train/test split, and model initialization
- Deterministic embedding generation (no dropout at inference)
- All configuration externalized in YAML
