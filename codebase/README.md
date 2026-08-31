# AI-Based Software Failure Analysis & Root Cause Prediction

**Project 1** – Advanced NLP-powered system that analyzes logs, stack traces, and error messages to predict the most likely root cause category and retrieve similar historical failures.

## Features

- **Synthetic Data Generator** – Creates realistic multi-service failure data with 9 root-cause categories
- **Advanced Preprocessing** – Cleans stack traces, removes noise (request IDs, timestamps, hex), extracts key phrases with spaCy
- **Sentence Embeddings** – Uses `sentence-transformers` (`all-MiniLM-L6-v2` by default, easily swappable)
- **Root Cause Classifier** – Logistic Regression / Random Forest / optional fine-tuned transformer head
- **Similarity Search** – FAISS-backed nearest-neighbor retrieval of historical failures
- **Full Prediction Pipeline** – Single function that returns predicted category + confidence + similar cases
- **Gradio Demo UI** – Interactive web interface for pasting errors
- **Configurable** – YAML config for categories, model names, paths
- **Colab-Ready** – Designed to run end-to-end in Google Colab (GPU recommended)
- **Extensible** – Clean modular structure for adding real log parsers, vector DBs, or fine-tuning

## Project Structure

```
ai-failure-root-cause/
├── config/
│   └── config.yaml
├── data/                     # Generated or real datasets land here
├── src/
│   ├── data/
│   │   ├── generator.py      # Synthetic failure generator
│   │   ├── preprocessing.py  # Cleaning + key-phrase extraction
│   │   └── dataset.py
│   ├── models/
│   │   ├── embeddings.py
│   │   ├── classifier.py
│   │   └── similarity.py
│   ├── pipeline/
│   │   └── predictor.py      # Main predict() API
│   ├── utils/
│   │   └── helpers.py
│   └── api/
│       └── gradio_app.py
├── scripts/
│   ├── generate_data.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── 01_colab_quickstart.ipynb
├── artifacts/                # Saved models, embeddings, plots
├── requirements.txt
└── README.md
```

## Quick Start (Google Colab)

1. Upload the entire unzipped folder to Colab (or clone/mount).
2. Open `notebooks/01_colab_quickstart.ipynb` **or** run the scripts below.

```python
# In a Colab cell
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm

# Generate data
!python scripts/generate_data.py --n-samples 3000

# Train
!python scripts/train.py

# Launch Gradio demo
!python -m src.api.gradio_app
```

## Root Cause Categories (default)

1. NullPointer
2. Configuration
3. Dependency
4. ResourceExhaustion
5. RaceCondition
6. Auth
7. Validation
8. Network
9. Database

## Usage Example

```python
from src.pipeline.predictor import RootCausePredictor

predictor = RootCausePredictor.load("artifacts")

result = predictor.predict("""
Service: payment-service
Error: java.lang.NullPointerException: Cannot invoke "com.example.User.getId()" because "user" is null
at com.example.PaymentService.process(PaymentService.java:112)
""")

print(result)
# {
#   "predicted_root_cause": "NullPointer",
#   "confidence": 0.94,
#   "similar_historical_failures": [...]
# }
```

## Extending the System

- Replace synthetic data with real logs (ELK, CloudWatch, Splunk export).
- Swap embedding model in `config/config.yaml`.
- Add fine-tuning with SetFit or Hugging Face Trainer.
- Persist embeddings in Chroma / Pinecone / Weaviate for production.
- Add SHAP / attention explanations.

## License

MIT – free for research and commercial use.
