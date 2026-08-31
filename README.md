# 🧠 AI-Based Software Failure Analysis & Root Cause Prediction

<p align="center">

<strong>Turning raw software failures into actionable root-cause intelligence using NLP, semantic embeddings, machine learning, and similarity search.</strong>

</p>

<p align="center">

  <a href="https://github.com/upadrastaharshavardhan/ai-failure-root-cause-NLP">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NLP-Sentence%20Transformers-FF6F00?style=for-the-badge" alt="NLP">
  <img src="https://img.shields.io/badge/ML-Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Vector%20Search-FAISS-00A67E?style=for-the-badge" alt="FAISS">
  <img src="https://img.shields.io/badge/Demo-Gradio-FF4B4B?style=for-the-badge" alt="Gradio">
  <img src="https://img.shields.io/badge/Research-August%202026-6C5CE7?style=for-the-badge" alt="Research">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">

</p>

---

## 🚀 What Is This Project?

Modern software systems generate enormous amounts of diagnostic information:

* application logs
* exceptions
* stack traces
* service names
* request IDs
* correlation IDs
* database errors
* authentication failures
* network failures
* dependency failures

When a production incident occurs, engineers often have to manually interpret this information and search through previous incidents to determine **what actually caused the failure**.

This project explores an NLP-based approach to automate that process.

### The system answers two questions:

> **1. What is the most likely root cause of this failure?**

and

> **2. Which historical failures look most similar?**

Instead of treating a failure as plain text, the system transforms it into a semantic representation and combines:

**Preprocessing → NLP Embeddings → ML Classification → Vector Similarity Search → Root-Cause Intelligence**

---

# 🎯 Research Objective

The primary research objective is to investigate whether modern NLP representations can effectively classify software failures and retrieve semantically similar historical incidents.

The research evaluates:

* semantic embeddings vs. traditional TF-IDF
* preprocessing impact
* classifier selection
* embedding model selection
* class balancing
* similarity retrieval quality
* classification performance
* inference latency
* failure-category ambiguity

The current experimental implementation uses a **synthetically generated multi-service failure dataset** designed to emulate realistic Java/Spring-style application failures.

---

# 📊 Experimental Results

## ⭐ Headline Results

The reported evaluation was performed on a **600-sample held-out test set** from a **3,000-sample synthetic dataset** using an 80/20 stratified split.

| Metric             |     Result |
| ------------------ | ---------: |
| 🎯 Accuracy        | **96.83%** |
| 🧠 Macro Precision |  **0.969** |
| 🧠 Macro Recall    |  **0.968** |
| 🏆 Macro F1        |  **0.967** |
| ⚖️ Weighted F1     |  **0.968** |
| 🔎 MRR@5           |  **0.912** |
| 🔎 Recall@1        |  **0.847** |
| 🔎 Recall@3        |  **0.953** |
| 🔎 Recall@5        |  **0.978** |
| 🔎 Precision@5     |  **0.891** |

These values come from the repository's experimental-results documentation.

> **Important:** These are research results on synthetic data, not a claim of production accuracy on real enterprise incident data.

---

# 🧪 Per-Class Performance

The experiment evaluates nine root-cause categories.

| Root Cause         | Precision | Recall |        F1 |
| ------------------ | --------: | -----: | --------: |
| NullPointer        |     0.985 |  0.970 |     0.977 |
| Configuration      |     0.971 |  0.985 | **0.978** |
| Dependency         |     0.955 |  0.940 |     0.947 |
| ResourceExhaustion |     0.970 |  0.955 |     0.962 |
| RaceCondition      |     0.940 |  0.955 |     0.947 |
| Auth               |     0.985 |  0.970 |     0.977 |
| Validation         |     0.970 |  0.985 |     0.977 |
| Network            |     0.955 |  0.970 |     0.962 |
| Database           |     0.940 |  0.955 |     0.947 |

The class distribution is intentionally close to uniform, which is why macro and weighted metrics are very similar.

---

# 🔬 Why NLP?

Traditional log-analysis systems often depend heavily on:

```text
Exact string matching
       ↓
Regex rules
       ↓
Keyword detection
       ↓
Predefined signatures
```

That approach can struggle when the same failure appears with:

* different request IDs
* different timestamps
* different services
* different stack-trace locations
* different wording
* different runtime context

This project instead investigates:

```text
Raw Failure
     │
     ▼
Noise Removal
     │
     ▼
Semantic Text Representation
     │
     ▼
Sentence Embedding
     │
     ├───────────────┐
     ▼               ▼
Classifier       Vector Search
     │               │
     ▼               ▼
Root Cause      Similar Failures
     │               │
     └───────┬───────┘
             ▼
      Failure Intelligence
```

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────────┐
                         │     Failure Input        │
                         │                          │
                         │ Logs / Error / Trace     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   NLP Preprocessing      │
                         │                          │
                         │ • Noise removal          │
                         │ • ID removal             │
                         │ • Timestamp removal      │
                         │ • Phrase extraction      │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                    ┌────────────────────────────────────┐
                    │       Sentence Transformer         │
                    │                                    │
                    │       all-MiniLM-L6-v2             │
                    │                                    │
                    │          384 dimensions             │
                    └───────────────┬────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
          ┌─────────────────────┐       ┌─────────────────────┐
          │ Root Cause           │       │ Similarity Search   │
          │ Classifier           │       │                     │
          │                     │       │ FAISS               │
          │ Logistic Regression │       │ IndexFlatIP         │
          │ / Random Forest     │       │                     │
          └──────────┬──────────┘       └──────────┬──────────┘
                     │                             │
                     ▼                             ▼
          ┌─────────────────────┐       ┌─────────────────────┐
          │ Predicted Root      │       │ Similar Historical  │
          │ Cause + Confidence  │       │ Failures            │
          └──────────┬──────────┘       └──────────┬──────────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    ▼
                         ┌──────────────────────────┐
                         │ Failure Analysis Result  │
                         └──────────────────────────┘
```

---

# 🧠 Core ML/NLP Pipeline

## 1️⃣ Failure Data Generation

The project includes a synthetic failure generator that creates realistic multi-service failure records.

Each generated record can contain:

```text
Service
+
Error Message
+
Request / Correlation Information
+
Stack Trace
+
Root Cause Label
```

The generator selects a root-cause category, samples representative error templates, selects a microservice, injects controlled noise, and constructs a Java/Spring-style stack trace.

---

## 2️⃣ Intelligent Preprocessing

Raw logs contain large amounts of information that may not contribute to root-cause classification.

The preprocessing layer removes dynamic identifiers such as:

```text
Request IDs
Correlation IDs
Trace IDs
UUIDs
Long hexadecimal values
Timestamps
```

It can also extract noun phrases using spaCy.

This helps the model focus on the **semantic failure signal** rather than incidental runtime identifiers.

---

# 3️⃣ Semantic Embeddings

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Configuration:

| Property         | Value            |
| ---------------- | ---------------- |
| Model            | all-MiniLM-L6-v2 |
| Vector Dimension | 384              |
| Normalization    | L2               |
| GPU Batch Size   | 64               |
| CPU Batch Size   | 16–32            |

Because vectors are L2-normalized, inner-product search can efficiently approximate cosine similarity.

---

# 4️⃣ Root Cause Classification

The primary classifier is:

```text
Logistic Regression
```

with:

```text
max_iter = 1000
class_weight = balanced
```

A Random Forest alternative is also supported.

The classifier maps the semantic representation of a failure to one of the nine root-cause categories.

---

# 5️⃣ Similarity Search

Classification answers:

> **What category is this failure?**

Similarity search answers:

> **Have we seen something like this before?**

The project uses:

```text
FAISS
IndexFlatIP
```

with L2-normalized vectors.

When FAISS is unavailable, the implementation can fall back to scikit-learn cosine-based nearest-neighbor search.

---

# 🔎 Retrieval Performance

| Retrieval Metric |     Score |
| ---------------- | --------: |
| Recall@1         | **0.847** |
| Recall@3         | **0.953** |
| Recall@5         | **0.978** |
| Precision@5      | **0.891** |
| MRR@5            | **0.912** |

This means the retrieval component frequently surfaces same-category historical failures within the top few results.

---

# 🧪 Ablation Study

One of the most valuable parts of this project is that the research does not only report the final model.

It also evaluates how individual design decisions affect performance.

| Configuration                |   Accuracy |  Macro F1 |     MRR@5 |
| ---------------------------- | ---------: | --------: | --------: |
| **MiniLM + LR + FAISS**      | **96.83%** | **0.967** | **0.912** |
| No preprocessing             |     94.17% |     0.941 |     0.887 |
| TF-IDF + Logistic Regression |     89.50% |     0.893 |         — |
| Random Forest                |     95.67% |     0.955 |     0.905 |
| all-mpnet-base-v2            | **97.33%** | **0.973** | **0.921** |
| No class weighting           |     95.50% |     0.953 |     0.908 |

The experiments show a substantial improvement over the TF-IDF baseline and measurable gains from preprocessing.

---

# ⚡ Latency

The reported latency profile is:

| Pipeline Stage |          GPU |          CPU |
| -------------- | -----------: | -----------: |
| Embedding      |      8–12 ms |     25–40 ms |
| Classification |        <1 ms |        <1 ms |
| FAISS Top-5    |       1–2 ms |       2–4 ms |
| **End-to-End** | **12–15 ms** | **30–45 ms** |

These measurements indicate that the classification and vector-search components are lightweight compared with embedding generation.

---

# 🧩 Root Cause Taxonomy

The current system supports nine categories:

```text
1. NullPointer
2. Configuration
3. Dependency
4. ResourceExhaustion
5. RaceCondition
6. Auth
7. Validation
8. Network
9. Database
```

This taxonomy is intentionally broad enough to demonstrate multi-class failure classification while remaining manageable for controlled experimentation.

---

# 💻 Example

```python
from src.pipeline.predictor import RootCausePredictor

predictor = RootCausePredictor.load("artifacts")

failure = """
Service: payment-service

Error:
java.lang.NullPointerException:
Cannot invoke "com.example.User.getId()"
because "user" is null

at com.example.PaymentService.process(
    PaymentService.java:112
)
"""

result = predictor.predict(failure)

print(result)
```

Example response:

```python
{
    "predicted_root_cause": "NullPointer",
    "confidence": 0.94,
    "similar_historical_failures": [
        ...
    ]
}
```

The prediction API is designed to return both classification information and similar historical cases.

---

# 📁 Repository Structure

```text
ai-failure-root-cause-NLP/
│
├── README.md
├── LICENSE
│
├── paper/
│   ├── AI_Failure_Root_Cause_Prediction_Research_Paper.pdf
│   └── AI_Failure_Root_Cause_Prediction_Research_Paper.md
│
├── docs/
│   ├── Methodology_Detailed.md
│   ├── Experimental_Results_Summary.md
│   ├── Data_Analysis_Report.md
│   └── Research_Analysis_and_Discussion.md
│
├── results/
│   └── Additional experimental outputs
│
└── codebase/
    │
    ├── README.md
    ├── requirements.txt
    │
    ├── config/
    │   └── config.yaml
    │
    ├── data/
    │   └── Generated / real datasets
    │
    ├── src/
    │   ├── data/
    │   │   ├── generator.py
    │   │   ├── preprocessing.py
    │   │   └── dataset.py
    │   │
    │   ├── models/
    │   │   ├── embeddings.py
    │   │   ├── classifier.py
    │   │   └── similarity.py
    │   │
    │   ├── pipeline/
    │   │   └── predictor.py
    │   │
    │   ├── utils/
    │   │   └── helpers.py
    │   │
    │   └── api/
    │       └── gradio_app.py
    │
    ├── scripts/
    │   ├── generate_data.py
    │   ├── train.py
    │   └── evaluate.py
    │
    ├── notebooks/
    │   └── 01_colab_quickstart.ipynb
    │
    └── tests/
```

The repository currently separates research material from the implementation, with dedicated `paper`, `docs`, `results`, and `codebase` areas. The implementation itself is organized into configuration, notebooks, scripts, source modules, and tests.

---

# 🛠️ Technology Stack

| Layer              | Technology                    |
| ------------------ | ----------------------------- |
| Language           | Python                        |
| NLP                | spaCy                         |
| Embeddings         | Sentence Transformers         |
| Default Model      | all-MiniLM-L6-v2              |
| ML                 | scikit-learn                  |
| Primary Classifier | Logistic Regression           |
| Alternative        | Random Forest                 |
| Vector Search      | FAISS                         |
| Fallback Search    | scikit-learn NearestNeighbors |
| Demo UI            | Gradio                        |
| Configuration      | YAML                          |
| Research           | Jupyter / Google Colab        |
| License            | MIT                           |

---

# 🚀 Quick Start

## 1. Clone

```bash
git clone https://github.com/upadrastaharshavardhan/ai-failure-root-cause-NLP.git

cd ai-failure-root-cause-NLP/codebase
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

# 🧪 Reproduce the Research

Generate the dataset:

```bash
python scripts/generate_data.py --n-samples 3000 --seed 42
```

Train the model:

```bash
python scripts/train.py
```

Evaluate:

```bash
python scripts/evaluate.py
```

The research documentation specifies seed `42` for data generation and experimental reproducibility.

---

# 🎨 Launch the Interactive Demo

```bash
python -m src.api.gradio_app
```

The Gradio interface provides an interactive way to paste a failure message and inspect the predicted root cause and similar historical failures.

---

# ☁️ Google Colab

The project is designed to support an end-to-end Google Colab workflow.

```python
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm

!python scripts/generate_data.py --n-samples 3000
!python scripts/train.py

!python -m src.api.gradio_app
```

GPU execution is recommended for faster embedding generation.

---

# 🔬 Research Documentation

The repository contains a complete research documentation layer.

### 📘 Research Paper

**AI-Based Software Failure Analysis and Root Cause Prediction using Natural Language Processing**

Available in:

```text
paper/
```

The repository contains both the Markdown source and PDF version of the research paper.

### 📐 Methodology

Detailed explanation of:

* data generation
* preprocessing
* regex normalization
* phrase extraction
* embedding strategy
* classification
* similarity search
* evaluation metrics
* reproducibility

→ `docs/Methodology_Detailed.md`

### 📊 Experimental Results

Contains:

* classification metrics
* per-class results
* retrieval metrics
* ablation study
* latency measurements
* statistical notes

→ `docs/Experimental_Results_Summary.md`

### 📈 Data Analysis

Dataset distribution, service distribution, noise characteristics and related analysis.

→ `docs/Data_Analysis_Report.md`

### 🧠 Research Discussion

Detailed interpretation of the experimental results, limitations, implications and future research directions.

→ `docs/Research_Analysis_and_Discussion.md`

---

# 🔍 Evaluation Methodology

The project evaluates two complementary capabilities.

## Classification

Given:

```text
Failure → Root Cause Category
```

Metrics include:

```text
Accuracy
Precision
Recall
Macro F1
Weighted F1
```

## Retrieval

Given:

```text
Failure → Similar Historical Failures
```

Metrics include:

```text
MRR@5
Recall@1
Recall@3
Recall@5
Precision@5
```

The metric definitions and evaluation methodology are documented in the repository's methodology documentation.

---

# 💡 Key Research Findings

### 1. Semantic embeddings outperform TF-IDF

The reported full system achieved **96.83% accuracy**, compared with **89.50%** for the TF-IDF + Logistic Regression baseline.

### 2. Preprocessing matters

Removing dynamic identifiers and timestamps improved the reported accuracy from **94.17% without preprocessing** to **96.83%** for the full configuration.

### 3. Larger embeddings can improve performance

The `all-mpnet-base-v2` configuration reached **97.33% accuracy**, compared with 96.83% for the default MiniLM configuration, at the cost of a larger/slower model.

### 4. Retrieval adds useful context

Even when classification confidence is not perfect, similar historical failures can provide additional evidence to an engineer.

### 5. Taxonomy boundaries remain challenging

The documented analysis identifies ambiguity around categories such as:

```text
Dependency ↔ Network
RaceCondition ↔ Database
```

These are examples where failure semantics can overlap.

---

# 🏭 From Research Prototype to Production

The current project is intentionally designed as an extensible research prototype.

A production implementation could replace the synthetic dataset with real enterprise telemetry:

```text
                  ┌──────────────────┐
                  │ Production Logs  │
                  └────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Log Ingestion      │
                 │ ELK / CloudWatch   │
                 │ Splunk / OpenLogs  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Failure Parser    │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ NLP Pipeline      │
                 └─────────┬─────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Root Cause Model      Vector Database
                 │                   │
                 └─────────┬─────────┘
                           ▼
                 ┌───────────────────┐
                 │ RCA Intelligence  │
                 └───────────────────┘
```

Potential production extensions include:

* ELK / Elasticsearch integration
* Splunk ingestion
* CloudWatch integration
* OpenTelemetry
* Chroma
* Pinecone
* Weaviate
* hierarchical root-cause taxonomy
* transformer fine-tuning
* SHAP explanations
* attention-based explanations
* human feedback loops
* drift detection
* online learning
* incident clustering
* automated incident summaries
* LLM-based RCA explanations
* multimodal telemetry analysis

Several of these extension directions are already identified in the implementation documentation.

---

# 🧭 Future Research Roadmap

```text
                    CURRENT
                       │
                       ▼
        ┌────────────────────────────┐
        │ Synthetic Failure Dataset  │
        └─────────────┬──────────────┘
                      │
                      ▼
        ┌────────────────────────────┐
        │ NLP Classification + RAG   │
        └─────────────┬──────────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
        Real Log Data      Better Models
              │                │
              ▼                ▼
       Domain Adaptation   Fine-Tuning
              │                │
              └───────┬────────┘
                      ▼
             Hierarchical RCA
                      │
                      ▼
             Multi-Service RCA
                      │
                      ▼
              Root Cause Graph
                      │
                      ▼
             Autonomous RCA
```

### Planned directions

* [ ] Replace synthetic failures with real production-like datasets
* [ ] Expand root-cause taxonomy
* [ ] Fine-tune domain-specific transformer models
* [ ] Add hierarchical classification
* [ ] Add incident clustering
* [ ] Add explainable AI
* [ ] Add temporal failure analysis
* [ ] Add service dependency graphs
* [ ] Add root-cause knowledge graph
* [ ] Add multimodal telemetry
* [ ] Add LLM-generated RCA explanations
* [ ] Add online learning
* [ ] Add model drift monitoring
* [ ] Evaluate on external benchmark datasets

---

# ⚠️ Limitations

This project should be interpreted as a **research prototype**, not a production incident-management system.

### Dataset limitation

The reported evaluation uses synthetically generated failures.

### Distribution limitation

Synthetic failure patterns may be cleaner and more separable than real production incidents.

### Taxonomy limitation

Only nine root-cause categories are currently evaluated.

### Generalization limitation

Performance on unseen organizations, applications, logging formats, programming languages, or infrastructure environments requires additional validation.

### Evaluation limitation

The reported primary metrics are based on a fixed held-out test set, with additional seed-based variance analysis documented in the experimental report.

---

# 🧑‍🔬 Reproducibility

The research emphasizes deterministic experimentation.

Key reproducibility controls include:

```text
Seed = 42
        │
        ├── Dataset generation
        ├── Train/test split
        ├── Model initialization
        └── Evaluation
```

Embedding generation is deterministic at inference, and configuration is externalized through YAML.

---

# 📚 Research Package

```text
paper/
    Research Paper PDF
    Research Paper Markdown

docs/
    Methodology
    Experimental Results
    Data Analysis
    Research Discussion

codebase/
    Complete implementation
    Training scripts
    Evaluation scripts
    Notebook
    Gradio demo
    Tests
```

This separation makes the repository useful for:

* academic review
* reproducibility
* experimentation
* ML research
* NLP research
* software engineering research
* AI-assisted QA research
* future production prototyping

---

# 🤝 Contributing

Contributions are welcome.

Potential contribution areas:

```text
Dataset
    ↓
Preprocessing
    ↓
Embedding Models
    ↓
Classifiers
    ↓
Retrieval
    ↓
Explainability
    ↓
Production Integrations
```

If you want to contribute:

```bash
git clone https://github.com/upadrastaharshavardhan/ai-failure-root-cause-NLP.git
cd ai-failure-root-cause-NLP
```

Create a branch:

```bash
git checkout -b feature/your-feature
```

Make your changes, add tests where appropriate, and open a pull request.

---

# 📜 License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

# 📖 Citation

If you use this project in academic or industrial research, please cite:

```text
Upadrasta Harsha Vardhan.

"AI-Based Software Failure Analysis and Root Cause Prediction
using Natural Language Processing."

Project 1 Research Documentation, August 2026.
```

### BibTeX

```bibtex
@misc{upadrasta2026failureanalysis,
  author       = {Upadrasta Harsha Vardhan},
  title        = {AI-Based Software Failure Analysis and Root Cause Prediction using Natural Language Processing},
  year         = {2026},
  month        = {August},
  note         = {Project 1 Research Documentation}
}
```

---

# 👨‍💻 Author

## Upadrasta Harsha Vardhan

AI / ML • NLP • Software Testing • Automation • Quality Engineering

This project explores the intersection of:

```text
Artificial Intelligence
        +
Natural Language Processing
        +
Machine Learning
        +
Software Engineering
        +
Quality Engineering
        +
Failure Analysis
```

---

# ⭐ If You Find This Useful

If this research is useful for your work or research:

**⭐ Star the repository**

**🍴 Fork the project**

**🧪 Reproduce the experiments**

**💡 Open an issue with ideas**

**🤝 Contribute improvements**

---

<p align="center">

<strong>From Failure Logs → Semantic Understanding → Root Cause Intelligence</strong>

<br><br>

Built for research, experimentation, and the future of AI-assisted software reliability.

</p>
