# Research Package – AI-Based Software Failure Analysis & Root Cause Prediction

**Project 1 – Complete Research Documentation Bundle**

This archive contains the full research paper, experimental results, methodology documentation, data analysis, and supporting materials for the NLP-based Root Cause Prediction system.

---

## Contents

```
research-paper-project1/
├── README.md                                      # This file
├── paper/
│   ├── AI_Failure_Root_Cause_Prediction_Research_Paper.pdf   # Main research paper (PDF)
│   └── AI_Failure_Root_Cause_Prediction_Research_Paper.md    # Source Markdown
├── docs/
│   ├── Methodology_Detailed.md
│   ├── Experimental_Results_Summary.md
│   ├── Data_Analysis_Report.md
│   └── Research_Analysis_and_Discussion.md
├── results/                                       # Place for additional plots / tables
├── figures/                                       # Place for additional figures
└── supplementary/
    └── (optional extra materials)
```

---

## Main Research Paper

**Title**: AI-Based Software Failure Analysis and Root Cause Prediction using Natural Language Processing

**Key Reported Metrics** (on 600-sample held-out test set):

| Metric              | Value    |
|---------------------|----------|
| Accuracy            | **96.83%** |
| Macro F1-Score      | **0.967** |
| MRR@5 (retrieval)   | **0.912** |
| Recall@5            | **0.978** |

The paper includes:

- Full problem motivation and related work
- Detailed methodology (preprocessing, embeddings, classification, similarity search)
- Complete experimental setup and dataset description
- Per-class metrics, confusion analysis, ablation study
- Latency measurements
- Qualitative error analysis
- Discussion of limitations, threats to validity, and future work
- Reproducibility appendix

---

## Supporting Documentation

1. **Methodology_Detailed.md** – Algorithms, regex patterns, model settings, metric definitions.
2. **Experimental_Results_Summary.md** – All tables of results in one place.
3. **Data_Analysis_Report.md** – Dataset statistics, class/service distribution, noise characteristics.
4. **Research_Analysis_and_Discussion.md** – Deeper scientific discussion and practical implications.

---

## How to Reproduce the Experiments

The accompanying software codebase (provided separately as `ai-failure-root-cause-project1.zip`) contains the full implementation. After installing dependencies:

```bash
python scripts/generate_data.py --n-samples 3000 --seed 42
python scripts/train.py
python scripts/evaluate.py
```

All reported numbers are reproducible with seed 42.

---

## Citation

If you use this work in academic or industrial research, please cite:

> AI-Based Software Failure Analysis and Root Cause Prediction using Natural Language Processing. Project 1 Research Documentation, August 2026.

---

## Contact / Extension

This package is designed to be self-contained for review and further research. The modular codebase allows easy replacement of the synthetic generator with real log data, swapping of embedding models, and addition of hierarchical classification or multi-modal features.
