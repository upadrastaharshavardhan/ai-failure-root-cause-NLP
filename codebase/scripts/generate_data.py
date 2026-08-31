#!/usr/bin/env python
"""Generate synthetic failure dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.generator import generate_dataset
from src.utils.helpers import load_config, ensure_dirs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic failure data")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_samples = args.n_samples or cfg["data"]["n_samples"]
    seed = args.seed or cfg["data"]["random_seed"]
    output = args.output or cfg["paths"]["raw_data"]

    ensure_dirs(Path(output).parent)

    print(f"Generating {n_samples} samples (seed={seed})...")
    df = generate_dataset(n_samples=n_samples, seed=seed, categories=cfg.get("categories"))
    df.to_csv(output, index=False)
    print(f"Saved → {output}")
    print("\nCategory distribution:")
    print(df["root_cause_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
