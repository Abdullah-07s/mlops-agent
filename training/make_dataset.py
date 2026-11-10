"""Generate the synthetic dataset:  python -m training.make_dataset --rows 500"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

from training.train import FEATURES, TARGET


def make_dataset(path: str | Path, rows: int = 500, seed: int = 42) -> tuple[int, int]:
    X, y = make_classification(
        n_samples=rows, n_features=len(FEATURES), n_informative=3, n_redundant=0, random_state=seed
    )
    df = pd.DataFrame(X, columns=FEATURES)
    df[TARGET] = y
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df.shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/train.csv")
    args = parser.parse_args()
    print("Dataset generated:", make_dataset(args.out, args.rows, args.seed))


if __name__ == "__main__":
    main()
