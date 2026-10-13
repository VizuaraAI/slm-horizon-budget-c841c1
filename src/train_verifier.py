"""Train the step-level verifier on traces you have already logged.

This is the one thing in the repo that is trained rather than measured. It is a
plain logistic regression on cheap run-time features (see
``interventions.step_features``), fit by gradient descent in numpy so there is
no extra dependency and nothing is hidden.

    python -m src.train_verifier --traces results/baseline/traces.jsonl \
        --out results/baseline/verifier.npz

Then point the config at it:

    python -m src.evaluate --intervention.kind verifier \
        --intervention.verifier_path results/baseline/verifier.npz

Why not let the model grade itself? Intrinsic self-correction without an
external signal is known to be unreliable, so the verifier is trained against
ground-truth step labels from logged traces instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .interventions import N_FEATURES


def features_from_record(step: dict) -> np.ndarray:
    """Rebuild the run-time feature vector from a logged step.

    Mirrors ``interventions.step_features`` exactly. Keep the two in sync: a
    verifier trained on features the agent cannot compute at run time is worse
    than no verifier at all.
    """
    answer = step["answer"]
    token = answer.strip().split()[-1].lower() if answer.strip() else ""
    horizon = max(int(step["horizon"]), 1)
    return np.array(
        [
            1.0,
            int(step["index"]) / horizon,
            min(len(answer), 200) / 200.0,
            1.0 if token else 0.0,
            1.0 if len(answer.split()) <= 3 else 0.0,
            1.0 if token.isdigit() else 0.0,
            0.0,
        ],
        dtype=np.float64,
    )


def load_dataset(trace_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for path in trace_paths:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                for step in json.loads(line)["steps"]:
                    rows.append(features_from_record(step))
                    labels.append(1.0 if step["step_correct"] else 0.0)
    if not rows:
        raise SystemExit("no steps found in the given traces")
    return np.vstack(rows), np.asarray(labels, dtype=np.float64)


def fit_logistic(
    x: np.ndarray, y: np.ndarray, epochs: int = 500, lr: float = 0.5, l2: float = 1e-3
) -> np.ndarray:
    w = np.zeros(x.shape[1], dtype=np.float64)
    n = x.shape[0]
    for _ in range(epochs):
        preds = 1.0 / (1.0 + np.exp(-x @ w))
        grad = x.T @ (preds - y) / n + l2 * w
        w -= lr * grad
    return w


def evaluate(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    preds = 1.0 / (1.0 + np.exp(-x @ w))
    hard = (preds >= 0.5).astype(np.float64)
    accuracy = float((hard == y).mean())
    base_rate = float(y.mean())
    # Brier score is the honest calibration number for a gate that will be
    # thresholded: accuracy alone hides an overconfident verifier.
    brier = float(np.mean((preds - y) ** 2))
    return {"accuracy": accuracy, "base_rate": base_rate, "brier": brier}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the step-level verifier.")
    parser.add_argument("--traces", nargs="+", required=True, help="one or more traces.jsonl")
    parser.add_argument("--out", default="results/verifier.npz")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    x, y = load_dataset(args.traces)
    assert x.shape[1] == N_FEATURES, "feature width drifted from interventions.step_features"

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(y))
    split = int(len(y) * (1.0 - args.val_fraction))
    train, val = order[:split], order[split:]

    w = fit_logistic(x[train], y[train], epochs=args.epochs, lr=args.lr, l2=args.l2)

    print("[verifier] train", evaluate(w, x[train], y[train]))
    if len(val):
        print("[verifier] val  ", evaluate(w, x[val], y[val]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, weights=w)
    print(f"[verifier] saved to {out}")
    print(
        "[verifier] note: if validation accuracy is close to the base rate, the "
        "features\n           carry no signal for this model. Add a real one "
        "(answer token logprob)\n           before concluding that verifier gating does not help."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
