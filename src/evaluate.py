"""Evaluation entry point: run a horizon sweep, fit the model, invert it.

    python -m src.evaluate --config configs/default.yaml
    python -m src.evaluate --config configs/default.yaml \
        --model.backend hf --model.name Qwen/Qwen2.5-1.5B-Instruct \
        --task.family fsm --intervention.kind vote --io.run_name fsm_vote
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import parse_config
from .estimate import bootstrap_fit, summarize
from .harness import run_sweep, write_records
from .horizon_model import naive_required_competence
from .interventions import build_intervention
from .solvers import build_solver


def main(argv: list[str] | None = None) -> int:
    cfg = parse_config(argv)
    cfg.setdefault("model", {})["seed"] = cfg.get("seed", 0)

    solver = build_solver(cfg["model"])
    intervention = build_intervention(cfg.get("intervention", {}))

    run_name = cfg["io"]["run_name"]
    results_dir = Path(cfg["io"]["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {run_name}  family={cfg['task']['family']}  "
          f"backend={cfg['model']['backend']}  intervention={intervention.name}")

    records = run_sweep(cfg, solver, intervention)
    traces_path = write_records(records, results_dir / "traces.jsonl")
    print(f"[run] wrote {len(records)} episodes to {traces_path}")

    raw = [json.loads(line) for line in open(traces_path, encoding="utf-8")]
    report = summarize(
        raw,
        targets=cfg["estimate"]["targets"],
        query_horizons=cfg["estimate"]["query_horizons"],
    )
    report["ci"] = bootstrap_fit(
        raw, n_resamples=int(cfg["estimate"]["bootstrap_resamples"]), seed=cfg.get("seed", 0)
    )
    report["config"] = {k: v for k, v in cfg.items() if not k.startswith("_")}

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_report(report)
    print(f"\n[run] summary written to {summary_path}")
    return 0


def print_report(report: dict) -> None:
    fit = report["fit"]
    ci = report.get("ci", {})

    def interval(key: str) -> str:
        lo, hi = ci.get(key, (float("nan"), float("nan")))
        return f"[{lo:.4f}, {hi:.4f}]"

    print("\n--- fitted horizon model -------------------------------------")
    print(f"  p_hat    (clean prefix accuracy)     {fit['p_hat']:.4f}  {interval('p_hat')}")
    print(f"  p_deg    (degraded prefix accuracy)  {fit['p_deg']:.4f}  {interval('p_deg')}")
    print(f"  delta    (self-conditioning penalty) {fit['delta']:.4f}  {interval('delta')}")
    print(f"  recovery (repair probability)        {fit['recovery']:.4f}  {interval('recovery')}")
    print(f"  het      (per-step spread, logit)    {fit['het']:.4f}  {interval('het')}")
    print(f"  steps    clean={fit['n_clean_steps']}  degraded={fit['n_degraded_steps']}")

    print("\n--- end-to-end success by horizon ----------------------------")
    for horizon, success in report["success_by_horizon"].items():
        print(f"  H={int(horizon):>4}   S={success:.3f}")

    print("\n--- horizon at 50% success -----------------------------------")
    print(f"  observed                  {report['h50_observed']:.2f}")
    print(f"  predicted by naive p^H    {report['h50_predicted_naive']:.2f}")
    print(f"  predicted by fitted model {report['h50_predicted_model']:.2f}")

    print("\n--- required per-step competence (the inversion) -------------")
    print(f"  {'T':>5} {'H':>5} {'naive p*':>10} {'model p*':>10} {'gap (pp)':>10}")
    for row in report["requirements"]:
        print(
            f"  {row['target']:>5.2f} {int(row['horizon']):>5} "
            f"{row['p_star_naive']:>10.4f} {row['p_star_model']:>10.4f} "
            f"{row['gap_pp']:>10.2f}"
        )
    print(
        "\n  A positive gap means the naive arithmetic asks for too little "
        "per-step\n  competence, so a model sized with T^(1/H) will miss the target."
    )


def quick_check() -> None:
    """Sanity check the inversion without running any model."""
    for horizon in (8, 16, 32, 64, 128):
        print(f"H={horizon:>4}  naive p* at T=0.5: {naive_required_competence(horizon, 0.5):.4f}")


if __name__ == "__main__":
    sys.exit(main())
