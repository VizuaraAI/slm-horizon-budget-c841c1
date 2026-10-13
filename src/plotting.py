"""Paper figures, rebuilt from the summary.json files that evaluate.py writes.

    python -m src.plotting --summaries results/*/summary.json --out figures/

Produces:
  requirement_curve.png   measured competence against the inverted requirement
  intervention_ledger.png reachable horizon per intervention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .horizon_model import HorizonModel, naive_required_competence  # noqa: E402

SLATE = "#4a5568"
AMBER = "#d97706"
BLUE = "#2563eb"


def load_summaries(paths: list[str]) -> dict[str, dict]:
    out = {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        name = data.get("config", {}).get("io", {}).get("run_name", Path(path).parent.name)
        out[name] = data
    return out


def plot_requirement_curve(summaries: dict[str, dict], out_path: Path, target: float = 0.5) -> None:
    horizons = [8, 16, 32, 64, 128]
    naive = [naive_required_competence(h, target) for h in horizons]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(horizons, naive, "--", color=SLATE, label=r"required: naive $T^{1/H}$")

    for name, data in summaries.items():
        fit = data["fit"]
        model = HorizonModel(
            p=fit["p_hat"], delta=fit["delta"], recovery=fit["recovery"], het=fit["het"]
        )
        corrected = [model.required_competence(h, target) for h in horizons]
        ax.plot(horizons, corrected, "-", color=AMBER, label=f"required: fitted ({name})")
        ax.axhline(fit["p_hat"], ls=":", color=BLUE, lw=1.0)
        ax.text(
            horizons[-1], fit["p_hat"], f"  measured p = {fit['p_hat']:.3f}",
            va="center", fontsize=8, color=BLUE,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("task horizon H (dependent steps)")
    ax.set_ylabel("per-step competence")
    ax.set_title(f"Required vs. measured per-step competence (T = {target})")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_intervention_ledger(summaries: dict[str, dict], out_path: Path) -> None:
    """Reachable horizon per run, annotated with what it cost per step.

    The cost column is the point of the ledger: a run that reaches a longer
    horizon by spending four forward passes per step is not obviously better
    than one that reaches a shorter horizon on one pass.
    """
    names, horizons, costs = [], [], []
    for name, data in sorted(summaries.items()):
        names.append(name)
        horizons.append(float(data.get("h50_observed", 0.0)))
        costs.append(float(data.get("mean_calls_per_step", 1.0)))

    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(names) + 1.8))
    bars = ax.barh(names, horizons, color=BLUE, height=0.55)
    span = max(horizons) if horizons else 1.0
    for bar, value, cost in zip(bars, horizons, costs):
        ax.text(bar.get_width() + 0.02 * span, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}  ({cost:.1f}x/step)", va="center", fontsize=7)
    ax.set_xlim(0, span * 1.45)
    ax.invert_yaxis()
    ax.set_xlabel("reachable horizon at 50% end-to-end success")
    ax.set_title("Horizon bought by each intervention")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", required=True)
    parser.add_argument("--out", default="figures")
    parser.add_argument("--target", type=float, default=0.5)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(args.summaries)

    plot_requirement_curve(summaries, out_dir / "requirement_curve.png", args.target)
    plot_intervention_ledger(summaries, out_dir / "intervention_ledger.png")
    print(f"[plot] wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
