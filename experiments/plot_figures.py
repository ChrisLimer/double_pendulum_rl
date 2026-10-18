"""Figures for docs/experiments.md (written to docs/img/).

    uv run --with matplotlib experiments/plot_figures.py

Reads experiments/results/{results.jsonl, robust_table.json} and the metrics/checkpoints of the
final run in runs/ (skipped if that run is not on this machine).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
IMG = ROOT / "docs" / "img"
FINAL_RUN = ROOT / "runs" / "2026-10-18_16-35-06"
sys.path.insert(0, str(ROOT))

# Reference palette (light mode): categorical slots in fixed order, chart chrome and ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def load_results():
    return [json.loads(line) for line in open(RES / "results.jsonl")]


def seeds(results, name):
    return sorted((r for r in results if r["name"] == name), key=lambda r: r["seed"])


def strip(ax, groups, title):
    """Per-seed success as dots, one column per configuration, with the mean as a short bar."""
    for i, (label, values) in enumerate(groups):
        x = i + np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([i])
        ax.scatter(x, values, s=42, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.hlines(np.mean(values), i - 0.25, i + 0.25, color=INK2, linewidth=2, zorder=2)
        ax.annotate(f"mean {np.mean(values):.0%}", (i + 0.28, np.mean(values)), va="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.5, len(groups) - 0.2)
    ax.set_ylim(-0.04, 1.04)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("swing-up success (evaluator)")
    ax.grid(axis="x", visible=False)
    ax.set_title(title)


def fig_seed_comparisons(results):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    strip(a, [("baseline", [r["success"] for r in seeds(results, "X_base")]),
              ("track bonus + es3\n+ upright-start signal", [r["success"] for r in seeds(results, "N_new")])],
          "Round 1 · 50 Hz, 60M steps, 6 seeds")
    strip(b, [("2 hidden layers", [r["success"] for r in seeds(results, "H_2L")]),
              ("3 hidden layers", [r["success"] for r in seeds(results, "H_3L")])],
          "Round 2 · 100 Hz, still starts, 60M steps, 4 seeds")
    fig.savefig(IMG / "seed_comparisons.png", dpi=160)


def fig_robustness():
    table = json.loads((RES / "robust_table.json").read_text())
    groups = ["3 layers", "+ easier physics", "+ start noise", "final run"]
    cases = [("hang_still", "hanging still"), ("hang_perturbed", "hanging, perturbed"),
             ("random", "random start"), ("push0.1_from_up", "pushes, level 0.1"), ("push0.2_from_up", "pushes, level 0.2")]
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    width = 0.19
    for gi, group in enumerate(groups):
        members = list(table[group].values())
        for ci, (case, _) in enumerate(cases):
            vals = [m[case]["success"] for m in members]
            x = ci + (gi - 1.5) * (width + 0.01)
            ax.bar(x, np.mean(vals), width, color=SERIES[gi], edgecolor=SURFACE, linewidth=1,
                   label=group if ci == 0 else None, zorder=2)
            if len(vals) > 1:  # seed range
                ax.vlines(x, min(vals), max(vals), color=INK2, linewidth=1.2, zorder=3)
    ax.set_xticks(range(len(cases)), [c[1] for c in cases])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("success (256 episodes per case)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9)
    ax.set_title("Robustness at 100 Hz · bars: mean over seeds (4, or 1 for the final run) · lines: seed range", pad=28)
    fig.savefig(IMG / "robustness.png", dpi=160)


def fig_final_run():
    if not (FINAL_RUN / "metrics.jsonl").exists():
        print("final run not found; skipping final_run_training.png")
        return
    from dprl import runs
    rows = [json.loads(line) for line in open(FINAL_RUN / "metrics.jsonl")]
    steps = np.array([r["env_steps"] for r in rows]) / 1e6

    def ema(key, alpha=0.02):
        out, v = [], np.nan
        for r in rows:
            x = r.get(key)
            if x is not None:
                v = x if np.isnan(v) else (1 - alpha) * v + alpha * x
            out.append(v)
        return np.array(out)

    ckpts = [runs.read_checkpoint_meta(p) for p in runs.list_checkpoints(FINAL_RUN)]
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    a.plot(steps, ema("swingup_success"), color=SERIES[0], linewidth=2, label="training episodes (smoothed)")
    a.scatter([c["env_steps"] / 1e6 for c in ckpts], [c["eval"]["success"] for c in ckpts], s=16,
              color=SERIES[1], edgecolor=SURFACE, linewidth=0.8, zorder=3, label="checkpoint test, hanging still")
    a.set_ylim(-0.03, 1.05)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a.set_ylabel("swing-up success")
    a.legend(loc="lower right", fontsize=9)
    a.set_title("Final run · 200M steps · swing-up success")
    for i, (key, label) in enumerate([("upright_share", "near-upright start share"), ("start_noise", "start noise"),
                                      ("push_level", "push level"), ("energy_weight", "energy shaping weight")]):
        y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        b.plot(steps, y, color=SERIES[i], linewidth=2, label=label)
    ease = np.array([1.0 if c.get("ease") is None else c["ease"] for c in ckpts])
    b.plot([c["env_steps"] / 1e6 for c in ckpts], ease, color=INK2, linewidth=1.5, linestyle="--", label="physics ease")
    b.set_ylim(-0.03, 1.05)
    b.set_xlim(0, steps[-1])
    b.set_xlabel("environment steps (millions)")
    b.set_ylabel("level")
    b.legend(loc="center right", fontsize=9, ncols=1)
    b.set_title("Curriculum")
    fig.savefig(IMG / "final_run_training.png", dpi=160)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    results = load_results()
    fig_seed_comparisons(results)
    fig_robustness()
    fig_final_run()
    print("written to", IMG)
