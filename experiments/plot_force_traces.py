"""Figure: the cart force while balancing, continuous vs discrete policies (docs/img/force_traces.png).

    uv run --with matplotlib experiments/plot_force_traces.py

One near-upright start (same for all), deterministic policies, real physics; 2 s after settling.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dprl import runs  # noqa: E402
from dprl.networks import actor, actor_mode  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def trace(run_glob, mode):
    run = sorted((ROOT / "runs").glob(run_glob))[0]
    _, cfg, env = runs.load_config(run)
    params, _ = runs.load_checkpoint(runs.list_checkpoints(run)[-1], cfg, env)
    s0 = env.reset(jax.random.key(11), mix=jnp.array([0.0, 0.0, 1.0]))

    def step(s, _):
        o = env.obs(s)
        a = actor_mode(params, o) if mode else actor(params, o)[0]
        applied, _ = env.apply_shield(s.q, s.qd, jnp.clip(a[0], -1, 1))
        s2, _, _, _ = env.step(s, a)
        return s2, (applied * env.max_force, s2.q[1], s2.q[2])

    _, (force, th1, th2) = jax.lax.scan(step, s0, None, length=env.steps(7.0))
    window = slice(env.steps(5.0), None)
    t = np.arange(env.steps(7.0))[window] * env.dt
    return t, np.asarray(force)[window], np.degrees(np.asarray(th1)[window]), np.degrees(np.asarray(th2)[window])


def main():
    cases = [("continuous (Gaussian mean)", "sym/*_sym_s1", False),
             ("3 levels, most likely level", "discrete/*_bins3_s2", True),
             ("3 levels, expected force", "discrete/*_bins3_s2", False)]
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)
    for ax, (title, pattern, mode), color in zip(axes, cases, SERIES):
        t, force, th1, th2 = trace(pattern, mode)
        ax.step(t, force, where="post", color=color, linewidth=1.4)
        ax.set_ylim(-32, 32)
        ax.set_ylabel("cart force [N]")
        ax.set_title(f"{title} · mean |F| {np.abs(force).mean():.1f} N · "
                     f"angles within {max(np.abs(th1).max(), np.abs(th2).max()):.2f}°")
    axes[-1].set_xlabel("time [s] (balancing, after settling)")
    out = ROOT / "docs" / "img" / "force_traces.png"
    fig.savefig(out, dpi=160)
    print("written", out)


if __name__ == "__main__":
    main()
