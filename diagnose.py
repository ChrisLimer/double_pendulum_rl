"""Behaviour of a checkpoint from hanging / random / upright starts (no pushes).

    uv run diagnose.py runs/<run> [ckpt_001234]     # default: latest checkpoint

Per start type: how often it reaches the top (and how fast), leaves the track (and
whether that happens before reaching the top), time upright, and how often it holds
the top for 5 s or more. Episodes stop counting once the cart leaves the track.
"""
import sys, jax, jax.numpy as jnp, numpy as np
from pathlib import Path
from dprl import runs
from dprl.networks import actor
run = Path(sys.argv[1]); _, cfg, env = runs.load_config(run)
path = runs.list_checkpoints(run)[-1] if len(sys.argv) < 3 else run / "checkpoints" / f"{sys.argv[2]}.npz"
params, meta = runs.load_checkpoint(path, cfg, env)
print(path.stem, meta["eval"], f"x_limit {env.x_limit} max_force {env.max_force}")
def ep(key, mix):
    s0 = env.reset(key, mix=mix)
    def step(c, _):
        s, off = c
        m, _ = actor(params, env.obs(s)); s2, r, term, _ = env.step(s, m)
        off = off | term
        return (s2, off), (s2.q, s2.qd, env.is_upright(s2.q) & ~off, off)
    _, out = jax.lax.scan(step, (s0, False), None, length=env.max_steps); return out
for name, mix in [("hanging", (1., 0., 0.)), ("random", (0., 1., 0.)), ("upright", (0., 0., 1.))]:
    q, qd, up, off = map(np.asarray, jax.vmap(lambda k: ep(k, mix))(jax.random.split(jax.random.key(7), 128)))
    reached = up.any(1)
    first = np.array([np.argmax(u) for u in up])
    def streak(u):
        best = cur = 0
        for v in u: cur = cur + 1 if v else 0; best = max(best, cur)
        return best
    st = np.array([streak(u) for u in up]) * env.dt
    off_any = off[:, -1]
    off_before_top = off_any & (~reached | (np.argmax(off, 1) < first))
    print(f"{name:8s} reach top {reached.mean():4.0%} (t~{np.median(first[reached]) * env.dt if reached.any() else np.nan:.1f}s) | "
          f"off-track {off_any.mean():4.0%} (before top {off_before_top.mean():4.0%}) | upright {up.mean():4.0%} | "
          f"streak>=5s {np.mean(st >= 5):4.0%} | last5s upright {up[:, -env.steps(5.0):].mean():4.0%}")
