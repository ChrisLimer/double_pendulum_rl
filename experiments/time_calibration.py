"""How good is the time-to-stable-upright prediction (critic heads 1-2)?

    uv run experiments/time_calibration.py runs/<group>/<run> [...]

Runs the deterministic policy from random and perturbed-hanging starts (real physics), and at
every step before the first stable-upright state compares the predicted time T_hat (and its
spread sigma) with the actual remaining time T. Reports the error at the episode start and over
all steps, and the calibration of the dashboard's "chance within +-0.5 s"
(erf(0.5 / (sigma sqrt 2))): the predicted chance against how often |T - T_hat| < 0.5 s really is.
"""
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dprl import runs  # noqa: E402
from dprl.env import stable_obs  # noqa: E402
from dprl.networks import actor, critic_outputs  # noqa: E402
from dprl.ppo import TIME_MAX  # noqa: E402

SPAN = 0.5


def analyse(run: Path, n=256):
    _, cfg, env = runs.load_config(run)
    params, meta = runs.load_checkpoint(runs.list_checkpoints(run)[-1], cfg, env)
    if not cfg.time_head:
        print(run.name, "has no time head")
        return

    def episode(key, mix, noise):
        s0 = env.reset(key, mix=jnp.array(mix), noise=noise)

        def step(s, _):
            o = env.obs(s)
            out = critic_outputs(params, o)
            mean = jnp.clip(out[1] * TIME_MAX, 0, TIME_MAX)
            std = jnp.sqrt(jnp.maximum(out[2] * TIME_MAX**2 - mean**2, 0.0))
            a, _ = actor(params, o)
            s2, _, term, _ = env.step(s, a)
            return s2, (mean, std, stable_obs(o), term)

        _, (mean, std, stable, term) = jax.lax.scan(step, s0, None, length=env.max_steps)
        return mean, std, stable, term

    print(f"== {run.relative_to(ROOT / 'runs')} ({meta['env_steps'] / 1e6:.0f}M steps)")
    for name, mix, noise in [("random", (0.0, 1.0, 0.0), 0.0), ("hanging, perturbed", (1.0, 0.0, 0.0), 1.0)]:
        mean, std, stable, term = map(np.asarray, jax.jit(jax.vmap(lambda k: episode(k, mix, noise)))(
            jax.random.split(jax.random.key(3), n)))
        reached = stable.any(1) & ~(term.cumsum(1) > 0).any(1)
        first = np.where(stable.any(1), stable.argmax(1), env.max_steps)
        start_err, pred, actual, conf = [], [], [], []
        for i in np.where(reached)[0]:
            remaining = (first[i] - np.arange(first[i])) * env.dt
            pred.append(mean[i, :first[i]]); actual.append(remaining)
            conf.append([math.erf(SPAN / (max(s, 1e-3) * math.sqrt(2))) for s in std[i, :first[i]]])
            start_err.append(mean[i, 0] - first[i] * env.dt)
        pred, actual, conf = np.concatenate(pred), np.concatenate(actual), np.concatenate(conf)
        hit = np.abs(pred - actual) < SPAN
        start_err = np.array(start_err)
        print(f"  {name}: reached stable upright in {reached.mean():.0%}; actual time from the start "
              f"median {np.median(first[reached]) * env.dt:.2f} s")
        print(f"    at the start: prediction error median {np.median(start_err):+.2f} s, "
              f"mean |error| {np.abs(start_err).mean():.2f} s")
        print(f"    all steps before stable: mean |error| {np.abs(pred - actual).mean():.2f} s, "
              f"within +-{SPAN} s {hit.mean():.0%} of the time")
        for lo, hi in [(0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]:
            sel = (conf >= lo) & (conf < hi)
            if sel.sum() > 50:
                print(f"    predicted chance {lo:.0%}-{min(hi, 1):.0%}: actually within +-{SPAN} s "
                      f"{hit[sel].mean():.0%} ({sel.mean():.0%} of steps)")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        analyse(Path(r).resolve())
