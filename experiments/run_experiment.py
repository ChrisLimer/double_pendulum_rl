"""One experiment: train with config overrides (no run folder, no dashboard), then evaluate.

    uv run experiments/run_experiment.py <name> <seed> <steps> '{"env": {...}, "ppo": {...}, "cur": {...}}'

Appends a JSON line to experiments/results/results.jsonl (checkpoint evaluator, off-track before
the top, robust_eval cases) and saves the final weights to experiments/results/params/<name>_<seed>.pkl.
Run many with experiments/queue.sh.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from dprl.env import DoublePendulumCart
from dprl.curriculum import Curriculum
from dprl.ppo import PPOConfig, make_trainer, make_evaluator
from dprl.networks import actor
name, seed, steps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
ov = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
ppo = ov.get("ppo", {})
if "hidden" in ppo: ppo["hidden"] = tuple(ppo["hidden"])
cfg = PPOConfig(seed=seed, total_timesteps=int(steps), **ppo)
env = DoublePendulumCart(shaping_gamma=cfg.gamma, **ov.get("env", {}))
cur = Curriculum(**ov.get("cur", {}))
init, chunk = make_trainer(cfg, env)
runner = init(jax.random.key(seed))
evaluate = make_evaluator(DoublePendulumCart(), 128)
upd, t0 = 0, time.time()
hist = []
while upd < cfg.num_updates:
    knobs = cur.knobs(upd / cfg.num_updates)
    runner, m = chunk(runner, knobs); m = jax.device_get(m)
    upd += cfg.updates_per_chunk
    ul = float(np.nanmean(np.asarray(m["swingup_success"], dtype=float)))
    cur.update(ul)
    if upd % 2000 == 0:
        hist.append((upd * cfg.steps_per_update, round(ul, 2), round(cur.upright_share, 2), round(cur.push_level, 2), round(cur.ease, 2), round(float(m["action_std"][-1]), 3)))
ev = {k: float(v) for k, v in jax.device_get(evaluate(runner.params, jax.random.key(123))).items()}
# off-track before top from hanging starts
envd = DoublePendulumCart()
def ep(key):
    s0 = envd.reset(key, mix=(1., 0., 0.))
    def step(c, _):
        s, off, up = c
        a, _ = actor(runner.params, envd.obs(s)); s2, _, term, _ = envd.step(s, a)
        up = up | (envd.is_upright(s2.q) & ~off)
        offb = off | term
        return (s2, offb, up), (term & ~off & ~up)
    (_, off, up), ob = jax.lax.scan(step, (s0, False, False), None, length=envd.max_steps)
    return off, up, ob.any()
off, up, ob = jax.jit(jax.vmap(ep))(jax.random.split(jax.random.key(7), 128))
ev.update(off=float(off.mean()), reach=float(up.mean()), off_before_top=float(ob.mean()),
          minutes=(time.time() - t0) / 60, final_cur=(cur.upright_share, cur.push_level, cur.ease), hist=hist[-4:])
(OUT / "params").mkdir(parents=True, exist_ok=True)
pickle.dump(jax.device_get(runner.params), open(OUT / "params" / f"{name}_{seed}.pkl", "wb"))
from robust_eval import make_robust  # noqa: E402
ev["robust"] = make_robust(DoublePendulumCart())(runner.params)
print(name, seed, json.dumps(ev))
with open(OUT / "results.jsonl", "a") as f:
    f.write(json.dumps({"name": name, "seed": seed, "steps": steps, "ov": ov, **ev}) + "\n")
