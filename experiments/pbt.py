"""Population-based training (the exploit half of Jaderberg et al. 2017) with a vmapped population.

    uv run experiments/pbt.py <name> <pop> <steps_per_member> <interval_steps> '{"env": .., "ppo": .., "cur": ..}'

The whole population trains in one process (jax.vmap over the PPO runner). Every interval the
members are scored with robust_eval (success from still + perturbed hanging starts, half weight for
random starts, minus 0.02 per second of settling time) and the bottom quarter is replaced by copies
of the top quarter (weights, optimizer state and curriculum), with fresh random keys.
At the end a greedy weight soup is tried: members are averaged into the best one, best first, while
the score does not drop. Results go to experiments/results/pbt.jsonl and params/.
Don't run it with the single-thread XLA_FLAGS from queue.sh: the vmapped population needs threads.
"""
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from dprl.env import DoublePendulumCart
from dprl.curriculum import Curriculum
from dprl.ppo import PPOConfig, make_trainer
from robust_eval import make_robust

name, pop, steps, interval = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
ov = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
ppo = ov.get("ppo", {})
if "hidden" in ppo: ppo["hidden"] = tuple(ppo["hidden"])
cfg = PPOConfig(seed=0, total_timesteps=int(steps), **ppo)
env = DoublePendulumCart(shaping_gamma=cfg.gamma, **ov.get("env", {}))
curs = [Curriculum(**ov.get("cur", {})) for _ in range(pop)]
init, chunk = make_trainer(cfg, env)
runners = jax.vmap(init)(jax.random.split(jax.random.key(1000 + hash(name) % 1000), pop))
vchunk = jax.jit(jax.vmap(chunk))
robust = make_robust(DoublePendulumCart(), n=64)
evaluator_env = DoublePendulumCart()

def score(params):
    r = robust(params)
    s = r["hang_still"]["success"] + r["hang_perturbed"]["success"] + 0.5 * r["random"]["success"]
    s -= 0.02 * r["hang_still"]["t_settled_med"]  # quicker settling is better
    return s, r

def stack_knobs(progress):
    ks = [c.knobs(progress) for c in curs]
    return jax.tree.map(lambda *x: jnp.stack(x), *ks)

upd, t0, log = 0, time.time(), []
every = max(1, int(interval / cfg.steps_per_update / cfg.updates_per_chunk))
n_chunk = 0
while upd < cfg.num_updates:
    runners, m = vchunk(runners, stack_knobs(upd / cfg.num_updates))
    m = jax.device_get(m)
    upd += cfg.updates_per_chunk; n_chunk += 1
    for i, c in enumerate(curs):
        c.update(float(np.nanmean(np.asarray(m["swingup_success"][i], dtype=float))))
    if n_chunk % every == 0 and upd < cfg.num_updates:
        scores = [score(jax.tree.map(lambda x: x[i], runners.params))[0] for i in range(pop)]
        order = np.argsort(scores)[::-1]
        q = max(1, pop // 4)
        src, dst = order[:q], order[-q:]
        idx = np.arange(pop); idx[dst] = src
        keys = jax.random.split(jax.random.key(upd), pop)
        runners = jax.tree.map(lambda x: x[idx], runners)._replace(key=keys)
        curs = [copy.deepcopy(curs[j]) for j in idx]
        log.append({"steps": upd * cfg.steps_per_update, "scores": [round(float(s), 3) for s in scores],
                    "copied": {int(d): int(s) for s, d in zip(src, dst)}})
        print(name, json.dumps(log[-1]), f"{(time.time() - t0) / 60:.0f} min", flush=True)

members = [jax.tree.map(lambda x: x[i], runners.params) for i in range(pop)]
results = [score(p) for p in members]
order = np.argsort([s for s, _ in results])[::-1]
# greedy soup: add members (best first) to a weight average while the score does not drop
soup, used, best = members[order[0]], [int(order[0])], results[order[0]][0]
for j in order[1:]:
    cand = jax.tree.map(lambda *x: sum(x) / (len(used) + 1), *[members[k] for k in used + [int(j)]])
    s, _ = score(cand)
    if s >= best:
        soup, used, best = cand, used + [int(j)], s
soup_score, soup_res = score(soup)
out = {"name": name, "pop": pop, "steps": steps, "ov": ov, "minutes": (time.time() - t0) / 60,
       "member_scores": [round(float(s), 3) for s, _ in results],
       "best_member": int(order[0]), "best": results[order[0]][1],
       "soup_members": used, "soup_score": round(float(soup_score), 3), "soup": soup_res,
       "curricula": [(round(c.upright_share, 2), round(c.push_level, 2), round(c.ease, 2)) for c in curs], "log": log}
(OUT / "params").mkdir(parents=True, exist_ok=True)
pickle.dump(jax.device_get(members[order[0]]), open(OUT / "params" / f"{name}_best.pkl", "wb"))
pickle.dump(jax.device_get(soup), open(OUT / "params" / f"{name}_soup.pkl", "wb"))
with open(OUT / "pbt.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print(name, "DONE", json.dumps({k: out[k] for k in ("member_scores", "best_member", "soup_members", "soup_score", "minutes")}))
