"""Evaluate the saved policies of the main experiment groups on the same fresh episodes
(robust_eval, 256 per case, key 4242) and write experiments/results/robust_table.json.

    uv run experiments/robust_table.py
"""
import json
import pickle
import sys
from pathlib import Path

import jax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from dprl import runs  # noqa: E402
from dprl.env import DoublePendulumCart  # noqa: E402
from robust_eval import make_robust  # noqa: E402

PARAMS = ROOT / "experiments" / "results" / "params"
GROUPS = {
    "3 layers": [f"B_3L_{s}" for s in (5, 6, 7, 8)],
    "+ easier physics": [f"G_g05_d01_{s}" for s in (1, 2, 3, 4)],
    "+ start noise": [f"EN_{s}" for s in (1, 2, 3, 4)],
    "population (best)": ["PBT_EN_best"],
}
RUNS = {"3 layers, full run": "2026-10-18_12-47-32", "final run": "2026-10-18_16-35-06"}

robust = make_robust(DoublePendulumCart(), n=256, push_levels=(0.1, 0.2))
key = jax.random.key(4242)
table = {}
for group, names in GROUPS.items():
    table[group] = {n: robust(pickle.load(open(PARAMS / f"{n}.pkl", "rb")), key) for n in names if (PARAMS / f"{n}.pkl").exists()}
for label, name in RUNS.items():
    run = runs.RUNS_DIR / name
    if run.exists():
        _, cfg, env = runs.load_config(run)
        params, _ = runs.load_checkpoint(runs.list_checkpoints(run)[-1], cfg, env)
        table[label] = {name: robust(params, key)}
out = ROOT / "experiments" / "results" / "robust_table.json"
out.write_text(json.dumps(table, indent=1))
for group, members in table.items():
    print(group, {case: [m[case]["success"] for m in members.values()] for case in next(iter(members.values()))})
