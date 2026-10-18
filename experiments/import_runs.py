"""Import experiment results (final weights only) as checkpoint-only runs in runs/imported/, so the
dashboard's run picker can show and simulate them.

    uv run experiments/import_runs.py EN G_g05     # experiments whose name starts with these
"""
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dprl import runs
from dprl.env import DoublePendulumCart
from dprl.curriculum import Curriculum
from dprl.ppo import PPOConfig
OUT = ROOT / "experiments" / "results"
prefixes = sys.argv[1:]
for line in open(OUT / "results.jsonl"):
    r = json.loads(line)
    if not any(r["name"].startswith(p) for p in prefixes):
        continue
    pkl = OUT / "params" / f"{r['name']}_{r['seed']}.pkl"
    if not pkl.exists():
        continue
    ppo = dict(r["ov"].get("ppo", {}))
    if "hidden" in ppo: ppo["hidden"] = tuple(ppo["hidden"])
    cfg = PPOConfig(seed=r["seed"], total_timesteps=int(r["steps"]), **ppo)
    env = DoublePendulumCart(shaping_gamma=cfg.gamma, **r["ov"].get("env", {}))
    cur = Curriculum(**r["ov"].get("cur", {}))
    run = runs.RUNS_DIR / "imported" / f"{r['name']}_seed{r['seed']}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = {"ppo": asdict(cfg), "env": env.to_dict(), "num_updates": cfg.num_updates,
              "steps_per_update": cfg.steps_per_update, "curriculum": asdict(cur),
              "note": "imported from an experiment-harness run: final weights only, no training logs",
              "robust_eval": r.get("robust")}
    (run / "config.json").write_text(json.dumps(config, indent=2))
    meta = {"update": cfg.num_updates, "env_steps": cfg.num_updates * cfg.steps_per_update,
            "eval": {k: r[k] for k in ("success", "swingup_time", "upright_last5")},
            "upright_share": r["final_cur"][0], "push_level": r["final_cur"][1]}
    runs.save_checkpoint(run, pickle.load(open(pkl, "rb")), meta)
    print("imported", run.name)
