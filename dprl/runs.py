"""On-disk layout shared by the trainer (writer) and the dashboard (reader).

runs/<timestamp>/
    config.json          env + PPO settings, written once at start
    metrics.jsonl        one JSON line per PPO update, appended as training runs
    checkpoints/
        ckpt_000120.npz  policy params + metadata, written atomically

The two processes only talk through these files, so neither can block the other.
"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import jax
import numpy as np

from .env import DoublePendulumCart
from .networks import init_params
from .ppo import PPOConfig

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def new_run(cfg: PPOConfig, env: DoublePendulumCart, curriculum=None) -> Path:
    run = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (run / "checkpoints").mkdir(parents=True)
    config = {"ppo": asdict(cfg), "env": env.to_dict(), "num_updates": cfg.num_updates,
              "steps_per_update": cfg.steps_per_update,
              "curriculum": asdict(curriculum) if curriculum else None}
    (run / "config.json").write_text(json.dumps(config, indent=2))
    return run


def latest_run() -> Path | None:
    runs = sorted(p for p in RUNS_DIR.glob("*") if (p / "config.json").exists())
    return runs[-1] if runs else None


def load_config(run: Path):
    config = json.loads((run / "config.json").read_text())
    ppo = config["ppo"]
    cfg = PPOConfig(**{**ppo, "hidden": tuple(ppo["hidden"])})
    # Ignore env fields from older versions of the code so old runs still open.
    fields = DoublePendulumCart.__dataclass_fields__
    env = DoublePendulumCart(**{k: v for k, v in config["env"].items() if k in fields})
    return config, cfg, env


def save_checkpoint(run: Path, params, meta: dict) -> Path:
    path = run / "checkpoints" / f"ckpt_{meta['update']:06d}.npz"
    arrays = {jax.tree_util.keystr(k): np.asarray(v) for k, v in jax.tree_util.tree_leaves_with_path(params)}
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:  # write elsewhere, then rename: readers never see a half file
        np.savez(f, __meta__=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)
    return path


def list_checkpoints(run: Path) -> list[Path]:
    return sorted((run / "checkpoints").glob("ckpt_*.npz"))


def load_checkpoint(path: Path, cfg: PPOConfig, env: DoublePendulumCart):
    """Rebuild the params pytree from a template network, filling leaves by name."""
    template = init_params(jax.random.key(0), env.obs_dim, env.act_dim, cfg.hidden)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    with np.load(path) as data:
        params = jax.tree_util.tree_unflatten(treedef, [data[jax.tree_util.keystr(k)] for k, _ in leaves])
        meta = json.loads(str(data["__meta__"]))
    return params, meta


def read_checkpoint_meta(path: Path) -> dict:
    with np.load(path) as data:
        return json.loads(str(data["__meta__"]))
