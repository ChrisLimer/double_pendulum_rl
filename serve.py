"""Dashboard server: live training metrics + checkpoints for the in-browser simulator.

    uv run serve.py                 # follows the newest run in runs/
    uv run serve.py --run runs/2026-10-17_22-00-00

The page can switch to any other run (runs/*, and checkpoint-only runs in runs/imported/*)
with its run picker, or by opening /?run=<name>.

It only reads files the trainer writes, so it can be started, stopped or restarted at
any time without touching training. The simulation itself runs in the browser
(static/sim.mjs): the page downloads a checkpoint's policy weights once and then steps
physics + policy locally at the control rate (100 Hz), which keeps it real-time and lets you push the
pendulum with no network round trip.
"""

import argparse
import base64
import functools
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from dprl import runs
from dprl.networks import actor

ROOT = Path(__file__).resolve().parent
pinned_run: Path | None = None
app = FastAPI()


def available_runs() -> list[Path]:
    """Training runs plus checkpoint-only runs imported into runs/imported/."""
    found = [p for p in runs.RUNS_DIR.glob("*") if (p / "config.json").exists()]
    found += [p for p in (runs.RUNS_DIR / "imported").glob("*") if (p / "config.json").exists()]
    return sorted(found)


def run_name(run: Path) -> str:
    try:
        return run.resolve().relative_to(runs.RUNS_DIR.resolve()).as_posix()
    except ValueError:
        return run.name


def current_run(name: str | None = None) -> Path:
    if name:  # only names of existing runs, so a query can't point anywhere else
        for run in available_runs():
            if run_name(run) == name:
                return run
        raise HTTPException(404, f"unknown run {name}")
    run = pinned_run or runs.latest_run()
    if run is None:
        raise HTTPException(404, "no runs yet; start training with `uv run train.py`")
    return run


@app.get("/api/runs")
def list_runs():
    return {"default": run_name(current_run()), "runs": [
        {"name": run_name(r), "checkpoints": len(runs.list_checkpoints(r)),
         "logs": (r / "metrics.jsonl").exists()} for r in available_runs()]}


@functools.lru_cache(maxsize=8)
def run_config(run: Path):
    return runs.load_config(run)


@functools.lru_cache(maxsize=32)
def checkpoint(path: Path, mtime: float):
    _, cfg, env = run_config(path.parent.parent)
    return runs.load_checkpoint(path, cfg, env)


@functools.lru_cache(maxsize=4096)
def checkpoint_meta(path: Path, mtime: float):
    return runs.read_checkpoint_meta(path)


def checkpoint_path(run: Path, name: str) -> Path:
    paths = runs.list_checkpoints(run)
    if not paths:
        raise HTTPException(404, "no checkpoints yet")
    path = paths[-1] if name == "latest" else run / "checkpoints" / f"{name}.npz"
    if not path.exists():
        raise HTTPException(404, f"unknown checkpoint {name}")
    return path


def env_for_browser(env):
    return {**env.to_dict(), "target_energy": env.target_energy}


@app.get("/api/status")
def status(run: str | None = None):
    run = current_run(run)
    config, _, _ = run_config(run)
    ckpts = runs.list_checkpoints(run)
    return {"run": run_name(run), "config": config, "num_checkpoints": len(ckpts),
            "latest_checkpoint": ckpts[-1].stem if ckpts else None}


@app.get("/api/metrics")
def metrics(since: int = 0, run: str | None = None):
    """Metric rows from index `since` on, so the page only fetches what's new."""
    run = current_run(run)
    path = run / "metrics.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines()[since:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # last line may be half-written
                break
    return {"run": run_name(run), "since": since, "rows": rows}


@app.get("/api/checkpoints")
def checkpoints(run: str | None = None):
    run = current_run(run)
    out = []
    for path in runs.list_checkpoints(run):
        meta = checkpoint_meta(path, path.stat().st_mtime)
        out.append({"name": path.stem, **meta})
    return {"run": run_name(run), "checkpoints": out}


@app.get("/api/policy")
def policy(ckpt: str = "latest", run: str | None = None):
    """Actor weights of a checkpoint as base64 float32 (row-major), plus the env config."""
    run = current_run(run)
    path = checkpoint_path(run, ckpt)
    params, meta = checkpoint(path, path.stat().st_mtime)
    _, _, env = run_config(run)
    b64 = lambda a: base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()  # noqa: E731
    layers = [{"shape": list(np.shape(layer["w"])), "w": b64(layer["w"]), "b": b64(layer["b"])}
              for layer in params["actor"]]
    return {"run": run_name(run), "checkpoint": path.stem, "meta": meta, "env": env_for_browser(env),
            "layers": layers}


@app.get("/api/parity")
def parity(ckpt: str = "latest", n: int = 64, seed: int = 0, run: str | None = None):
    """Reference values from the Python/JAX code, for static/parity.html to compare the
    JavaScript simulator against: one control step from random states, with random cart
    and tip forces, plus the policy's action for each state."""
    run = current_run(run)
    path = checkpoint_path(run, ckpt)
    params, _ = checkpoint(path, path.stat().st_mtime)
    _, _, env = run_config(run)
    k1, k2, k3, k4 = jax.random.split(jax.random.key(seed), 4)
    states = jax.vmap(lambda k: env.reset(k, mix=(0.0, 1.0, 0.0)))(jax.random.split(k1, n))
    q, qd = states.q, states.qd * 2
    force = jax.random.uniform(k2, (n,), minval=-env.max_force, maxval=env.max_force)
    tip = jax.random.uniform(k3, (n, 2), minval=-15.0, maxval=15.0)
    next_q, next_qd = jax.vmap(lambda q, qd, f, t: env.integrate(q, qd, f, (t[0], t[1])))(q, qd, force, tip)
    obs = jax.vmap(env.obs)(states._replace(qd=qd))
    mean, _ = actor(params, obs)
    out = dict(q=q, qd=qd, force=force, tip=tip, next_q=next_q, next_qd=next_qd, obs=obs,
               action=jnp.clip(mean[:, 0], -1, 1),
               pend_energy=jax.vmap(env.pendulum_energy)(q, qd))
    return {"checkpoint": path.stem, **{k: np.asarray(v).tolist() for k, v in out.items()}}


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")


def main():
    global pinned_run
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, help="show this run instead of following the newest one")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = p.parse_args()
    pinned_run = args.run.resolve() if args.run else None

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
