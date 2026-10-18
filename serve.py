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
from dprl.ppo import TIME_MAX

ROOT = Path(__file__).resolve().parent
pinned_run: Path | None = None
app = FastAPI()


def available_runs() -> list[Path]:
    """Runs in runs/ and in group folders runs/<group>/ (runs/imported/ holds checkpoint-only
    runs imported from experiments)."""
    return runs.all_runs()


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


def group_of(run: Path) -> str | None:
    return run.parent.name if run.parent.resolve() != runs.RUNS_DIR.resolve() else None


@app.get("/api/leaderboard")
def leaderboard(group: str | None = None, run: str | None = None):
    """The runs of a group (default: the shown run's group), ranked by the evaluation score of
    their latest checkpoint."""
    if group is None:
        group = group_of(current_run(run))
    if group is None:
        return {"group": None, "runs": []}
    rows = []
    for r in available_runs():
        if group_of(r) != group:
            continue
        ckpts = runs.list_checkpoints(r)
        if not ckpts:
            continue
        meta = checkpoint_meta(ckpts[-1], ckpts[-1].stat().st_mtime)
        config, _, _ = run_config(r)
        rows.append({"name": run_name(r), "label": config.get("label") or r.name, "checkpoint": ckpts[-1].stem,
                     "env_steps": meta.get("env_steps"), "progress": meta.get("update", 0) / config["num_updates"],
                     "eval": meta.get("eval", {})})
    rows.sort(key=lambda row: -(row["eval"].get("score") or row["eval"].get("success") or 0))
    return {"group": group, "runs": rows}


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


@functools.lru_cache(maxsize=256)
def value_scale(path: Path, mtime: float) -> float:
    """Factor from the critic's output to return units. Runs trained with reward_norm divide
    rewards by the running std of the return; that std is stored in the checkpoint (newer runs)
    or logged as return_std in metrics.jsonl at the checkpoint's update."""
    meta = checkpoint_meta(path, mtime)
    if meta.get("value_scale"):
        return float(meta["value_scale"])
    run = path.parent.parent
    config, _, _ = run_config(run)
    if not config["ppo"].get("reward_norm"):
        return 1.0
    update, scale = meta.get("update", 0), 1.0
    with open(run / "metrics.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row.get("return_std"):
                scale = row["return_std"]
            if row["update"] >= update:
                break
    return float(scale)


@app.get("/api/policy")
def policy(ckpt: str = "latest", run: str | None = None):
    """Actor and critic weights of a checkpoint as base64 float32 (row-major), the env config, and
    value_scale (critic output x value_scale = predicted discounted return)."""
    run = current_run(run)
    path = checkpoint_path(run, ckpt)
    params, meta = checkpoint(path, path.stat().st_mtime)
    _, _, env = run_config(run)
    b64 = lambda a: base64.b64encode(np.asarray(a, dtype="<f4").tobytes()).decode()  # noqa: E731
    pack = lambda net: [{"shape": list(np.shape(layer["w"])), "w": b64(layer["w"]), "b": b64(layer["b"])}  # noqa: E731
                        for layer in params[net]]
    return {"run": run_name(run), "checkpoint": path.stem, "meta": meta, "env": env_for_browser(env),
            "layers": pack("actor"), "critic": pack("critic"),
            "value_scale": value_scale(path, path.stat().st_mtime),
            # mirror-symmetric networks (see dprl/networks.py) and the time-to-stable heads
            "symmetric": "sym" in params, "time_max": TIME_MAX if np.shape(params["critic"][-1]["b"])[0] >= 3 else None}


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
    # Track shield on cart states near the ends, with random policy actions.
    k5, k6 = jax.random.split(k4)
    cart = jax.random.uniform(k5, (n, 2), minval=jnp.array([-env.x_limit, -6.0]), maxval=jnp.array([env.x_limit, 6.0]))
    shield_q, shield_qd = q.at[:, 0].set(cart[:, 0]), qd.at[:, 0].set(cart[:, 1])
    shield_a = jax.random.uniform(k6, (n,), minval=-1.0, maxval=1.0)
    shielded, shield_active = jax.vmap(env.apply_shield)(shield_q, shield_qd, shield_a)
    out = dict(q=q, qd=qd, force=force, tip=tip, next_q=next_q, next_qd=next_qd, obs=obs,
               action=jnp.clip(mean[:, 0], -1, 1),
               pend_energy=jax.vmap(env.pendulum_energy)(q, qd),
               shield_q=shield_q, shield_qd=shield_qd, shield_a=shield_a,
               shielded=shielded, shield_active=shield_active.astype(jnp.float32))
    return {"checkpoint": path.stem, **{k: np.asarray(v).tolist() for k, v in out.items()}}


@app.middleware("http")
async def no_stale_pages(request, call_next):
    """Revalidate the page and its scripts on every load, so an edited app.js / sim.mjs is
    picked up by a normal reload (the server answers 304 when nothing changed)."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


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
