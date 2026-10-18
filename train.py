"""Train PPO on the double pendulum cart and (by default) launch the dashboard alongside.

    uv run train.py                       # train + dashboard at http://127.0.0.1:8000
    uv run train.py --total-steps 5e6 --no-server

The dashboard runs as a separate process (serve.py) and reads the run folder, so it never
slows down or blocks training. It keeps running after training ends; Ctrl+C stops both.
"""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

from dprl.env import DoublePendulumCart
from dprl.curriculum import Curriculum
from dprl.ppo import PPOConfig, make_evaluator, make_trainer
from dprl.runs import new_run, save_checkpoint

ROOT = Path(__file__).resolve().parent


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_dashboard(port: int):
    url = f"http://127.0.0.1:{port}"
    if port_in_use(port):
        print(f"Port {port} is busy; assuming the dashboard is already running at {url} "
              "(it always shows the newest run).")
        return None
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve.py"), "--port", str(port)])
    print(f"Dashboard: {url}")
    return proc


def to_json(value):
    value = float(value)
    return value if math.isfinite(value) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--total-steps", type=float, default=PPOConfig.total_timesteps)
    p.add_argument("--num-envs", type=int, default=PPOConfig.num_envs)
    p.add_argument("--seed", type=int, default=PPOConfig.seed)
    p.add_argument("--ckpt-every", type=int, default=200, help="save a checkpoint every N updates")
    p.add_argument("--energy-shaping", choices=["potential", "additive"], default="potential",
                   help="how the energy reward is added (see DoublePendulumCart)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-server", action="store_true", help="don't launch the dashboard")
    args = p.parse_args()

    cfg = PPOConfig(seed=args.seed, total_timesteps=int(args.total_steps), num_envs=args.num_envs)
    env = DoublePendulumCart(shaping_gamma=cfg.gamma, energy_shaping=args.energy_shaping)
    curriculum = Curriculum()
    run = new_run(cfg, env, curriculum)
    print(f"Run folder: {run.relative_to(ROOT)}  ({cfg.num_updates} updates, "
          f"{cfg.steps_per_update} env steps each, JAX backend: {jax.default_backend()})")

    server = None if args.no_server else start_dashboard(args.port)
    init, train_chunk = make_trainer(cfg, env)
    evaluate = make_evaluator(env)
    runner = init(jax.random.key(cfg.seed))
    eval_key = jax.random.key(cfg.seed + 1)  # same test episodes for every checkpoint

    def checkpoint(update, row):
        results = {k: to_json(v) for k, v in jax.device_get(evaluate(runner.params, eval_key)).items()}
        save_checkpoint(run, runner.params, {
            "update": update, "env_steps": update * cfg.steps_per_update, "time": time.time(),
            "eval": results,
            "push_level": curriculum.push_level,
            "upright_share": curriculum.upright_share,
            "start_noise": curriculum.start_noise,
            "ease": curriculum.ease,
            "upright_late": row and row["upright_late"],
        })
        return results

    checkpoint(0, None)  # the untrained policy, so the viewer has something from the start
    update, last_ckpt, t_start, row = 0, 0, time.time(), None
    try:
        with open(run / "metrics.jsonl", "a") as log:
            while update < cfg.num_updates:
                t0 = time.time()
                progress = update / cfg.num_updates
                knobs = curriculum.knobs(progress)
                runner, metrics = train_chunk(runner, knobs)
                metrics = jax.device_get(metrics)  # blocks until the chunk is done
                sps = cfg.updates_per_chunk * cfg.steps_per_update / (time.time() - t0)

                for i in range(cfg.updates_per_chunk):
                    update += 1
                    row = {k: to_json(np.asarray(v)[i]) for k, v in metrics.items()}
                    row.update(update=update, env_steps=update * cfg.steps_per_update,
                               wall_time=time.time() - t_start, sps=sps,
                               energy_weight=float(knobs["energy_weight"]), push_level=curriculum.push_level,
                               upright_share=curriculum.upright_share, start_noise=curriculum.start_noise,
                               ease=curriculum.ease)
                    log.write(json.dumps(row) + "\n")
                log.flush()
                upright_late = np.nanmean(np.asarray(metrics["upright_late"], dtype=float))
                success = np.nanmean(np.asarray(metrics["swingup_success"], dtype=float))
                curriculum.update(float(success))

                msg = ""
                if update - last_ckpt >= args.ckpt_every or update >= cfg.num_updates:
                    ev = checkpoint(update, row)
                    last_ckpt = update
                    msg = f"  | ckpt eval: swing-up success {ev['success']:.0%}"
                print(f"update {update:5d}/{cfg.num_updates}  steps {update * cfg.steps_per_update:>11,}  "
                      f"upright(late) {upright_late:5.2f}  swing-up success {success:4.2f}  upright share {curriculum.upright_share:.2f}  noise {curriculum.start_noise:.2f}  "
                      f"push {curriculum.push_level:.2f}  w_E {float(knobs['energy_weight']):.2f}  {sps:,.0f} steps/s{msg}", flush=True)

        print(f"Training finished in {time.time() - t_start:.0f} s.")
        if server:
            print("Dashboard still running; press Ctrl+C to stop it.")
            server.wait()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not cut the save short
        if update > last_ckpt:
            checkpoint(update, row)
            print(f"\nInterrupted; saved checkpoint at update {update}.")
    finally:
        if server:
            server.terminate()


if __name__ == "__main__":
    main()
    # Skip interpreter teardown: after a Ctrl+C mid-computation, JAX's CPU thread pool can
    # wait forever on an interrupted worker while shutting down. Everything is saved by now.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
