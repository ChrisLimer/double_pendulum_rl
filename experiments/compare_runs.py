"""Compare the final checkpoints of runs on the same fresh episodes, with and without the
mirror-symmetry rule from the dashboard (act with the side, s or mirror(s), whose critic value is
higher; switch only if the other side is better by 2% and the current one was used >= 0.25 s).

    uv run experiments/compare_runs.py runs/phaseA/*          # writes experiments/results/compare_<group>.json

Deterministic policy, real physics, track shield as configured in each run, 256 episodes per case.
"""
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dprl import runs  # noqa: E402
from dprl.networks import actor, critic  # noqa: E402

CASES = {  # name: (start mix, start noise, cart spread, push level)
    "still": ((1.0, 0.0, 0.0), 0.0, 0.0, 0.0),
    "perturbed": ((1.0, 0.0, 0.0), 1.0, 0.0, 0.0),
    "random": ((0.0, 1.0, 0.0), 0.0, 0.0, 0.0),
    "edge": ((0.0, 1.0, 0.0), 0.0, 1.0, 0.0),
    "push0.1": ((0.0, 0.0, 1.0), 0.0, 0.0, 0.1),
    "push0.2": ((0.0, 0.0, 1.0), 0.0, 0.0, 0.2),
    "push0.3": ((0.0, 0.0, 1.0), 0.0, 0.0, 0.3),
}
MARGIN, DWELL_SECONDS = 0.02, 0.25


def make_eval(env, params, n=256):
    dwell = env.steps(DWELL_SECONDS)

    def episode(key, case, mirror_mode):
        mix, noise, cart_spread, push = case
        k0, kp = jax.random.split(key)
        s0 = env.reset(k0, mix=jnp.array(mix), noise=noise, cart_spread=cart_spread)

        def step(c, k):
            s, off, side, since = c
            s = env.sample_push(k, s, push)
            tip = (jnp.where(s.push_left > 0, s.push_force, 0.0), 0.0)
            o, om = env.obs(s), env.obs(s._replace(q=-s.q, qd=-s.qd))
            if mirror_mode:
                v, vm = critic(params, o), critic(params, om)
                cur, other = jnp.where(side < 0, vm, v), jnp.where(side < 0, v, vm)
                first = side == 0
                flip = ~first & (other - cur > MARGIN * jnp.abs(cur)) & (since >= dwell)
                side = jnp.where(first, jnp.where(vm > v, -1, 1), jnp.where(flip, -side, side))
                since = jnp.where(first | flip, 0, since + 1)
            a = jnp.where(side < 0, -actor(params, om)[0], actor(params, o)[0])
            s2, _, term, _ = env.step(s, a, tip_force=tip)
            off = off | term
            return (s2, off, side, since), (env.is_upright(s2.q) & ~off, off, s2.shielded)

        init = (s0, False, jnp.int32(0 if mirror_mode else 1), jnp.int32(0))
        _, (up, off, shielded) = jax.lax.scan(step, init, jax.random.split(kp, env.max_steps))
        last = env.steps(5.0)
        success = (up[-last:].mean() >= 0.9) & ~off[-1]
        settled = jnp.where(up[-1], env.max_steps - jnp.argmax(jnp.flip(~up)), env.max_steps) * env.dt
        return success, off[-1], settled, shielded.mean()

    def run(key):
        out = {}
        for mirror_mode in (False, True):
            for i, (name, case) in enumerate(CASES.items()):
                keys = jax.random.split(jax.random.fold_in(key, i), n)
                s, off, settled, shield = jax.jit(jax.vmap(lambda k: episode(k, case, mirror_mode)))(keys)
                out[f"{'auto' if mirror_mode else 'off'}/{name}"] = {
                    "success": float(s.mean()), "off_track": float(off.mean()),
                    "settled_median": float(np.median(settled)), "settled_p90": float(np.percentile(settled, 90)),
                    "shield_rate": float(shield.mean())}
        return out

    return run


def main():
    results = {}
    for run in map(Path, sys.argv[1:]):
        if not (run / "config.json").exists() or not runs.list_checkpoints(run):
            continue
        config, cfg, env = runs.load_config(run)
        params, meta = runs.load_checkpoint(runs.list_checkpoints(run)[-1], cfg, env)
        label = config.get("label") or run.name
        results[label] = {"run": str(run.resolve().relative_to(ROOT)), "env_steps": meta.get("env_steps"),
                          **make_eval(env, params)(jax.random.key(2024))}
        r = results[label]
        print(f"{label:10s} " + " ".join(f"{c}:{r['off/' + c]['success']:.2f}->{r['auto/' + c]['success']:.2f}"
                                          for c in CASES), flush=True)
    group = Path(sys.argv[1]).parent.name
    out = ROOT / "experiments" / "results" / f"compare_{group}.json"
    out.write_text(json.dumps(results, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
