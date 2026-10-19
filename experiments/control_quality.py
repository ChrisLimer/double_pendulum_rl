"""How does a policy control, beyond success: smoothness, effort and precision.

    uv run experiments/control_quality.py runs/<group>/* [...]   # writes experiments/results/control_<group>.json
    uv run experiments/control_quality.py --decode mean runs/...  # discrete policies: expected force

Discrete (categorical) policies are decoded either as the most likely force level ("mode", what
the evaluator and dashboard use) or as the expected force sum_k p_k F_k ("mean"), which is a
smooth function of the state.

Deterministic policy, real physics, track shield as configured, no pushes, 128 episodes per case:
    balance   near-upright starts; statistics over the last 15 s (after settling)
    swing-up  hanging still and random starts; time to settle (upright from then on)
Per case: angle and cart-position RMS while balancing, mean |force|, mean |change of force| per
step (jerk), action sign changes per second (chattering), mechanical power put in by the cart
force (mean |F * cart speed|), and the share of steps at full force.
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
from dprl.networks import actor, actor_mode  # noqa: E402

CASES = {"balance": (0.0, 0.0, 1.0, 0.0), "swing-up, hanging still": (1.0, 0.0, 0.0, 0.0),
         "swing-up, random": (0.0, 1.0, 0.0, 0.0), "pushes 0.2": (0.0, 0.0, 1.0, 0.2)}


def evaluate(run: Path, n=128, decode="mode"):
    config, cfg, env = runs.load_config(run)
    params, meta = runs.load_checkpoint(runs.list_checkpoints(run)[-1], cfg, env)

    def policy(o):
        if cfg.action_bins and decode == "mode":
            return actor_mode(params, o)  # the most likely force level
        return actor(params, o)[0]  # Gaussian mean, or the expected force of a discrete policy

    def episode(key, case):
        *mix, push = case
        k0, kp = jax.random.split(key)
        s0 = env.reset(k0, mix=jnp.array(mix))

        def step(s, k):
            s = env.sample_push(k, s, push)
            tip = (jnp.where(s.push_left > 0, s.push_force, 0.0), 0.0)
            a = policy(env.obs(s))
            applied, _ = env.apply_shield(s.q, s.qd, jnp.clip(a[0], -1, 1))
            s2, _, term, _ = env.step(s, a, tip_force=tip)
            return s2, (applied, s2.q, s2.qd, env.is_upright(s2.q), term)

        _, out = jax.lax.scan(step, s0, jax.random.split(kp, env.max_steps))
        return out

    result = {"label": config.get("label") or run.name, "action_bins": cfg.action_bins, "decode": decode,
              "env_steps": meta.get("env_steps")}
    for name, case in CASES.items():
        a, q, qd, up, term = map(np.asarray, jax.jit(jax.vmap(lambda k: episode(k, case)))(jax.random.split(jax.random.key(5), n)))
        force = a * env.max_force
        window = slice(env.steps(5.0), None)  # the last 15 s
        flips = (np.diff(np.sign(np.where(np.abs(a) < 1e-6, 0, a)), axis=1) != 0).sum(1) / (env.max_steps * env.dt)
        settled = np.where(up[:, -1], env.max_steps - np.argmax(np.flip(~up, 1), 1), env.max_steps) * env.dt
        last = env.steps(5.0)
        result[name] = {
            "success": float(((up[:, -last:].mean(1) >= 0.9) & ~term.any(1)).mean()),
            "off_track": float(term.any(1).mean()),
            "settle_median": float(np.median(settled)),
            "angle_rms_deg": float(np.degrees(np.sqrt((q[:, window, 1:] ** 2).mean()))),
            "cart_rms_m": float(np.sqrt((q[:, window, 0] ** 2).mean())),
            "force_mean_abs": float(np.abs(force).mean()),
            "force_jerk": float(np.abs(np.diff(force, axis=1)).mean()),
            "sign_changes_per_s": float(flips.mean()),
            "full_force_share": float((np.abs(a) > 0.99).mean()),
            "power_mean_W": float(np.abs(force * qd[..., 0]).mean()),
        }
    return result


def main():
    args, decode = sys.argv[1:], "mode"
    if args[:1] == ["--decode"]:
        decode, args = args[1], args[2:]
    rows = []
    for r in args:
        run = Path(r).resolve()
        if (run / "config.json").exists() and runs.list_checkpoints(run):
            rows.append(evaluate(run, decode=decode))
            b = rows[-1]["balance"]
            print(f"{rows[-1]['label']:12s} balance: angle {b['angle_rms_deg']:.2f} deg, cart {b['cart_rms_m']:.3f} m, "
                  f"|F| {b['force_mean_abs']:.1f} N, jerk {b['force_jerk']:.1f} N/step, "
                  f"{b['sign_changes_per_s']:.0f} sign changes/s | swing-up still {rows[-1]['swing-up, hanging still']['settle_median']:.2f} s, "
                  f"random {rows[-1]['swing-up, random']['settle_median']:.2f} s | success still/random/push0.2 "
                  f"{rows[-1]['swing-up, hanging still']['success']:.2f}/{rows[-1]['swing-up, random']['success']:.2f}/"
                  f"{rows[-1]['pushes 0.2']['success']:.2f}", flush=True)
    group = Path(args[0]).resolve().parent.name
    out = ROOT / "experiments" / "results" / f"control_{group}_{decode}.json"
    out.write_text(json.dumps(rows, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
