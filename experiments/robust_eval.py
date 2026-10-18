"""Stricter test of a policy than the checkpoint evaluator, used for the analysis in docs/experiments.md.

    uv run experiments/robust_eval.py runs/<run> [ckpt_001234]    # default: latest checkpoint
    uv run experiments/robust_eval.py experiments/results/params/<name>_<seed>.pkl

Deterministic policy, real physics, 128 episodes per case:
    hang_still        the training/evaluator start (hanging still, cart centred)
    hang_perturbed    cart within 1 m and 0.5 m/s, links within 0.3 rad and 0.5 rad/s of hanging
    random            fully random angles and +-3 rad/s
    push<L>_from_up   near-upright start with random tip pushes at push level L (L * 10 N)
Per case: success (on the track and upright >= 90% of the last 5 s), off-track rate, median time
of first reaching the top and of settling (upright from then to the end), max cart excursion and
action jerk (mean |a_t - a_t-1|, lower is smoother).
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dprl.env import DoublePendulumCart
from dprl.networks import actor

def make_robust(env, n=128, push_levels=(0.2, 0.4)):
    last = env.steps(5.0)
    def episode(params, key, mix, pushes, hang_noise):
        k0, k1, kp = jax.random.split(key, 3)
        s0 = env.reset(k0, mix=mix)
        # "perturbed hanging": old distribution (cart +-1 m, +-0.3 rad, +-0.5 rad/s)
        u = jax.random.uniform(k1, (5,), minval=-1, maxval=1)
        s0 = jax.lax.cond(hang_noise, lambda s: s._replace(
            q=jnp.array([u[0], jnp.pi + 0.3 * u[1], jnp.pi + 0.3 * u[2]]),
            qd=jnp.array([0.5 * u[3], 0.5 * u[4], 0.0])), lambda s: s, s0)
        def step(c, k):
            s, off = c
            s = env.sample_push(k, s, pushes)
            tip = (jnp.where(s.push_left > 0, s.push_force, 0.0), 0.0)
            a, _ = actor(params, env.obs(s))
            s2, _, term, _ = env.step(s, a, tip_force=tip)
            off = off | term
            return (s2, off), (env.is_upright(s2.q) & ~off, off, a[0], s2.q[0])
        _, (up, off, a, x) = jax.lax.scan(step, (s0, False), jax.random.split(kp, env.max_steps))
        reached = up.any()
        first = jnp.where(reached, jnp.argmax(up), env.max_steps)
        # time from which it stays upright to the end
        settled_from = env.max_steps - jnp.argmax(jnp.flip(~up))
        settled_from = jnp.where(up[-1], settled_from, env.max_steps)
        succ = (up[-last:].mean() >= 0.9) & ~off[-1]
        return dict(success=succ, first_up=first * env.dt, settled=settled_from * env.dt, off=off[-1],
                    max_x=jnp.abs(x).max(), jerk=jnp.abs(jnp.diff(a)).mean(), up_last5=up[-last:].mean())
    @jax.jit
    def run(params, key):
        out = {}
        cases = {"hang_still": ((1., 0., 0.), 0.0, False), "hang_perturbed": ((1., 0., 0.), 0.0, True),
                 "random": ((0., 1., 0.), 0.0, False)}
        for lvl in push_levels:
            cases[f"push{lvl}_from_up"] = ((0., 0., 1.), lvl, False)
        for name, (mix, pushes, hn) in cases.items():
            r = jax.vmap(lambda k: episode(params, k, jnp.array(mix), pushes, hn))(jax.random.split(key, n))
            out[name] = {"success": r["success"].mean(), "off": r["off"].mean(), "up_last5": r["up_last5"].mean(),
                         "t_first_up_med": jnp.median(r["first_up"]), "t_settled_med": jnp.median(r["settled"]),
                         "t_settled_p90": jnp.percentile(r["settled"], 90), "max_x_med": jnp.median(r["max_x"]),
                         "jerk": r["jerk"].mean()}
        return out
    return lambda params, key=jax.random.key(99): jax.tree.map(lambda v: round(float(v), 3), run(params, key))

def show(res):
    for k, v in res.items():
        print(f"  {k:17s} " + " ".join(f"{kk}={vv}" for kk, vv in v.items()))

if __name__ == "__main__":
    from pathlib import Path
    from dprl import runs
    p = Path(sys.argv[1])
    if p.is_dir():
        _, cfg, env = runs.load_config(p); ck = runs.list_checkpoints(p)[-1] if len(sys.argv) < 3 else p / "checkpoints" / f"{sys.argv[2]}.npz"
        params, _ = runs.load_checkpoint(ck, cfg, env); print(ck)
    else:
        import pickle
        params = pickle.load(open(p, "rb"))
        env = DoublePendulumCart()
    show(make_robust(env)(params))
