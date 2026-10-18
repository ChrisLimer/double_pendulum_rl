"""PPO (clipped objective + GAE) in the PureJaxRL style: the whole update, including the
environment rollout, is one jitted function. Many updates are fused with lax.scan and the
host only sees the results between chunks (to log metrics and write checkpoints).

Two standard normalizations (see "What matters in on-policy RL", Andrychowicz et al. 2021):
- rewards are divided by the running std of the discounted return, so values are O(1) (raw
  returns are in the hundreds, which also made the 0.2 value clipping very slow); on by default;
- observations can be standardized with running statistics (obs_norm); export_params() folds
  that into the first layer, so saved checkpoints are plain MLPs on raw observations. Off by
  default: with it, 2 of 4 seeds collapsed early (see docs/experiments.md).
The runner also keeps a replay buffer of hard states (see env.REPLAY_LOOKBACK) that the
curriculum mixes into the episode starts."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .env import DoublePendulumCart, EnvState, SUCCESS_FRACTION, SUCCESS_WINDOW
from .networks import actor, critic, gaussian_logp, init_params


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    total_timesteps: int = 200_000_000  # ~70 min at ~50k steps/s on an M-series CPU
    num_envs: int = 64            # small batches = more updates per sample; learned balance faster
    num_steps: int = 128          # rollout length per env per update
    update_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.9975         # ~4 s horizon at 100 Hz: a swing-up takes 2-4 s
    gae_lambda: float = 0.975     # 0.95 at 50 Hz, same time scale
    clip_eps: float = 0.2
    ent_coef: float = 0.005       # without it the better critic let the policy std collapse early
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    hidden: tuple = (256, 256, 256)  # 2 layers never learned the swing-up at 100 Hz (0/4 seeds); 3 did (4/4)
    updates_per_chunk: int = 10   # updates fused into one jit call between host syncs
    obs_norm: bool = False        # running observation standardization (2 of 4 seeds got stuck with it;
                                  # the observations are already scaled by hand)
    reward_norm: bool = True      # divide rewards by the running std of the discounted return
    replay_size: int = 1024       # hard-state buffer for replay starts

    @property
    def steps_per_update(self):
        return self.num_envs * self.num_steps

    @property
    def num_updates(self):
        return self.total_timesteps // self.steps_per_update


class Stats(NamedTuple):
    """Running mean / variance (parallel Welford)."""
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @staticmethod
    def zeros(shape=()):
        return Stats(jnp.zeros(shape), jnp.ones(shape), jnp.float32(1e-4))

    def update(self, batch):
        b_mean, b_var, b_count = batch.mean(0), batch.var(0), batch.shape[0]
        delta, total = b_mean - self.mean, self.count + b_count
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / total
        return Stats(self.mean + delta * b_count / total, m2 / total, total)


class Runner(NamedTuple):
    params: dict
    opt_state: optax.OptState
    env_state: EnvState
    obs: jax.Array          # raw observations
    key: jax.Array
    obs_stats: Stats
    ret: jax.Array          # running discounted return per env (for reward scaling)
    ret_stats: Stats
    replay_s: jax.Array     # (replay_size, 6) hard states (q, qd)
    replay_ptr: jax.Array
    replay_n: jax.Array


def export_params(params, obs_stats: Stats, enabled=True):
    """Fold the observation standardization into the first layer of actor and critic:
    ((o - m) / s) W + b = o (W / s) + (b - (m / s) W)."""
    if not enabled:
        return params
    std = jnp.sqrt(obs_stats.var + 1e-8)

    def fold(layers):
        first = layers[0]
        w = first["w"] / std[:, None]
        return [{"w": w, "b": first["b"] - (obs_stats.mean / std) @ first["w"]}, *layers[1:]]

    return {**params, "actor": fold(params["actor"]), "critic": fold(params["critic"])}


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict


def masked_mean(x, mask):
    n = mask.sum()
    return jnp.where(n > 0, (x * mask).sum() / jnp.maximum(n, 1), jnp.nan)


def make_trainer(cfg: PPOConfig, env: DoublePendulumCart):
    """Returns (init, train_chunk).

    train_chunk(runner, knobs) -> (runner, metrics stacked per update).
    knobs is the curriculum's dict of traced values (see Curriculum.knobs), so changing them
    between chunks does not recompile.
    """
    batch_size = cfg.steps_per_update
    minibatch_size = batch_size // cfg.num_minibatches
    schedule = optax.linear_schedule(
        cfg.lr, 0.0, cfg.num_updates * cfg.update_epochs * cfg.num_minibatches
    )
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(schedule, eps=1e-5))

    def init(key):
        k_params, k_env, key = jax.random.split(key, 3)
        params = init_params(k_params, env.obs_dim, env.act_dim, cfg.hidden)
        env_state = jax.vmap(env.reset)(jax.random.split(k_env, cfg.num_envs))
        obs = jax.vmap(env.obs)(env_state)
        return Runner(params, tx.init(params), env_state, obs, key, Stats.zeros(env.obs_dim),
                      jnp.zeros(cfg.num_envs), Stats.zeros((1,)), jnp.zeros((cfg.replay_size, 6)),
                      jnp.int32(0), jnp.int32(0))

    def normalize(runner, obs):
        if not cfg.obs_norm:
            return obs
        return (obs - runner.obs_stats.mean) / jnp.sqrt(runner.obs_stats.var + 1e-8)

    def env_step(carry, _):
        runner, knobs = carry
        key, k_act, k_env = jax.random.split(runner.key, 3)
        obs_n = normalize(runner, runner.obs)
        mean, log_std = actor(runner.params, obs_n)
        action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
        logp = gaussian_logp(action, mean, log_std)
        value = critic(runner.params, obs_n)
        env_knobs = {**knobs, "replay_s": runner.replay_s, "replay_n": runner.replay_n}
        env_state, obs, reward, info = jax.vmap(env.step_autoreset, in_axes=(0, 0, 0, None))(
            jax.random.split(k_env, cfg.num_envs), runner.env_state, action, env_knobs
        )
        info["raw_reward"] = reward
        ret, ret_stats = runner.ret, runner.ret_stats
        if cfg.reward_norm:
            ret = ret * cfg.gamma + reward
            ret_stats = ret_stats.update(ret[:, None])
            reward = reward / jnp.sqrt(ret_stats.var[0] + 1e-8)
            ret = jnp.where(info["done"], 0.0, ret)
        # Time-limit truncation is not a real ending: bootstrap with the value of the state we
        # were cut off in, folded into the reward, so GAE can treat every `done` the same way.
        final_value = critic(runner.params, normalize(runner, info["final_obs"]))
        reward = reward + cfg.gamma * final_value * info["truncated"]
        info["raw_obs"] = runner.obs
        transition = Transition(obs_n, action, logp, value, reward, info["done"], info)
        runner = runner._replace(env_state=env_state, obs=obs, key=key, ret=ret, ret_stats=ret_stats)
        return (runner, knobs), transition

    def add_replay(runner, traj):
        """Write up to 64 new hard states per update into the ring buffer."""
        mask = traj.info["replay_candidate"].reshape(-1)
        states = traj.info["replay_state"].reshape(-1, 6)
        order = jnp.argsort(~mask)[:64]  # candidates first
        valid = mask[order]
        pos = (runner.replay_ptr + jnp.cumsum(valid) - 1) % cfg.replay_size
        pos = jnp.where(valid, pos, cfg.replay_size)  # out of range -> dropped
        replay_s = runner.replay_s.at[pos].set(states[order], mode="drop")
        k = valid.sum()
        return runner._replace(replay_s=replay_s, replay_ptr=(runner.replay_ptr + k) % cfg.replay_size,
                               replay_n=jnp.minimum(runner.replay_n + k, cfg.replay_size))

    def compute_gae(traj: Transition, last_value):
        def step(carry, t: Transition):
            gae, next_value = carry
            not_done = 1.0 - t.done
            delta = t.reward + cfg.gamma * next_value * not_done - t.value
            gae = delta + cfg.gamma * cfg.gae_lambda * not_done * gae
            return (gae, t.value), gae

        _, adv = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
        return adv, adv + traj.value

    def loss_fn(params, mb):
        obs, action, old_logp, old_value, adv, target = mb
        mean, log_std = actor(params, obs)
        logp = gaussian_logp(action, mean, log_std)
        log_ratio = logp - old_logp
        ratio = jnp.exp(log_ratio)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg_loss = -jnp.minimum(
            ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
        ).mean()

        value = critic(params, obs)
        value_clipped = old_value + jnp.clip(value - old_value, -cfg.clip_eps, cfg.clip_eps)
        v_loss = 0.5 * jnp.maximum((value - target) ** 2, (value_clipped - target) ** 2).mean()

        entropy = jnp.sum(params["log_std"] + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
        loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
        aux = {
            "loss": loss,
            "pg_loss": pg_loss,
            "v_loss": v_loss,
            "entropy": entropy,
            "approx_kl": ((ratio - 1) - log_ratio).mean(),
            "clip_frac": (jnp.abs(ratio - 1) > cfg.clip_eps).mean(),
        }
        return loss, aux

    def update(carry, _):
        runner, knobs = carry
        (runner, _), traj = jax.lax.scan(env_step, (runner, knobs), None, length=cfg.num_steps)
        adv, target = compute_gae(traj, critic(runner.params, normalize(runner, runner.obs)))
        data = (traj.obs, traj.action, traj.logp, traj.value, adv, target)
        data = jax.tree.map(lambda x: x.reshape((batch_size, *x.shape[2:])), data)

        def epoch(carry, _):
            params, opt_state, key = carry
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, batch_size)
            minibatches = jax.tree.map(
                lambda x: x[perm].reshape((cfg.num_minibatches, minibatch_size, *x.shape[1:])), data
            )

            def minibatch_step(carry, mb):
                params, opt_state = carry
                grads, aux = jax.grad(loss_fn, has_aux=True)(params, mb)
                updates, opt_state = tx.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), aux

            (params, opt_state), aux = jax.lax.scan(minibatch_step, (params, opt_state), minibatches)
            return (params, opt_state, key), aux

        (params, opt_state, key), aux = jax.lax.scan(
            epoch, (runner.params, runner.opt_state, runner.key), None, length=cfg.update_epochs
        )
        runner = runner._replace(params=params, opt_state=opt_state, key=key)
        if cfg.obs_norm:
            runner = runner._replace(obs_stats=runner.obs_stats.update(traj.info["raw_obs"].reshape(-1, env.obs_dim)))
        runner = add_replay(runner, traj)

        info = traj.info
        metrics = jax.tree.map(jnp.mean, aux)
        metrics.update(
            episodes=info["done"].sum(),
            ep_return=masked_mean(info["ep_return"], info["done"]),
            off_track=info["terminated"].sum(),  # episodes ended by leaving the track
            upright=info["upright"].mean(),
            upright_late=masked_mean(info["upright"], info["late"]),
            # ... in episodes from near-upright starts only (drives the curriculum)
            upright_late_up=masked_mean(info["upright"], info["late"] & info["start_upright"]),
            recovery=masked_mean(info["upright"], info["recovery_window"]),
            swingup_rate=masked_mean(info["reached_top"], info["swingup_episode"]),
            swingup_time=masked_mean(info["swingup_time"], info["swingup_episode"] & info["reached_top"]),
            # evaluator-style success of hanging starts (drives the curriculum)
            swingup_success=masked_mean(info["swingup_success"], info["swingup_episode"]),
            action_std=jnp.exp(params["log_std"]).mean(),
            shield_rate=info["shielded"].mean(),  # fraction of steps the track shield overrode
            reward_raw=info["raw_reward"].mean(),
            return_std=jnp.sqrt(runner.ret_stats.var[0]),
            replay_n=runner.replay_n,
        )
        return (runner, knobs), metrics

    @jax.jit
    def train_chunk(runner: Runner, knobs):
        (runner, _), metrics = jax.lax.scan(update, (runner, knobs), None, length=cfg.updates_per_chunk)
        return runner, metrics

    return jax.jit(init), train_chunk


def export_runner(cfg: PPOConfig, runner: Runner):
    """Policy + critic params on raw observations (what checkpoints store and evaluators use)."""
    return export_params(runner.params, runner.obs_stats, cfg.obs_norm)


EVAL_CASES = {
    # name: (start mix, start noise, cart spread, push level)
    "still": ((1.0, 0.0, 0.0), 0.0, 0.0, 0.0),        # hanging still, cart centred
    "perturbed": ((1.0, 0.0, 0.0), 1.0, 0.0, 0.0),    # hanging, cart and links perturbed
    "random": ((0.0, 1.0, 0.0), 0.0, 0.0, 0.0),       # any angles, +-3 rad/s
    "edge": ((0.0, 1.0, 0.0), 0.0, 1.0, 0.0),         # random, cart up to +-2.5 m and +-1.5 m/s
    "push": ((0.0, 0.0, 1.0), 0.0, 0.0, 0.2),         # near upright, tip pushes at level 0.2
}


def make_evaluator(env: DoublePendulumCart, num_episodes: int = 64):
    """Checkpoint test: deterministic policy (the Gaussian mean), real physics, 20 s episodes,
    num_episodes per case in EVAL_CASES. An episode succeeds if the cart never leaves the track
    and the pendulum is upright >= 90% of the last 5 s.

    Returns evaluate(params, key) -> dict with
        success        success from hanging still (the headline number)
        swingup_time   mean time to first reach upright from hanging still (successful episodes) [s]
        upright_last5  mean fraction of the last 5 s spent upright, hanging still
        success_<case> success per case, off_track_<case> off-track rate per case
        score          mean success over all cases
    """
    last = env.steps(SUCCESS_WINDOW)

    def episode(params, key, case):
        mix, noise, cart_spread, push = case
        k_reset, k_push = jax.random.split(key)
        state0 = env.reset(k_reset, mix=jnp.array(mix), noise=noise, cart_spread=cart_spread)

        def step(carry, k):
            state, off_track = carry
            state = env.sample_push(k, state, push)
            tip = (jnp.where(state.push_left > 0, state.push_force, 0.0), 0.0)
            mean, _ = actor(params, env.obs(state))
            state, _, terminated, _ = env.step(state, mean, tip_force=tip)
            off_track = off_track | terminated  # the episode is over once the cart leaves the track
            return (state, off_track), (env.is_upright(state.q) & ~off_track, off_track)

        _, (upright, off_track) = jax.lax.scan(step, (state0, False), jax.random.split(k_push, env.max_steps))
        upright_last = upright[-last:].mean()
        success = (upright_last >= SUCCESS_FRACTION) & ~off_track[-1]
        first_up = jnp.where(upright.any(), jnp.argmax(upright) + 1, -1)
        return success, first_up * env.dt, upright_last, off_track[-1]

    @jax.jit
    def evaluate(params, key):
        out = {}
        for i, (name, case) in enumerate(EVAL_CASES.items()):
            keys = jax.random.split(jax.random.fold_in(key, i), num_episodes)
            success, first_up, upright_last, off = jax.vmap(lambda k: episode(params, k, case))(keys)
            out[f"success_{name}"] = success.mean()
            out[f"off_track_{name}"] = off.mean()
            if name == "still":
                out["success"] = success.mean()
                out["swingup_time"] = masked_mean(first_up, success)
                out["upright_last5"] = upright_last.mean()
        out["score"] = jnp.mean(jnp.array([out[f"success_{n}"] for n in EVAL_CASES]))
        return out

    return evaluate
