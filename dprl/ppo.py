"""PPO (clipped objective + GAE) in the PureJaxRL style: the whole update, including the
environment rollout, is one jitted function. Many updates are fused with lax.scan and the
host only sees the results between chunks (to log metrics and write checkpoints)."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .env import DoublePendulumCart, EnvState
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
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    hidden: tuple = (256, 256, 256)  # 2 layers never learned the swing-up at 100 Hz (0/4 seeds); 3 did (4/4)
    updates_per_chunk: int = 10   # updates fused into one jit call between host syncs

    @property
    def steps_per_update(self):
        return self.num_envs * self.num_steps

    @property
    def num_updates(self):
        return self.total_timesteps // self.steps_per_update


class Runner(NamedTuple):
    params: dict
    opt_state: optax.OptState
    env_state: EnvState
    obs: jax.Array
    key: jax.Array


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
        return Runner(params, tx.init(params), env_state, obs, key)

    def env_step(carry, _):
        runner, knobs = carry
        key, k_act, k_env = jax.random.split(runner.key, 3)
        mean, log_std = actor(runner.params, runner.obs)
        action = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
        logp = gaussian_logp(action, mean, log_std)
        value = critic(runner.params, runner.obs)
        env_state, obs, reward, info = jax.vmap(env.step_autoreset, in_axes=(0, 0, 0, None))(
            jax.random.split(k_env, cfg.num_envs), runner.env_state, action, knobs
        )
        # Time-limit truncation is not a real ending: bootstrap with the value of the state we
        # were cut off in, folded into the reward, so GAE can treat every `done` the same way.
        reward = reward + cfg.gamma * critic(runner.params, info["final_obs"]) * info["truncated"]
        transition = Transition(runner.obs, action, logp, value, reward, info["done"], info)
        runner = runner._replace(env_state=env_state, obs=obs, key=key)
        return (runner, knobs), transition

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
        adv, target = compute_gae(traj, critic(runner.params, runner.obs))
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
        )
        return (runner, knobs), metrics

    @jax.jit
    def train_chunk(runner: Runner, knobs):
        (runner, _), metrics = jax.lax.scan(update, (runner, knobs), None, length=cfg.updates_per_chunk)
        return runner, metrics

    return jax.jit(init), train_chunk


def make_evaluator(env: DoublePendulumCart, num_episodes: int = 64):
    """Checkpoint test: deterministic policy from hanging starts, no pushes.

    Returns evaluate(params, key) -> dict with
        success       fraction of episodes upright >= 90% of the last 5 s, never off track
        swingup_time  mean time to first reach upright (successful episodes) [s]
        upright_last5 mean fraction of the last 5 s spent upright
    """
    last = int(5.0 / env.dt)

    def episode(params, key):
        state0 = env.reset(key, mix=(1.0, 0.0, 0.0))

        def step(carry, _):
            state, off_track = carry
            mean, _ = actor(params, env.obs(state))
            state, _, terminated, _ = env.step(state, mean)
            off_track = off_track | terminated  # the episode is over once the cart leaves the track
            return (state, off_track), (env.is_upright(state.q) & ~off_track, off_track)

        _, (upright, off_track) = jax.lax.scan(step, (state0, False), None, length=env.max_steps)
        upright_last = upright[-last:].mean()
        success = (upright_last >= 0.9) & ~off_track[-1]
        first_up = jnp.where(upright.any(), jnp.argmax(upright) + 1, -1)
        return success, first_up * env.dt, upright_last

    @jax.jit
    def evaluate(params, key):
        success, first_up, upright_last = jax.vmap(episode, in_axes=(None, 0))(
            params, jax.random.split(key, num_episodes)
        )
        return {
            "success": success.mean(),
            "swingup_time": masked_mean(first_up, success),
            "upright_last5": upright_last.mean(),
        }

    return evaluate
