"""Tiny actor-critic MLPs in plain JAX. Params are a nested dict of arrays (a pytree).

Mirror symmetry (optional, params contain the marker leaf "sym"): the cart-pendulum is symmetric
under s -> M(s) (x, angles and velocities negated) with a -> -a. A symmetric policy and critic are
built by averaging the network over a state and its mirror image:

    mean(s)  = (f(o) - f(M o)) / 2 + tie_break(o)   so mean(M s) = -mean(s) (away from the tie-break)
    value(s) = (g(o) + g(M o)) / 2                  so value(M s) = value(s)

An equivariant mean is exactly 0 in states that are their own mirror image (cart centred, links
exactly vertical, nothing moving): neither direction is better there, so a deterministic policy
would wait forever when hanging perfectly still. tie_break() adds a small push in a fixed
direction only within ~0.05 (in observation units) of the hanging-still fixed point, which breaks
the tie; everywhere else the policy is exactly symmetric. (At the upright fixed point a zero action
is right, so no tie-break there.)

The critic can have extra outputs (heads); value() is output 0, see ppo.py for the others.

Discrete actions (optional, params contain "bins", the K force levels in [-1, 1], symmetric
around 0): the actor outputs K logits of a categorical policy instead of a Gaussian mean. The
mirror of action bin k is bin K-1-k, so the symmetric version averages f(o) with the reversed
f(M o). actor() then returns the expected force sum_k p_k bins_k as its "mean" (plus the same
tie-break as the continuous policy), so every deterministic user of actor() (evaluators,
diagnostics) works unchanged. The expected force is a smooth function of the state; the most
likely level (argmax) balances just as well but switches between large forces ~70 times per
second (see docs/experiments.md). actor_mode() gives that decoding for comparison."""

import math

import jax
import jax.numpy as jnp


def init_mlp(key, sizes, out_scale):
    """Orthogonal init: sqrt(2) gain on hidden layers, out_scale on the last one."""
    layers = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = out_scale if i == len(sizes) - 2 else math.sqrt(2)
        w = jax.nn.initializers.orthogonal(scale)(k, (n_in, n_out))
        layers.append({"w": w, "b": jnp.zeros(n_out)})
    return layers


def mlp(layers, x):
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


# Mirror of an observation (x, sin1, cos1, sin2, cos2, xd, w1, w2): cosines are unchanged.
MIRROR = jnp.array([-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
TIE_BREAK_FORCE = 0.3   # action units (x max_force)
TIE_BREAK_WIDTH = 0.05  # observation units


def init_params(key, obs_dim, act_dim, hidden=(64, 64), symmetric=False, critic_heads=1, action_bins=0):
    ka, kc = jax.random.split(key)
    params = {
        "actor": init_mlp(ka, [obs_dim, *hidden, action_bins or act_dim], out_scale=0.01),
        "critic": init_mlp(kc, [obs_dim, *hidden, critic_heads], out_scale=1.0),
        "log_std": jnp.full(act_dim, -0.5),  # state-independent Gaussian std
    }
    if symmetric:
        params["sym"] = jnp.zeros(())  # marker: the dict structure is static under jit
    if action_bins:
        params["bins"] = jnp.linspace(-1.0, 1.0, action_bins)
    return params


def tie_break(obs):
    """A fixed push near the hanging-still fixed point of the mirror map (see module docstring)."""
    asym = obs[..., [0, 1, 3, 5, 6, 7]]  # the components the mirror negates
    near = jnp.exp(-jnp.sum(asym**2, axis=-1, keepdims=True) / TIE_BREAK_WIDTH**2)
    hanging = jnp.clip(-obs[..., 2:3], 0.0, 1.0)  # lower link pointing down
    return TIE_BREAK_FORCE * near * hanging


def mlp_and_mirror(layers, obs):
    """The network on obs and on its mirror image, in one batched pass (much cheaper than two)."""
    out = mlp(layers, jnp.stack([obs, obs * MIRROR]))
    return out[0], out[1]


def policy_logits(params, obs):
    """Categorical policy over params["bins"]: (..., K) logits."""
    if "sym" in params:
        direct, mirrored = mlp_and_mirror(params["actor"], obs)
        return (direct + mirrored[..., ::-1]) / 2
    return mlp(params["actor"], obs)


def actor(params, obs):
    """Gaussian policy: returns (mean, log_std). Works for single or batched obs.
    For a discrete policy, "mean" is the most likely force level (and log_std is unused)."""
    if "bins" in params:
        bins = jnp.asarray(params["bins"])  # loaded checkpoints hold numpy arrays
        level = (jax.nn.softmax(policy_logits(params, obs)) @ bins)[..., None]
        if "sym" in params:
            level = level + tie_break(obs)
        return level, jnp.zeros_like(level)
    if "sym" in params:
        direct, mirrored = mlp_and_mirror(params["actor"], obs)
        mean = (direct - mirrored) / 2 + tie_break(obs)
    else:
        mean = mlp(params["actor"], obs)
    return mean, jnp.broadcast_to(params["log_std"], mean.shape)


def actor_mode(params, obs):
    """Discrete policies only: the most likely force level (argmax; lowest bin on ties)."""
    bins = jnp.asarray(params["bins"])
    return bins[jnp.argmax(policy_logits(params, obs), axis=-1)][..., None]


def critic_outputs(params, obs):
    """All critic heads, (..., n). Mirror-invariant when the params are symmetric."""
    if "sym" in params:
        direct, mirrored = mlp_and_mirror(params["critic"], obs)
        return (direct + mirrored) / 2
    return mlp(params["critic"], obs)


def critic(params, obs):
    return critic_outputs(params, obs)[..., 0]


def gaussian_logp(x, mean, log_std):
    return jnp.sum(-0.5 * ((x - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
