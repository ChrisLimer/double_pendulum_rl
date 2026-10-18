"""Tiny actor-critic MLPs in plain JAX. Params are a nested dict of arrays (a pytree)."""

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


def init_params(key, obs_dim, act_dim, hidden=(64, 64)):
    ka, kc = jax.random.split(key)
    return {
        "actor": init_mlp(ka, [obs_dim, *hidden, act_dim], out_scale=0.01),
        "critic": init_mlp(kc, [obs_dim, *hidden, 1], out_scale=1.0),
        "log_std": jnp.full(act_dim, -0.5),  # state-independent Gaussian std
    }


def actor(params, obs):
    """Gaussian policy: returns (mean, log_std). Works for single or batched obs."""
    mean = mlp(params["actor"], obs)
    return mean, jnp.broadcast_to(params["log_std"], mean.shape)


def critic(params, obs):
    return mlp(params["critic"], obs)[..., 0]


def gaussian_logp(x, mean, log_std):
    return jnp.sum(-0.5 * ((x - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
