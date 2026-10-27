"""The original closed-form 2-link dynamics (before the n-link generalization), kept as a
reference: the generic n-link code must reproduce it exactly for two links."""
import jax.numpy as jnp


def two_link_dynamics(q, qd, force, tip_force=(0.0, 0.0), gravity_scale=1.0, damping=0.0,
                      M=1.0, m1=0.3, m2=0.3, l1=0.5, l2=0.5, g=9.81):
    g = g * gravity_scale
    _, th1, th2 = q
    _, w1, w2 = qd
    fx, fy = tip_force
    c1, s1, c2, s2 = jnp.cos(th1), jnp.sin(th1), jnp.cos(th2), jnp.sin(th2)
    c12, s12 = jnp.cos(th1 - th2), jnp.sin(th1 - th2)
    D = jnp.array([
        [M + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2],
        [(m1 + m2) * l1 * c1, (m1 + m2) * l1**2, m2 * l1 * l2 * c12],
        [m2 * l2 * c2, m2 * l1 * l2 * c12, m2 * l2**2],
    ])
    rhs = jnp.array([
        force + fx + (m1 + m2) * l1 * s1 * w1**2 + m2 * l2 * s2 * w2**2,
        (m1 + m2) * g * l1 * s1 - m2 * l1 * l2 * s12 * w2**2 + l1 * (fx * c1 - fy * s1) - damping * w1,
        m2 * g * l2 * s2 + m2 * l1 * l2 * s12 * w1**2 + l2 * (fx * c2 - fy * s2) - damping * w2,
    ])
    return jnp.linalg.solve(D, rhs)
