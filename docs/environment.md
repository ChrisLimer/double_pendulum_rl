# Environment

Everything is in [dprl/env.py](../dprl/env.py) as pure JAX functions on an `EnvState` named tuple, so
thousands of environments are stepped with `vmap` inside the jitted training loop.

## System and dynamics

The environment is a cart with a chain of n links: two by default, three or more by configuration
(`PendulumCart(link_masses=..., link_lengths=...)`, or `train.py --links N`). The old name
`DoublePendulumCart` still works.

Generalized coordinates `q = (x, th_1, …, th_n)`:

- `x`: cart position [m]
- `th_i`: angle of link i (1 = lowest) from vertical-up [rad], wrapped to [-π, π). These are
  absolute angles, not relative to the link below.

The links are massless rods with point masses at their upper ends.

| Parameter | Value | |
|---|---|---|
| cart mass | 1.0 kg | |
| link masses | 0.3 kg each | `link_masses`, n = its length |
| link lengths | 0.5 m each | `link_lengths` |
| max cart force | 30 N | 20 N could not pump enough energy within the track |
| track | ±3 m | leaving it ends the episode |
| control rate | 100 Hz (`dt = 0.01`) | was 50 Hz in the first round of experiments |
| integrator | RK4, 2 substeps (5 ms); 4 substeps (2.5 ms) for 3+ links | force held constant over the step |
| episode | 20 s (2000 steps) | time-limit truncations are bootstrapped, not terminal |

`dynamics()` solves `D(q) q̈ = rhs` for the accelerations, in closed form for any n. Let
μ_i = Σ_{k≥i} m_k be the mass carried by link i, and μ_ij = μ_max(i,j):

```
D_xx = M + Σ m      D_xi = μ_i l_i cos θ_i      D_ij = μ_ij l_i l_j cos(θ_i − θ_j)
rhs_x = F + f_x + Σ_i μ_i l_i sin θ_i ω_i²
rhs_i = g μ_i l_i sin θ_i − Σ_j μ_ij l_i l_j sin(θ_i − θ_j) ω_j² + l_i (f_x cos θ_i − f_y sin θ_i) − c ω_i
```

For n = 2 this is exactly the original double-pendulum formula (`tests/reference_two_link.py`;
the tests compare the two). A horizontal force on the tip (the random pushes, or the mouse in the dashboard) enters as the generalized force `Jᵀf`,
with `J` the Jacobian of the tip position. Two extra arguments exist for the curriculum only
(see [training.md](training.md)): `gravity_scale` and `damping`, a drag torque `−c·ω` on each link
("thick air"). They are 1 and 0 outside training, so the evaluator, the diagnostics and the
dashboard always use the real physics.

The tests check this for 1 to 4 links:
- energy conservation without force
- the power balance of the tip force, `dE/dt = f · v_tip`
- that upright is an unstable equilibrium

With 3+ links the motion is faster. 5 ms RK4 steps lose ~2·10⁻⁴ of the energy in 2 s of chaotic
motion, and 2.5 ms steps keep it at float32 noise, hence 4 substeps.

## Observation and action

`obs = (x / 3, sin θ_1, cos θ_1, …, sin θ_n, cos θ_n, ẋ / 3, ω_1 / 10, …, ω_n / 10)`, 2 + 3n values
(8 for the double pendulum, 11 for the triple). Sine and cosine avoid the wrap-around jump at ±π.
`obs_layout()` gives the indices. The mirror image of an observation negates everything except
the cosines. The action `a ∈ [−1, 1]` is the cart force divided by
30 N (clipped).

## Reward

Per step, `r = task + track_bonus + energy_shaping`, as in `DoublePendulumCart.step`:

**Task reward** (`reward()`, clipped at ≥ 0 so ending the episode early is never attractive):

```
broad   = Π_i (1 + cos θ_i)/2                         # 1 upright, 0 if a link hangs down
sharp   = exp(−Σ_i θ_i² / (2 · 0.25²))                # narrow bump around vertical
upright = 0.5 · broad + 0.5 · sharp
task    = max(0, upright − 0.05 (x/3)² − 0.5 · sharp · (x/3)² − 0.01 a² − 0.01 · broad · Σ_i ω_i²)
```

The `0.5 · sharp · (x/3)²` term (`upright_centring`, added in phase A) pulls the cart back to
the centre while balancing. With only the 0.05 penalty, the final round-3 policy balanced up to
1.5 m off centre and then lacked the track to recover from a push.
The broad term alone is very flat near the top (still 0.7 at 0.8 rad off), so wobbling was almost
as good as balancing. The sharp half fixes that. The spin penalty is weighted by `broad`, so it acts
near the top (settle instead of whirling through) and not during the swing-up.

**Track bonus** (`track_reward()`): `+0.1 · (1 − (x/3)²)` on every step on the track, outside the
clip. While the pendulum hangs, the clipped task reward is 0. The cart-position penalty then had
no effect, and leaving the track cost nothing, so early policies drove the cart off the end in
~60% of swing-ups. The bonus keeps a reason to stay centred in every state, and because it is
positive it never makes termination attractive. It was one of the two changes that fixed the
off-track problem; see [experiments.md](experiments.md).

**Shield penalty**: `−0.3` for every step in which the track shield (below) overrode the action,
so the policy learns not to rely on it.

**Energy shaping** (`energy_shaping_reward()`): potential-based, `γ·Φ(s') − Φ(s)` with

```
E      = pendulum energy seen from the cart (cart motion not counted)
E*     = energy standing still upright (= g((m1+m2) l1 + m2 l2))
Φ(s)   = energy_scale · w_E · clip(1 − |E − E*| / (2 E*), 0, 1)
```

- Why `2E*` and not `E*`: the pendulum energy runs from `−E*` (hanging still) to `+E*` (upright
  still), so `|E − E*|` goes from `2E*` to 0. Dividing by `2E*` maps the whole swing-up onto
  0 → 1. With `E*`, Φ would be clipped to 0 for every `E ≤ 0`, the whole lower half of the swing,
  and the shaping would be silent exactly when pumping starts.
- Potential-based shaping provably leaves the optimal policy unchanged and can't be farmed by
  spinning at the target energy. An additive `w_E · closeness` bonus (still available as
  `energy_shaping="additive"`) was exploited that way. On termination Φ(s') = 0.
- `energy_scale = 3`. 10 made the early swing-up pump so hard that the cart left the track.
- `w_E` decays linearly from 1 to 0.1 over the first 40% of training (curriculum).

The scale of the task reward terms (the upright weights and the 0.05 / 0.01 / 0.01 penalties) was
agreed at the start of the project and has not been changed. Everything else was added on top and
is documented here.

## Episode starts

`reset(key, mix, spread, noise)` samples one of three start types with probabilities `mix`:

| Type | Start state |
|---|---|
| hanging | `noise = 0`: cart centred and at rest, links within 0.02 rad of straight down and still. `noise = 1`: cart within ±1 m and ±0.5 m/s, links within ±0.3 rad and ±0.5 rad/s. |
| fully random | any angles, ±3 rad/s, cart within ±1 m |
| near upright | cart within ±1 m, links within ±0.4·spread rad and ±spread rad/s |

The checkpoint evaluator starts hanging still (`noise = 0`), the start the task is defined by.
The curriculum sets `mix` and `noise` during training.

## Track shield

`apply_shield()` runs before every physics step, in training, evaluation and the browser
simulator alike. If braking at `shield_decel = 9 m/s²` could not stop the cart `shield_margin =
0.25 m` before the end of the track, it overrides the action with full force toward the centre.
Full force on the cart alone would give 18.75 m/s²; half of that leaves room for the reaction of
the swinging links. Far from the ends it never interferes. This is a "shield" in the sense of
[Alshiekh et al. 2018](https://arxiv.org/abs/1708.08611): a fixed safety layer that sits after
the learner and corrects only unsafe actions. The policy is trained with it in the loop.

It is not a mathematical guarantee: a violent swing can still drag the cart out. The checkpoint
evaluator reports the remaining off-track rate per test case.

## Hard starts and replay

- `reset(..., cart_spread)` widens the cart state of all starts from ±1 m / ±0.5 m/s up to
  ±2.5 m / ±1.5 m/s, close to the ends. The curriculum ties it to the start-noise level.
- During training, each environment keeps a 0.5 s ring buffer of its states. When the shield
  starts braking, or the cart leaves the track, the state from 0.5 s earlier is a *hard state*.
  The trainer collects these into a 1024-entry replay buffer, and the curriculum starts up to 20%
  of the episodes from them: a simple form of prioritized level replay, aimed at the failures.

## Pushes

`sample_push()`: with probability `dt / 2.5 s` per step, a horizontal push on the tip starts. It
lasts 0.1–0.2 s and has a random sign, with force `level · 10 N · U(0.3, 1)`. The push level comes
from the curriculum. The runs so far trained up to level ≈ 0.2.

## The JavaScript port

[static/sim.mjs](../static/sim.mjs) re-implements `dynamics`, `integrate`, `obs` and the policy MLP so
the dashboard can simulate checkpoints in the browser. Keep the two in sync. The parity page
(`/parity.html` on the dashboard) compares one step from 200 random states against the Python code.
