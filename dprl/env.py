"""Double pendulum on a cart, as pure JAX functions (jit/vmap friendly).

Generalized coordinates q = (x, th1, th2):
    x    cart position [m]
    th1  angle of the lower link from vertical-up [rad], wrapped to [-pi, pi)
    th2  angle of the upper link from vertical-up [rad]  (absolute, not relative)
Both links are massless rods with point masses m1, m2 at their ends.
The action is a horizontal force on the cart, a in [-1, 1] scaled by max_force.

The task covers swing-up, balancing and recovery: training episodes start hanging still or
near upright (the curriculum sets the mix), only end when the cart leaves the track (or at
the time limit), and random pushes on the tip knock the pendulum around.

static/sim.mjs is a JavaScript port of dynamics/integrate/obs for the browser simulator;
keep the two in sync (static/parity.html checks them against each other).
"""

from dataclasses import asdict, dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

UPRIGHT_ANGLE = 0.3   # both links within this of vertical counts as "upright" for metrics
# Settle-in time before "upright" counts for the upright_late metric: 1 s for episodes that
# start upright, 5 s for episodes that start down (time to swing up).
SETTLE_TIME_UP, SETTLE_TIME_DOWN = 1.0, 5.0
# A swing-up episode succeeds if it stays on the track and is upright for >= 90% of its last 5 s
# (the same test as ppo.make_evaluator).
SUCCESS_WINDOW, SUCCESS_FRACTION = 5.0, 0.9
# Default initial-state mixture: (hanging still, fully random, near upright).
DEFAULT_START_MIX = (0.5, 0.0, 0.5)
# Replay starts: states this long before the shield fired or the cart left the track.
REPLAY_LOOKBACK = 0.5
# "Stable upright" for the time-to-stable prediction: both links within STABLE_ANGLE of vertical,
# both link rates below STABLE_RATE and the cart slower than STABLE_CART_SPEED.
STABLE_ANGLE, STABLE_RATE, STABLE_CART_SPEED = 0.2, 1.5, 1.0


def stable_obs(obs):
    """Is the observed state stable upright? Works on raw observations (single or batched)."""
    cos_limit = jnp.cos(STABLE_ANGLE)
    return ((obs[..., 2] > cos_limit) & (obs[..., 4] > cos_limit)
            & (jnp.abs(obs[..., 6]) * 10.0 < STABLE_RATE) & (jnp.abs(obs[..., 7]) * 10.0 < STABLE_RATE)
            & (jnp.abs(obs[..., 5]) * 3.0 < STABLE_CART_SPEED))


class EnvState(NamedTuple):
    q: jax.Array            # (3,) positions
    qd: jax.Array           # (3,) velocities
    t: jax.Array            # step counter within the episode
    ep_return: jax.Array    # running return, for logging
    push_left: jax.Array    # steps left of the current push (0 = no push)
    push_force: jax.Array   # horizontal force on the tip during the push [N]
    since_push: jax.Array   # steps since the last push started
    push_while_up: jax.Array  # was the pendulum upright when that push started?
    first_up: jax.Array     # first step it was upright (-1 = not yet)
    start_down: jax.Array   # did the episode start with the lower link below horizontal?
    start_upright: jax.Array  # was it a near-upright start (not hanging or fully random)?
    up_in_window: jax.Array   # upright steps within the last SUCCESS_WINDOW of the episode
    shielded: jax.Array     # did the track shield override the action in the last step?
    start_replay: jax.Array  # did the episode start from a replayed hard state?
    hist: jax.Array         # (H, 6) ring buffer of recent (q, qd), H = REPLAY_LOOKBACK / dt


def wrap(angle):
    return jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi


@dataclass(frozen=True)
class DoublePendulumCart:
    cart_mass: float = 1.0
    m1: float = 0.3
    m2: float = 0.3
    l1: float = 0.5
    l2: float = 0.5
    g: float = 9.81
    max_force: float = 30.0   # 20 N could not pump enough energy within the track
    dt: float = 0.01          # control period (100 Hz)
    substeps: int = 2         # RK4 substeps per control step (5 ms, as at 50 Hz x 4)
    max_steps: int = 2000     # 20 s episodes
    x_limit: float = 3.0      # cart leaves the track -> terminate
    off_track_penalty: float = 0.0  # one-off reward when the cart leaves the track
    # Track bonus: + track_bonus * (1 - (x / x_limit)^2) every step the cart is on the track,
    # added outside the >= 0 clip of the task reward. While the pendulum hangs the clipped task
    # reward is 0, so the cart-position penalty had no effect there and leaving the track cost
    # nothing; the swing-up then drove the cart off the end in ~60% of hanging starts. This term
    # keeps a reason to stay centred (and on the track) in every state. It is positive, so it
    # never makes ending the episode attractive.
    track_bonus: float = 0.1
    # Centring while balancing: - upright_centring * sharp * (x / x_limit)^2 inside the task reward,
    # with sharp the narrow upright bump (so it only acts near the top). The 0.05 cart-position
    # penalty alone let the policy balance 1.5 m off centre, leaving no room to recover from a push.
    upright_centring: float = 0.5
    # Track shield: overrides the action with full force toward the centre when the cart could
    # not stop before the end of the track by braking at shield_decel (a conservative fraction of
    # max_force / total mass, because the swinging links push back on the cart) plus shield_margin.
    # The policy is trained with the shield in the loop and pays shield_penalty per overridden step,
    # so it learns not to need it. static/sim.mjs implements the same shield.
    shield: bool = True
    shield_decel: float = 9.0     # [m/s^2]; max_force / total mass = 18.75
    shield_margin: float = 0.25   # [m]
    shield_penalty: float = 0.3
    # Random pushes on the tip (scaled by the curriculum's push level in [0, 1]).
    push_max_force: float = 10.0
    push_interval: float = 2.5    # mean seconds between pushes
    push_min_steps: int = 10      # push lasts 0.1-0.2 s
    push_max_steps: int = 20
    # Energy reward, phi = clip(1 - |E - E*| / (2 E*), 0, 1), weighted by the curriculum's w_E:
    #   "potential": add gamma*phi(s') - phi(s) (scaled by energy_scale). Rewards moving toward
    #                the target energy but not staying there, so it can't be farmed by spinning
    #                at the right energy, and it provably keeps the optimal policy unchanged.
    #   "additive":  add w_E * phi(s') every step (simple, but the policy may learn to keep the
    #                pendulum swinging at E* instead of balancing).
    energy_shaping: str = "potential"
    upright_sigma: float = 0.25   # width of the sharp upright bonus [rad]
    energy_scale: float = 3.0     # 10 made the early swing-up pump hard enough to leave the track
    shaping_gamma: float = 0.9975  # must equal the PPO discount for the potential form

    obs_dim = 8
    act_dim = 1

    def to_dict(self):
        return asdict(self)

    # ---- physics ------------------------------------------------------------

    def dynamics(self, q, qd, force, tip_force=(0.0, 0.0), gravity_scale=1.0, damping=0.0):
        """Accelerations qdd from  D(q) qdd = tau - C(q, qd) qd - G(q).

        force acts horizontally on the cart; tip_force = (fx, fy) acts on the upper mass and
        enters as generalized forces J^T f, with J the Jacobian of the tip position.
        gravity_scale and damping (a drag torque -damping * w on each link, like a thick
        atmosphere) only differ from 1 and 0 during training, set by the curriculum.
        """
        M, m1, m2, l1, l2, g = self.cart_mass, self.m1, self.m2, self.l1, self.l2, self.g * gravity_scale
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

    def energy(self, q, qd):
        """Total mechanical energy of cart + pendulum (used by the tests)."""
        xd = qd[0]
        v2x, v2y = self.tip_velocity(q, qd)
        v1x = xd + self.l1 * jnp.cos(q[1]) * qd[1]
        v1y = -self.l1 * jnp.sin(q[1]) * qd[1]
        kinetic = 0.5 * (self.cart_mass * xd**2 + self.m1 * (v1x**2 + v1y**2) + self.m2 * (v2x**2 + v2y**2))
        return kinetic + self.potential(q)

    def potential(self, q):
        _, th1, th2 = q
        return self.g * ((self.m1 + self.m2) * self.l1 * jnp.cos(th1) + self.m2 * self.l2 * jnp.cos(th2))

    def pendulum_energy(self, q, qd, gravity_scale=1.0):
        """Energy of the links as seen from the cart (cart motion excluded), so sliding
        the cart doesn't count. Equals target_energy when standing still upright."""
        _, th1, th2 = q
        _, w1, w2 = qd
        v2x = self.l1 * jnp.cos(th1) * w1 + self.l2 * jnp.cos(th2) * w2
        v2y = -self.l1 * jnp.sin(th1) * w1 - self.l2 * jnp.sin(th2) * w2
        kinetic = 0.5 * (self.m1 * (self.l1 * w1) ** 2 + self.m2 * (v2x**2 + v2y**2))
        return kinetic + gravity_scale * self.potential(q)

    @property
    def target_energy(self):
        return self.g * ((self.m1 + self.m2) * self.l1 + self.m2 * self.l2)

    def tip_velocity(self, q, qd):
        _, th1, th2 = q
        xd, w1, w2 = qd
        return (xd + self.l1 * jnp.cos(th1) * w1 + self.l2 * jnp.cos(th2) * w2,
                -self.l1 * jnp.sin(th1) * w1 - self.l2 * jnp.sin(th2) * w2)

    def integrate(self, q, qd, force, tip_force=(0.0, 0.0), gravity_scale=1.0, damping=0.0):
        """Advance one control period with fixed-step RK4 (forces held constant)."""
        h = self.dt / self.substeps

        def f(y):
            return jnp.concatenate([y[3:], self.dynamics(y[:3], y[3:], force, tip_force, gravity_scale, damping)])

        y = jnp.concatenate([q, qd])
        for _ in range(self.substeps):  # unrolled at trace time
            k1 = f(y)
            k2 = f(y + 0.5 * h * k1)
            k3 = f(y + 0.5 * h * k2)
            k4 = f(y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q = y[:3].at[1:].set(wrap(y[1:3]))
        return q, y[3:]

    def is_upright(self, q):
        return (jnp.cos(q[1]) > jnp.cos(UPRIGHT_ANGLE)) & (jnp.cos(q[2]) > jnp.cos(UPRIGHT_ANGLE))

    # ---- RL interface -------------------------------------------------------

    def steps(self, seconds: float) -> int:
        return round(seconds / self.dt)

    def reset(self, key, mix=DEFAULT_START_MIX, spread=1.0, noise=0.0, cart_spread=0.0) -> EnvState:
        """Sample a start from a mixture of (hanging, fully random, near upright).
        Hanging: at noise = 0 the cart is centred and at rest and the links hang still within
        0.02 rad of straight down; noise in [0, 1] widens this up to cart within 1 m and 0.5 m/s,
        links within 0.3 rad and 0.5 rad/s.
        Near upright: cart within 1 m, up to 0.4 * spread rad and spread rad/s.
        cart_spread in [0, 1] widens the cart state of all starts (hanging scaled by noise) from
        +-1 m and +-0.5 m/s up to +-2.5 m and +-1.5 m/s: hard starts close to the ends."""
        k_mode, k_x, k_th, k_w = jax.random.split(key, 4)
        mode = jax.random.choice(k_mode, 3, p=jnp.array(mix))
        u = lambda k, lo, hi: jax.random.uniform(k, (2,), minval=lo, maxval=hi)  # noqa: E731
        ks = jax.random.split(k_th, 3)
        kw = jax.random.split(k_w, 3)
        th = jnp.select(
            [mode == 0, mode == 1],
            [jnp.pi + u(ks[0], -1.0, 1.0) * (0.02 + 0.28 * noise), u(ks[1], -jnp.pi, jnp.pi)],
            u(ks[2], -0.4 * spread, 0.4 * spread),
        )
        w = jnp.select([mode == 0, mode == 1], [u(kw[0], -0.5, 0.5) * noise, u(kw[1], -3.0, 3.0)], u(kw[2], -spread, spread))
        x, xd = jax.random.uniform(k_x, (2,), minval=-1.0, maxval=1.0)
        x, xd = x * (1.0 + 1.5 * cart_spread), xd * (0.5 + 1.0 * cart_spread)
        x, xd = jnp.where(mode == 0, noise * x, x), jnp.where(mode == 0, noise * xd, xd)
        q = jnp.array([x, *wrap(th)])
        qd = jnp.array([xd, *w])
        i0, f0 = jnp.int32(0), jnp.float32(0.0)
        return self.make_state(q, qd, mode == 2)

    def make_state(self, q, qd, start_upright=False, start_replay=False) -> EnvState:
        i0, f0 = jnp.int32(0), jnp.float32(0.0)
        hist = jnp.tile(jnp.concatenate([q, qd]), (self.steps(REPLAY_LOOKBACK), 1))
        return EnvState(q, qd, i0, f0, i0, f0, jnp.int32(10**6), jnp.bool_(False), jnp.int32(-1),
                        jnp.cos(q[1]) < 0, jnp.asarray(start_upright, bool), i0, jnp.bool_(False),
                        jnp.asarray(start_replay, bool), hist)

    def obs(self, state: EnvState):
        x, th1, th2 = state.q
        xd, w1, w2 = state.qd
        return jnp.array([
            x / self.x_limit,
            jnp.sin(th1), jnp.cos(th1),
            jnp.sin(th2), jnp.cos(th2),
            xd / 3.0, w1 / 10.0, w2 / 10.0,
        ])

    def reward(self, q, qd, a):
        """Task reward (without energy shaping)."""
        x, th1, th2 = q
        _, w1, w2 = qd
        # Upright term, 1 at the top: half a broad term that still guides from far away
        # (0 when a link hangs down) and half a sharp bump around vertical. The broad term
        # alone is very flat near the top (still 0.7 at 0.8 rad off), which made wobbling
        # nearly as good as balancing and balancing slow to learn.
        broad = (1 + jnp.cos(th1)) / 2 * (1 + jnp.cos(th2)) / 2
        sharp = jnp.exp(-(th1**2 + th2**2) / (2 * self.upright_sigma**2))
        upright = 0.5 * broad + 0.5 * sharp
        r = (upright
             - 0.05 * (x / self.x_limit) ** 2
             - self.upright_centring * sharp * (x / self.x_limit) ** 2
             - 0.01 * a**2
             - 0.01 * broad * (w1**2 + w2**2))  # near the top: settle, don't whirl through
        # Kept >= 0 so that ending the episode early (leaving the track) is never attractive.
        return jnp.maximum(r, 0.0)

    def track_reward(self, q, terminated):
        """Track bonus (see track_bonus): highest with the cart centred, 0 once it has left."""
        return jnp.where(terminated, 0.0, self.track_bonus * (1.0 - (q[0] / self.x_limit) ** 2))

    def energy_closeness(self, q, qd, gravity_scale=1.0):
        """1 at the energy of the pendulum standing still upright (E*), falling linearly to 0
        at hanging still (-E*), so a swing-up gets a signal all the way from the bottom."""
        target = gravity_scale * self.target_energy
        energy_err = jnp.abs(self.pendulum_energy(q, qd, gravity_scale) - target) / (2 * target)
        return jnp.clip(1.0 - energy_err, 0.0, 1.0)

    def energy_shaping_reward(self, state, q, qd, terminated, energy_weight, gravity_scale=1.0):
        if self.energy_shaping == "additive":
            return energy_weight * self.energy_closeness(q, qd, gravity_scale)
        scale = self.energy_scale * energy_weight
        phi = scale * self.energy_closeness(state.q, state.qd, gravity_scale)
        phi_next = jnp.where(terminated, 0.0, scale * self.energy_closeness(q, qd, gravity_scale))
        return self.shaping_gamma * phi_next - phi

    def apply_shield(self, q, qd, a):
        """Track shield (see shield): returns (action, overridden?)."""
        x, v = q[0], qd[0]
        direction = jnp.where(jnp.abs(v) > 0.01, jnp.sign(v), jnp.sign(x))  # toward this end
        room = self.x_limit - direction * x
        stopping = jnp.maximum(direction * v, 0.0) ** 2 / (2 * self.shield_decel)
        active = self.shield & (stopping + self.shield_margin >= room) & (direction * a > -1.0)
        return jnp.where(active, -direction, a), active

    def step(self, state: EnvState, action, energy_weight=0.0, tip_force=(0.0, 0.0), max_steps=None,
             gravity_scale=1.0, damping=0.0):
        """One control step. Returns (new_state, reward, terminated, truncated)."""
        a = jnp.clip(action[0], -1.0, 1.0)
        applied, shielded = self.apply_shield(state.q, state.qd, a)
        q, qd = self.integrate(state.q, state.qd, applied * self.max_force, tip_force, gravity_scale, damping)
        terminated = jnp.abs(q[0]) > self.x_limit
        reward = (self.reward(q, qd, a)
                  + self.energy_shaping_reward(state, q, qd, terminated, energy_weight, gravity_scale)
                  + self.track_reward(q, terminated)
                  - self.shield_penalty * shielded
                  - self.off_track_penalty * terminated)
        t = state.t + 1
        truncated = t >= (self.max_steps if max_steps is None else max_steps)
        first_up = jnp.where((state.first_up < 0) & self.is_upright(q), t, state.first_up)
        hist = state.hist.at[state.t % state.hist.shape[0]].set(jnp.concatenate([state.q, state.qd]))
        new = state._replace(q=q, qd=qd, t=t, ep_return=state.ep_return + reward, first_up=first_up,
                             shielded=shielded, hist=hist)
        return new, reward, terminated, truncated

    def sample_push(self, key, state: EnvState, push_level):
        """Maybe start a random push on the tip; count down a running one."""
        k_start, k_len, k_mag, k_sign = jax.random.split(key, 4)
        start = (state.push_left == 0) & (jax.random.uniform(k_start) < self.dt / self.push_interval) & (push_level > 0)
        length = jax.random.randint(k_len, (), self.push_min_steps, self.push_max_steps + 1)
        magnitude = push_level * self.push_max_force * jax.random.uniform(k_mag, minval=0.3, maxval=1.0)
        sign = jnp.where(jax.random.bernoulli(k_sign), 1.0, -1.0)
        return state._replace(
            push_left=jnp.where(start, length, jnp.maximum(state.push_left - 1, 0)),
            push_force=jnp.where(start, sign * magnitude, state.push_force),
            since_push=jnp.where(start, 0, state.since_push + 1),
            push_while_up=jnp.where(start, self.is_upright(state.q), state.push_while_up),
        )

    def step_autoreset(self, key, state: EnvState, action, knobs):
        """Training step: random pushes, then step, then reset finished episodes in place.

        knobs holds the curriculum settings (traced, so they can change without recompiling):
        energy_weight, push_level, start_mix (3 probabilities), start_spread, start_noise,
        cart_spread, episode_steps, gravity_scale, damping, and replay starts: replay_prob and a
        buffer replay_s (B, 6) of (q, qd) with replay_n valid entries.
        Returns (state, obs, reward, info). info["final_obs"] is the observation before the
        reset, so the trainer can bootstrap the value of time-limit truncations.
        """
        k_push, k_reset = jax.random.split(key)
        state = self.sample_push(k_push, state, knobs["push_level"])
        tip_force = (jnp.where(state.push_left > 0, state.push_force, 0.0), 0.0)
        next_state, reward, terminated, truncated = self.step(
            state, action, knobs["energy_weight"], tip_force, knobs["episode_steps"],
            knobs["gravity_scale"], knobs["damping"],
        )
        done = terminated | truncated
        upright = self.is_upright(next_state.q)
        in_window = next_state.t > knobs["episode_steps"] - self.steps(SUCCESS_WINDOW)
        next_state = next_state._replace(up_in_window=next_state.up_in_window + (upright & in_window))
        settle = jnp.where(next_state.start_down, self.steps(SETTLE_TIME_DOWN), self.steps(SETTLE_TIME_UP))
        info = {
            "terminated": terminated,
            "truncated": truncated & ~terminated,
            "done": done,
            "final_obs": self.obs(next_state),
            "ep_return": next_state.ep_return,
            "upright": upright,
            "late": next_state.t >= settle,
            "start_upright": next_state.start_upright,
            # 2-3 s after a push that hit an upright pendulum: is it upright again?
            "recovery_window": next_state.push_while_up & (next_state.since_push >= self.steps(2.0))
                               & (next_state.since_push < self.steps(3.0)),
            "swingup_episode": done & next_state.start_down & ~next_state.start_replay,
            "shielded": next_state.shielded,
            # hard state for the replay buffer: REPLAY_LOOKBACK before the shield first fires in a
            # stretch, or before the cart leaves the track (hist[t % H] is the oldest entry)
            "replay_candidate": ((next_state.shielded & ~state.shielded) | terminated)
                                & (next_state.t > next_state.hist.shape[0]),
            "replay_state": next_state.hist[next_state.t % next_state.hist.shape[0]],
            "swingup_time": next_state.first_up * self.dt,
            "reached_top": next_state.first_up >= 0,
            # same test as the evaluator: on the track and upright >= 90% of the last 5 s
            "swingup_success": truncated & ~terminated
                               & (next_state.up_in_window >= SUCCESS_FRACTION * self.steps(SUCCESS_WINDOW)),
        }
        k_reset, k_replay, k_pick = jax.random.split(k_reset, 3)
        reset_state = self.reset(k_reset, knobs["start_mix"], knobs["start_spread"], knobs["start_noise"],
                                 knobs.get("cart_spread", 0.0))
        replay_s, replay_n = knobs.get("replay_s", jnp.zeros((1, 6))), knobs.get("replay_n", 0)
        pick = replay_s[jax.random.randint(k_pick, (), 0, jnp.maximum(replay_n, 1))]
        replayed = self.make_state(pick[:3], pick[3:], start_replay=True)
        use_replay = (replay_n > 0) & (jax.random.uniform(k_replay) < knobs.get("replay_prob", 0.0))
        reset_state = jax.tree.map(lambda r, n: jnp.where(use_replay, r, n), replayed, reset_state)
        state = jax.tree.map(lambda r, n: jnp.where(done, r, n), reset_state, next_state)
        return state, self.obs(state), reward, info
