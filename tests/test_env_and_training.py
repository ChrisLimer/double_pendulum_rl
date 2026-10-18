import jax
import jax.numpy as jnp
import numpy as np

from dprl import runs
from dprl.curriculum import Curriculum
from dprl.env import DoublePendulumCart
from dprl.ppo import PPOConfig, make_evaluator, make_trainer


def test_energy_is_conserved_without_force():
    env = DoublePendulumCart()
    q, qd = jnp.array([0.0, 0.4, -0.3]), jnp.array([0.5, 0.0, 1.0])
    e0 = env.energy(q, qd)
    step = jax.jit(lambda q, qd: env.integrate(q, qd, 0.0))
    for _ in range(100):  # 2 s of free, chaotic motion
        q, qd = step(q, qd)
    assert abs(env.energy(q, qd) - e0) / abs(e0) < 1e-4


def test_tip_force_does_work_at_the_tip():
    """Power balance checks the J^T f generalized forces: dE/dt = f . v_tip."""
    env = DoublePendulumCart(dt=1e-3, substeps=1)
    q, qd = jnp.array([0.2, 0.7, -1.1]), jnp.array([0.3, -1.0, 2.0])
    f = (3.0, -2.0)
    v = env.tip_velocity(q, qd)
    q2, qd2 = env.integrate(q, qd, 0.0, f)
    v2 = env.tip_velocity(q2, qd2)
    power = f[0] * (v[0] + v2[0]) / 2 + f[1] * (v[1] + v2[1]) / 2
    assert np.isclose((env.energy(q2, qd2) - env.energy(q, qd)) / env.dt, power, rtol=1e-2)


def test_upright_is_an_equilibrium_but_unstable():
    env = DoublePendulumCart()
    q, qd = env.integrate(jnp.zeros(3), jnp.zeros(3), 0.0)
    assert np.allclose(q, 0) and np.allclose(qd, 0)
    q, _ = env.integrate(jnp.array([0.0, 0.05, 0.05]), jnp.zeros(3), 0.0)
    assert q[1] > 0.05


def test_energy_target_and_reward():
    env = DoublePendulumCart()
    up, down, still = jnp.zeros(3), jnp.array([0.0, jnp.pi, jnp.pi]), jnp.zeros(3)
    assert np.isclose(env.pendulum_energy(up, still), env.target_energy)
    # Cart motion is not counted as pendulum energy.
    assert np.isclose(env.pendulum_energy(up, jnp.array([2.0, 0, 0])), env.target_energy)
    assert np.isclose(env.reward(up, still, 0.0), 1.0)
    assert np.isclose(env.reward(down, still, 0.0), 0.0)
    assert np.isclose(env.energy_closeness(up, still), 1.0) and np.isclose(env.energy_closeness(down, still), 0.0)
    # Never negative, even when spinning fast through the top.
    assert env.reward(up, jnp.array([3.0, 30.0, -30.0]), 1.0) >= 0


def test_potential_shaping_cannot_be_farmed():
    """Staying at the target energy earns nothing extra (slightly less, by (1-gamma)*phi);
    the additive form pays every step."""
    pot, add = DoublePendulumCart(), DoublePendulumCart(energy_shaping="additive")
    state = pot.reset(jax.random.key(0))._replace(q=jnp.zeros(3), qd=jnp.zeros(3))
    r_pot = pot.energy_shaping_reward(state, state.q, state.qd, False, 1.0)
    r_add = add.energy_shaping_reward(state, state.q, state.qd, False, 1.0)
    assert r_pot <= 0 and np.isclose(r_add, 1.0)
    # Pumping energy toward the target is rewarded.
    low = state._replace(q=jnp.array([0.0, jnp.pi, jnp.pi]))
    assert pot.energy_shaping_reward(low, jnp.array([0.0, 2.0, 2.0]), jnp.zeros(3), False, 1.0) > 0


def test_track_bonus_applies_while_hanging_and_stops_off_track():
    """The clipped task reward is 0 while hanging; the track bonus still rewards staying centred,
    and is never negative, so leaving the track always loses it."""
    env = DoublePendulumCart()
    down, still = jnp.array([0.0, jnp.pi, jnp.pi]), jnp.zeros(3)
    assert np.isclose(env.reward(down, still, 0.0), 0.0)
    centred = env.track_reward(down, False)
    near_end = env.track_reward(down.at[0].set(0.9 * env.x_limit), False)
    assert np.isclose(centred, env.track_bonus) and 0 < near_end < centred
    assert env.track_reward(down.at[0].set(1.1 * env.x_limit), True) == 0
    state = env.reset(jax.random.key(0))._replace(q=down, qd=still)
    _, r, _, _ = env.step(state, jnp.zeros(1))
    assert np.isclose(r, env.track_bonus, atol=1e-3)


def test_reset_mixture_covers_hanging_and_upright():
    env = DoublePendulumCart()
    states = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 2000))
    down = np.cos(states.q[:, 1]) < -0.9
    up = np.cos(states.q[:, 1]) > 0.9
    assert 0.4 < down.mean() < 0.6 and 0.4 < up.mean() < 0.6 and np.all(down | up)
    assert np.all(np.abs(states.q[:, 1:]) <= np.pi)
    # Hanging starts are centred and still.
    assert np.allclose(states.q[down, 0], 0) and np.allclose(states.qd[down], 0)
    assert np.all(np.cos(states.q[down, 1:]) < -0.999)


def test_pushes_happen_at_full_level_only():
    env = DoublePendulumCart()
    state = env.reset(jax.random.key(0))
    keys = jax.random.split(jax.random.key(1), 2000)
    starts = jax.vmap(lambda k: env.sample_push(k, state, 1.0).push_left > 0)(keys).mean()
    assert np.isclose(starts, env.dt / env.push_interval, atol=0.005)
    assert not jax.vmap(lambda k: env.sample_push(k, state, 0.0).push_left > 0)(keys).any()


def test_training_success_matches_the_evaluator_test():
    """swingup_success: on the track and upright >= 90% of the last 5 s of a full episode."""
    env = DoublePendulumCart()
    knobs = Curriculum(episode_steps=600).knobs(0.0)
    still_up = env.reset(jax.random.key(0))._replace(q=jnp.zeros(3), qd=jnp.zeros(3))

    def run(state):
        def step(s, k):
            s2, _, _, info = env.step_autoreset(k, s, jnp.zeros(1), knobs)
            return s2, (info["done"], info["swingup_success"])
        _, (done, success) = jax.lax.scan(step, state, jax.random.split(jax.random.key(1), 600))
        assert done[-1] and not done[:-1].any()
        return bool(success[-1])

    assert run(still_up)  # stays balanced at the unstable equilibrium with no force
    assert not run(still_up._replace(q=jnp.array([0.0, jnp.pi, jnp.pi])))  # hangs the whole time


def test_curriculum_eases_physics_first():
    c = Curriculum()
    k = c.knobs(0.0)
    assert np.isclose(k["gravity_scale"], 0.5) and np.isclose(k["damping"], 0.1)
    for _ in range(50):
        c.update(0.95)
    k = c.knobs(0.0)
    assert c.ease == 0.0 and k["gravity_scale"] == 1.0 and k["damping"] == 0.0 and c.upright_share == 0.5
    c.update(0.95)
    assert np.isclose(c.upright_share, 0.48)
    for _ in range(2):
        c.update(0.5)  # failing: back to more near-upright starts, then easier physics again
    assert c.upright_share == 0.5 and np.isclose(c.ease, 0.02)


def test_easier_physics_is_slower():
    env = DoublePendulumCart()
    q0, qd0 = jnp.array([0.0, 0.3, 0.3]), jnp.zeros(3)
    real = env.integrate(q0, qd0, 0.0)[1]
    eased = env.integrate(q0, qd0, 0.0, gravity_scale=0.5, damping=0.1)[1]
    assert 0 < eased[1] < real[1]


def test_curriculum():
    c = Curriculum(gravity_relief=0.0, damping=0.0)
    assert c.energy_weight(0.0) == 1.0 and np.isclose(c.energy_weight(0.2), 0.55)
    assert np.isclose(c.energy_weight(0.9), 0.1)
    assert c.start_mix() == (0.5, 0.0, 0.5)
    c.update(0.5)  # failing: already at the maximum near-upright share
    assert c.upright_share == 0.5 and c.push_level == 0.0
    for _ in range(20 + 50 + 5):
        c.update(0.95)
    assert np.isclose(c.upright_share, 0.1) and c.start_noise == 1.0 and np.isclose(c.push_level, 0.02 * 5)
    assert np.allclose(c.start_mix(), (0.9 * 0.7, 0.9 * 0.3, 0.1))
    c.update(float("nan"))
    c.update(0.85)  # in between: no change
    assert np.isclose(c.push_level, 0.1)
    for _ in range(5 + 50 + 1):
        c.update(0.5)  # pushes go first, then start noise, then near-upright starts come back
    assert c.push_level == 0.0 and c.start_noise == 0.0 and np.isclose(c.upright_share, 0.12)


def test_training_chunk_eval_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    env = DoublePendulumCart()
    cfg = PPOConfig(total_timesteps=8 * 32 * 4, num_envs=8, num_steps=32, num_minibatches=4,
                    updates_per_chunk=2, hidden=(32, 32))
    init, train_chunk = make_trainer(cfg, env)
    c = Curriculum(push_level=0.5)
    runner, metrics = train_chunk(init(jax.random.key(0)), c.knobs(0.1))
    assert metrics["loss"].shape == (2,)
    assert np.isfinite(metrics["loss"]).all() and np.isfinite(metrics["upright"]).all()
    assert "swingup_success" in metrics  # the curriculum signal
    assert np.isfinite(metrics["time_loss"]).all() and np.isfinite(metrics["time_to_stable"]).all()

    ev = make_evaluator(env, num_episodes=4)(runner.params, jax.random.key(0))
    assert 0.0 <= float(ev["success"]) <= 1.0

    run = runs.new_run(cfg, env, Curriculum())
    runs.save_checkpoint(run, runner.params, {"update": 2, "env_steps": 512})
    assert runs.latest_run() == run
    _, cfg2, env2 = runs.load_config(run)
    params, meta = runs.load_checkpoint(runs.list_checkpoints(run)[-1], cfg2, env2)
    assert meta["update"] == 2 and env2 == env
    jax.tree.map(np.testing.assert_array_equal, params, runner.params)


def test_shield_keeps_the_cart_on_the_track():
    """Pushing full force toward the end from a fast start: the shield brakes in time."""
    env = DoublePendulumCart()
    state = env.reset(jax.random.key(0))._replace(q=jnp.array([1.0, 0.0, 0.0]), qd=jnp.array([4.0, 0.0, 0.0]))

    def step(s, _):
        s, _, term, _ = env.step(s, jnp.ones(1))
        return s, (s.q[0], s.shielded, term)

    _, (x, shielded, term) = jax.lax.scan(step, state, None, length=300)
    assert not term.any() and x.max() < env.x_limit and shielded.any()
    # Far from the ends it never interferes.
    a, active = env.apply_shield(jnp.zeros(3), jnp.array([1.0, 0.0, 0.0]), 0.7)
    assert not active and a == 0.7
    # Without the shield the same start leaves the track.
    free = DoublePendulumCart(shield=False)
    _, (_, _, term) = jax.lax.scan(lambda s, _: (lambda r: (r[0], (r[0].q[0], r[0].shielded, r[2])))(free.step(s, jnp.ones(1))),
                                   state, None, length=300)
    assert term.any()


def test_upright_centring():
    env = DoublePendulumCart()
    centred = env.reward(jnp.zeros(3), jnp.zeros(3), 0.0)
    off = env.reward(jnp.array([1.5, 0.0, 0.0]), jnp.zeros(3), 0.0)
    assert np.isclose(centred - off, (0.05 + env.upright_centring) * 0.25, atol=1e-6)


def test_exported_params_fold_the_observation_normalization():
    from dprl.networks import actor, critic, init_params
    from dprl.ppo import Stats, export_params
    params = init_params(jax.random.key(0), 8, 1, (16, 16))
    stats = Stats(jnp.linspace(-1, 1, 8), jnp.linspace(0.5, 2.0, 8), jnp.float32(100))
    obs = jax.random.normal(jax.random.key(1), (5, 8))
    norm = (obs - stats.mean) / jnp.sqrt(stats.var + 1e-8)
    folded = export_params(params, stats)
    np.testing.assert_allclose(actor(folded, obs)[0], actor(params, norm)[0], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(critic(folded, obs), critic(params, norm), rtol=1e-4, atol=1e-5)


def test_replay_starts_use_the_buffer():
    env = DoublePendulumCart()
    knobs = Curriculum().knobs(0.0)
    hard = jnp.array([[2.0, 0.5, -0.5, 1.0, 2.0, -2.0]])
    knobs.update(replay_s=hard, replay_n=jnp.int32(1), replay_prob=jnp.float32(1.0), episode_steps=jnp.int32(1))
    state = env.reset(jax.random.key(0))
    state, _, _, _ = env.step_autoreset(jax.random.key(1), state, jnp.zeros(1), knobs)  # episode ends, reset
    assert state.start_replay and np.allclose(state.q, hard[0, :3]) and np.allclose(state.qd, hard[0, 3:])


def test_symmetric_policy_and_critic():
    from dprl.networks import MIRROR, TIE_BREAK_FORCE, actor, critic_outputs, init_params
    env = DoublePendulumCart()
    params = init_params(jax.random.key(0), 8, 1, (32, 32), symmetric=True, critic_heads=3)
    params["actor"][-1]["w"] = params["actor"][-1]["w"] * 100  # a non-trivial policy
    states = jax.vmap(lambda k: env.reset(k, mix=(0.0, 1.0, 0.0)))(jax.random.split(jax.random.key(1), 64))
    obs = jax.vmap(env.obs)(states)
    mean, _ = actor(params, obs)
    mirrored, _ = actor(params, obs * MIRROR)
    far = (np.cos(states.q[:, 1]) > -0.5)  # away from the hanging tie-break region
    np.testing.assert_allclose(mirrored[far], -mean[far], atol=1e-5)
    np.testing.assert_allclose(critic_outputs(params, obs * MIRROR), critic_outputs(params, obs), atol=1e-5)
    # Fixed points of the mirror map: upright -> exactly 0; hanging still -> the tie-break push.
    up = env.obs(env.reset(jax.random.key(0))._replace(q=jnp.zeros(3), qd=jnp.zeros(3)))
    down = env.obs(env.reset(jax.random.key(0))._replace(q=jnp.array([0.0, jnp.pi, jnp.pi]), qd=jnp.zeros(3)))
    assert np.isclose(actor(params, up)[0][0], 0.0, atol=1e-6)
    assert np.isclose(actor(params, down)[0][0], TIE_BREAK_FORCE, atol=1e-4)


def test_stable_upright_test():
    from dprl.env import stable_obs
    env = DoublePendulumCart()
    s = env.reset(jax.random.key(0))
    assert stable_obs(env.obs(s._replace(q=jnp.array([1.0, 0.1, -0.1]), qd=jnp.array([0.5, 1.0, -1.0]))))
    assert not stable_obs(env.obs(s._replace(q=jnp.array([0.0, 0.3, 0.0]), qd=jnp.zeros(3))))
    assert not stable_obs(env.obs(s._replace(q=jnp.zeros(3), qd=jnp.array([0.0, 2.0, 0.0]))))


def test_discrete_policy_trains_and_is_symmetric(tmp_path, monkeypatch):
    from dprl.networks import MIRROR, actor, init_params, policy_logits
    params = init_params(jax.random.key(0), 8, 1, (32, 32), symmetric=True, action_bins=5)
    params["actor"][-1]["w"] = params["actor"][-1]["w"] * 100
    obs = jax.vmap(DoublePendulumCart().obs)(jax.vmap(lambda k: DoublePendulumCart().reset(k, mix=(0.0, 1.0, 0.0)))(
        jax.random.split(jax.random.key(1), 32)))
    np.testing.assert_allclose(policy_logits(params, obs * MIRROR), policy_logits(params, obs)[:, ::-1], atol=1e-5)
    level, _ = actor(params, obs)
    assert set(np.round(np.asarray(level[:, 0]), 3)) <= {-1.0, -0.5, 0.0, 0.5, 1.0}
    # One training chunk with a 3-level policy, then a checkpoint round trip.
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    env = DoublePendulumCart()
    cfg = PPOConfig(total_timesteps=8 * 32 * 4, num_envs=8, num_steps=32, num_minibatches=4,
                    updates_per_chunk=2, hidden=(32, 32), action_bins=3)
    init, train_chunk = make_trainer(cfg, env)
    runner, metrics = train_chunk(init(jax.random.key(0)), Curriculum().knobs(0.1))
    assert np.isfinite(metrics["loss"]).all() and (metrics["action_std"] > 0.5).all()
    run = runs.new_run(cfg, env, Curriculum())
    runs.save_checkpoint(run, runner.params, {"update": 2, "env_steps": 512})
    _, cfg2, env2 = runs.load_config(run)
    loaded, _ = runs.load_checkpoint(runs.list_checkpoints(run)[-1], cfg2, env2)
    np.testing.assert_array_equal(loaded["bins"], [-1.0, 0.0, 1.0])
