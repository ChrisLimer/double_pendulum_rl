# Training

## PPO

[dprl/ppo.py](../dprl/ppo.py) is PPO with a clipped objective and GAE in the
[PureJaxRL](https://github.com/luchris429/purejaxrl) style. The environment rollout, the advantage
computation and all epochs of minibatch updates are one jitted function, and `lax.scan` fuses 10
updates per call (`train_chunk`). The host only sees metrics between chunks, which is where the
curriculum is updated and checkpoints are written.

| Setting | Value | Note |
|---|---|---|
| parallel envs × rollout | 64 × 128 steps | small batches = more updates per sample |
| epochs × minibatches | 4 × 8 | |
| discount γ | 0.9975 | ~4 s horizon at 100 Hz (0.995 at 50 Hz) |
| GAE λ | 0.975 | same time scale as 0.95 at 50 Hz |
| clip / value coef / grad norm | 0.2 / 0.5 / 0.5 | value loss clipped as well |
| learning rate | 3e-4, linear decay to 0 | Adam, eps 1e-5 |
| networks | actor and critic MLPs, 3 × 256, tanh | 2 layers never learned the swing-up at 100 Hz |
| policy | Gaussian, state-independent log std (init −0.5) | the evaluator uses the mean |
| entropy bonus | 0 | |
| total | 200M env steps | ~70 min at ~50k steps/s on one Apple-silicon CPU |

- **Truncation:** at the time limit the value of the state the episode was cut off in is
  bootstrapped (`reward += γ V(s_final)`), so GAE can treat every `done` the same way. Only leaving
  the track is a real termination.
- **Knobs:** the curriculum values (`knobs`) are traced arrays, not Python constants, so changing
  them never triggers recompilation.

## Curriculum

[dprl/curriculum.py](../dprl/curriculum.py) runs on the host between chunks, using one signal: the
swing-up success rate of the hanging-start training episodes. Its test is the evaluator's: on the
track and upright ≥ 90% of the last 5 s.

While the success rate stays above 0.9, the curriculum moves one stage forward at a time
(0.02 per chunk):

0. **Easier physics → real physics.** Training starts with gravity × 0.5 and a link drag of
   0.1 N·m·s/rad, so motions are slower and easier to catch. The `ease` level goes from 1 to 0.
1. **Fewer near-upright starts.** Their share goes from 50% down to 10% (the floor keeps balancing
   practised), so more episodes practise the swing-up.
2. **Start noise 0 → 1.** Hanging starts get perturbed (cart off centre and moving, links swinging),
   and up to 30% of the non-upright starts become fully random. This keeps the policy from
   overfitting to the single hanging-still trajectory.
3. **Push level 0 → 1.** Random pushes on the tip get stronger.

Below 0.8 it goes back: pushes first, then start noise, then more near-upright starts, then
easier physics.

Separately, the energy-shaping weight `w_E` decays on a fixed schedule (1 → 0.1 over 40% of
training).

[docs/img/final_run_training.png](img/final_run_training.png) shows how the stages played out in
the final run. Physics was real by ~35M steps, near-upright starts were at the floor by ~38M and
start noise at its maximum by ~70M; pushes then crept up to ~0.2.

## Run folders

[dprl/runs.py](../dprl/runs.py) defines the on-disk format shared by the trainer (writer) and the
dashboard (reader):

```
runs/<timestamp>/
    config.json          env + PPO + curriculum settings, written once
    metrics.jsonl        one JSON line per PPO update (losses, upright fractions, swing-up rate,
                         curriculum levels, steps/s, ...)
    checkpoints/
        ckpt_000200.npz  policy params + metadata (eval result, curriculum state), written atomically
```

`runs/` is not under version control. A checkpoint is saved every 200 updates (~1.6M steps) and
evaluated on 64 hanging-still episodes. That evaluation, `make_evaluator`, uses the deterministic
policy and real physics, and computes success, time to the top and upright fraction in the last 5 s.

## Metrics worth watching

| Metric | Meaning |
|---|---|
| `swingup_success` | the curriculum signal (above) |
| `swingup_rate` | hanging-start episodes that ever reached the top |
| `upright_late` | upright fraction after a settle-in time (1 s from upright starts, 5 s from hanging) |
| `upright_late_up` | the same, near-upright starts only (balancing) |
| `recovery` | upright 2–3 s after a push that hit an upright pendulum |
| `action_std` | policy std; it collapsing early (≈0.005) meant the 2-layer policy had stopped exploring |

## Evaluating a checkpoint

```bash
uv run diagnose.py runs/<run> [ckpt_001234]
uv run experiments/robust_eval.py runs/<run> [ckpt_001234]
```

- **diagnose.py** reports reach-the-top rate, off-track before and after the top, and hold streaks
  for hanging (still), random and upright starts.
- **robust_eval.py** adds perturbed hanging starts, pushes at chosen levels, settling times, cart
  excursion and action jerk.
