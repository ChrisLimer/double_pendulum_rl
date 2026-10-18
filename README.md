# Double pendulum on a cart: one PPO policy for swing-up, balance and recovery

A single neural-network policy, trained with PPO in pure JAX, that swings a double inverted
pendulum up from hanging, balances it on a cart with a finite track, and recovers from pushes
and arbitrary starting states. Training runs on a laptop CPU, and a browser dashboard shows
the training curves live and simulates any checkpoint in real time, so you can push the
pendulum with the mouse.

![Final run: swing-up success and curriculum over 200M steps](docs/img/final_run_training.png)

## Results (final run, 200M steps, ~70 min on an Apple-silicon CPU)

Deterministic policy, real physics, 256 test episodes per case:

| Start / test | Success | Time to upright and settled |
|---|---|---|
| Hanging still (cart centred) | **100%** | 1.62 s, a single clean swing, no second try |
| Hanging, perturbed (cart ±1 m, links ±0.3 rad, swinging) | **100%** | 1.6 s (90th percentile 1.7 s) |
| Fully random angles and velocities | 91% | ~2.1 s |
| Near upright | 99% | – |
| Random tip pushes every ~2.5 s, level 0.1 / 0.2 (0.3–1 N / 0.6–2 N for 0.1–0.2 s) | 98% / 72% | – |

Success means: on the track for the whole 20 s episode and upright (both links within 0.3 rad)
for at least 90% of its last 5 s. How we got there, including what did not work, is in
[docs/experiments.md](docs/experiments.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 and the dependencies).

```bash
uv run train.py
```

That trains for 200M steps and opens the dashboard at http://127.0.0.1:8000. Run folders go to
`runs/<timestamp>/`. Other entry points:

```bash
uv run train.py --total-steps 2e7 --no-server   # a short run without the dashboard
uv run serve.py                                  # dashboard only; follows the newest run
uv run diagnose.py runs/<run>                    # behaviour by start type (hanging / random / upright)
uv run experiments/robust_eval.py runs/<run>     # the stricter test used for the table above
uv run pytest                                    # tests (physics, rewards, curriculum, training step)
```

## How it works, in short

- **Physics** ([dprl/env.py](dprl/env.py)): Lagrangian dynamics of a cart with two point-mass links,
  RK4 at 100 Hz control (2 substeps of 5 ms), 30 N max force, track of ±3 m (leaving it ends the
  episode). Pure functions, so the environment is `jit`/`vmap`-ed together with the policy.
- **Reward**: upright term (half broad cosine, half sharp Gaussian), small cart-position, action and
  spin penalties, a centring *track bonus*, and potential-based *energy shaping* that fades out.
- **PPO** ([dprl/ppo.py](dprl/ppo.py)): PureJaxRL style, the rollout and the update are one jitted
  function; 64 envs × 128 steps per update, 3×256 tanh MLPs, γ = 0.9975.
- **Curriculum** ([dprl/curriculum.py](dprl/curriculum.py)), driven by the swing-up success of the
  hanging starts: easier physics (half gravity plus link drag) fades to real physics, then the
  share of near-upright starts drops from 50% to 10%, then starts get perturbed and random, and
  finally random pushes get stronger.
- **Dashboard** ([serve.py](serve.py), [static/](static/)): a small FastAPI server that only reads run
  folders; the page runs physics and policy in the browser ([static/sim.mjs](static/sim.mjs), a JS port
  checked against the Python code by `parity.html`).

## Documentation

| | |
|---|---|
| [docs/environment.md](docs/environment.md) | dynamics, observations, reward terms, starts, pushes |
| [docs/training.md](docs/training.md) | PPO setup, curriculum, metrics, run folders, evaluation |
| [docs/dashboard.md](docs/dashboard.md) | the live dashboard, browser simulator and parity check |
| [docs/experiments.md](docs/experiments.md) | the full experiment history and analysis, with figures |
| [experiments/README.md](experiments/README.md) | the scripts used for the multi-seed experiments |

## Layout

```
dprl/            env.py (physics + rewards), ppo.py (trainer + evaluator), curriculum.py,
                 networks.py (MLPs), runs.py (run folder format)
train.py         training entry point (+ dashboard)
serve.py         dashboard server
diagnose.py      per-start-type behaviour of a checkpoint
static/          dashboard page, browser simulator, parity check
experiments/     multi-seed experiment harness, robustness eval, population training, figures
docs/            documentation and figures (images in Git LFS)
tests/           pytest suite
```
