# Double (or n-link) pendulum on a cart: one PPO policy for swing-up, balance and recovery

A single neural-network policy, trained with PPO in pure JAX, that swings a double inverted
pendulum up from hanging, balances it on a cart with a finite track, and recovers from pushes
and arbitrary starting states. Training runs on a laptop CPU, and a browser dashboard shows
the training curves live and simulates any checkpoint in real time, so you can push the
pendulum with the mouse.

![Round-3 run: swing-up success and curriculum over 200M steps](docs/img/final_run_training.png)

## Results (current defaults, 4 seeds × 80M steps, ~2.5 h each with 6 runs sharing a laptop CPU)

Deterministic policy, real physics, 256 test episodes per case and seed:

| Start / test | Success | Off-track | Notes |
|---|---|---|---|
| Hanging still (cart centred) | **100%** | 0% | upright and settled at ~1.8 s, a single clean swing |
| Hanging, perturbed (cart ±1 m, links ±0.3 rad, swinging) | **100%** | 0% | |
| Fully random angles and velocities | **100%** | 0% | settled in 1.5 s median, 90% within 2.3 s |
| Random, cart near the ends (±2.5 m, ±1.5 m/s) | **100%** | 0% | |
| Tip pushes every ~2.5 s, level 0.1 / 0.2 / 0.3 | 100% / 71% / 31% | 0% | strong pushes knock it over, but it stays on the track |

Success means: on the track for the whole 20 s episode and upright (both links within 0.3 rad)
for at least 90% of its last 5 s. A track shield brakes the cart if it could not otherwise stop
before the end. Before it existed, up to 30% of the pushed episodes left the track. How we got
here, including what did not work, is in [docs/experiments.md](docs/experiments.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.13 and the dependencies).

```bash
uv run train.py
```

That trains for 200M steps and opens the dashboard at http://127.0.0.1:8000. Run folders go to
`runs/<timestamp>/`. Other entry points:

```bash
uv run train.py --total-steps 2e7 --no-server   # a short run without the dashboard
uv run train.py --links 3                        # a triple pendulum (any n works)
uv run serve.py                                  # dashboard only; follows the newest run
uv run diagnose.py runs/<run>                    # behaviour by start type (hanging / random / upright)
uv run experiments/robust_eval.py runs/<run>     # the stricter test used for the table above
uv run pytest                                    # tests (physics, rewards, curriculum, training step)
```

## How it works, in short

- **Physics** ([dprl/env.py](dprl/env.py)): Lagrangian dynamics of a cart with a chain of n point-mass
  links (two by default; `--links N`),
  RK4 at 100 Hz control (2 substeps of 5 ms), 30 N max force, track of ±3 m (leaving it ends the
  episode). Pure functions, so the environment is `jit`/`vmap`-ed together with the policy.
- **Reward**: upright term (half broad cosine, half sharp Gaussian), small cart-position, action and
  spin penalties, centring while upright, a *track bonus*, and potential-based *energy shaping* that
  fades out. A *track shield* overrides the action when the cart could not stop before the end.
- **PPO** ([dprl/ppo.py](dprl/ppo.py)): PureJaxRL style, the rollout and the update are one jitted
  function; 64 envs × 128 steps per update, 3×256 tanh MLPs, γ = 0.9975, reward scaling, entropy 0.005.
  The policy and critic are mirror-symmetric by construction (left/right swing-ups are learned as
  one), and the critic also predicts the time until the pendulum is stable upright. A discrete
  (categorical) action space is available too: even 3 force levels work as well, if decoded by
  their expected force.
- **Curriculum** ([dprl/curriculum.py](dprl/curriculum.py)), driven by the swing-up success of the
  hanging starts: easier physics (half gravity plus link drag) fades to real physics, then the
  share of near-upright starts drops from 50% to 10%, then starts get perturbed, random, close to
  the ends of the track and replayed from past near-failures, and finally random pushes get stronger.
- **Dashboard** ([serve.py](serve.py), [static/](static/)): a small FastAPI server that only reads run
  folders; the page runs physics and policy in the browser ([static/sim.mjs](static/sim.mjs), a JS port
  checked against the Python code by `parity.html`). It shows the critic's predicted value, can
  pick the swing direction by mirror symmetry, and ranks runs of a `--group` on a leaderboard with
  the best four simulated side by side.

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
