# Dashboard

```bash
uv run serve.py                          # follows the newest run in runs/
uv run serve.py --run runs/<timestamp>   # opens on this run
uv run serve.py --port 8001              # or PORT=8001 uv run serve.py
```

`train.py` starts it automatically unless you pass `--no-server`. The dashboard is a separate
process that only reads run folders, so it can be started, stopped or restarted at any time
without touching training.

## What the page shows

- **Header:** the run, a progress bar with steps and throughput, and the **show** picker, which
  switches to any run in `runs/` or `runs/<group>/` (or `?run=<name>` in the URL). Runs in `runs/imported/` are
  experiment results with final weights only (see
  [experiments/README.md](../experiments/README.md)); they have no training curves.
- **Charts:** each metric from `metrics.jsonl` (see [training.md](training.md)), raw and smoothed,
  updated every 2 s.
- **Simulator:** the selected checkpoint (or "Latest", which follows new checkpoints) running live
  in the browser at the control rate.
  - **Start** picks the start type: hanging still (the evaluator start), hanging perturbed, mixed,
    fully random or near upright.
  - **Pushing:** drag the tip with the mouse, or hold ← / → to push it with the force set by the
    slider.
  - **Space** starts a new episode, and **Endless** removes the 20 s limit.
  - **checkpoint test** shows the checkpoint's evaluator result.
  - **predicted value** shows the critic's estimate of the discounted return (γ = 0.9975, ~4 s
    horizon) for the current state and for its mirror image, in reward units. Runs trained with
    reward scaling are converted back with the scale logged at that checkpoint. Balancing
    upright is worth about 400.
  - **Discrete action** (only for discrete policies): show the policy with its expected force
    (smooth, the default) or with its most likely level (bang-bang).
  - **stable upright** shows the predicted time until the pendulum is stable upright, its spread,
    and the chance of reaching it within ±0.5 s of the prediction. "Stable upright" means links
    within 0.2 rad, rates below 1.5 rad/s, cart below 1 m/s. After the event it shows when it
    happened and what was predicted at the start.
  - **Symmetry:** for mirror-symmetric checkpoints the symmetry is built into the network, and the
    selector has no effect. For older checkpoints: the system is symmetric under x → −x, angles → −angles (velocities negated),
    force → −force, but a policy can learn a preferred swing direction.
    - "Auto" acts with whichever of s and mirror(s) the critic values higher. When acting
      mirrored, the action is −π(mirror(s)).
    - It switches only when the other side is ≥ 2% better and the current side has been used
      for ≥ 0.25 s.
    - "Always mirrored" shows the opposite tendency.
    - On random starts "auto" raised the baseline's success from 88% to 95% (see
      [experiments.md](experiments.md)).

- **Top runs** (only for runs in a `--group`): a leaderboard of the group ranked by the latest
  checkpoint's score, and the best four simulated side by side from the same start with
  deterministic actions, for a direct comparison. The cart turns orange while the track shield
  brakes.

## How the simulation works

The server sends the actor weights as base64 float32 (`/api/policy`) together with the env
config. [static/sim.mjs](../static/sim.mjs) then steps the physics (RK4) and the MLP policy in
JavaScript, in real time with no network round trip, which is what makes pushing responsive.

Because this duplicates `dprl/env.py`, the **Python ↔ JS physics check** link (`/parity.html`)
compares one control step from 200 random states, with random cart and tip forces, against the
JAX code (`/api/parity`): next state, observation, energy and the policy's action. Run it after
changing either side. At 100 Hz with the 3-layer policy the maximum relative error is ~1e-7.

## API

| Endpoint | Returns |
|---|---|
| `/api/runs` | available runs and the default one |
| `/api/leaderboard?group=&run=` | the runs of a group ranked by their latest checkpoint's score |
| `/api/status?run=` | config, number of checkpoints |
| `/api/metrics?since=&run=` | metric rows from index `since` |
| `/api/checkpoints?run=` | checkpoint names + metadata |
| `/api/policy?ckpt=&run=` | actor and critic weights, env config, `value_scale` (critic output → return units) |
| `/api/parity?n=&seed=&run=` | reference values for the parity page |

`run` is optional. It is matched against the list of existing runs, so a query can't read
anything else on disk.
