# Experiments and analysis

This is the history of how the policy got from "balances, but drives off the track when swinging
up" to "swings up in ~1.6 s every time, recovers from random states, and does not leave the
track". It includes what
did not work. The raw results are in [experiments/results/](../experiments/results/), and the
scripts are described in [experiments/README.md](../experiments/README.md).

## Goal and how success is measured

One policy that swings the pendulum up from hanging, balances it, recovers from pushes, and
re-swings if it falls. The headline metric is the checkpoint evaluator's
(`ppo.make_evaluator`): deterministic policy, real physics, 20 s episodes from a hanging start. An
episode succeeds if the cart never leaves the track and the pendulum is upright (both links within
0.3 rad) for at least 90% of the last 5 s.

Two things to keep in mind when reading the numbers:

- **Single seeds are very noisy.** In round 1 the same configuration scored anywhere from 0% to
  29% across seeds. Every conclusion below is from 3–6 seeds, and the curriculum's dynamics were
  the main source of the noise.
- **The definition of "hanging start" changed between rounds.** Round 1 used a perturbed start
  (cart ±1 m, links ±0.3 rad, ±0.5 rad/s); from round 2 on the evaluator starts hanging still at
  the centre. The numbers are only comparable within a round. The robustness test below checks
  both start types.

Compute: one Apple-silicon laptop CPU. PPO with this small model reaches ~70k steps/s at 50 Hz and
~32–50k at 100 Hz with 3 layers. Single-threaded XLA
(`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`) is just as fast
per process, so ~8 experiments ran in parallel ([experiments/queue.sh](../experiments/queue.sh)).

## Round 1: the cart leaves the track before reaching the top (50 Hz)

**Starting point.** The policy balanced well (~95% upright from near-upright starts), but from
hanging starts the cart left the track in ~60% of episodes before ever reaching the top. Swing-up
success was 15–27%.

**Diagnosis**, from rolling out the policy and reading the training metrics:

1. **Violent pumping.** From hanging, the policy pumped energy so fast that the cart reached the end
   of the track within ~1.2 s, with the energy still rising. The energy shaping at scale 10 paid
   for energy gain immediately; the penalty for leaving the track (losing Φ) came later.
2. **Leaving the track cost nothing while hanging.** The task reward is clipped at ≥ 0 and is 0
   while the pendulum hangs, so the cart-position penalty had no effect there. Nearly every
   training episode ended off the track.
3. **The curriculum fed back on itself.** Its signal mixed upright and hanging starts. Failed
   swing-ups pulled it down, which removed hanging starts from training, and in some seeds it
   collapsed back to difficulty 0, withdrawing swing-up practice when it was needed most.
4. **Policy std collapse** (to ~0.05), which reduced exploration.

**What was tried** (12–30M steps, 2–3 seeds unless noted):

| Change | Result |
|---|---|
| off-track penalty 50 | 0% (1 seed); the policy became timid |
| GAE λ = 0.98 | 0% (1 seed) |
| std floor (log std ≥ −1.6) / entropy bonus 0.01 | no clear effect |
| `energy_scale` 1 | worse (6%) |
| `energy_scale` 3 alone | 21% at 30M, but 15% at 60M (no better than baseline) |
| track bonus 0.1 alone | rarely off-track, but the curriculum stalled at difficulty ~0.45 |
| **track bonus 0.1 + `energy_scale` 3 + curriculum driven by upright starts only** | **best** |

**Result** (60M steps, 6 seeds each, [seed_comparisons.png](img/seed_comparisons.png), left):
34% mean success vs 13% for the baseline. The new configuration won 30 of 36 seed pairs (one-sided
Mann-Whitney p ≈ 0.02), and off-track-before-top fell from ~55% to 0–3% in 4 of 6 seeds.

![Per-seed success in round 1 (left) and round 2 (right)](img/seed_comparisons.png)

**A bug found on the way.** The first full run with the new curriculum signal scored only 16%. Its
"upright starts" were defined as "lower link above horizontal", which also counted many fully
random starts that can't be caught within the 1 s settle time. That held the signal at ~0.7, below
the 0.8 raise threshold, so the curriculum stalled and never trained on wide upright starts.
Counting only the near-upright start type fixed it.

**Full runs (120M steps):**

| | Old settings | New settings |
|---|---|---|
| reach the top | 38% | 78% |
| off-track before the top | 62% | **0%** |
| upright in the last 5 s | 25% | 44% |
| success | 27% | 28% |

The off-track problem was solved, but the policy now reached the top and could not hold it, so
overall success did not move. Catching and holding the top became the bottleneck.

## Round 2: 100 Hz, still starts, and a deeper network

The second round changed the setup:

- Control at 100 Hz (`dt` 0.01, 2 RK4 substeps), with γ, λ and the push duration rescaled to keep
  the same time constants.
- Hanging starts are hanging still with the cart centred.
- Near-upright starts are mixed in at 50%, and their share decreases while swing-up success is
  above 90% and increases below 80%. The earlier difficulty curriculum (episode length,
  start-spread, random starts) was dropped.
- Optionally, a third hidden layer.

With 2 hidden layers this **did not learn at all**: 0% success in 4 of 4 seeds at 60M. The policy
std collapsed to ~0.005 within 16M steps, so exploration stopped, and the success-driven curriculum
never moved. With **3 hidden layers** the same setup reached 100%, 78%, 100%, 100%
([seed_comparisons.png](img/seed_comparisons.png), right), and its std stayed at 0.03–0.06.

The full 3-layer run (120M) succeeded 100% from hanging still. It reached the top at 1.62 s and
stayed there, with no second attempt, using at most 1.25 m of track. But it had **overfitted to
that one start**: 63% success from perturbed hanging starts, 16% from random starts, and it fell to
almost any push.

## Round 3: robustness and speed

Three ideas were tested on top of the 3-layer setup.

**Easier physics first (`G`, 4 seeds, 60M).** Training starts with half gravity and a drag torque
of 0.1 N·m·s/rad on each link, and the curriculum fades both out before anything else. All 4 seeds
reached 100% from hanging still (3-layer baseline: 3 of 4), they settled faster (median 1.6–2.2 s
vs 2.6–3.0 s), and push recovery improved. Perturbed and random starts were still weak. We found
no prior work using exactly this gravity/drag curriculum; the closest are domain randomization
and environment curricula.

**Start noise stage (`EN` = easier physics + start noise, 4 seeds, 120M).** Once the near-upright
share is at its floor, hanging starts are progressively perturbed and fully random starts are
mixed in, before pushes. This removed the overfitting: 100% from still and from perturbed hanging
starts in every seed, settling at 1.6–1.9 s, and 93–99% from random starts.

**Population-based training (4 members, 80M steps each).** All members train in one process (the
PPO runner `vmap`-ed over the population). Every 8M steps they are ranked on a robustness score,
and the weakest is replaced by a copy of the best: weights, optimizer state and curriculum, with
new random keys. This is the "exploit" half of
[PBT (Jaderberg et al. 2017)](https://arxiv.org/abs/1711.09846). At the end a greedy weight soup
([PAPA](https://arxiv.org/abs/2304.03094) / model-soup style) averages members into the best one
while the score doesn't drop.

- The best member matched the best independent seeds: 100% / 100% from still / perturbed hanging,
  99% from random starts. The 4 independent seeds scored 93–99% on random starts, using 120M steps
  each against 80M here.
- Averaging weights never helped. Any mix of the best member with another scored worse, so the
  "soup" was the best member alone.
- After ~40M steps the members were so similar that exploit mostly swapped near-equals.
- For this problem PBT is roughly as good as "train 4 seeds and keep the best". It is not a clear
  win.

**LQR as a teacher (dropped).** We linearized the real dynamics at the top, solved the discrete
Riccati equation, and estimated by simulation where clipped LQR reliably catches the pendulum.
That region is tiny (about ±0.1 rad, ≤ 0.5% of the test box) for any weighting tried, so
regularizing the policy toward LQR would only touch states the policy already handles. A
long-horizon trajectory-optimization teacher (iLQR swing-up with track and force limits) would be
the version worth trying; see the next steps below.

![Robustness of the round-3 configurations](img/robustness.png)

The figure above evaluates every saved policy on the same 256 fresh episodes per case
([experiments/robust_table.py](../experiments/robust_table.py),
[results/robust_table.json](../experiments/results/robust_table.json)):

| Configuration (mean over seeds) | Hanging still | Hanging, perturbed | Random | Push 0.1 | Push 0.2 |
|---|---|---|---|---|---|
| 3 layers (4 seeds, 60M) | 99% | 70% | 42% | 50% | 7% |
| + easier physics (4 seeds, 60M) | 100% | 61% | 48% | 95% | 45% |
| + start noise (4 seeds, 120M) | **100%** | **100%** | **96%** | **99%** | **72%** |
| population, best member (80M) | 100% | 100% | 99% | 100% | 68% |
| **round-3 final run** (200M) | **100%** | **100%** | 91% | 98% | 72% |

## The round-3 final run

`uv run train.py` with the round-3 defaults (easier physics + start noise, 3 × 256, 100 Hz, 200M steps,
~70 min). Its curve is at the top of the [README](../README.md):

- Checkpoint success reached 100% at ~25M steps and stayed there from ~55M.
- The training success (smoothed) stays at ~85% because by then training episodes include
  perturbed and random starts and pushes.

The final checkpoint:

- **Hanging still:** 100%, upright at 1.62 s and settled, a single clean swing. Max cart
  excursion 1.2 m; action jerk (mean |Δa| per step) 0.013.
- **Hanging, perturbed:** 100%, settled at 1.6 s (90th percentile 1.7 s).
- **Random starts:** 91%, settled at ~2.1 s. **Near upright:** 99%.
- **Pushes:** level 0.1 98%, level 0.2 72%, level 0.3 32%. The curriculum had only reached
  level ≈ 0.2.

## Round 4: never leave the track ("phase A")

**What the failures looked like.** The round-3 policy only failed on random starts and pushes,
and it did so by leaving the track. Rollouts of the failures showed two patterns:

- Failing episodes reached a peak cart speed of ~5.7 m/s, against ~3.9 m/s in successful ones,
  but their peak energy was no higher.
- Under pushes, the policy was often balancing 1–1.7 m off centre when the push came. The
  recovery manoeuvre then needed more track than was left.

So the problem was not the energy "panicking" upward, but catches and recoveries that ignore the
ends of the track.

**Changes** (in [env.py](../dprl/env.py), [ppo.py](../dprl/ppo.py), [curriculum.py](../dprl/curriculum.py)):

1. **Track shield.** A braking override whenever the cart could not stop 0.25 m before the end at
   9 m/s². The policy trains with it in the loop and pays 0.3 per overridden step. This is
   shielding as in [Alshiekh et al. 2018](https://arxiv.org/abs/1708.08611); Recovery RL, the
   learned version of the same idea, was the alternative.
2. **Centring while balancing.** `−0.5 · sharp · (x/3)²`, acting only near the top.
3. **Normalization.** Rewards are scaled by the running return std. Observations were standardized
   too, and that was later dropped (see below).
4. **Hard starts.** The cart state of the starts widens up to ±2.5 m / ±1.5 m/s. Up to 20% of the
   episodes start from a replay buffer of states recorded 0.5 s before the shield fired or the
   cart left the track.
5. **A richer checkpoint test** with five cases, and a leaderboard on the dashboard.

**Runs** (120M steps each, group `phaseA`):

- **4 baseline seeds:** the round-3 defaults, with all of the above off.
- **4 seeds with all of it on:** two of these (seeds 1 and 2) never got going. Their policy std
  collapsed to ~0.02–0.04 while the physics was still eased, and at 33M steps they could not
  even balance, so I stopped them.
- **`Afix`:** phase A without the observation normalization and with an entropy bonus of 0.005,
  restarted on the same two seeds.

Two things seemed to drive the collapse:

- With reward scaling the critic fits far better (value loss ~0.002 vs ~40 without it), so the
  policy gradient consistently shrinks the std.
- Observation normalization amplified the barely varying cart inputs in those seeds.

**Results:** every final checkpoint on the same 256 fresh episodes per case
([experiments/compare_runs.py](../experiments/compare_runs.py), data in
[results/compare_phaseA.json](../experiments/results/compare_phaseA.json)). The table shows mean
success, and the worst off-track rate over the seeds in brackets.

| | still | perturbed | random | edge | push 0.1 | push 0.2 | push 0.3 | settle, random p90 |
|---|---|---|---|---|---|---|---|---|
| baseline (4 seeds) | 100% | 100% | 93% (9%) | 89% (13%) | 99% | 76% (14%) | 33% (**30%**) | 4.8 s |
| phase A with obs. norm. (2 surviving of 4) | 100% | 100% | 99% (1%) | 98% (1%) | 99% | 58% (0%) | 23% (0%) | 6.8 s |
| **`Afix` (2 seeds)** | **100%** | **100%** | **100%** (1%) | **100%** (0%) | **100%** | 75% (**0%**) | 33% (**0%**) | **3.2 s** |

- **Leaving the track is essentially solved.** In the `Afix` runs it happens in at most 0.8% of
  any case, and never in seed 1. The baseline left the track in up to 30% of the episodes under
  pushes.
- **Strong pushes still knock it over.** Level 0.3 is a push of 0.9–3 N for 0.1–0.2 s every
  ~2.5 s. But it now falls *safely*, without leaving the track, and swings back up.
- The swing-up from hanging still takes 1.65 s, against 1.76 s for the baseline.

**Mirror symmetry.** The dynamics are symmetric (x, angles and velocities negated, force negated),
but the critic often rated a state and its mirror image very differently during swing-ups (e.g.
192 vs 332): the policies learn a preferred swing direction. The dashboard's "auto" rule acts with
the side the critic values higher (−π(mirror(s)) when mirrored), with a little hysteresis. With the
same episodes:

| random starts | success off → auto | settle p90 off → auto |
|---|---|---|
| baseline | 93% → 98% | 4.8 s → 2.9 s |
| phase A with obs. norm. | 99% → 100% | 6.8 s → 4.8 s |
| `Afix` | 100% → 100% | 3.2 s → 2.4 s |

That this helps means the learned bias costs performance. Building the symmetry into training
would remove it rather than correct it afterwards; see the next steps below.

## Round 5: a mirror-symmetric policy and a time-to-stable prediction

**Symmetry by construction.** Round 4 showed that the policies learn a preferred swing direction.
That bias cost performance: picking the better of s and mirror(s) in the viewer improved random
starts. Round 5 builds the symmetry into the networks ([networks.py](../dprl/networks.py)):

```
mean(s)  = ( f(o) − f(M o) ) / 2 + tie_break(o)        ⇒  mean(M s) = −mean(s)
value(s) = ( g(o) + g(M o) ) / 2                        ⇒  value(M s) = value(s)
```

Here M is the mirror (x, angles and velocities negated). This holds exactly for any weights, and
PPO stays on-policy: the Gaussian N(mean, σ) is itself mirror-symmetric. Both passes run as one
batched call, but backpropagating through twice the batch still made training ~1.75× slower per
step on the CPU.

**The special case.** In states that are their own mirror image (cart centred, links exactly
vertical, nothing moving), an equivariant mean is exactly 0, because neither direction is better.
Upright that is correct: stand still. Hanging still it would wait forever. `tie_break()` therefore
adds a small fixed push (0.3 × max force) that fades out within ~0.05 (observation units) of the
hanging-still point, so the symmetry is exact everywhere else. The tests check equivariance, the
zero action at the upright fixed point and the tie-break at the hanging one.

**Time to stable upright.** Two extra critic outputs predict T, the time until the state is
"stable upright", and E[T²]:
- "Stable upright" means both links within 0.2 rad of vertical, link rates below 1.5 rad/s and
  the cart slower than 1 m/s.
- They are learned by TD(λ) like the value: T(s) = 0 if s is stable, else dt + T(s′). Leaving the
  track counts as 20 s.
- The dashboard shows T ± σ, where σ = √(E[T²] − T²), and the chance of being within ±0.5 s of
  the prediction, assuming a normal distribution.

**Runs:** 80M steps in group `sym`, 4 symmetric seeds and 2 non-symmetric seeds, all with the time
heads. The table compares their final checkpoints on 256 fresh episodes per case
([results/compare_sym.json](../experiments/results/compare_sym.json)):

| | random: settle median / p90 | hanging still: settle | push 0.2 / 0.3 | other cases |
|---|---|---|---|---|
| **symmetric** (4 seeds) | **1.54 s / 2.32 s** | 1.77 s | 71% / 31% | 100%, 0% off-track |
| not symmetric (2 seeds) | 2.20 s / 3.57 s | 1.88 s | 73% / 35% | 100%, 0% off-track |
| phase A `Afix` (2 seeds, 120M) | 2.13 s / 3.18 s | 1.65 s | 75% / 33% | 100%, ≤ 1% off-track |

- **Faster recovery from random states.** The symmetric policy settles ~30% faster. That is as
  good as the old policies with the viewer's mirror trick (`Afix` + "auto": 1.46 s / 2.39 s), but
  now built in.
- **Much faster early learning:**

  ![Learning speed](img/learning_speed_symmetry.png)

  At 5M steps the symmetric seeds score 0.54–0.70, the non-symmetric ones ~0.05. All configurations
  converge to 0.92–0.96 by ~30M steps.
- **No gain at the limits.** Push recovery is the same within seed noise, and swinging up from
  exactly hanging still is ~0.1 s slower than `Afix` (the tie-break start).

**How good is the time prediction?** We compared predicted with actual time to stable upright
along whole episodes ([experiments/time_calibration.py](../experiments/time_calibration.py)):

| | random start: error at the start | all steps: within ±0.5 s | perturbed hanging: error at the start |
|---|---|---|---|
| symmetric (2 seeds) | 0.31–0.39 s | 88–94% | 0.08–0.09 s |
| not symmetric (2 seeds) | 0.57–0.97 s | 63–84% | 0.09–0.19 s |

The symmetric critic predicts much better, plausibly because it learns one function for both
directions. The "chance within ±0.5 s" is roughly calibrated, though somewhat under-confident: in
steps where it says 30–60%, the actual rate is 89–94%.

## Round 6: discrete actions

**Question.** Would a discrete action space work as well or better: just {−30, 0, +30} N, or more
levels for finer control? Bang-bang control is time-optimal for many swing-up problems. And a
categorical policy can represent "push left *or* right" as two separate options, where a Gaussian
can only average them.

**Implementation** ([networks.py](../dprl/networks.py), `PPOConfig.action_bins`):
- The actor outputs K logits of a categorical distribution over K evenly spaced force levels in
  [−30, 30] N ([Tang & Agrawal 2020](https://arxiv.org/abs/1901.10500)).
- The mirror of level k is level K−1−k, so the symmetric version averages the logits of a state
  with the reversed logits of its mirror.
- PPO samples a level per step, with log-probabilities from the softmax and a categorical entropy
  bonus.
- Everything else (shield, curriculum, time heads) is unchanged.

**Runs** (60M steps, symmetric, group `discrete`): K = 3, 7 and 21, 2 seeds each.

**Two ways to turn the policy into one deterministic force:**

- *Most likely level* (argmax). For K = 3 this is true bang-bang control: only −30, 0 or +30 N.
- *Expected force* F̄(s) = Σₖ pₖ(s)·Fₖ, which for K = 3 is 30 N · (p₊ − p₋). It is a smooth
  function of the state even though only three levels were ever applied in training. At the
  exactly symmetric hanging-still state F̄ = 0, so it gets the same tie-break as the continuous
  policy.

**Results** (256 episodes per case, expected-force decoding,
[compare_discrete.json](../experiments/results/compare_discrete.json)):

| | still, perturbed, random, edge, push 0.1 | push 0.2 / 0.3 | off-track | settle: still / random median / p90 |
|---|---|---|---|---|
| continuous, symmetric (4 seeds, 80M) | 100% | 71% / 31% | 0% | 1.77 s / 1.54 s / 2.32 s |
| 3 levels (2 seeds, 60M) | 100% | 71% / 33% | 0% | 1.73 s / 1.39 s / 2.25 s |
| 7 levels (2 seeds, 60M) | 100% | 66% / 30% | 0% | 1.74 s / 1.44 s / 2.26 s |
| 21 levels (2 seeds, 60M) | 100% | 67% / 30% | 0% | 1.70 s / 1.37 s / 2.27 s |

Learning speed was about the same as the continuous symmetric policy: slightly slower in the
first ~5M steps, equal or slightly ahead from ~20M.

**How they control while balancing**
([experiments/control_quality.py](../experiments/control_quality.py), 128 near-upright episodes,
last 15 s):

| | mean force | force change per step | sign changes per s | angle RMS |
|---|---|---|---|---|
| continuous | 0.8–0.9 N | 0.1–0.2 N | 26–42 | 0.02–0.05° |
| discrete, most likely level | 2.3–14 N | 3.5–27 N | 25–73 | 0.06–0.14° |
| discrete, expected force | 0.7–0.8 N | 0.1 N | 9–15 | 0.03–0.06° |

![Cart force while balancing](img/force_traces.png)

**Analysis.**

1. **Three force levels are enough to learn the task.** The 3-level policy learned as fast and
   reached the same success as the continuous and finer discrete policies. Swinging up is close
   to bang-bang anyway, and the swing-ups were even slightly faster (random-start settling
   1.4 s vs 1.5 s).
2. **Decoded by the most likely level, the 3-level policy balances by "dithering":** short ±30 N
   pulse pairs, up to ~70 sign changes per second (see the figure). It is as precise as the
   continuous policy in angle, but on real hardware this would mean vibration, motor heating and
   excited unmodelled dynamics. More levels did not remove it reliably: the argmax of a 21-level
   policy also jumped between distant levels in some seeds.
3. **The expected force gives a smooth controller from the same network.** It was as smooth as the
   continuous policy or smoother (fewer sign changes), with the same success and swing-up times,
   because the probabilities pₖ(s) vary smoothly with the state. This is the categorical
   analogue of using the Gaussian's mean.
4. **Number of levels.** 3, 7 and 21 performed the same within seed noise. With expected-force
   decoding the resolution of the output does not depend on K, so a small K is fine; K = 3 is
   the simplest.
5. **What a discrete policy adds.** It can represent both "full left" and "full right" as likely
   at the same time, which a Gaussian cannot. The expected force averages them away, but that only
   matters where both are genuinely good, i.e. at the symmetric hanging-still state, where the
   tie-break handles it.

**Conclusion.** Discrete actions work as well as continuous ones here, and 3 levels are enough,
as long as the deployed controller uses the expected force. Use the most likely level only if
the actuator can *only* do on/off; then expect dithering while balancing. The continuous
policy stays the default. The discrete option is kept (`--ppo action_bins=K`), and the dashboard
can show both decodings.

## Lessons

1. **Look at trajectories before tuning.** Two rollouts showed the cart leaving the track within
   1.2 s with energy still rising, which pointed straight at the energy shaping. The aggregate
   metrics only said "60% off-track".
2. **Clipped rewards have dead zones.** A penalty inside a `max(0, ·)` does nothing wherever the
   rest is 0.
3. **A curriculum signal must measure the skill it gates.** Mixing start types made the signal
   depend on the difficulty level itself, and the curriculum then stalled or collapsed.
4. **Capacity matters more than expected.** 2 → 3 hidden layers was the difference between 0 of 4
   and 4 of 4 seeds learning the 100 Hz swing-up.
5. **One fixed start overfits.** A policy trained only from hanging still learns one trajectory.
   Perturbing the starts once that trajectory is mastered made it robust, with no loss of speed.
6. **Easing the physics** (lower gravity, link drag) made learning more reliable and the swing-up
   faster.
7. **Use 3+ seeds.** Several single-seed conclusions in round 1 reversed with more seeds or
   longer training. In round 4, 2 of 4 seeds of a configuration failed outright.
8. **A safety layer in the training loop beats penalties.** A large off-track penalty made
   round-1 policies timid (0% success). A shield that simply prevents the failure, plus a small
   penalty for needing it, removed track exits without hurting the swing-up.
9. **A better critic can hurt exploration.** Reward scaling made learning faster, but without an
   entropy bonus the policy std could collapse before the swing-up was found.
10. **Build known symmetries into the model.** The mirror-symmetric policy learned the swing-up
    from far fewer samples, recovered faster from random states and predicted better. The only
    cost was the exactly symmetric hanging-still state, which needs an explicit tie-break.
11. **Discrete is not the same as bang-bang.** A 3-level categorical policy learned the task as
    well as a continuous one; how it is decoded decides whether it chatters (most likely level)
    or controls smoothly (expected force).

## Open issues and next steps

- **Pushes.** Recovery is good up to level 0.1 and partial at 0.2 (75%); level 0.3 still knocks
  it over, now safely. Longer training lets the push stage continue; a slower push ramp or a
  separate recovery signal might work better.
- **A learned switch to a "calm-down" controller** (Recovery RL style). This is a second policy
  output that hands control to an energy-dissipating, braking controller in states the critic
  rates as risky. The shield now covers the track limit, so this would mainly target the strong
  pushes.
- **A trajectory-optimization teacher.** An iLQR swing-up with track and force limits, used for
  behaviour-cloning pretraining plus a decaying regularizer on the policy's own states (DAgger
  style), as in guided policy search.
- **Robustness to model error** (masses, lengths, friction, sensor noise and delay) via domain
  randomization, if this is ever to run on hardware.
- **Off-policy methods** (SAC) are the usual choice in the swing-up literature and may be more
  sample-efficient. PPO with a few minutes of CPU per 10M steps was enough here.

## References

- Jaderberg et al., *Population Based Training of Neural Networks*, 2017. [arXiv:1711.09846](https://arxiv.org/abs/1711.09846)
- Jolicoeur-Martineau et al., *PopulAtion Parameter Averaging (PAPA)*, 2023. [arXiv:2304.03094](https://arxiv.org/abs/2304.03094)
- *Solving the swing-up and balance task for the Acrobot and Pendubot with SAC* (SAC + LQR handover, 100 Hz training). [arXiv:2312.11311](https://arxiv.org/abs/2312.11311)
- *Combining Deep RL and Local Control for the Acrobot Swing-up and Balance Task*. [arXiv:2012.11663](https://arxiv.org/abs/2012.11663)
- Hesse et al., *A Reinforcement Learning Strategy for the Swing-Up of the Double Pendulum on a Cart*, Procedia Manufacturing 24, 2018 (PILCO). [record](https://ris.uni-paderborn.de/record/22996)
- Ho et al., *Applying DDPG to Swing-Up and Balance Control for a Double Inverted Pendulum on a Cart*, Robotica & Management 28(2), 2023. [link](https://rm.reviste.ubbcluj.ro/issues/vol-28-no-2-december-2023/applying-ddpg-algorithm-to-swing-up-and-balance-control-for-a-double-inverted-pendulum-on-a-cart/)
- DeepMind Control Suite, cartpole `two_poles` swing-up task. [cartpole.py](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/cartpole.py)
- Ng, Harada, Russell, *Policy invariance under reward transformations* (potential-based shaping), ICML 1999.
- Lu et al., PureJaxRL. [github.com/luchris429/purejaxrl](https://github.com/luchris429/purejaxrl)
