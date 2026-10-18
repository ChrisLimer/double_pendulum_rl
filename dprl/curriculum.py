"""Training curriculum, updated on the host between jitted chunks.

- Energy shaping weight: decays linearly with training progress (it guides the swing-up
  early on; the upright term has to take over later).
- Start mix: every episode starts either hanging still (cart centred, at rest) or near
  upright. Near-upright starts teach balancing, which the swing-up needs at the end; they
  begin at upright_share = 50% of episodes.
- Easier physics (optional, ease in [0, 1]): gravity scaled by 1 - gravity_relief * ease and a
  drag torque damping * ease on each link, so motions are slower and easier to catch. At
  ease = 0 the physics is the real one (the evaluator and the dashboard always use that).
- Performance-driven, from the swing-up success rate of the hanging starts (on the track and
  upright >= 90% of the last 5 s, the evaluator's test), in stages:
    0. while success > raise_above, ease goes down to 0 (real physics) first;
    1. then the near-upright share shrinks toward min_upright_share, so more episodes
       practise the swing-up;
    2. then start noise rises to 1: hanging starts get perturbed (cart off centre and moving,
       links off vertical and swinging), up to random_share of the non-upright starts are
       fully random, the cart state of all starts widens toward the ends of the track
       (cart_spread), and up to replay_share of the episodes start from replayed hard states
       (recorded shortly before the track shield fired or the cart left the track), so the
       policy does not overfit to the one hanging-still trajectory;
    3. once that is at the maximum, random pushes on the tip get stronger.
  While success < lower_below it goes back the other way: pushes, start noise, more
  near-upright starts (up to max_upright_share), then easier physics.
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class Curriculum:
    energy_start: float = 1.0
    energy_end: float = 0.1
    energy_decay_frac: float = 0.4   # reach energy_end after this fraction of training
    raise_above: float = 0.9         # swing-up success above this -> harder
    lower_below: float = 0.8         # ... below this -> easier
    step: float = 0.02               # change per chunk
    max_upright_share: float = 0.5   # share of near-upright starts at the beginning
    min_upright_share: float = 0.1   # keep some, so balancing is never forgotten
    episode_steps: int = 2000        # 20 s at 100 Hz
    # Easier physics at the start: 4/4 seeds reached 100% from hanging still with it, settling
    # faster (1.6-2.2 s vs 2.6-3.0 s); the evaluator always uses the real physics.
    gravity_relief: float = 0.5      # gravity scale at ease = 1 is 1 - gravity_relief
    damping: float = 0.1             # link drag [N m s/rad] at ease = 1
    random_share: float = 0.3        # of the non-upright starts, at start noise 1
    replay_share: float = 0.2        # episodes starting from replayed hard states, at start noise 1
    wide_starts: bool = True         # widen the cart state of the starts with start noise
    noise_stage: bool = True         # stage 2 on/off
    upright_share: float = 0.5
    start_noise: float = 0.0
    push_level: float = 0.0
    ease: float = 1.0

    def energy_weight(self, progress: float) -> float:
        frac = min(progress / self.energy_decay_frac, 1.0)
        return self.energy_start + frac * (self.energy_end - self.energy_start)

    def start_mix(self):
        """(hanging, fully random, near upright)"""
        rand = (1.0 - self.upright_share) * self.random_share * self.start_noise
        return (1.0 - self.upright_share - rand, rand, self.upright_share)

    def knobs(self, progress: float) -> dict:
        """Values passed into the jitted training chunk."""
        return {
            "energy_weight": jnp.float32(self.energy_weight(progress)),
            "push_level": jnp.float32(self.push_level),
            "start_mix": jnp.array(self.start_mix(), dtype=jnp.float32),
            "start_spread": jnp.float32(1.0),
            "start_noise": jnp.float32(self.start_noise),
            "cart_spread": jnp.float32(self.start_noise if self.wide_starts else 0.0),
            "replay_prob": jnp.float32(self.replay_share * self.start_noise),
            "episode_steps": jnp.int32(self.episode_steps),
            "gravity_scale": jnp.float32(1.0 - self.gravity_relief * self.ease),
            "damping": jnp.float32(self.damping * self.ease),
        }

    @property
    def eased(self) -> bool:
        return self.ease > 0 and (self.gravity_relief > 0 or self.damping > 0)

    def update(self, swingup_success: float) -> None:
        """swingup_success: evaluator-style success rate of the hanging starts."""
        if math.isnan(swingup_success):
            return
        if swingup_success > self.raise_above:
            if self.eased:
                self.ease = max(0.0, self.ease - self.step)
            elif self.upright_share > self.min_upright_share:
                self.upright_share = max(self.min_upright_share, self.upright_share - self.step)
            elif self.noise_stage and self.start_noise < 1.0:
                self.start_noise = min(1.0, self.start_noise + self.step)
            else:
                self.push_level = min(1.0, self.push_level + self.step)
        elif swingup_success < self.lower_below:
            if self.push_level > 0.0:
                self.push_level = max(0.0, self.push_level - self.step)
            elif self.start_noise > 0.0:
                self.start_noise = max(0.0, self.start_noise - self.step)
            elif self.upright_share < self.max_upright_share or not (self.gravity_relief > 0 or self.damping > 0):
                self.upright_share = min(self.max_upright_share, self.upright_share + self.step)
            else:
                self.ease = min(1.0, self.ease + self.step)
