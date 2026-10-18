// JavaScript port of the environment physics (dprl/env.py) and the policy network
// (dprl/networks.py), so the browser can simulate a checkpoint live at the control rate (100 Hz).
// Keep in sync with the Python code; static/parity.html checks the two against each other.

export const UPRIGHT_ANGLE = 0.3;

export function wrap(a) {
  const twoPi = 2 * Math.PI;
  return ((((a + Math.PI) % twoPi) + twoPi) % twoPi) - Math.PI;
}

// D(q) qdd = rhs, see DoublePendulumCart.dynamics. tip = [fx, fy] acts on the upper mass.
export function dynamics(env, q, qd, force, tip) {
  const { cart_mass: M, m1, m2, l1, l2, g } = env;
  const [, th1, th2] = q;
  const [, w1, w2] = qd;
  const [fx, fy] = tip;
  const c1 = Math.cos(th1), s1 = Math.sin(th1), c2 = Math.cos(th2), s2 = Math.sin(th2);
  const c12 = Math.cos(th1 - th2), s12 = Math.sin(th1 - th2);
  const D = [
    [M + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2],
    [(m1 + m2) * l1 * c1, (m1 + m2) * l1 * l1, m2 * l1 * l2 * c12],
    [m2 * l2 * c2, m2 * l1 * l2 * c12, m2 * l2 * l2],
  ];
  const rhs = [
    force + fx + (m1 + m2) * l1 * s1 * w1 * w1 + m2 * l2 * s2 * w2 * w2,
    (m1 + m2) * g * l1 * s1 - m2 * l1 * l2 * s12 * w2 * w2 + l1 * (fx * c1 - fy * s1),
    m2 * g * l2 * s2 + m2 * l1 * l2 * s12 * w1 * w1 + l2 * (fx * c2 - fy * s2),
  ];
  return solve3(D, rhs);
}

function det3(m) {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
       - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
       + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

function solve3(A, b) {  // Cramer's rule; D is always well conditioned here
  const d = det3(A);
  return [0, 1, 2].map((col) => det3(A.map((row, i) => row.map((v, j) => (j === col ? b[i] : v)))) / d);
}

// One control period of RK4 with forces held constant (DoublePendulumCart.integrate).
export function integrate(env, q, qd, force, tip = [0, 0]) {
  const h = env.dt / env.substeps;
  const f = (y) => [y[3], y[4], y[5], ...dynamics(env, y.slice(0, 3), y.slice(3), force, tip)];
  const add = (y, k, s) => y.map((v, i) => v + s * k[i]);
  let y = [...q, ...qd];
  for (let n = 0; n < env.substeps; n++) {
    const k1 = f(y);
    const k2 = f(add(y, k1, 0.5 * h));
    const k3 = f(add(y, k2, 0.5 * h));
    const k4 = f(add(y, k3, h));
    y = y.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  }
  return { q: [y[0], wrap(y[1]), wrap(y[2])], qd: y.slice(3) };
}

export function obs(env, q, qd) {
  const [x, th1, th2] = q;
  const [xd, w1, w2] = qd;
  return [x / env.x_limit, Math.sin(th1), Math.cos(th1), Math.sin(th2), Math.cos(th2),
          xd / 3.0, w1 / 10.0, w2 / 10.0];
}

export function pendulumEnergy(env, q, qd) {
  const { m1, m2, l1, l2, g } = env;
  const [, th1, th2] = q;
  const [, w1, w2] = qd;
  const v2x = l1 * Math.cos(th1) * w1 + l2 * Math.cos(th2) * w2;
  const v2y = -l1 * Math.sin(th1) * w1 - l2 * Math.sin(th2) * w2;
  const kinetic = 0.5 * (m1 * (l1 * w1) ** 2 + m2 * (v2x * v2x + v2y * v2y));
  return kinetic + g * ((m1 + m2) * l1 * Math.cos(th1) + m2 * l2 * Math.cos(th2));
}

export function isUpright(q) {
  return Math.cos(q[1]) > Math.cos(UPRIGHT_ANGLE) && Math.cos(q[2]) > Math.cos(UPRIGHT_ANGLE);
}

export function points(env, q) {  // cart pivot, lower mass, tip (world coordinates, y up)
  const [x, th1, th2] = q;
  const p1 = [x + env.l1 * Math.sin(th1), env.l1 * Math.cos(th1)];
  return [[x, 0], p1, [p1[0] + env.l2 * Math.sin(th2), p1[1] + env.l2 * Math.cos(th2)]];
}

export function tipVelocity(env, q, qd) {
  const [, th1, th2] = q;
  const [xd, w1, w2] = qd;
  return [xd + env.l1 * Math.cos(th1) * w1 + env.l2 * Math.cos(th2) * w2,
          -env.l1 * Math.sin(th1) * w1 - env.l2 * Math.sin(th2) * w2];
}

// Initial states, same distributions as DoublePendulumCart.reset.
// mode: "still" (hanging still, the evaluator start), "hanging" (perturbed), "mix", "random" or "upright".
export function reset(mode = "mix") {
  const u = (lo, hi) => lo + (hi - lo) * Math.random();
  if (mode === "mix") {
    const r = Math.random();
    mode = r < 0.4 ? "hanging" : r < 0.7 ? "random" : "upright";
  }
  if (mode === "still") {  // the evaluator's start (DoublePendulumCart.reset, hanging, noise 0)
    return { q: [0, wrap(Math.PI + u(-0.02, 0.02)), wrap(Math.PI + u(-0.02, 0.02))], qd: [0, 0, 0] };
  }
  let th, w;
  // "hanging": perturbed hanging start (reset with noise 1)
  if (mode === "hanging") { th = [Math.PI + u(-0.3, 0.3), Math.PI + u(-0.3, 0.3)]; w = [u(-0.5, 0.5), u(-0.5, 0.5)]; }
  else if (mode === "random") { th = [u(-Math.PI, Math.PI), u(-Math.PI, Math.PI)]; w = [u(-3, 3), u(-3, 3)]; }
  else { th = [u(-0.4, 0.4), u(-0.4, 0.4)]; w = [u(-1, 1), u(-1, 1)]; }
  return { q: [u(-1, 1), wrap(th[0]), wrap(th[1])], qd: [u(-0.5, 0.5), w[0], w[1]] };
}

// Track shield, as DoublePendulumCart.apply_shield: full force toward the centre when braking at
// shield_decel could not stop the cart shield_margin before the end. Returns [action, active].
// Runs trained before the shield existed have no env.shield and are simulated without it.
export function shield(env, q, qd, a) {
  if (env.shield !== true) return [a, false];
  const x = q[0], v = qd[0];
  const dir = Math.abs(v) > 0.01 ? Math.sign(v) : Math.sign(x);
  const room = env.x_limit - dir * x;
  const stopping = Math.max(dir * v, 0) ** 2 / (2 * env.shield_decel);
  const active = stopping + env.shield_margin >= room && dir * a > -1;
  return active ? [-dir, true] : [a, false];
}

// ---- policy -----------------------------------------------------------------

function f32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

// From the /api/policy response: [{shape: [n_in, n_out], w, b}] with base64 float32.
export function loadPolicy(layers, symmetric = false, bins = null) {
  const net = layers.map((l) => ({ nIn: l.shape[0], nOut: l.shape[1], w: f32(l.w), b: f32(l.b) }));
  net.symmetric = symmetric;
  net.bins = bins;  // force levels of a categorical policy (null: Gaussian mean)
  return net;
}

// Deterministic action: the mean of the Gaussian policy, clipped to [-1, 1].
// Forward pass of an MLP from loadPolicy() (tanh hidden layers, linear output).
function mlp(layers, o) {
  let x = Float64Array.from(o);
  layers.forEach((layer, n) => {
    const { nIn, nOut, w, b } = layer;
    const y = Float64Array.from(b);
    for (let i = 0; i < nIn; i++) {
      const xi = x[i], row = i * nOut;
      for (let j = 0; j < nOut; j++) y[j] += xi * w[row + j];
    }
    if (n < layers.length - 1) for (let j = 0; j < nOut; j++) y[j] = Math.tanh(y[j]);
    x = y;
  });
  return x;
}

// Mirror symmetry, as dprl/networks.py: an observation's mirror image negates everything but the
// cosines; symmetric networks average over a state and its mirror, plus a small tie-break push
// near the hanging-still fixed point (where an exactly symmetric policy would output 0).
const MIRROR = [-1, -1, 1, -1, 1, -1, -1, -1];
const TIE_BREAK_FORCE = 0.3, TIE_BREAK_WIDTH = 0.05;
const mirrorObs = (o) => o.map((v, i) => v * MIRROR[i]);

function tieBreak(o) {
  const d2 = [0, 1, 3, 5, 6, 7].reduce((acc, i) => acc + o[i] * o[i], 0);
  return TIE_BREAK_FORCE * Math.exp(-d2 / TIE_BREAK_WIDTH ** 2) * Math.min(Math.max(-o[2], 0), 1);
}

// loadPolicy(layers, symmetric) marks the network; the functions below apply the symmetry.
function actorMean(policy, o) {
  if (policy.bins) {  // categorical policy: the most likely force level (lowest bin on ties)
    let logits = mlp(policy, o);
    if (policy.symmetric) {
      const mirrored = mlp(policy, mirrorObs(o));
      logits = logits.map((v, k) => (v + mirrored[logits.length - 1 - k]) / 2);
    }
    let best = 0;
    for (let k = 1; k < logits.length; k++) if (logits[k] > logits[best]) best = k;
    return policy.bins[best];
  }
  const m = mlp(policy, o)[0];
  return policy.symmetric ? (m - mlp(policy, mirrorObs(o))[0]) / 2 + tieBreak(o) : m;
}

export function criticOutputs(critic, o) {
  const out = mlp(critic, o);
  if (!critic.symmetric) return out;
  const mirrored = mlp(critic, mirrorObs(o));
  return out.map((v, i) => (v + mirrored[i]) / 2);
}

// Deterministic action: the Gaussian policy's mean, clipped to [-1, 1].
export function policyAction(policy, o) {
  return Math.max(-1, Math.min(1, actorMean(policy, o)));
}

// The critic's estimate (multiply by the checkpoint's value_scale for return units).
export function value(critic, o) {
  return criticOutputs(critic, o)[0];
}

// Predicted time to stable upright [s] and its spread, from critic heads 1 and 2
// (T / time_max and E[T^2] / time_max^2), or null for networks without them.
export function timeToStable(critic, o, timeMax) {
  if (!timeMax) return null;
  const out = criticOutputs(critic, o);
  if (out.length < 3) return null;
  const mean = Math.min(Math.max(out[1] * timeMax, 0), timeMax);
  const second = Math.max(out[2] * timeMax * timeMax, mean * mean);
  return { mean, std: Math.sqrt(second - mean * mean) };
}

// The test the time prediction is trained for (env.stable_obs).
export function isStable(o) {
  const c = Math.cos(0.2);
  return o[2] > c && o[4] > c && Math.abs(o[6]) * 10 < 1.5 && Math.abs(o[7]) * 10 < 1.5 && Math.abs(o[5]) * 3 < 1.0;
}

// Mirror image of a state: the system is symmetric under x -> -x, angles -> -angles (all
// velocities negated); the matching action is -a.
export function mirror(q, qd) {
  return { q: q.map((v) => -v), qd: qd.map((v) => -v) };
}
