// JavaScript port of the environment physics (dprl/env.py) and the policy network
// (dprl/networks.py), so the browser can simulate a checkpoint live at the control rate (100 Hz).
// Keep in sync with the Python code; static/parity.html checks the two against each other.

export const UPRIGHT_ANGLE = 0.3;

export function wrap(a) {
  const twoPi = 2 * Math.PI;
  return ((((a + Math.PI) % twoPi) + twoPi) % twoPi) - Math.PI;
}

// The chain: masses m, lengths l and the mass carried by each link, mu_i = sum_{k>=i} m_k.
// Runs from before the n-link code have m1/m2/l1/l2 instead of link_masses/link_lengths.
export function chain(env) {
  const m = env.link_masses ?? [env.m1, env.m2], l = env.link_lengths ?? [env.l1, env.l2];
  const mu = m.map((_, i) => m.slice(i).reduce((a, b) => a + b, 0));
  return { n: m.length, m, l, mu };
}

// Solve A x = b by Gaussian elimination with partial pivoting (A is small: (n + 1) x (n + 1)).
function solve(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

// D(q) qdd = rhs, see PendulumCart.dynamics (dprl/env.py). tip = [fx, fy] acts on the top mass.
export function dynamics(env, q, qd, force, tip) {
  const { n, m, l, mu } = chain(env);
  const th = q.slice(1), w = qd.slice(1), [fx, fy] = tip, g = env.g;
  const D = Array.from({ length: n + 1 }, () => new Array(n + 1).fill(0));
  const rhs = new Array(n + 1).fill(0);
  D[0][0] = env.cart_mass + m.reduce((a, b) => a + b, 0);
  rhs[0] = force + fx;
  for (let i = 0; i < n; i++) {
    const c = Math.cos(th[i]), s = Math.sin(th[i]);
    D[0][i + 1] = D[i + 1][0] = mu[i] * l[i] * c;
    rhs[0] += mu[i] * l[i] * s * w[i] * w[i];
    rhs[i + 1] = g * mu[i] * l[i] * s + l[i] * (fx * c - fy * s);
    for (let j = 0; j < n; j++) {
      const pair = mu[Math.max(i, j)] * l[i] * l[j];
      D[i + 1][j + 1] = pair * Math.cos(th[i] - th[j]);
      rhs[i + 1] -= pair * Math.sin(th[i] - th[j]) * w[j] * w[j];
    }
  }
  return solve(D, rhs);
}

// One control period of RK4 with forces held constant (PendulumCart.integrate).
export function integrate(env, q, qd, force, tip = [0, 0]) {
  const h = env.dt / env.substeps, k = q.length;
  const f = (y) => [...y.slice(k), ...dynamics(env, y.slice(0, k), y.slice(k), force, tip)];
  const add = (y, d, s) => y.map((v, i) => v + s * d[i]);
  let y = [...q, ...qd];
  for (let n = 0; n < env.substeps; n++) {
    const k1 = f(y);
    const k2 = f(add(y, k1, 0.5 * h));
    const k3 = f(add(y, k2, 0.5 * h));
    const k4 = f(add(y, k3, h));
    y = y.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  }
  return { q: [y[0], ...y.slice(1, k).map(wrap)], qd: y.slice(k) };
}

// x / x_limit, (sin, cos) per link, cart speed / 3, link rates / 10 (PendulumCart.obs).
export function obs(env, q, qd) {
  const th = q.slice(1);
  return [q[0] / env.x_limit, ...th.flatMap((a) => [Math.sin(a), Math.cos(a)]), qd[0] / 3.0, ...qd.slice(1).map((v) => v / 10.0)];
}

// Velocities of the point masses relative to the cart (cart motion excluded).
function relativeVelocities(env, q, qd) {
  const { l } = chain(env);
  let vx = 0, vy = 0;
  return l.map((li, i) => {
    vx += li * Math.cos(q[i + 1]) * qd[i + 1];
    vy -= li * Math.sin(q[i + 1]) * qd[i + 1];
    return [vx, vy];
  });
}

export function pendulumEnergy(env, q, qd) {
  const { m, l, mu } = chain(env);
  const kinetic = relativeVelocities(env, q, qd).reduce((acc, [vx, vy], i) => acc + 0.5 * m[i] * (vx * vx + vy * vy), 0);
  return kinetic + env.g * l.reduce((acc, li, i) => acc + mu[i] * li * Math.cos(q[i + 1]), 0);
}

export function isUpright(q) {
  return q.slice(1).every((a) => Math.cos(a) > Math.cos(UPRIGHT_ANGLE));
}

export function points(env, q) {  // cart pivot, then each mass up to the tip (world coordinates, y up)
  const { l } = chain(env);
  const pts = [[q[0], 0]];
  l.forEach((li, i) => {
    const [px, py] = pts[pts.length - 1];
    pts.push([px + li * Math.sin(q[i + 1]), py + li * Math.cos(q[i + 1])]);
  });
  return pts;
}

export function tipVelocity(env, q, qd) {
  const rel = relativeVelocities(env, q, qd);
  const [vx, vy] = rel[rel.length - 1];
  return [qd[0] + vx, vy];
}

// Initial states, same distributions as PendulumCart.reset, for env's number of links.
// mode: "still" (hanging still, the evaluator start), "hanging" (perturbed), "mix", "random" or "upright".
export function reset(mode = "mix", env = null) {
  const n = env ? chain(env).n : 2;
  const u = (lo, hi) => lo + (hi - lo) * Math.random();
  const each = (fn) => Array.from({ length: n }, fn);
  if (mode === "mix") {
    const r = Math.random();
    mode = r < 0.4 ? "hanging" : r < 0.7 ? "random" : "upright";
  }
  if (mode === "still") {  // the evaluator's start (reset, hanging, noise 0)
    return { q: [0, ...each(() => wrap(Math.PI + u(-0.02, 0.02)))], qd: new Array(n + 1).fill(0) };
  }
  let th, w;
  // "hanging": perturbed hanging start (reset with noise 1)
  if (mode === "hanging") { th = each(() => Math.PI + u(-0.3, 0.3)); w = each(() => u(-0.5, 0.5)); }
  else if (mode === "random") { th = each(() => u(-Math.PI, Math.PI)); w = each(() => u(-3, 3)); }
  else { th = each(() => u(-0.4, 0.4)); w = each(() => u(-1, 1)); }
  return { q: [u(-1, 1), ...th.map(wrap)], qd: [u(-0.5, 0.5), ...w] };
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
// Observation layout for n links (obs length 2 + 3n), as env.obs_layout.
function layout(len) {
  const n = (len - 2) / 3, idx = (f) => Array.from({ length: n }, (_, i) => f(i));
  return { n, cos: idx((i) => 2 + 2 * i), xd: 1 + 2 * n, w: idx((i) => 2 + 2 * n + i) };
}
// Mirror: everything negated except the cosines.
const mirrorSigns = (len) => { const s = new Array(len).fill(-1); layout(len).cos.forEach((i) => (s[i] = 1)); return s; };
const TIE_BREAK_FORCE = 0.3, TIE_BREAK_WIDTH = 0.05;
const mirrorObs = (o) => { const s = mirrorSigns(o.length); return o.map((v, i) => v * s[i]); };

function tieBreak(o) {
  const s = mirrorSigns(o.length);
  const d2 = o.reduce((acc, v, i) => acc + (s[i] < 0 ? v * v : 0), 0);
  return TIE_BREAK_FORCE * Math.exp(-d2 / TIE_BREAK_WIDTH ** 2) * Math.min(Math.max(-o[2], 0), 1);
}

// loadPolicy(layers, symmetric) marks the network; the functions below apply the symmetry.
// Categorical policy: the expected force (default, as dprl/networks.py actor) or, with
// policy.decode === "mode", the most likely level (lowest bin on ties).
function discreteAction(policy, o) {
  let logits = mlp(policy, o);
  if (policy.symmetric) {
    const mirrored = mlp(policy, mirrorObs(o));
    logits = logits.map((v, k) => (v + mirrored[logits.length - 1 - k]) / 2);
  }
  if (policy.decode === "mode") {
    let best = 0;
    for (let k = 1; k < logits.length; k++) if (logits[k] > logits[best]) best = k;
    return policy.bins[best];
  }
  const top = Math.max(...logits);
  const p = logits.map((v) => Math.exp(v - top));
  const z = p.reduce((a, b) => a + b, 0);
  const mean = p.reduce((acc, v, k) => acc + v * policy.bins[k], 0) / z;
  return policy.symmetric ? mean + tieBreak(o) : mean;
}

function actorMean(policy, o) {
  if (policy.bins) return discreteAction(policy, o);
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
  const lay = layout(o.length), c = Math.cos(0.2);
  return lay.cos.every((i) => o[i] > c) && lay.w.every((i) => Math.abs(o[i]) * 10 < 1.5) && Math.abs(o[lay.xd]) * 3 < 1.0;
}

// Mirror image of a state: the system is symmetric under x -> -x, angles -> -angles (all
// velocities negated); the matching action is -a.
export function mirror(q, qd) {
  return { q: q.map((v) => -v), qd: qd.map((v) => -v) };
}
