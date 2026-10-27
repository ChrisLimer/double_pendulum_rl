// Dashboard: polls the server for training metrics/checkpoints, and simulates the chosen
// checkpoint live in the browser (physics + policy from sim.mjs) so you can push it around.
import * as sim from "./sim.mjs";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmtSteps = (n) => n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : `${n}`;

// =============================================================================
// Live metric charts
// =============================================================================

const CHARTS = [
  ["upright_late", "Upright fraction after settling"],
  ["upright_late_up", "Upright after settling, near-upright starts"],
  ["swingup_success", "Swing-up success, hanging starts (curriculum signal)"],
  ["swingup_rate", "Reached the top (hanging starts)"],
  ["swingup_time", "Time to swing up (s)"],
  ["recovery", "Upright 2–3 s after a push"],
  ["upright_share", "Near-upright start share (curriculum)"],
  ["start_noise", "Start noise (curriculum)"],
  ["push_level", "Push level (curriculum)"],
  ["energy_weight", "Energy reward weight (schedule)"],
  ["ep_return", "Episode return (shaped)"],
  ["loss", "Total loss"],
  ["v_loss", "Value loss"],
  ["pg_loss", "Policy loss"],
  ["approx_kl", "Approx. KL"],
  ["action_std", "Policy std (exploration)"],
];
const EMA_ALPHA = 0.1;  // smoothing for the bold line; the faint line is the raw value
const charts = {};

function makeChart(key, title) {
  const box = document.createElement("div");
  box.className = "panel chart";
  const canvas = document.createElement("canvas");
  box.appendChild(canvas);
  $("charts").appendChild(box);
  const accent = css("--accent"), grid = css("--border"), muted = css("--muted");
  const chart = new Chart(canvas, {
    type: "line",
    data: { datasets: [
      { data: [], borderColor: accent + "44", borderWidth: 1, pointRadius: 0, spanGaps: true },
      { data: [], borderColor: accent, borderWidth: 2, pointRadius: 0, spanGaps: true },
    ] },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false, parsing: false, normalized: true,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: { legend: { display: false }, title: { display: true, text: title, color: css("--text") },
                 tooltip: { callbacks: { title: (items) => `${fmtSteps(items[0].parsed.x)} steps` } } },
      scales: {
        x: { type: "linear", grid: { color: grid }, ticks: { color: muted, maxTicksLimit: 5, callback: fmtSteps } },
        y: { grid: { color: grid }, ticks: { color: muted, maxTicksLimit: 5 } },
      },
    },
  });
  charts[key] = { chart, ema: null };
}

function resetCharts() {
  for (const c of Object.values(charts)) {
    c.chart.data.datasets.forEach((d) => (d.data = []));
    c.ema = null;
    c.chart.update();
  }
}

function addRows(rows) {
  for (const [key, c] of Object.entries(charts)) {
    for (const row of rows) {
      const y = row[key];
      if (y === null || y === undefined) continue;
      c.ema = c.ema === null ? y : c.ema + EMA_ALPHA * (y - c.ema);
      c.chart.data.datasets[0].data.push({ x: row.env_steps, y });
      c.chart.data.datasets[1].data.push({ x: row.env_steps, y: c.ema });
    }
    c.chart.update();
  }
}

// =============================================================================
// Polling the server
// =============================================================================

const state = { run: null, metricsSince: 0, lastRow: null, numUpdates: null, checkpoints: [] };

// ?run=<name> shows that run; without it the server's default (pinned or newest) run.
const RUN = new URLSearchParams(location.search).get("run");

async function getJSON(url) {
  if (RUN) url += (url.includes("?") ? "&" : "?") + `run=${encodeURIComponent(RUN)}`;
  const r = await fetch(url);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  return body;
}

async function loadRunPicker() {
  const d = await getJSON("/api/runs");
  const select = $("run-select");
  select.replaceChildren(new Option(`Default (${d.default})`, ""));
  for (const r of [...d.runs].reverse()) {
    const label = `${r.name} · ${r.checkpoints} ckpt${r.logs ? "" : " · no logs"}`;
    select.add(new Option(label, r.name));
  }
  select.value = RUN || "";
}

async function pollMetrics() {
  const d = await getJSON(`/api/metrics?since=${state.metricsSince}`);
  if (d.run !== state.run) {  // a new training run started: start the plots over
    state.run = d.run;
    state.metricsSince = 0;
    state.lastRow = null;
    resetCharts();
    $("run-name").textContent = d.run;
    if (d.since !== 0) return pollMetrics();
  }
  if (d.rows.length) {
    addRows(d.rows);
    state.metricsSince += d.rows.length;
    state.lastRow = d.rows[d.rows.length - 1];
  }
}

async function pollStatus() {
  const d = await getJSON("/api/status");
  state.numUpdates = d.config.num_updates;
  const row = state.lastRow;
  if (!row) {
    if (d.config.note) $("progress-text").textContent = d.config.note;
    return;
  }
  const frac = row.update / state.numUpdates;
  $("progress-fill").style.width = `${(100 * frac).toFixed(1)}%`;
  const elapsed = `${Math.floor(row.wall_time / 60)}m ${Math.floor(row.wall_time % 60)}s`;
  $("progress-text").textContent = frac >= 1
    ? `done · ${fmtSteps(row.env_steps)} steps in ${elapsed}`
    : `update ${row.update}/${state.numUpdates} · ${fmtSteps(row.env_steps)} steps · ${elapsed} · ${Math.round(row.sps).toLocaleString()} steps/s`;
}

async function pollCheckpoints() {
  const d = await getJSON("/api/checkpoints");
  state.checkpoints = d.checkpoints;
  const signature = d.run + d.checkpoints.map((c) => c.name).join();
  if (signature === state.ckptSignature) return;  // unchanged: don't close an open dropdown
  state.ckptSignature = signature;
  const select = $("ckpt");
  const selected = select.value;
  const options = [new Option("Latest (auto-follow)", "latest")];
  for (const c of [...d.checkpoints].reverse()) {
    const ev = c.eval ? ` · swing-up ${(100 * c.eval.success).toFixed(0)}%` : "";
    options.push(new Option(`${c.name} · ${fmtSteps(c.env_steps)} steps${ev}`, c.name));
  }
  select.replaceChildren(...options);
  select.value = options.some((o) => o.value === selected) ? selected : "latest";
}

function poll(fn, ms) {
  (async () => {
    for (;;) {
      try { await fn(); } catch (e) { /* server not ready / no run yet: try again later */ }
      await sleep(ms);
    }
  })();
}

// =============================================================================
// Live simulation in the browser
// =============================================================================

const HOLD_SECONDS = 1.5;    // pause after an episode ends before the next one starts
const SPRING = 40;           // N/m, mouse drag pulls the tip like a spring...
const DAMPING = 3;           // N*s/m ...with some damping so it doesn't jiggle
const policies = new Map();  // checkpoint name -> {policy, env, meta}; weights are fetched once

const live = {
  env: null, policy: null, ckpt: null, meta: null,
  q: null, qd: null, prevQ: null, t: 0, uprightTime: 0, acc: 0, lastFrame: null,
  ended: null, endReason: "", loading: false, token: 0,
  action: 0, tipForce: [0, 0], keys: new Set(), mouse: null, trail: [],
};

async function getPolicy(name) {
  if (!policies.has(name)) {
    const d = await getJSON(`/api/policy?ckpt=${encodeURIComponent(name)}`);
    policies.set(d.checkpoint, { policy: sim.loadPolicy(d.layers, d.symmetric, d.bins), env: d.env, meta: d.meta,
                                 critic: d.critic ? sim.loadPolicy(d.critic, d.symmetric) : null,
                                 valueScale: d.value_scale ?? 1, timeMax: d.time_max, symmetric: d.symmetric });
  }
  return { name, ...policies.get(name) };
}

// Load the selected (or newest) checkpoint and start a fresh episode with it.
async function startEpisode() {
  const token = ++live.token;
  live.loading = true;
  try {
    let name = $("ckpt").value;
    if (name === "latest") {
      await pollCheckpoints().catch(() => {});
      if (!state.checkpoints.length) throw new Error("waiting for the first checkpoint…");
      name = state.checkpoints[state.checkpoints.length - 1].name;
    }
    const p = await getPolicy(name);
    if (token !== live.token) return;
    Object.assign(live, { env: p.env, policy: p.policy, ckpt: p.name, meta: p.meta, critic: p.critic,
                          valueScale: p.valueScale, timeMax: p.timeMax, symmetric: p.symmetric });
    $("decode-label").hidden = !p.policy.bins;
    p.policy.decode = $("decode").value;
    resetEpisode();
  } catch (e) {
    if (token !== live.token) return;
    $("info-status").textContent = e.message;
    await sleep(2000);
  } finally {
    if (token === live.token) live.loading = false;
  }
}

function resetEpisode() {
  const s = sim.reset($("start-mode").value, live.env);
  Object.assign(live, { q: s.q, qd: s.qd, prevQ: s.q, t: 0, uprightTime: 0, acc: 0, ended: null, trail: [],
                        side: 0, sideSince: 0, flips: 0, predAtStart: null, stableAt: null });
}

// External force on the tip from the user: mouse spring while dragging, or arrow keys.
function userTipForce() {
  const env = live.env, max = +$("push-strength").value;
  if (live.mouse) {
    const tip = sim.points(env, live.q).at(-1), v = sim.tipVelocity(env, live.q, live.qd);
    let f = [0, 1].map((i) => SPRING * (live.mouse[i] - tip[i]) - DAMPING * v[i]);
    const mag = Math.hypot(...f);
    if (mag > max) f = f.map((c) => (c * max) / mag);
    return f;
  }
  const dir = (live.keys.has("ArrowRight") ? 1 : 0) - (live.keys.has("ArrowLeft") ? 1 : 0);
  return [dir * max, 0];
}

// Symmetry: the system is the same under the mirror map (x, angles, velocities -> negated; force
// -> -force), but a policy can learn a preferred swing direction. "auto" evaluates the critic on
// the state and on its mirror image and acts with the side that predicts the higher return:
// a(s) = pi(s), or a(s) = -pi(mirror(s)) when acting mirrored. It only switches when the other
// side is better by SWITCH_MARGIN (relative) and the current side has been used for SWITCH_DWELL,
// so it does not flip-flop between two nearly equal estimates.
const SWITCH_MARGIN = 0.02, SWITCH_DWELL = 0.25;

function chooseAction() {
  const env = live.env, mode = $("symmetry").value;
  const o = sim.obs(env, live.q, live.qd);
  const m = sim.mirror(live.q, live.qd), om = sim.obs(env, m.q, m.qd);
  live.value = live.critic ? sim.value(live.critic, o) * live.valueScale : null;
  live.stable = sim.isStable(o);
  live.time = live.critic ? sim.timeToStable(live.critic, o, live.timeMax) : null;
  if (live.time && live.predAtStart === null) live.predAtStart = live.time;
  if (live.stable && live.stableAt === null) live.stableAt = live.t;
  live.valueMirror = live.critic ? sim.value(live.critic, om) * live.valueScale : null;
  if (mode === "off" || !live.critic) live.side = 1;
  else if (mode === "mirror") live.side = -1;
  else {
    const [cur, other] = live.side === -1 ? [live.valueMirror, live.value] : [live.value, live.valueMirror];
    if (live.side === 0) live.side = live.valueMirror > live.value ? -1 : 1;
    else if (other - cur > SWITCH_MARGIN * Math.abs(cur) && live.t - live.sideSince >= SWITCH_DWELL) {
      live.side = -live.side;
      live.sideSince = live.t;
      live.flips += 1;
    }
  }
  return live.side === -1 ? -sim.policyAction(live.policy, om) : sim.policyAction(live.policy, o);
}

function controlStep() {
  const env = live.env;
  [live.action, live.shielded] = sim.shield(env, live.q, live.qd, chooseAction());
  live.tipForce = userTipForce();
  live.prevQ = live.q;
  ({ q: live.q, qd: live.qd } = sim.integrate(env, live.q, live.qd, live.action * env.max_force, live.tipForce));
  live.t += env.dt;
  if (sim.isUpright(live.q)) live.uprightTime += env.dt;
  live.trail.push(sim.points(env, live.q).at(-1));
  if (live.trail.length > 50) live.trail.shift();

  if (Math.abs(live.q[0]) > env.x_limit) end("cart left the track");
  else if (!$("endless").checked && live.t >= env.max_steps * env.dt - 1e-9) end("20 s episode finished");
}

function end(reason) {
  live.ended = performance.now();
  live.endReason = reason;
}

function frame(now) {
  requestAnimationFrame(frame);
  // Catch up on slow frames so sim time stays real time; longer gaps (tab in the background)
  // pause the simulation instead of fast-forwarding through it.
  const gap = live.lastFrame === null ? 0 : (now - live.lastFrame) / 1000;
  const wall = gap > 1.0 ? 0 : gap;
  live.lastFrame = now;
  if (live.policy && live.q && live.ended === null) {
    live.acc += wall;
    while (live.acc >= live.env.dt && live.ended === null) {  // fixed control-rate steps, real time
      controlStep();
      live.acc -= live.env.dt;
    }
  }
  const idle = !live.q || (live.ended !== null && now - live.ended > HOLD_SECONDS * 1000);
  if (idle && !live.loading) startEpisode();
  draw();
}

// ---- drawing ---------------------------------------------------------------

let view = null;  // world <-> pixel mapping, also used for mouse input

function draw() {
  const canvas = $("sim-canvas");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!live.q) return;

  const env = live.env;
  const { m: masses, l: lengths } = sim.chain(env);
  const L = lengths.reduce((a, b) => a + b, 0);
  // Equal scale on both axes; room for the pendulum both above and below the track.
  const scale = Math.min(W / (2 * (env.x_limit + 0.35)), H / (2 * L + 0.4));
  view = { scale, ox: W / 2, oy: H / 2 };
  const px = (x, y) => [view.ox + x * scale, view.oy - y * scale];
  const text = css("--text"), muted = css("--muted"), accent = css("--accent"), bad = css("--bad");

  // Interpolate between the last two physics steps for smooth 60+ fps drawing.
  const a = Math.min(live.acc / env.dt, 1);
  const q = live.q.map((v, i) => (i === 0 ? live.prevQ[0] + a * (v - live.prevQ[0])
                                          : live.prevQ[i] + a * sim.wrap(v - live.prevQ[i])));
  const pts = sim.points(env, q), pivot = pts[0], tip = pts.at(-1);

  // Track with end stops.
  ctx.lineWidth = 2;
  ctx.strokeStyle = muted;
  ctx.beginPath(); ctx.moveTo(...px(-env.x_limit - 0.2, -0.1)); ctx.lineTo(...px(env.x_limit + 0.2, -0.1)); ctx.stroke();
  ctx.strokeStyle = bad;
  for (const sx of [-env.x_limit, env.x_limit]) {
    ctx.beginPath(); ctx.moveTo(...px(sx, -0.15)); ctx.lineTo(...px(sx, 0.05)); ctx.stroke();
  }

  // Tip trail (last second).
  ctx.lineWidth = 1.5;
  for (let k = 1; k < live.trail.length; k++) {
    ctx.strokeStyle = `rgba(76, 194, 255, ${(0.5 * k) / live.trail.length})`;
    ctx.beginPath(); ctx.moveTo(...px(...live.trail[k - 1])); ctx.lineTo(...px(...live.trail[k])); ctx.stroke();
  }

  // Cart force from the policy (full scale = 0.8 m).
  arrow(ctx, px(pivot[0] + Math.sign(live.action) * 0.25, 0),
        px(pivot[0] + Math.sign(live.action) * (0.25 + 0.8 * Math.abs(live.action)), 0), accent);

  // Cart, links and masses.
  const [cx, cy] = px(pivot[0] - 0.2, 0.08);
  ctx.fillStyle = live.shielded ? "#ffb04c" : "#3a4250";  // orange while the track shield brakes
  ctx.fillRect(cx, cy, 0.4 * scale, 0.16 * scale);
  ctx.strokeStyle = text;
  ctx.lineWidth = Math.max(3, 0.03 * scale);
  ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(...px(...pivot));
  for (const p of pts.slice(1)) ctx.lineTo(...px(...p));
  ctx.stroke();
  ctx.fillStyle = accent;
  pts.slice(1).forEach((p, i) => {
    ctx.beginPath(); ctx.arc(...px(...p), Math.sqrt(masses[i]) * 0.1 * scale, 0, 2 * Math.PI); ctx.fill();
  });

  // User force on the tip (0.05 m per N), and the drag line to the mouse.
  const orange = "#ffb04c";
  if (live.mouse) {
    ctx.setLineDash([4, 4]); ctx.strokeStyle = orange; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(...px(...tip)); ctx.lineTo(...px(...live.mouse)); ctx.stroke();
    ctx.setLineDash([]);
  }
  const [fx, fy] = live.tipForce;
  if (Math.hypot(fx, fy) > 0.1) arrow(ctx, px(...tip), px(tip[0] + 0.05 * fx, tip[1] + 0.05 * fy), orange);

  ctx.fillStyle = text;
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillText(`${live.ckpt}   t = ${live.t.toFixed(2)} s`, 10, 20);
  if (live.ended !== null) { ctx.fillStyle = muted; ctx.fillText(live.endReason, 10, 38); }

  updateInfo();
}

function arrow(ctx, [x0, y0], [x1, y1], color) {
  const len = Math.hypot(x1 - x0, y1 - y0);
  if (len < 2) return;
  const ux = (x1 - x0) / len, uy = (y1 - y0) / len;
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1 + 8 * ux, y1 + 8 * uy);
  ctx.lineTo(x1 - 6 * uy, y1 + 6 * ux);
  ctx.lineTo(x1 + 6 * uy, y1 - 6 * ux);
  ctx.fill();
}

// Normal approximation of the predicted time to stable upright: the chance that the actual
// time is within +-CONFIDENCE_SPAN of the prediction.
const CONFIDENCE_SPAN = 0.5;
function erf(x) {  // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return Math.sign(x) * y;
}

function stableText() {
  if (!live.time) return "– (this checkpoint has no time prediction)";
  const reached = live.stableAt === null ? "" : ` · first stable at ${live.stableAt.toFixed(2)} s` +
    (live.predAtStart ? ` (predicted ${live.predAtStart.mean.toFixed(2)} s at the start)` : "");
  if (live.stable) return `stable now${reached}`;
  const { mean, std } = live.time;
  const p = std > 1e-3 ? erf(CONFIDENCE_SPAN / (std * Math.SQRT2)) : 1;
  return `in ${mean.toFixed(2)} s ± ${std.toFixed(2)} s · ${Math.round(100 * p)}% chance within ±${CONFIDENCE_SPAN} s${reached}`;
}

function updateInfo() {
  const m = live.meta;
  $("info-ckpt").textContent = `${live.ckpt} (update ${m.update}, ${fmtSteps(m.env_steps)} env steps)`;
  const ev = m.eval;
  $("info-eval").textContent = ev
    ? `swing-up ${(100 * ev.success).toFixed(0)}% of 64 hanging starts` +
      (ev.swingup_time == null ? "" : ` · ${ev.swingup_time.toFixed(1)} s avg`)
    : "–";
  const limit = $("endless").checked ? "∞" : `${(live.env.max_steps * live.env.dt).toFixed(0)} s`;
  $("info-time").textContent = `${live.t.toFixed(2)} s / ${limit}`;
  const frac = live.t > 0 ? live.uprightTime / live.t : 0;
  $("info-upright").textContent = `${(100 * frac).toFixed(0)}% of the time`;
  const v = (x) => (x == null ? "–" : x.toFixed(1));
  $("info-value").textContent = live.value == null ? "–"
    : `V(s) ${v(live.value)} · V(mirror) ${v(live.valueMirror)}`;
  $("info-stable").textContent = stableText();
  $("info-side").textContent = live.symmetric ? "built into the network (mirror-symmetric policy)"
    : $("symmetry").value === "off" ? "off"
    : `acting ${live.side === -1 ? "mirrored" : "as is"} · ${live.flips} switch${live.flips === 1 ? "" : "es"}`;
  const up = sim.isUpright(live.q);
  $("info-status").textContent = live.ended !== null ? `${live.endReason} · loading next…`
    : live.shielded ? "track shield braking" : live.mouse || live.tipForce[0] ? "being pushed" : up ? "balancing" : "swinging up";
}

// ---- input -----------------------------------------------------------------

function mouseWorld(e) {
  const r = $("sim-canvas").getBoundingClientRect();
  return [(e.clientX - r.left - view.ox) / view.scale, (view.oy - (e.clientY - r.top)) / view.scale];
}

const canvas = $("sim-canvas");
canvas.addEventListener("pointerdown", (e) => {
  if (!view) return;
  canvas.setPointerCapture(e.pointerId);
  canvas.classList.add("dragging");
  live.mouse = mouseWorld(e);
});
canvas.addEventListener("pointermove", (e) => { if (live.mouse) live.mouse = mouseWorld(e); });
for (const ev of ["pointerup", "pointercancel"]) {
  canvas.addEventListener(ev, () => { live.mouse = null; canvas.classList.remove("dragging"); });
}

window.addEventListener("keydown", (e) => {
  if (e.target instanceof Element && e.target.matches("select, input[type=range]")) return;
  if (e.key === "ArrowLeft" || e.key === "ArrowRight") { live.keys.add(e.key); e.preventDefault(); }
  if (e.key === " ") { e.preventDefault(); startEpisode(); }
});
window.addEventListener("keyup", (e) => live.keys.delete(e.key));
window.addEventListener("blur", () => live.keys.clear());

$("push-strength").addEventListener("input", (e) => ($("push-strength-value").textContent = `${e.target.value} N`));
$("ckpt").addEventListener("change", startEpisode);
$("start-mode").addEventListener("change", startEpisode);
$("symmetry").addEventListener("change", startEpisode);
$("decode").addEventListener("change", startEpisode);
$("restart").addEventListener("click", startEpisode);
$("run-select").addEventListener("change", (e) => {
  const url = new URL(location.href);
  if (e.target.value) url.searchParams.set("run", e.target.value); else url.searchParams.delete("run");
  location.href = url.toString();
});
// After using a control, give the keyboard back to the simulator (arrow keys would
// otherwise change the dropdown/slider, and Space would re-click the button).
for (const el of document.querySelectorAll(".controls select, .controls input, .controls button")) {
  el.addEventListener("change", () => el.blur());
  el.addEventListener("click", () => { if (el.tagName === "BUTTON" || el.type === "checkbox") el.blur(); });
}

// =============================================================================
// Wire it up
// =============================================================================

for (const [key, title] of CHARTS) makeChart(key, title);
loadRunPicker().catch(() => {});
if (RUN) $("parity-link").href = `parity.html?run=${encodeURIComponent(RUN)}`;
poll(pollMetrics, 2000);
poll(pollStatus, 2000);
poll(pollCheckpoints, 3000);
requestAnimationFrame(frame);
