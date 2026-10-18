// "Top runs": the leaderboard of the shown run's group (runs/<group>/*, ranked by the score of
// their latest checkpoint) and up to TOP_N of them simulated side by side from the same start,
// with deterministic actions (the policy mean), so the best members can be compared directly.
import * as sim from "./sim.mjs";

const TOP_N = 4;
const $ = (id) => document.getElementById(id);
const RUN = new URLSearchParams(location.search).get("run");
const pct = (v) => (v == null ? "–" : `${Math.round(100 * v)}%`);
const CASES = [["success", "still"], ["success_perturbed", "perturbed"], ["success_random", "random"],
               ["success_edge", "edge"], ["success_push", "push"]];

const top = { group: null, rows: [], sims: [], signature: "", acc: 0, last: null, ended: 0 };

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

function renderTable() {
  const head = `<tr><th>#</th><th>run</th><th>steps</th><th>score</th>${CASES.map(([, l]) => `<th>${l}</th>`).join("")}</tr>`;
  const body = top.rows.map((r, i) => {
    const current = r.name === RUN ? " class=current" : "";
    const steps = r.env_steps == null ? "–" : `${(r.env_steps / 1e6).toFixed(0)}M${r.progress < 1 ? "…" : ""}`;
    return `<tr${current}><td>${i + 1}</td><td><a href="?run=${encodeURIComponent(r.name)}">${r.label}</a></td>` +
      `<td>${steps}</td><td><b>${r.eval.score == null ? "–" : r.eval.score.toFixed(2)}</b></td>` +
      CASES.map(([k]) => `<td>${pct(r.eval[k])}</td>`).join("") + "</tr>";
  }).join("");
  $("top-table").innerHTML = head + body;
}

async function refresh() {
  const d = await fetchJSON(`/api/leaderboard${RUN ? `?run=${encodeURIComponent(RUN)}` : ""}`);
  $("top-section").hidden = !d.group;
  if (!d.group) return;
  top.group = d.group;
  top.rows = d.runs;
  $("top-group").textContent = d.group;
  renderTable();
  const best = d.runs.slice(0, TOP_N);
  const signature = best.map((r) => r.name + r.checkpoint).join();
  if (signature === top.signature) return;
  top.signature = signature;
  const policies = await Promise.all(best.map((r) =>
    fetchJSON(`/api/policy?ckpt=${r.checkpoint}&run=${encodeURIComponent(r.name)}`)));
  const grid = $("top-grid");
  grid.replaceChildren();
  top.sims = best.map((r, i) => {
    const box = document.createElement("div");
    box.className = "top-sim";
    box.innerHTML = `<canvas></canvas><div class="small num"><b>#${i + 1}</b> ${r.label} · ${r.checkpoint}<br><span class="muted"></span></div>`;
    grid.append(box);
    return { row: r, env: policies[i].env, policy: sim.loadPolicy(policies[i].layers, policies[i].symmetric),
             canvas: box.querySelector("canvas"), status: box.querySelector(".muted") };
  });
  restart();
}

function restart() {
  const s = sim.reset($("top-start").value);  // the same start for every run
  for (const t of top.sims) Object.assign(t, { q: [...s.q], qd: [...s.qd], t: 0, up: 0, ended: null, shielded: false });
  top.ended = 0;
}

function step(t) {
  if (t.ended) return;
  const env = t.env;
  [t.action, t.shielded] = sim.shield(env, t.q, t.qd, sim.policyAction(t.policy, sim.obs(env, t.q, t.qd)));
  ({ q: t.q, qd: t.qd } = sim.integrate(env, t.q, t.qd, t.action * env.max_force, [0, 0]));
  t.t += env.dt;
  if (sim.isUpright(t.q)) t.up += env.dt;
  if (Math.abs(t.q[0]) > env.x_limit) t.ended = "left the track";
  else if (t.t >= env.max_steps * env.dt - 1e-9) t.ended = "20 s done";
}

function draw(t) {
  const c = t.canvas, ctx = c.getContext("2d"), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const env = t.env, L = env.l1 + env.l2;
  const scale = Math.min(W / (2 * (env.x_limit + 0.3)), H / (2 * L + 0.3));
  const px = (x, y) => [W / 2 + x * scale, H / 2 - y * scale];
  ctx.strokeStyle = "#6b7280"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(...px(-env.x_limit, -0.08)); ctx.lineTo(...px(env.x_limit, -0.08)); ctx.stroke();
  const [pivot, p1, tip] = sim.points(env, t.q);
  const [cx, cy] = px(pivot[0] - 0.2, 0.07);
  ctx.fillStyle = t.shielded ? "#ffb04c" : "#3a4250";
  ctx.fillRect(cx, cy, 0.4 * scale, 0.14 * scale);
  ctx.strokeStyle = t.ended === "left the track" ? "#ff6b6b" : "#e6e8eb"; ctx.lineWidth = 2.5; ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(...px(...pivot)); ctx.lineTo(...px(...p1)); ctx.lineTo(...px(...tip)); ctx.stroke();
  t.status.textContent = `${t.t.toFixed(1)} s · upright ${Math.round((100 * t.up) / Math.max(t.t, 1e-9))}%` +
    (t.ended ? ` · ${t.ended}` : t.shielded ? " · shield" : "");
}

function frame(now) {
  if (top.sims.length) {
    const gap = top.last === null ? 0 : Math.min((now - top.last) / 1000, 0.1);
    top.acc += gap;
    const dt = top.sims[0].env.dt;
    while (top.acc >= dt) { top.sims.forEach(step); top.acc -= dt; }
    top.sims.forEach(draw);
    if (top.sims.every((t) => t.ended)) {
      top.ended += gap;
      if (top.ended > 1.5) restart();
    }
  }
  top.last = now;
  requestAnimationFrame(frame);
}

$("top-restart").addEventListener("click", restart);
$("top-start").addEventListener("change", restart);
refresh().catch(() => {});
setInterval(() => refresh().catch(() => {}), 15000);
requestAnimationFrame(frame);
