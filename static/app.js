"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  model: null,          // /api/model payload
  scale: "original",    // "original" | "normalized"
  theta: [0, 0],        // in the current scale
  path: [],             // gradient-descent path, [[t0, t1, J], ...] in the current scale
  alpha: { original: 0.02, normalized: 0.3 },
  lang: loadPref("lang", "en"),
  drag: "orbit",
  running: null,        // { stop(), kind } of the active GD animation
  camera: null,         // 3D camera: set by the View buttons, updated on drag
  iter: 0,              // GD iterations since the path started
  status: null,         // { key, vars, tone } of the GD status line
};

const $ = (id) => document.getElementById(id);
const surfaceEl = $("surface");
const lineEl = $("linePlot");

function loadPref(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch (e) { return fallback; }
}
function savePref(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* storage unavailable */ }
}

function t(key, vars) {
  let s = (I18N[state.lang] && I18N[state.lang][key]) ?? I18N.en[key] ?? key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, v);
  return s;
}

function fmt(v, d = 3) {
  if (!Number.isFinite(v)) return "—";
  const s = v.toFixed(d);
  return s === `-${(0).toFixed(d)}` ? (0).toFixed(d) : s;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---------------------------------------------------------------------------
// Math (mirrors app/model.py; per-frame values are computed here so the UI
// never waits on a network round trip through the tunnel)
// ---------------------------------------------------------------------------
const scaleData = () => state.model.scales[state.scale];
const feature = () => scaleData().feature;
const ys = () => state.model.data.y;

function predict(theta, f = feature()) {
  return f.map((v) => theta[0] + theta[1] * v);
}

function costAndGrad(theta) {
  const f = feature(), y = ys(), m = y.length;
  let j = 0, g0 = 0, g1 = 0;
  for (let i = 0; i < m; i++) {
    const e = theta[0] + theta[1] * f[i] - y[i];
    j += e * e;
    g0 += e;
    g1 += e * f[i];
  }
  return { J: j / (2 * m), grad: [g0 / m, g1 / m] };
}

// Same line expressed in the other scale: h = t0 + t1*x = n0 + n1*z, z = (x-mu)/sigma
function toNormalized([t0, t1]) {
  const { mean, std } = state.model.stats;
  return [t0 + t1 * mean, t1 * std];
}
function toOriginal([n0, n1]) {
  const { mean, std } = state.model.stats;
  const t1 = n1 / std;
  return [n0 - t1 * mean, t1];
}
function convert(theta, from, to) {
  if (from === to) return theta.slice();
  return to === "normalized" ? toNormalized(theta) : toOriginal(theta);
}

function startTheta() {
  // A deliberately poor starting line, defined in the original scale so that
  // switching scales shows the same line.
  const [a, b] = state.model.scales.original.theta_star;
  return convert([a - 20, b + 3], "original", state.scale);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function configureSliders() {
  const r = scaleData().range;
  [["t0Range", r.t0], ["t1Range", r.t1]].forEach(([id, [lo, hi]]) => {
    const el = $(id);
    el.min = lo;
    el.max = hi;
    el.step = (hi - lo) / 1000;
  });
  $("alpha").value = state.alpha[state.scale];
}

function syncInputs(skip) {
  const [a, b] = state.theta;
  $("t0Range").value = a;
  $("t1Range").value = b;
  if (skip !== "t0Num") $("t0Num").value = +a.toFixed(4);
  if (skip !== "t1Num") $("t1Num").value = +b.toFixed(4);
}

let frame = null;
function scheduleRender() {
  if (frame) return;
  frame = requestAnimationFrame(() => { frame = null; render(); });
}

function setTheta(theta, { keepPath = false, skip } = {}) {
  if (!keepPath) { state.path = []; state.iter = 0; stopRun(); setStatus(null); }
  state.theta = theta;
  syncInputs(skip);
  scheduleRender();
}

function bindControls() {
  $("t0Range").addEventListener("input", (e) => setTheta([+e.target.value, state.theta[1]]));
  $("t1Range").addEventListener("input", (e) => setTheta([state.theta[0], +e.target.value]));
  $("t0Num").addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    if (Number.isFinite(v)) setTheta([v, state.theta[1]], { skip: "t0Num" });
  });
  $("t1Num").addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    if (Number.isFinite(v)) setTheta([state.theta[0], v], { skip: "t1Num" });
  });
  ["t0Num", "t1Num"].forEach((id) => $(id).addEventListener("change", () => syncInputs()));

  $("alpha").addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    if (Number.isFinite(v) && v >= 0) state.alpha[state.scale] = v;
  });
  $("stepBtn").addEventListener("click", gdStep);
  $("runBtn").addEventListener("click", runGD);
  $("convBtn").addEventListener("click", runToConvergence);
  $("optBtn").addEventListener("click", () => setTheta(scaleData().theta_star.slice()));
  $("resetBtn").addEventListener("click", () => setTheta(startTheta()));

  document.querySelectorAll("[data-scale]").forEach((btn) =>
    btn.addEventListener("click", () => setScale(btn.dataset.scale)));
  document.querySelectorAll("[data-view]").forEach((btn) =>
    btn.addEventListener("click", () => setView(btn.dataset.view)));
  document.querySelectorAll("[data-drag]").forEach((btn) =>
    btn.addEventListener("click", () => setDrag(btn.dataset.drag)));

  $("themeBtn").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    savePref("theme", next);
    render();
  });
  $("langBtn").addEventListener("click", () => {
    state.lang = state.lang === "en" ? "pt" : "en";
    savePref("lang", state.lang);
    applyLanguage();
    render();
  });
}

function setScale(scale) {
  if (scale === state.scale) return;
  stopRun();
  const from = state.scale;
  state.theta = convert(state.theta, from, scale);
  state.path = state.path.map(([a, b, j]) => [...convert([a, b], from, scale), j]);
  state.scale = scale;
  document.querySelectorAll("[data-scale]").forEach((b) =>
    b.classList.toggle("active", b.dataset.scale === scale));
  configureSliders();
  syncInputs();
  render();
}

// ---------------------------------------------------------------------------
// Gradient descent
// ---------------------------------------------------------------------------
const GD_TOL = 1e-4;        // converged when ‖∇J‖ < GD_TOL
const GD_MAX_ITER = 100000;

// One update θ ← θ − α∇J, without rendering. Returns "ok" | "converged" | "diverged".
function gdAdvance() {
  const { J, grad } = costAndGrad(state.theta);
  if (state.path.length === 0) state.path.push([...state.theta, J]);
  if (Math.hypot(grad[0], grad[1]) < GD_TOL) return "converged";
  const a = state.alpha[state.scale];
  const next = [state.theta[0] - a * grad[0], state.theta[1] - a * grad[1]];
  const nj = costAndGrad(next).J;
  if (!Number.isFinite(nj) || nj > 1e12 || Math.abs(next[0]) > 1e6 || Math.abs(next[1]) > 1e6) {
    return "diverged";
  }
  state.theta = next;
  state.path.push([...next, nj]);
  state.iter++;
  return Math.hypot(...costAndGrad(next).grad) < GD_TOL ? "converged" : "ok";
}

function afterAdvance(result) {
  syncInputs();
  scheduleRender();
  const n = state.iter;
  if (result === "converged") setStatus("statusConverged", { n, tol: GD_TOL }, "ok");
  else if (result === "diverged") setStatus("statusDiverged", { n }, "bad");
  else if (result === "max") setStatus("statusMax", { n }, "bad");
  else setStatus("statusRunning", { n, j: fmt(costAndGrad(state.theta).J, 4) });
}

function gdStep() {
  stopRun();
  afterAdvance(gdAdvance());
}

function runGD() {
  stopRun();
  let n = 0;
  const id = setInterval(() => {
    const r = gdAdvance();
    afterAdvance(r);
    if (++n >= 30 || r !== "ok") stopRun();
  }, 70);
  state.running = { stop: () => clearInterval(id), kind: "steps" };
  updateRunUi();
}

// Run until ‖∇J‖ < GD_TOL. The iteration budget grows with elapsed time
// (not frame count), so the first, largest moves are visible and a slow run
// (thousands of iterations in the original scale) still ends in a few
// seconds on any machine.
function runToConvergence() {
  if (state.running && state.running.kind === "converge") { stopRun(); return; }
  stopRun();
  let raf = 0, cancelled = false, start = null;
  const iter0 = state.iter;
  const tick = (now) => {
    if (cancelled) return;
    if (start === null) start = now;
    const ms = now - start;
    const target = iter0 + 1 + Math.floor(ms / 80 + (ms / 400) ** 3);
    let r = "ok";
    for (let k = 0; state.iter < target && k < 5000 && r === "ok"; k++) {
      r = gdAdvance();
      if (r === "ok" && state.iter >= GD_MAX_ITER) r = "max";
    }
    afterAdvance(r);
    if (r === "ok") raf = requestAnimationFrame(tick);
    else stopRun();
  };
  state.running = { stop: () => { cancelled = true; cancelAnimationFrame(raf); }, kind: "converge" };
  updateRunUi();
  raf = requestAnimationFrame(tick);
}

function stopRun() {
  if (state.running) state.running.stop();
  state.running = null;
  updateRunUi();
}

function updateRunUi() {
  const converging = !!(state.running && state.running.kind === "converge");
  const btn = $("convBtn");
  btn.textContent = t(converging ? "stop" : "converge");
  btn.classList.toggle("stop", converging);
}

function setStatus(key, vars, tone = "") {
  state.status = key ? { key, vars, tone } : null;
  renderStatus();
}

function renderStatus() {
  const el = $("gdStatus"), st = state.status;
  el.textContent = st ? t(st.key, st.vars) : "";
  el.dataset.tone = st ? st.tone : "";
}

// ---------------------------------------------------------------------------
// Camera / drag mode
// ---------------------------------------------------------------------------
const VIEWS = {
  iso: { eye: { x: 1.55, y: -1.55, z: 0.95 }, up: { x: 0, y: 0, z: 1 } },
  top: { eye: { x: 0, y: 0, z: 2.4 }, up: { x: 0, y: 1, z: 0 } },
  t0: { eye: { x: 0, y: -2.4, z: 0.15 }, up: { x: 0, y: 0, z: 1 } },
  t1: { eye: { x: 2.4, y: 0, z: 0.15 }, up: { x: 0, y: 0, z: 1 } },
};

function setView(name) {
  if (!HAS_WEBGL) return;
  state.camera = { ...VIEWS[name], center: { x: 0, y: 0, z: 0 } };
  render();
}

function setDrag(mode) {
  state.drag = mode;
  document.querySelectorAll("[data-drag]").forEach((b) =>
    b.classList.toggle("active", b.dataset.drag === mode));
  if (HAS_WEBGL) Plotly.relayout(surfaceEl, { "scene.dragmode": mode });
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function applyLanguage() {
  document.documentElement.lang = state.lang === "pt" ? "pt-BR" : "en";
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  $("langBtn").textContent = state.lang === "en" ? "PT" : "EN";
  updateRunUi();
  renderStatus();
  document.title = t("title");
}

function render() {
  if (!state.model) return;
  const { J, grad } = costAndGrad(state.theta);
  renderFormula();
  renderMetrics(J, grad);
  renderTable();
  renderSurface(J, grad);
  renderLine();
}

function signed(v, d = 2) {
  return v < 0 ? `- ${fmt(-v, d)}` : `+ ${fmt(v, d)}`;
}

function renderFormula() {
  const [a, b] = state.theta;
  const v = state.scale === "normalized" ? "z" : "x";
  const tex = `h(${v}) = \\theta_0 + \\theta_1 ${v} = ${fmt(a, 2)} ${signed(b, 2)}\\,${v}`;
  katex.render(tex, $("formula"), { throwOnError: false });

  const m = ys().length;
  katex.render(
    `J(\\theta)=\\frac{1}{2m}\\sum_{i=1}^{m}\\big(h(${v}^{(i)})-y^{(i)}\\big)^2,\\quad m=${m}`,
    $("costFormula"), { throwOnError: false });

  const note = $("normNote");
  if (state.scale === "normalized") {
    const [o0, o1] = toOriginal(state.theta);
    const { mean, std } = state.model.stats;
    note.textContent = t("normNote", { mu: fmt(mean, 2), sigma: fmt(std, 3), o0: fmt(o0, 2), o1: fmt(o1, 3) });
    note.hidden = false;
  } else {
    note.hidden = true;
  }
}

function renderMetrics(J, grad) {
  $("jVal").textContent = fmt(J, 3);
  $("g0Val").textContent = fmt(grad[0], 3);
  $("g1Val").textContent = fmt(grad[1], 3);
  $("jStar").textContent = fmt(scaleData().J_star, 3);
}

function renderTable() {
  const x = state.model.data.x, y = ys(), f = feature();
  const norm = state.scale === "normalized";
  const yhat = predict(state.theta);
  const cols = [t("colX"), ...(norm ? [t("colZ")] : []), t("colY"), t("colPred"), t("colErr"), t("colSq")];
  $("dataTable").tHead.innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;

  let sse = 0;
  const rows = x.map((xi, i) => {
    const e = y[i] - yhat[i];
    sse += e * e;
    return `<tr><td>${fmt(xi, 1)}</td>${norm ? `<td>${fmt(f[i], 3)}</td>` : ""}` +
      `<td>${fmt(y[i], 1)}</td><td class="pred">${fmt(yhat[i], 2)}</td>` +
      `<td class="${e < 0 ? "neg" : ""}">${fmt(e, 2)}</td><td>${fmt(e * e, 2)}</td></tr>`;
  });
  $("dataTable").tBodies[0].innerHTML = rows.join("");

  const pad = cols.length - 2;
  $("dataTable").tFoot.innerHTML =
    `<tr><td colspan="${pad}"></td><td>${t("sum")}</td><td>${fmt(sse, 2)}</td></tr>` +
    `<tr><td colspan="${pad}"></td><td>J = Σ/2m</td><td>${fmt(sse / (2 * y.length), 3)}</td></tr>`;
}

function themeLayout() {
  const text = cssVar("--text"), grid = cssVar("--grid"), card = cssVar("--card");
  return { text, grid, card, muted: cssVar("--muted") };
}

function axis3d(title, range, c) {
  return {
    title: { text: title, font: { color: c.text } },
    range,
    autorange: false,
    color: c.text,
    gridcolor: c.grid,
    zerolinecolor: c.grid,
    showbackground: true,
    backgroundcolor: c.card,
    showspikes: false,
  };
}

const HAS_WEBGL = (() => {
  try {
    const cv = document.createElement("canvas");
    return !!(cv.getContext("webgl2") || cv.getContext("webgl"));
  } catch (e) { return false; }
})();

function renderSurface(J, grad) {
  if (!HAS_WEBGL) {
    surfaceEl.innerHTML = `<div class="no-webgl">${t("noWebgl")}</div>`;
    return;
  }
  const s = scaleData();
  const c = themeLayout();
  const dark = document.documentElement.dataset.theme === "dark";
  const [a, b] = state.theta;
  const r0 = s.range.t0, r1 = s.range.t1;
  const w0 = r0[1] - r0[0], w1 = r1[1] - r1[0];
  const zMax = Math.max(...s.grid.J.map((row) => Math.max(...row))) * 1.02;
  const lift = zMax * 0.008; // keep overlays from sinking into the surface

  const surface = {
    type: "surface",
    name: t("traceSurface"),
    x: s.grid.t0, y: s.grid.t1, z: s.grid.J,
    colorscale: "Viridis",
    opacity: 0.9,
    showscale: false,
    contours: {
      z: { show: true, usecolormap: true, project: { z: true }, width: 1 },
    },
    lighting: { ambient: 0.75, diffuse: 0.6, specular: 0.1, roughness: 0.9 },
    hovertemplate: "θ₀ %{x:.2f}<br>θ₁ %{y:.3f}<br>J %{z:.2f}<extra></extra>",
  };

  // Tangent plane at the current point: z = J + g0 (t0 - a) + g1 (t1 - b)
  const h0 = w0 * 0.1, h1 = w1 * 0.1;
  const px = [a - h0, a + h0], py = [b - h1, b + h1];
  const pz = py.map((yy) => px.map((xx) => J + grad[0] * (xx - a) + grad[1] * (yy - b)));
  const plane = {
    type: "surface",
    name: t("tracePlane"),
    x: px, y: py, z: pz,
    surfacecolor: [[0, 0], [0, 0]],
    colorscale: [[0, cssVar("--plane")], [1, cssVar("--plane")]],
    cmin: 0, cmax: 1,
    showscale: false,
    opacity: 0.55,
    showlegend: true,
    hoverinfo: "skip",
  };

  // Descent direction -∇J, drawn along the tangent plane; length is scaled
  // relative to the axis ranges so it is always visible.
  const gn = Math.hypot(grad[0] / w0, grad[1] / w1);
  let arrow = { x: [], y: [], z: [] };
  if (Math.hypot(grad[0], grad[1]) > GD_TOL * 10) {
    const k = 0.2 / gn;
    const d0 = -k * grad[0], d1 = -k * grad[1];
    arrow = { x: [a, a + d0], y: [b, b + d1], z: [J + lift, J + grad[0] * d0 + grad[1] * d1 + lift] };
  }
  const arrowLine = {
    type: "scatter3d", mode: "lines+markers", name: t("traceGrad"),
    ...arrow,
    line: { color: cssVar("--arrow"), width: 8 },
    marker: { size: [0, 7], symbol: "diamond", color: cssVar("--arrow") },
    hoverinfo: "skip",
  };

  const drop = {
    type: "scatter3d", mode: "lines+markers", showlegend: false,
    x: [a, a], y: [b, b], z: [J, 0],
    line: { color: cssVar("--point"), width: 3, dash: "dash" },
    marker: { size: [0, 4], color: cssVar("--point") },
    hoverinfo: "skip",
  };

  const current = {
    type: "scatter3d", mode: "markers", name: t("traceCurrent"),
    x: [a], y: [b], z: [J + lift],
    marker: { size: 9, color: cssVar("--point"), line: { color: dark ? "#000" : "#fff", width: 2 } },
    hovertemplate: "θ₀ %{x:.2f}<br>θ₁ %{y:.3f}<br>J %{z:.3f}<extra></extra>",
  };

  const [s0, s1] = s.theta_star;
  const optimum = {
    type: "scatter3d", mode: "markers", name: t("traceOptimum"),
    x: [s0], y: [s1], z: [s.J_star],
    marker: { size: 5, symbol: "x", color: c.text },
    hovertemplate: "θ* (%{x:.2f}, %{y:.3f})<br>J* %{z:.3f}<extra></extra>",
  };

  // Long runs have thousands of tiny steps; draw only visibly distinct points.
  const pts = [];
  state.path.forEach((p, i) => {
    const q = pts[pts.length - 1];
    if (!q || i === state.path.length - 1 || Math.hypot((p[0] - q[0]) / w0, (p[1] - q[1]) / w1) > 0.003) pts.push(p);
  });
  const path = {
    type: "scatter3d", mode: "lines+markers", name: t("tracePath"),
    x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), z: pts.map((p) => p[2] + lift),
    line: { color: c.text, width: 4 },
    marker: { size: 3, color: c.text },
    hoverinfo: "skip",
  };

  const layout = {
    uirevision: "keep",
    paper_bgcolor: c.card,
    font: { family: "Inter, sans-serif", color: c.text, size: 12 },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    showlegend: true,
    legend: { orientation: "h", x: 0, y: 1, bgcolor: "rgba(0,0,0,0)", font: { color: c.text } },
    scene: {
      uirevision: "keep",
      dragmode: state.drag,
      aspectmode: "manual",
      aspectratio: { x: 1, y: 1, z: 0.75 },
      camera: state.camera,
      xaxis: axis3d("θ₀", r0, c),
      yaxis: axis3d("θ₁", r1, c),
      zaxis: axis3d("J(θ)", [0, zMax], c),
    },
  };

  Plotly.react(surfaceEl, [surface, plane, drop, arrowLine, path, optimum, current], layout,
    { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["toImage"] });
}

function renderLine() {
  const c = themeLayout();
  const f = feature(), y = ys();
  const norm = state.scale === "normalized";
  const yhat = predict(state.theta);
  const fMin = Math.min(...f), fMax = Math.max(...f);
  const pad = (fMax - fMin) * 0.08;
  const lx = [fMin - pad, fMax + pad];
  const yMin = Math.min(...y), yMax = Math.max(...y);

  const resid = { x: [], y: [] };
  f.forEach((v, i) => { resid.x.push(v, v, null); resid.y.push(y[i], yhat[i], null); });

  const traces = [
    { type: "scatter", mode: "lines", name: t("traceResid"), ...resid,
      line: { color: cssVar("--resid"), width: 1.5, dash: "dot" }, hoverinfo: "skip" },
    { type: "scatter", mode: "lines", name: t("traceLine"),
      x: lx, y: predict(state.theta, lx), line: { color: cssVar("--line"), width: 3 },
      hovertemplate: `${norm ? "z" : "x"} %{x:.2f}<br>ŷ %{y:.2f}<extra></extra>` },
    { type: "scatter", mode: "markers", name: t("traceData"),
      x: f, y, marker: { color: cssVar("--data"), size: 8 },
      hovertemplate: `${norm ? "z" : "x"} %{x:.2f}<br>y %{y:.1f}<extra></extra>` },
  ];

  const ax = (title, range) => ({
    title: { text: title }, range, color: c.muted, gridcolor: c.grid, zerolinecolor: c.grid,
  });
  const layout = {
    paper_bgcolor: c.card, plot_bgcolor: c.card,
    font: { family: "Inter, sans-serif", color: c.text, size: 12 },
    margin: { l: 56, r: 12, t: 8, b: 44 },
    showlegend: true,
    legend: { orientation: "h", x: 0, y: 1.12 },
    xaxis: ax(norm ? t("axisZ") : t("axisX"), lx),
    yaxis: ax(t("axisY"), [yMin - 25, yMax + 20]),
  };
  Plotly.react(lineEl, traces, layout, { responsive: true, displaylogo: false, displayModeBar: false });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function init() {
  applyLanguage();
  bindControls();
  const res = await fetch("/api/model");
  state.model = await res.json();
  configureSliders();
  state.camera = { ...VIEWS.iso, center: { x: 0, y: 0, z: 0 } };
  state.theta = startTheta();
  syncInputs();
  render();

  // Click a point on the cost surface to jump there.
  // Remember where the user rotated to, so redraws never move the camera.
  if (HAS_WEBGL) surfaceEl.on("plotly_relayout", (ev) => {
    if (ev["scene.camera"]) state.camera = ev["scene.camera"];
  });
  if (HAS_WEBGL) surfaceEl.on("plotly_click", (ev) => {
    const p = ev.points && ev.points[0];
    if (p && p.curveNumber === 0) setTheta([p.x, p.y]);
  });
}

init().catch((err) => {
  console.error(err);
  $("formula").textContent = "Failed to load /api/model: " + err.message;
});
