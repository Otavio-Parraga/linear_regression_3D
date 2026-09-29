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
  interacting: false,   // pointer/wheel gesture on the 3D plot in progress
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

const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

function fmt(v, d = 3) {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1e5) {           // e.g. after divergence: 3.14×10¹¹ keeps the layout intact
    const [mant, exp] = v.toExponential(1).split("e");
    return `${mant}×10${String(+exp).replace(/./g, (ch) => SUP[ch])}`;
  }
  const s = v.toFixed(d);
  return s === `-${(0).toFixed(d)}` ? (0).toFixed(d) : s;
}

// Same rule for KaTeX formulas.
function texNum(v, d = 2) {
  if (Math.abs(v) >= 1e5) {
    const [mant, exp] = v.toExponential(2).split("e");
    return `${mant}\\times10^{${+exp}}`;
  }
  return fmt(v, d);
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function withAlpha(hex, alpha) {
  const n = parseInt(hex.replace("#", ""), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
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

function inGrid([a, b]) {
  const r = scaleData().range;
  return a >= r.t0[0] && a <= r.t0[1] && b >= r.t1[0] && b <= r.t1[1];
}

// Gradient descent on this quadratic converges iff α < 2/λmax(H), where
// H = (1/m) Σ [[1, f], [f, f²]] is the Hessian of J.
function stableAlpha() {
  const f = feature(), m = f.length;
  const mf = f.reduce((s, v) => s + v, 0) / m;
  const mf2 = f.reduce((s, v) => s + v * v, 0) / m;
  const lmax = (1 + mf2) / 2 + Math.hypot((1 - mf2) / 2, mf);
  const amax = 2 / lmax;
  const p = 10 ** (Math.floor(Math.log10(amax)) - 1);
  return +(Math.floor(amax / p + 1e-3) * p).toPrecision(2);
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
  $("alphaHint").textContent = t("alphaHint", { amax: stableAlpha() });
}

function syncInputs(skip) {
  const [a, b] = state.theta;
  $("t0Range").value = a;
  $("t1Range").value = b;
  if (skip !== "t0Num") $("t0Num").value = +a.toFixed(4);
  if (skip !== "t1Num") $("t1Num").value = +b.toFixed(4);
  $("t0Range").setAttribute("aria-valuetext", `θ0 = ${fmt(a, 2)}`);
  $("t1Range").setAttribute("aria-valuetext", `θ1 = ${fmt(b, 3)}`);
  const r = scaleData().range;
  $("t0Out").hidden = a >= r.t0[0] && a <= r.t0[1];
  $("t1Out").hidden = b >= r.t1[0] && b <= r.t1[1];
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
  const reset = () => setTheta(startTheta());
  $("resetBtn").addEventListener("click", reset);
  $("statusReset").addEventListener("click", reset);

  // S = one step, R = run to convergence (again = stop), Esc = stop
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest("input, textarea, select")) return;
    const k = e.key.toLowerCase();
    if (k === "s") { e.preventDefault(); gdStep(); }
    else if (k === "r") { e.preventDefault(); runToConvergence(); }
    else if (k === "escape") stopRun();
  });

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
  document.querySelectorAll("[data-lang]").forEach((btn) =>
    btn.addEventListener("click", () => {
      if (btn.dataset.lang === state.lang) return;
      state.lang = btn.dataset.lang;
      savePref("lang", state.lang);
      applyLanguage();
      render();
    }));
}

function markActive(selector, key, value) {
  document.querySelectorAll(selector).forEach((b) => {
    const on = b.dataset[key] === value;
    b.classList.toggle("active", on);
    b.setAttribute("aria-pressed", on);
  });
}

function setScale(scale) {
  if (scale === state.scale) return;
  stopRun();
  const from = state.scale;
  state.theta = convert(state.theta, from, scale);
  state.path = state.path.map(([a, b, j]) => [...convert([a, b], from, scale), j]);
  state.scale = scale;
  markActive("[data-scale]", "scale", scale);
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
  else if (result === "diverged") setStatus("statusDiverged", { n, amax: stableAlpha() }, "bad");
  else if (result === "max") setStatus("statusMax", { n }, "bad");
  else setStatus("statusRunning", { n, j: fmt(costAndGrad(state.theta).J, 4) });
}

function gdStep() {
  stopRun();
  afterAdvance(gdAdvance());
}

function runGD() {
  if (state.running && state.running.kind === "steps") { stopRun(); return; }
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
  const kind = state.running && state.running.kind;
  $("convBtn").textContent = t(kind === "converge" ? "stop" : "converge");
  $("convBtn").classList.toggle("stop", kind === "converge");
  $("runBtn").textContent = t(kind === "steps" ? "stop" : "run");
  $("runBtn").classList.toggle("stop", kind === "steps");
}

function setStatus(key, vars, tone = "") {
  state.status = key ? { key, vars, tone } : null;
  renderStatus();
}

function renderStatus() {
  const el = $("gdStatus"), st = state.status;
  el.hidden = !st;
  if (!st) return;
  // Progress updates every frame; only the final outcome is announced.
  el.setAttribute("aria-live", st.tone ? "polite" : "off");
  el.dataset.tone = st.tone;
  el.title = st.tone === "ok" ? t("statusTol", { tol: GD_TOL }) : "";
  el.querySelector(".gd-text").textContent = t(st.key, st.vars);
  $("statusReset").hidden = st.tone !== "bad";
}

// ---------------------------------------------------------------------------
// Camera / drag mode
// ---------------------------------------------------------------------------
const VIEWS = {
  iso: { eye: { x: 1.5, y: -1.55, z: 0.95 }, up: { x: 0, y: 0, z: 1 } },
  top: { eye: { x: 0, y: 0, z: 2.1 }, up: { x: 0, y: 1, z: 0 } },
  t0: { eye: { x: 0, y: -2.1, z: 0.12 }, up: { x: 0, y: 0, z: 1 } },
  t1: { eye: { x: 2.1, y: 0, z: 0.12 }, up: { x: 0, y: 0, z: 1 } },
};

const VIEW_CENTER = { x: 0, y: 0, z: -0.16 };

function setView(name) {
  if (!HAS_WEBGL) return;
  state.camera = { ...VIEWS[name], center: VIEW_CENTER };
  markActive("[data-view]", "view", name);
  render();
}

function setDrag(mode) {
  state.drag = mode;
  markActive("[data-drag]", "drag", mode);
  if (HAS_WEBGL) Plotly.relayout(surfaceEl, { "scene.dragmode": mode });
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function applyLanguage() {
  document.documentElement.lang = state.lang === "pt" ? "pt-BR" : "en";
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-label]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nLabel)));
  markActive("[data-lang]", "lang", state.lang);
  $("keysHint").innerHTML = t("keysHint", { s: "<kbd>S</kbd>", r: "<kbd>R</kbd>", esc: "<kbd>Esc</kbd>" });
  if (state.model) $("alphaHint").textContent = t("alphaHint", { amax: stableAlpha() });
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
  const s = scaleData(), [a, b] = state.theta, v = state.scale === "normalized" ? "z" : "x";
  surfaceEl.setAttribute("aria-label", t("surfaceAria", {
    a: fmt(a, 2), b: fmt(b, 3), j: fmt(J, 3), js: fmt(s.J_star, 3), s0: fmt(s.theta_star[0], 2), s1: fmt(s.theta_star[1], 3),
  }));
  lineEl.setAttribute("aria-label", t("lineAria", { a: fmt(a, 2), b: fmt(b, 3), v, j: fmt(J, 3) }));
}

function signed(v, d = 2) {
  return v < 0 ? `- ${texNum(-v, d)}` : `+ ${texNum(v, d)}`;
}

function renderFormula() {
  const [a, b] = state.theta;
  const v = state.scale === "normalized" ? "z" : "x";
  const tex = `h(${v}) = \\theta_0 + \\theta_1 ${v} = ${texNum(a, 2)} ${signed(b, 2)}\\,${v}`;
  katex.render(tex, $("formula"), { throwOnError: false });

  const m = ys().length;
  katex.render(
    `J(\\theta)=\\frac{1}{2m}\\sum_{i=1}^{m}\\big(h(${v}^{(i)})-y^{(i)}\\big)^2`,
    $("costFormula"), { throwOnError: false });
  $("mCount").textContent = `m = ${m}`;

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
  $("gnVal").textContent = fmt(Math.hypot(grad[0], grad[1]), 3);
  $("jStar").textContent = fmt(scaleData().J_star, 3);
}

function renderTable() {
  const x = state.model.data.x, y = ys(), f = feature();
  const norm = state.scale === "normalized";
  const yhat = predict(state.theta);
  const cols = [t("colX"), ...(norm ? [t("colZ")] : []), t("colY"), t("colPred"), t("colErr"), t("colSq")];
  $("dataTable").tHead.innerHTML = `<tr>${cols.map((c) => `<th scope="col">${c}</th>`).join("")}</tr>`;

  const errs = y.map((yi, i) => yi - yhat[i]);
  const sqs = errs.map((e) => e * e);
  const sse = sqs.reduce((acc, v) => acc + v, 0);
  const maxSq = Math.max(...sqs) || 1;
  const rows = x.map((xi, i) =>
    `<tr><td>${fmt(xi, 1)}</td>${norm ? `<td>${fmt(f[i], 3)}</td>` : ""}` +
    `<td>${fmt(y[i], 1)}</td><td class="pred">${fmt(yhat[i], 2)}</td>` +
    `<td>${fmt(errs[i], 2)}</td>` +
    `<td class="sq" style="--w:${(100 * sqs[i] / maxSq).toFixed(1)}%">${fmt(sqs[i], 2)}</td></tr>`);
  $("dataTable").tBodies[0].innerHTML = rows.join("");

  const pad = cols.length - 2;
  $("dataTable").tFoot.innerHTML =
    `<tr><td colspan="${pad}"></td><td class="lbl">${t("sum")}</td><td>${fmt(sse, 2)}</td></tr>` +
    `<tr><td colspan="${pad}"></td><td class="lbl">J = Σ/2m</td><td class="j">${fmt(sse / (2 * y.length), 3)}</td></tr>`;
}

function themeLayout() {
  const fs = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
  return {
    text: cssVar("--text"), muted: cssVar("--muted"), grid: cssVar("--grid"),
    card: cssVar("--card"), inset: cssVar("--inset"), border: cssVar("--border-strong"),
    tick: Math.round(fs * 0.8), title: Math.round(fs * 0.95),
  };
}

function hoverLabel(c) {
  return { bgcolor: c.card, bordercolor: c.border, font: { color: c.text, family: "Atkinson Hyperlegible Mono, monospace", size: c.tick } };
}

function axis3d(title, range, c) {
  return {
    title: { text: title, font: { color: c.text, size: c.title } },
    tickfont: { color: c.muted, size: c.tick },
    range,
    autorange: false,
    gridcolor: c.grid,
    zerolinecolor: c.grid,
    showline: false,
    nticks: 6,
    showbackground: false,
    backgroundcolor: c.inset,
    showspikes: false,
  };
}

const HAS_WEBGL = (() => {
  try {
    const cv = document.createElement("canvas");
    return !!(cv.getContext("webgl2") || cv.getContext("webgl"));
  } catch (e) { return false; }
})();

// Height of the 3D box: always the whole landscape of the current scale, fixed,
// so moving θ or running GD never rescales the view.
function surfaceTop() {
  const s = scaleData();
  s.gridMax ??= Math.max(...s.grid.J.map((row) => Math.max(...row)));
  return s.gridMax;
}

function renderSurface(J, grad) {
  // While the user rotates or zooms, any redraw (even a data-only restyle)
  // re-applies Plotly's stored camera and fights the gesture. Hold the 3D
  // figure still; the run keeps going and the figure catches up on release.
  if (state.interacting && surfaceEl.data) return;
  if (!HAS_WEBGL) {
    surfaceEl.innerHTML = `<div class="no-webgl">${t("noWebgl")}</div>`;
    return;
  }
  const s = scaleData();
  const c = themeLayout();
  const [a, b] = state.theta;
  const r0 = s.range.t0, r1 = s.range.t1;
  const w0 = r0[1] - r0[0], w1 = r1[1] - r1[0];
  const cap = surfaceTop();
  const lift = cap * 0.01; // keep overlays from sinking into the surface
  const col = {
    point: cssVar("--point"), path: cssVar("--path"), deriv: cssVar("--deriv"),
    opt: cssVar("--optimum"), optFill: cssVar("--opt-fill"), optEdge: cssVar("--opt-edge"), accent: cssVar("--accent"), mesh: cssVar("--mesh"), derivStrong: cssVar("--deriv-strong"), planeFill: cssVar("--plane-fill"),
  };
  // z on the tangent plane at (a, b): J + g0 (t0 - a) + g1 (t1 - b)
  const onPlane = (xx, yy) => J + grad[0] * (xx - a) + grad[1] * (yy - b);

  const surface = {
    type: "surface",
    x: s.grid.t0, y: s.grid.t1, z: s.grid.J,
    colorscale: [[0, cssVar("--surf-0")], [0.28, cssVar("--surf-1")], [0.62, cssVar("--surf-2")], [1, cssVar("--surf-3")]],
    cmin: 0, cmax: cap,
    opacity: 0.97,
    showscale: false,
    contours: {
      // fine mesh along θ₀ and θ₁, and level curves projected on the floor
      x: { show: true, start: r0[0], end: r0[1], size: w0 / 12, color: withAlpha(col.mesh, 0.1), width: 1, highlight: false },
      y: { show: true, start: r1[0], end: r1[1], size: w1 / 12, color: withAlpha(col.mesh, 0.1), width: 1, highlight: false },
      z: {
        show: true, start: cap / 14, end: cap, size: cap / 14,
        color: withAlpha(col.accent, 0.5), width: 1.5, highlight: false,
        project: { z: true },
      },
    },
    lighting: { ambient: 0.62, diffuse: 0.72, specular: 0.28, roughness: 0.55, fresnel: 0.25 },
    lightposition: { x: -20000, y: -40000, z: 60000 },
    hovertemplate: "θ₀ %{x:.2f}<br>θ₁ %{y:.3f}<br>J %{z:.2f}<extra></extra>",
  };

  // Tangent plane patch + its outline
  const h0 = w0 * 0.1, h1 = w1 * 0.1;
  const px = [a - h0, a + h0], py = [b - h1, b + h1];
  const plane = {
    type: "surface",
    x: px, y: py, z: py.map((yy) => px.map((xx) => onPlane(xx, yy) + lift * 0.5)),
    surfacecolor: [[0, 0], [0, 0]],
    colorscale: [[0, col.planeFill], [1, col.planeFill]], cmin: 0, cmax: 1,
    showscale: false, opacity: 0.38, hoverinfo: "skip",
  };
  const corners = [[px[0], py[0]], [px[1], py[0]], [px[1], py[1]], [px[0], py[1]], [px[0], py[0]]];
  const planeEdge = {
    type: "scatter3d", mode: "lines", hoverinfo: "skip",
    x: corners.map((p) => p[0]), y: corners.map((p) => p[1]), z: corners.map(([xx, yy]) => onPlane(xx, yy) + lift * 0.5),
    line: { color: col.derivStrong, width: 5 },
  };
  // Slices of the plane along θ₀ and θ₁ through the point: their slopes are ∂J/∂θ₀ and ∂J/∂θ₁.
  const slices = {
    type: "scatter3d", mode: "lines", hoverinfo: "skip",
    x: [px[0], px[1], null, a, a], y: [b, b, null, py[0], py[1]],
    z: [onPlane(px[0], b), onPlane(px[1], b), null, onPlane(a, py[0]), onPlane(a, py[1])].map((v) => (v === null ? null : v + lift * 0.5)),
    line: { color: col.derivStrong, width: 2.5, dash: "dot" },
  };

  // −∇J: direction only (length is fixed relative to the axes), with a head
  // drawn in the tangent plane so it reads as an arrow from any angle.
  let arrow = { x: [], y: [], z: [] };
  if (Math.hypot(grad[0], grad[1]) > GD_TOL * 10) {
    const L = 0.3, head = 0.07, ang = 0.5;
    const gn = Math.hypot(grad[0] / w0, grad[1] / w1);
    const u = [-grad[0] / w0 / gn, -grad[1] / w1 / gn];            // unit, axis-relative
    const toTheta = ([uu, vv]) => [a + uu * w0, b + vv * w1];
    const tip = [u[0] * L, u[1] * L];
    const rot = (th) => [
      tip[0] + head * (-u[0] * Math.cos(th) + u[1] * Math.sin(th)),
      tip[1] + head * (-u[1] * Math.cos(th) - u[0] * Math.sin(th)),
    ];
    const pts = [[0, 0], tip, null, rot(ang), tip, rot(-ang)].map((p) => (p ? toTheta(p) : null));
    arrow = {
      x: pts.map((p) => (p ? p[0] : null)),
      y: pts.map((p) => (p ? p[1] : null)),
      z: pts.map((p) => (p ? onPlane(p[0], p[1]) + lift : null)),
    };
  }
  const arrowTrace = {
    type: "scatter3d", mode: "lines", hoverinfo: "skip", ...arrow,
    line: { color: col.derivStrong, width: 9 },
  };

  // Long runs have thousands of tiny steps; draw only visibly distinct points.
  // A path that leaves the box (divergence) is cut at the box edge.
  const pts = [];
  const inside = (p) => inGrid(p) && p[2] <= cap;
  for (let i = 0; i < state.path.length; i++) {
    const p = state.path[i], q = pts[pts.length - 1];
    if (!inside(p)) {
      if (q) pts.push([Math.min(Math.max(p[0], r0[0]), r0[1]), Math.min(Math.max(p[1], r1[0]), r1[1]), Math.min(p[2], cap)]);
      break;
    }
    if (!q || i === state.path.length - 1 || Math.hypot((p[0] - q[0]) / w0, (p[1] - q[1]) / w1) > 0.003) pts.push(p);
  }
  const onChart = inside([a, b, J]);
  const hide = (tr) => (onChart ? tr : { ...tr, x: [], y: [], z: [] });
  const path = {
    type: "scatter3d", mode: "lines+markers", hoverinfo: "skip",
    x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), z: pts.map((p) => p[2] + lift),
    line: { color: col.path, width: 6 },
    marker: { size: 3, color: col.path },
  };
  // Shadows on the floor, so the path and points also read in the Top view.
  const pathShadow = {
    type: "scatter3d", mode: "lines", hoverinfo: "skip", opacity: 0.7,
    x: pts.map((p) => p[0]), y: pts.map((p) => p[1]), z: pts.map(() => 0),
    line: { color: col.path, width: 4 },
  };

  const drop = (xx, yy, zz, color) => ({
    type: "scatter3d", mode: "lines+markers", hoverinfo: "skip",
    x: [xx, xx], y: [yy, yy], z: [zz, 0],
    line: { color, width: 3, dash: "dash" },
    marker: { size: [0, 5], color, symbol: "circle" },
  });

  const current = {
    type: "scatter3d", mode: "markers",
    x: [a], y: [b], z: [J + lift],
    marker: { size: 12, color: col.point, line: { color: "#ffffff", width: 3 } },
    hovertemplate: "θ₀ %{x:.2f}<br>θ₁ %{y:.3f}<br>J %{z:.3f}<extra></extra>",
  };

  const halo = {
    type: "scatter3d", mode: "markers", hoverinfo: "skip", opacity: 0.28,
    x: [a], y: [b], z: [J + lift],
    marker: { size: 30, color: col.point, line: { width: 0 } },
  };

  const [s0, s1] = s.theta_star;
  const optimum = {
    type: "scatter3d", mode: "markers",
    x: [s0], y: [s1], z: [s.J_star + lift],
    marker: { size: 8, symbol: "diamond", color: col.optFill, line: { color: col.optEdge, width: 2 } },
    hovertemplate: "θ* (%{x:.2f}, %{y:.3f})<br>J* %{z:.3f}<extra></extra>",
  };

  const eye = (state.camera && state.camera.eye) || VIEWS.iso.eye;
  const topDown = Math.abs(eye.z) > 3 * Math.hypot(eye.x, eye.y);

  const layout = {
    uirevision: "keep",
    paper_bgcolor: c.card,
    font: { family: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, sans-serif", color: c.text, size: c.tick },
    margin: { l: 28, r: 8, t: 0, b: 0 },
    showlegend: false,
    hoverlabel: hoverLabel(c),
    scene: {
      uirevision: "keep",
      dragmode: state.drag,
      aspectmode: "manual",
      aspectratio: { x: 1, y: 1, z: 0.74 },
      camera: state.camera,
      xaxis: axis3d("θ₀", r0, c),
      yaxis: axis3d("θ₁", r1, c),
      // the floor; the height axis is hidden when looking straight down, where its labels would pile up
      zaxis: { ...axis3d(topDown ? "" : "J(θ)", [0, cap], c), showbackground: true, showticklabels: !topDown },
    },
  };

  const traces = [
    surface, hide(plane), hide(planeEdge), hide(slices), pathShadow, drop(s0, s1, s.J_star, col.opt),
    hide(drop(a, b, J, col.point)), path, hide(arrowTrace), optimum, hide(halo), hide(current),
  ];

  Plotly.react(surfaceEl, traces, layout, { responsive: true, displaylogo: false, displayModeBar: false });
}

function renderLine() {
  const c = themeLayout();
  const f = feature(), y = ys();
  const norm = state.scale === "normalized";
  const yhat = predict(state.theta);
  const fMin = Math.min(...f), fMax = Math.max(...f);
  const pad = (fMax - fMin) * 0.08;
  const lx = [fMin - pad, fMax + pad];
  const yMin = Math.min(...y), yMax = Math.max(...y), ySpan = yMax - yMin;

  const resid = { x: [], y: [] };
  f.forEach((v, i) => { resid.x.push(v, v, null); resid.y.push(y[i], yhat[i], null); });

  const traces = [
    { type: "scatter", mode: "lines", name: t("traceResid"), ...resid,
      line: { color: cssVar("--resid"), width: 2, dash: "dot" }, hoverinfo: "skip" },
    { type: "scatter", mode: "lines", name: t("traceLine"),
      x: lx, y: predict(state.theta, lx), line: { color: cssVar("--line"), width: 3.5 },
      hovertemplate: `${norm ? "z" : "x"} %{x:.2f}<br>ŷ %{y:.2f}<extra></extra>` },
    { type: "scatter", mode: "markers", name: t("traceData"),
      x: f, y, marker: { color: cssVar("--data"), size: 10, line: { color: c.card, width: 2 } },
      hovertemplate: `${norm ? "z" : "x"} %{x:.2f}<br>y %{y:.1f}<extra></extra>` },
  ];

  const ax = (title, range) => ({
    title: { text: title, font: { size: c.tick, color: c.muted } }, range,
    tickfont: { size: c.tick, color: c.muted }, gridcolor: c.grid, zeroline: false,
    linecolor: c.border, ticks: "", automargin: true,
  });
  const layout = {
    paper_bgcolor: c.card, plot_bgcolor: c.card,
    font: { family: "Atkinson Hyperlegible Next, Atkinson Hyperlegible, sans-serif", color: c.text, size: c.tick },
    margin: { l: 8, r: 8, t: 4, b: 8 },
    showlegend: true,
    legend: { orientation: "h", x: 0, y: 1, yanchor: "bottom", font: { size: c.tick, color: c.muted } },
    hoverlabel: hoverLabel(c),
    xaxis: ax(norm ? t("axisZ") : t("axisX"), lx),
    yaxis: ax(t("axisY"), [yMin - 0.3 * ySpan, yMax + 0.3 * ySpan]),
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
  state.camera = { ...VIEWS.iso, center: VIEW_CENTER };
  markActive("[data-view]", "view", "iso");
  state.theta = startTheta();
  syncInputs();
  render();

  // Track the camera while it moves (drag, zoom), not only when it stops:
  // a running GD redraws every frame, and a stale camera would snap the view
  // back mid-drag.
  const followCamera = (ev) => {
    if (!ev["scene.camera"]) return;
    state.camera = ev["scene.camera"];
    markActive("[data-view]", "view", null); // user moved away from the preset
  };
  if (HAS_WEBGL) {
    surfaceEl.on("plotly_relayouting", followCamera);
    surfaceEl.on("plotly_relayout", followCamera);
  }

  // Click a point on the cost surface to jump there.
  // Plotly's 3D "click" fires when the button goes down, so the start of every
  // rotation would count as "move θ here". Hold the clicked point until the
  // button is released and apply it only if the pointer barely moved; rotating
  // then never moves θ or interrupts a run.
  let downAt = null, dragged = false, pending = null, wheelTimer = 0;
  surfaceEl.addEventListener("pointerdown", (e) => {
    downAt = [e.clientX, e.clientY]; dragged = false; pending = null;
    state.interacting = true;
  }, true);
  surfaceEl.addEventListener("wheel", () => {
    state.interacting = true;
    clearTimeout(wheelTimer);
    wheelTimer = setTimeout(() => { state.interacting = false; scheduleRender(); }, 250);
  }, { capture: true, passive: true });
  window.addEventListener("pointermove", (e) => {
    if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) dragged = true;
  }, true);
  window.addEventListener("pointerup", () => {
    if (!downAt) return;
    if (pending && !dragged) setTheta(pending);
    downAt = null; pending = null;
    state.interacting = false;
    scheduleRender();   // one full redraw with the camera where the user left it
  }, true);
  if (HAS_WEBGL) surfaceEl.on("plotly_click", (ev) => {
    const p = ev.points && ev.points[0];
    if (!p || p.curveNumber !== 0) return;
    if (downAt) pending = [p.x, p.y];      // button still down: decide on release
    else if (!dragged) setTheta([p.x, p.y]);
  });
}

init().catch((err) => {
  console.error(err);
  $("formula").textContent = "Failed to load /api/model: " + err.message;
});
