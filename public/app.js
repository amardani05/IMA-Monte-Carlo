/* IMA Monte Carlo front end.
   Form <-> payload, SEC auto-fill with provenance, POST /api/simulate, SVG panels,
   reconciliation, share links, correlation editor, PNG export, pin-and-compare. */

const PRESET = {
  ticker: "MYRG", company_name: "MYR Group", current_price: 271.0,
  bear_price: 222.32, base_price: 323.01, bull_price: 362.32,
  current_revenue: 3510.0, current_ebitda_margin: 0.061, current_ev: 4310.0,
  current_net_debt: 110.0, shares_outstanding: 15.5,
  rev_cagr: [0.057, 0.069, 0.075], ebitda_margin: [0.071, 0.075, 0.077],
  ev_ebitda_multiple: [12.4, 17.0, 18.5], net_debt_change_pct: [0.0, 0.015],
  share_dilution_pct: [0.01, 0.01], n_simulations: 100000, horizon_years: 2, random_seed: 42,
};
const DEFAULT_CORR = [
  [1.00, 0.40, 0.35, -0.15, 0.10],
  [0.40, 1.00, 0.25, -0.10, 0.05],
  [0.35, 0.25, 1.00, -0.05, 0.00],
  [-0.15, -0.10, -0.05, 1.00, 0.20],
  [0.10, 0.05, 0.00, 0.20, 1.00],
];
const CORR_LABELS = ["Rev", "Margin", "Mult", "Debt", "Dilut"];
const TRI = ["rev_cagr", "ebitda_margin", "ev_ebitda_multiple"];
const NORM = ["net_debt_change_pct", "share_dilution_pct"];
const SCALARS = ["current_price", "bear_price", "base_price", "bull_price", "current_revenue",
  "current_ebitda_margin", "current_ev", "current_net_debt", "shares_outstanding",
  "n_simulations", "horizon_years", "random_seed"];

const C = {
  blue: "#2a63b0", red: "#b1271f", amber: "#9a6300", green: "#1f6d44", purple: "#574ca6",
  text: "#101820", muted: "#64707e", faint: "#8b959f", grid: "#e9edf2", axis: "#c9d1db", panel: "#ffffff",
};

const $ = (id) => document.getElementById(id);
const form = $("pitchForm");
let lastResult = null;
let lastPayload = null;
let pinned = null;
let multipleIsPlaceholder = false;
let multipleEdited = false;

/* Formatting */
const money = (v, d = 2) => "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const money0 = (v) => money(v, 0);
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const spct = (v, d = 1) => (v >= 0 ? "+" : "") + pct(v, d);
const mult = (v, d = 1) => Number(v).toFixed(d) + "x";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const round = (v, d) => Math.round(v * 10 ** d) / 10 ** d;
const REF_FMT = { rev_cagr: (v) => pct(v, 1), ebitda_margin: (v) => pct(v, 2), ev_ebitda_multiple: (v) => mult(v),
  net_debt_change_pct: (v) => spct(v, 1), share_dilution_pct: (v) => spct(v, 2) };

/* Correlation grid */
function buildCorrGrid(matrix) {
  const g = $("corrGrid");
  let html = `<div class="h"></div>` + CORR_LABELS.map((l) => `<div class="h">${l}</div>`).join("");
  for (let i = 0; i < 5; i++) {
    html += `<div class="h row">${CORR_LABELS[i]}</div>`;
    for (let j = 0; j < 5; j++) {
      const diag = i === j;
      html += `<input type="number" step="0.05" min="-1" max="1" data-i="${i}" data-j="${j}" value="${matrix[i][j]}"
        ${diag ? 'class="diag" readonly tabindex="-1"' : ""} aria-label="Correlation ${CORR_LABELS[i]} with ${CORR_LABELS[j]}">`;
    }
  }
  g.innerHTML = html;
  g.querySelectorAll("input:not(.diag)").forEach((el) => {
    el.addEventListener("input", () => {
      const v = parseFloat(el.value);
      const twin = g.querySelector(`input[data-i="${el.dataset.j}"][data-j="${el.dataset.i}"]`);
      if (twin) twin.value = el.value;
      const st = $("corrStatus");
      if (Number.isFinite(v) && Math.abs(v) > 1) { st.textContent = "Correlations must be between -1 and 1."; st.className = "hint bad"; }
      else { st.textContent = "Mirrored. Positive-definiteness is checked when you run."; st.className = "hint"; }
    });
  });
}
function readCorr() {
  const m = DEFAULT_CORR.map((r) => r.slice());
  $("corrGrid").querySelectorAll("input").forEach((el) => {
    const v = parseFloat(el.value);
    if (Number.isFinite(v)) m[+el.dataset.i][+el.dataset.j] = v;
  });
  for (let i = 0; i < 5; i++) m[i][i] = 1;
  return m;
}
function corrIsDefault(m) { return m.every((r, i) => r.every((v, j) => Math.abs(v - DEFAULT_CORR[i][j]) < 1e-9)); }

/* Form <-> payload */
function fillForm(p) {
  form.ticker.value = p.ticker || "";
  form.company_name.value = p.company_name || "";
  SCALARS.forEach((k) => { if (form[k] && p[k] !== undefined && p[k] !== null) form[k].value = p[k]; });
  TRI.forEach((k) => { if (p[k]) { form[`${k}_low`].value = p[k][0]; form[`${k}_mode`].value = p[k][1]; form[`${k}_high`].value = p[k][2]; } });
  NORM.forEach((k) => { if (p[k]) { form[`${k}_mean`].value = p[k][0]; form[`${k}_std`].value = p[k][1]; } });
  buildCorrGrid(p.correlation_matrix || DEFAULT_CORR);
  if (p.correlation_matrix && !corrIsDefault(p.correlation_matrix)) $("corrDetails").open = true;
}
function readForm() {
  const num = (name) => {
    const el = form[name]; const v = el.value.trim();
    if (v === "" || !Number.isFinite(Number(v))) {
      el.classList.add("invalid");
      throw new Error(`"${el.parentElement.childNodes[0].textContent.trim()}" needs a number.`);
    }
    el.classList.remove("invalid");
    return Number(v);
  };
  form.querySelectorAll("input").forEach((el) => el.classList.remove("invalid"));
  const payload = { ticker: form.ticker.value.trim().toUpperCase() || "TICKER",
    company_name: form.company_name.value.trim() || form.ticker.value.trim() || "Untitled" };
  SCALARS.forEach((k) => { payload[k] = num(k); });
  TRI.forEach((k) => { payload[k] = [num(`${k}_low`), num(`${k}_mode`), num(`${k}_high`)]; });
  NORM.forEach((k) => { payload[k] = [num(`${k}_mean`), num(`${k}_std`)]; });
  const corr = readCorr();
  if (!corrIsDefault(corr)) payload.correlation_matrix = corr;
  return payload;
}

/* URL state */
const b64e = (s) => btoa(unescape(encodeURIComponent(s))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const b64d = (s) => decodeURIComponent(escape(atob(s.replace(/-/g, "+").replace(/_/g, "/"))));
function writeUrl(payload) {
  try { history.replaceState(null, "", "#s=" + b64e(JSON.stringify(payload))); } catch (_) { /* ignore */ }
}
function readUrl() {
  const m = location.hash.match(/^#s=(.+)$/);
  if (!m) return null;
  try { return JSON.parse(b64d(m[1])); } catch (_) { return null; }
}

/* Provenance and reference chips */
function clearProvenance() {
  document.querySelectorAll(".src-badge, .calib-note").forEach((el) => el.remove());
  document.querySelectorAll(".needs-input").forEach((el) => el.classList.remove("needs-input"));
  document.querySelectorAll(".refs").forEach((el) => { el.innerHTML = ""; });
  $("sources").hidden = true; $("sourcesList").innerHTML = "";
}
function badge(label, title, tone) {
  const b = document.createElement("span");
  b.className = "src-badge" + (tone ? ` ${tone}` : ""); b.textContent = label; b.title = title; return b;
}
function setMode(driver, value) {
  const isTri = TRI.includes(driver);
  const el = form[`${driver}_${isTri ? "mode" : "mean"}`];
  el.value = value;
  if (isTri) {
    const lo = form[`${driver}_low`], hi = form[`${driver}_high`];
    if (parseFloat(lo.value) > value) lo.value = value;
    if (parseFloat(hi.value) < value) hi.value = value;
  }
  if (driver === "ev_ebitda_multiple") multipleEdited = true;
  refreshChipState(driver);
}
function refreshChipState(driver) {
  const isTri = TRI.includes(driver);
  const cur = parseFloat(form[`${driver}_${isTri ? "mode" : "mean"}`].value);
  document.querySelectorAll(`.refs[data-refs="${driver}"] .chip[data-value]`).forEach((ch) => {
    ch.classList.toggle("active", Math.abs(parseFloat(ch.dataset.value) - cur) < 1e-9);
  });
}
function renderRefs(driver, ref, note) {
  const box = document.querySelector(`.refs[data-refs="${driver}"]`);
  if (!box) return;
  const f = REF_FMT[driver];
  let html = "";
  const settable = [];
  if (ref && ref.current !== undefined) settable.push(["Current", ref.current]);
  if (ref && ref.hist_median !== undefined) settable.push(["Hist. median", ref.hist_median]);
  if (ref && ref.hist_mean !== undefined) settable.push(["Hist. mean", ref.hist_mean]);
  if (settable.length) {
    html += `<span class="cap">Set mode to</span>`;
    html += settable.map(([l, v]) => `<button type="button" class="chip" data-value="${v}" title="Set the mode to ${l.toLowerCase()} ${f(v)}">${l} ${f(v)}</button>`).join("");
  }
  if (ref && ref.raw_min !== undefined && ref.raw_max !== undefined) {
    html += `<span class="chip static" title="Full filed range before trimming">Filed ${f(ref.raw_min)} to ${f(ref.raw_max)}${ref.n ? `, n=${ref.n}` : ""}</span>`;
  } else if (ref && ref.n) {
    html += `<span class="chip static">n=${ref.n}</span>`;
  }
  if (ref && ref.fcf_yield_mean !== undefined) {
    html += `<span class="chip static" title="Free cash flow as a share of enterprise value">FCF yield ${pct(ref.fcf_yield_mean, 1)}/yr</span>`;
  }
  if (note) {
    const weak = /Generic|Insufficient|Placeholder|Capped|too few|No usable/i.test(note);
    html += `<p class="calib-note${weak ? " weak" : ""}">${esc(note)}</p>`;
  }
  box.innerHTML = html;
  box.querySelectorAll(".chip[data-value]").forEach((ch) => ch.addEventListener("click", () => setMode(driver, parseFloat(ch.dataset.value))));
  refreshChipState(driver);
}
function applyProvenance(prov, refs) {
  document.querySelectorAll("label[data-field]").forEach((label) => {
    const note = prov[label.dataset.field]; if (!note) return;
    const fromSec = note.startsWith("SEC EDGAR"), derived = note.startsWith("derived");
    if (fromSec || derived) label.prepend(badge(fromSec ? "SEC" : "derived", note, derived ? "derived" : ""));
  });
  [...TRI, ...NORM].forEach((d) => renderRefs(d, refs[d], prov[d]));
  $("sourcesList").innerHTML = Object.entries(prov).map(([k, v]) => `<dt>${esc(k.replace(/_/g, " "))}</dt><dd>${esc(v)}</dd>`).join("");
  $("sources").hidden = false;
}
function showLookup(kind, html) { const el = $("lookupMsg"); el.className = `lookup-msg ${kind}`; el.innerHTML = html; el.hidden = false; }

function fillFromPrefill(data) {
  const f = data.fields;
  clearProvenance();
  form.ticker.value = f.ticker; form.company_name.value = f.company_name;
  ["current_revenue", "current_ebitda_margin", "current_ev", "current_net_debt", "shares_outstanding", "horizon_years"]
    .forEach((k) => { if (f[k] !== null && f[k] !== undefined) form[k].value = f[k]; });
  if (f.current_price !== null && f.current_price !== undefined) form.current_price.value = f.current_price;
  TRI.forEach((k) => { form[`${k}_low`].value = f[k][0]; form[`${k}_mode`].value = f[k][1]; form[`${k}_high`].value = f[k][2]; });
  NORM.forEach((k) => { form[`${k}_mean`].value = f[k][0]; form[`${k}_std`].value = f[k][1]; });

  applyProvenance(data.provenance, data.reference || {});
  multipleIsPlaceholder = Boolean(data.needs_price); multipleEdited = false;

  const c = data.company;
  const bits = [`<strong>${esc(c.name)}</strong>, CIK ${c.cik}`, `FY${c.latest_fy} EBITDA ${money0(c.ebitda_fy_m)}M (${esc(c.ebitda_basis)})`];
  if (c.current_ev_ebitda) bits.push(`current ${mult(c.current_ev_ebitda)} EV/EBITDA`);
  let html = `<p>${bits.join(" &middot; ")}</p>`;
  if (c.revenue_ttm_m) html += `<p class="fine">Form uses FY${c.latest_fy} revenue so it pairs with the FY margin. ${esc(c.revenue_ttm_label)} revenue is ${money0(c.revenue_ttm_m)}M if you prefer an LTM basis; set the margin to match before switching.</p>`;
  if (data.needs_price) {
    html += `<p class="warn">No quote provider is configured, so enter the current price yourself. Everything else is filled; enterprise value and the exit multiple update when you do.</p>`;
    form.current_price.value = "";
    document.querySelector('label[data-field="current_price"]').classList.add("needs-input");
  }
  html += `<p class="warn">Bear, base and bull are yours to set. They are the DCF output this model exists to test.</p>`;
  html += `<p class="warn">The driver modes are held at today's values, which makes the untouched model a status-quo projection. Use the chips under each driver, or type your view into the mode.</p>`;
  (data.warnings || []).forEach((w) => { html += `<p class="warn">${esc(w)}</p>`; });
  html += `<p class="fine"><a href="${esc(c.filings_url)}" target="_blank" rel="noopener">Check against the filings on EDGAR</a></p>`;
  showLookup("ok", html);
  ["bear_price", "base_price", "bull_price"].forEach((k) => { form[k].value = ""; form[k].closest("label").classList.add("needs-input"); });
}

function recomputeEV() {
  const price = parseFloat(form.current_price.value), shares = parseFloat(form.shares_outstanding.value), nd = parseFloat(form.current_net_debt.value);
  if (![price, shares, nd].every(Number.isFinite) || price <= 0 || shares <= 0) return;
  const ev = price * shares + nd;
  form.current_ev.value = round(ev, 1);
  const label = document.querySelector('label[data-field="current_ev"]');
  if (label && !label.querySelector(".src-badge")) label.prepend(badge("derived", "derived: price x shares + net debt", "derived"));
  document.querySelector('label[data-field="current_price"]')?.classList.remove("needs-input");
  const ebitda = parseFloat(form.current_revenue.value) * parseFloat(form.current_ebitda_margin.value);
  if (!Number.isFinite(ebitda) || ebitda <= 0) return;
  const m = ev / ebitda;
  if (multipleIsPlaceholder && !multipleEdited) {
    form.ev_ebitda_multiple_low.value = round(m * 0.8, 4);
    form.ev_ebitda_multiple_mode.value = round(m, 4);
    form.ev_ebitda_multiple_high.value = round(m * 1.2, 4);
    renderRefs("ev_ebitda_multiple", { current: round(m, 2) },
      `Anchored on the current ${mult(m)}, plus or minus 20%. Your comps view belongs in the mode.`);
  }
}

async function autofill() {
  const ticker = form.ticker.value.trim().toUpperCase();
  if (!ticker) { showLookup("err", "Enter a ticker first."); return; }
  const btn = $("autofillBtn"); btn.disabled = true; btn.textContent = "Loading";
  showLookup("busy", `Reading ${esc(ticker)} filings from SEC EDGAR.`);
  try {
    const params = new URLSearchParams({ horizon: form.horizon_years.value || "2" });
    const typed = parseFloat(form.current_price.value);
    if (Number.isFinite(typed) && typed > 0) params.set("price", String(typed));
    const res = await fetch(`/api/company/${encodeURIComponent(ticker)}?${params}`);
    const data = await res.json().catch(() => ({ error: `Server returned ${res.status}` }));
    if (!res.ok) throw new Error(data.error || `Server returned ${res.status}`);
    fillFromPrefill(data);
  } catch (err) { showLookup("err", esc(err.message)); }
  finally { btn.disabled = false; btn.textContent = "Auto-fill from SEC"; }
}

/* SVG helpers */
const W = 560, H = 300, PAD = { t: 14, r: 18, b: 40, l: 52 }, IW = W - PAD.l - PAD.r, IH = H - PAD.t - PAD.b;
const frame = (inner, label) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(label)}" font-family="IBM Plex Sans, ui-sans-serif, system-ui, sans-serif">${inner}</svg>`;
function axes({ xTicks, yTicks, xLabel, yLabel }) {
  let s = "";
  yTicks.forEach(({ y, label }) => {
    s += `<line x1="${PAD.l}" y1="${y}" x2="${PAD.l + IW}" y2="${y}" stroke="${C.grid}"/>`;
    s += `<text x="${PAD.l - 8}" y="${y + 3.5}" fill="${C.muted}" font-size="10" text-anchor="end">${esc(label)}</text>`;
  });
  xTicks.forEach(({ x, label }) => { s += `<text x="${x}" y="${PAD.t + IH + 15}" fill="${C.muted}" font-size="10" text-anchor="middle">${esc(label)}</text>`; });
  s += `<line x1="${PAD.l}" y1="${PAD.t + IH}" x2="${PAD.l + IW}" y2="${PAD.t + IH}" stroke="${C.axis}"/>`;
  if (xLabel) s += `<text x="${PAD.l + IW / 2}" y="${H - 6}" fill="${C.muted}" font-size="10.5" text-anchor="middle">${esc(xLabel)}</text>`;
  if (yLabel) s += `<text transform="translate(12,${PAD.t + IH / 2}) rotate(-90)" fill="${C.muted}" font-size="10.5" text-anchor="middle">${esc(yLabel)}</text>`;
  return s;
}
function niceTicks(min, max, count = 5) {
  const span = max - min || 1, raw = span / count, mag = 10 ** Math.floor(Math.log10(raw)), n = raw / mag;
  const step = (n >= 5 ? 10 : n >= 2 ? 5 : n >= 1 ? 2 : 1) * mag, out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(v);
  return out;
}

function histChart(r) {
  const { edges, density } = r.histogram, m = r.meta, s = r.stats;
  const xMin = edges[0], xMax = edges[edges.length - 1], yMax = Math.max(...density) * 1.12 || 1;
  const sx = (v) => PAD.l + ((v - xMin) / (xMax - xMin)) * IW, sy = (v) => PAD.t + IH - (v / yMax) * IH;
  let g = axes({ xTicks: niceTicks(xMin, xMax, 6).map((v) => ({ x: sx(v), label: money0(v) })),
    yTicks: niceTicks(0, yMax, 4).map((v) => ({ y: sy(v), label: v.toFixed(3) })), xLabel: "Terminal share price ($)", yLabel: "Density" });
  density.forEach((d, i) => {
    if (d <= 0) return;
    const x = sx(edges[i]), w = Math.max(sx(edges[i + 1]) - x - 0.4, 0.6);
    g += `<rect x="${x.toFixed(2)}" y="${sy(d).toFixed(2)}" width="${w.toFixed(2)}" height="${(PAD.t + IH - sy(d)).toFixed(2)}" fill="${C.blue}" opacity="0.72"/>`;
  });
  const marks = [["Bear", m.bear_price, C.red, "4 3"], ["Base", m.base_price, C.amber, "4 3"], ["Bull", m.bull_price, C.green, "4 3"],
    ["Spot", m.current_price, C.text, ""], ["Median", s.median_price, C.purple, "1 3"]];
  marks.forEach(([, v, color, dash]) => {
    if (v < xMin || v > xMax) return;
    g += `<line x1="${sx(v).toFixed(2)}" y1="${PAD.t}" x2="${sx(v).toFixed(2)}" y2="${PAD.t + IH}" stroke="${color}" stroke-width="1.6" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`;
  });
  const lw = 112, lh = marks.length * 14 + 10;
  const rightSide = sx(s.median_price) < PAD.l + IW / 2;
  const lx = rightSide ? PAD.l + IW - lw - 6 : PAD.l + 6, ly = PAD.t + 5;
  g += `<rect x="${lx}" y="${ly}" width="${lw}" height="${lh}" rx="4" fill="${C.panel}" opacity="0.94" stroke="${C.grid}"/>`;
  marks.forEach(([label, v, color], i) => {
    const y = ly + 15 + i * 14;
    g += `<line x1="${lx + 8}" y1="${y - 3.5}" x2="${lx + 22}" y2="${y - 3.5}" stroke="${color}" stroke-width="2"/>`;
    g += `<text x="${lx + 27}" y="${y}" fill="${C.muted}" font-size="9.5">${esc(label)} ${money0(v)}</text>`;
  });
  return frame(g, `Distribution of simulated terminal share price with bear, base, bull, spot and median marked. Median ${money(s.median_price)}.`);
}
function barsChart(r) {
  const m = r.meta, s = r.stats;
  const data = [{ label: "Below bear", sub: `under ${money0(m.bear_price)}`, v: s.below_bear, c: C.red },
    { label: "Bear to base", sub: `${money0(m.bear_price)} to ${money0(m.base_price)}`, v: s.bear_to_base, c: C.amber },
    { label: "Base to bull", sub: `${money0(m.base_price)} to ${money0(m.bull_price)}`, v: s.base_to_bull, c: C.blue },
    { label: "Above bull", sub: `over ${money0(m.bull_price)}`, v: s.above_bull, c: C.green }];
  const yMax = Math.max(...data.map((d) => d.v)) * 1.25 || 1, sy = (v) => PAD.t + IH - (v / yMax) * IH;
  let g = axes({ xTicks: [], yTicks: niceTicks(0, yMax, 4).map((v) => ({ y: sy(v), label: pct(v, 0) })), yLabel: "Probability" });
  const slot = IW / data.length, bw = Math.min(slot * 0.56, 74);
  data.forEach((d, i) => {
    const cx = PAD.l + slot * (i + 0.5), y = sy(d.v);
    g += `<rect x="${(cx - bw / 2).toFixed(2)}" y="${y.toFixed(2)}" width="${bw.toFixed(2)}" height="${(PAD.t + IH - y).toFixed(2)}" rx="3" fill="${d.c}" opacity="0.88"/>`;
    g += `<text x="${cx.toFixed(2)}" y="${(y - 7).toFixed(2)}" fill="${C.text}" font-size="12.5" font-weight="600" text-anchor="middle">${pct(d.v)}</text>`;
    g += `<text x="${cx.toFixed(2)}" y="${PAD.t + IH + 15}" fill="${C.muted}" font-size="10.5" text-anchor="middle">${esc(d.label)}</text>`;
    g += `<text x="${cx.toFixed(2)}" y="${PAD.t + IH + 27}" fill="${C.faint}" font-size="9.5" text-anchor="middle">${esc(d.sub)}</text>`;
  });
  return frame(g, `Probability of finishing below bear ${pct(s.below_bear)}, between bear and base ${pct(s.bear_to_base)}, between base and bull ${pct(s.base_to_bull)}, above bull ${pct(s.above_bull)}.`);
}
function cdfChart(r) {
  const { x, y } = r.cdf, m = r.meta, s = r.stats, xMin = 0, xMax = r.histogram.edges[r.histogram.edges.length - 1];
  const sx = (v) => PAD.l + ((v - xMin) / (xMax - xMin)) * IW, sy = (v) => PAD.t + IH - v * IH;
  let g = axes({ xTicks: niceTicks(xMin, xMax, 6).map((v) => ({ x: sx(v), label: money0(v) })),
    yTicks: [0, 0.25, 0.5, 0.75, 1].map((v) => ({ y: sy(v), label: pct(v, 0) })), xLabel: "Terminal share price ($)", yLabel: "Cumulative probability" });
  const pts = x.map((v, i) => (v <= xMax ? `${sx(v).toFixed(2)},${sy(y[i]).toFixed(2)}` : null)).filter(Boolean).join(" ");
  g += `<polyline points="${pts}" fill="none" stroke="${C.blue}" stroke-width="1.8"/>`;
  [["Bear", m.bear_price, C.red, s.p_at_least_bear], ["Base", m.base_price, C.amber, s.p_at_least_base], ["Bull", m.bull_price, C.green, s.p_at_least_bull]]
    .forEach(([label, v, color, p]) => {
      if (v < xMin || v > xMax) return;
      const px = sx(v), py = sy(1 - p);
      g += `<line x1="${px.toFixed(2)}" y1="${PAD.t}" x2="${px.toFixed(2)}" y2="${PAD.t + IH}" stroke="${color}" stroke-width="1.4" stroke-dasharray="4 3"/>`;
      g += `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="3" fill="${color}"/>`;
      const anchor = px > PAD.l + IW * 0.62 ? "end" : "start", dx = anchor === "end" ? -8 : 8;
      const ly = Math.min(Math.max(py - 8, PAD.t + 10), PAD.t + IH - 4);
      g += `<text x="${(px + dx).toFixed(2)}" y="${ly.toFixed(2)}" fill="${color}" font-size="10" font-weight="600" text-anchor="${anchor}">P(at least ${label}) ${pct(p)}</text>`;
    });
  if (m.current_price <= xMax) g += `<line x1="${sx(m.current_price).toFixed(2)}" y1="${PAD.t}" x2="${sx(m.current_price).toFixed(2)}" y2="${PAD.t + IH}" stroke="${C.text}" opacity="0.4"/>`;
  return frame(g, `Cumulative distribution of terminal price. Probability of at least bear ${pct(s.p_at_least_bear)}, base ${pct(s.p_at_least_base)}, bull ${pct(s.p_at_least_bull)}.`);
}
function tornadoChart(r) {
  const rows = r.tornado.map((d) => ({ ...d, lo: Math.min(d.low, d.high), hi: Math.max(d.low, d.high) })), median = r.stats.median_price;
  const L = { t: 14, r: 46, b: 40, l: 132 }, iw = W - L.l - L.r, ih = H - L.t - L.b;
  const xMin = Math.min(...rows.map((d) => d.lo), median), xMax = Math.max(...rows.map((d) => d.hi), median), padX = (xMax - xMin) * 0.08 || 1;
  const lo = xMin - padX, hi = xMax + padX, sx = (v) => L.l + ((v - lo) / (hi - lo)) * iw;
  let g = "";
  niceTicks(lo, hi, 5).forEach((v) => {
    g += `<line x1="${sx(v).toFixed(2)}" y1="${L.t}" x2="${sx(v).toFixed(2)}" y2="${L.t + ih}" stroke="${C.grid}"/>`;
    g += `<text x="${sx(v).toFixed(2)}" y="${L.t + ih + 15}" fill="${C.muted}" font-size="10" text-anchor="middle">${money0(v)}</text>`;
  });
  const slot = ih / rows.length, bh = Math.min(slot * 0.5, 22);
  rows.forEach((d, i) => {
    const cy = L.t + slot * (i + 0.5);
    g += `<rect x="${sx(d.lo).toFixed(2)}" y="${(cy - bh / 2).toFixed(2)}" width="${Math.max(sx(d.hi) - sx(d.lo), 1).toFixed(2)}" height="${bh}" rx="3" fill="${C.blue}" opacity="0.78"/>`;
    g += `<text x="${L.l - 9}" y="${(cy + 3.5).toFixed(2)}" fill="${C.text}" font-size="10.5" text-anchor="end">${esc(d.name)}</text>`;
    g += `<text x="${(sx(d.lo) - 5).toFixed(2)}" y="${(cy + 3.5).toFixed(2)}" fill="${C.red}" font-size="9.5" text-anchor="end">${money0(d.lo)}</text>`;
    g += `<text x="${(sx(d.hi) + 5).toFixed(2)}" y="${(cy + 3.5).toFixed(2)}" fill="${C.green}" font-size="9.5">${money0(d.hi)}</text>`;
  });
  g += `<line x1="${sx(median).toFixed(2)}" y1="${L.t}" x2="${sx(median).toFixed(2)}" y2="${L.t + ih}" stroke="${C.purple}" stroke-width="1.5" stroke-dasharray="2 3"/>`;
  g += `<line x1="${L.l}" y1="${L.t + ih}" x2="${L.l + iw}" y2="${L.t + ih}" stroke="${C.axis}"/>`;
  g += `<text x="${L.l + iw / 2}" y="${H - 6}" fill="${C.muted}" font-size="10.5" text-anchor="middle">Median terminal price ($). Bar spans the driver's P10 to P90.</text>`;
  const top = rows[rows.length - 1];
  return frame(g, `Driver sensitivity. The widest swing is ${top.name}, moving the median from ${money0(top.lo)} to ${money0(top.hi)}.`);
}

/* Tables and panels */
const seCell = (p, se) => `<td class="num">${pct(p)}<span class="se">&plusmn;${(se * 100).toFixed(2)}</span></td>`;
function summaryTables(r) {
  const s = r.stats;
  const dist = [["5th percentile", money(s.p5)], ["25th percentile", money(s.p25)], ["Median", money(s.median_price)], ["Mean", money(s.mean_price)],
    ["75th percentile", money(s.p75)], ["95th percentile", money(s.p95)], ["Standard deviation", money(s.std_price)]];
  const tbl = (cap, rows) => `<table><caption>${esc(cap)}</caption><tbody>${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${esc(v)}</td></tr>`).join("")}</tbody></table>`;
  const thresh = `<table><caption>Cumulative thresholds, with sampling error</caption><tbody>
    <tr><td>P(at least bear)</td>${seCell(s.p_at_least_bear, s.se.p_at_least_bear)}</tr>
    <tr><td>P(at least base)</td>${seCell(s.p_at_least_base, s.se.p_at_least_base)}</tr>
    <tr><td>P(at least bull)</td>${seCell(s.p_at_least_bull, s.se.p_at_least_bull)}</tr>
    <tr><td>Median return, total</td><td class="num">${spct(s.median_return)}</td></tr>
    <tr><td>Median return, annualised</td><td class="num">${spct(s.median_return_annualized)}</td></tr>
    <tr><td>Mean return, annualised</td><td class="num">${spct(s.mean_return_annualized)}</td></tr>
  </tbody></table>`;
  return tbl("Terminal price distribution", dist) + thresh;
}
function sensitivityTables(r) {
  return Object.entries(r.sensitivity).map(([name, rows]) => `<div class="sens-block"><h4>${esc(name)}</h4><table>
    <thead><tr><th>Driver quintile</th><th>Paths</th><th>Median price</th><th>P(at least base)</th></tr></thead>
    <tbody>${rows.map((b) => `<tr><td>${esc(b.bucket)}</td><td class="num">${b.count.toLocaleString()}</td><td class="num">${money(b.median_price)}</td><td class="num">${pct(b.p_above_base)}</td></tr>`).join("")}</tbody></table></div>`).join("");
}
function reconciliation(r, payload) {
  const rc = r.reconciliation, s = r.stats, m = r.meta;
  const gap = rc.median_price / rc.mode_path_price - 1;
  $("reconSub").textContent = `Mode-path price ${money(rc.mode_path_price)}. Simulated median ${money(rc.median_price)}, ${spct(gap)} against it: the ranges carry ${gap < 0 ? "more downside than upside" : "more upside than downside"}.`;
  const hiMult = payload.ev_ebitda_multiple[2], hiMargin = payload.ebitda_margin[2], loMult = payload.ev_ebitda_multiple[0], loMargin = payload.ebitda_margin[0];
  const row = (key, c) => {
    const im = c.implied_multiple_at_mode_margin, ig = c.implied_margin_at_mode_multiple;
    const multOut = im !== null && (im > hiMult || im < loMult), margOut = ig !== null && (ig > hiMargin || ig < loMargin);
    return `<tr class="${key}"><td>${key[0].toUpperCase() + key.slice(1)}</td><td class="num">${money(c.target)}</td><td class="num">${spct(c.vs_spot, 0)}</td>
      ${seCell(c.p_at_least, s.se[`p_at_least_${key}`])}
      <td class="num${multOut ? " flag" : ""}" title="${multOut ? "Outside your multiple range" : ""}">${im === null ? "n/a" : mult(im)}</td>
      <td class="num${margOut ? " flag" : ""}" title="${margOut ? "Outside your margin range" : ""}">${ig === null ? "n/a" : pct(ig, 2)}</td></tr>`;
  };
  $("reconTable").innerHTML = `<table>
    <thead><tr><th>Case</th><th>Target</th><th>vs spot</th><th>P(at least)</th><th>Multiple needed at mode margin ${pct(rc.mode_drivers.ebitda_margin, 2)}</th><th>Margin needed at mode multiple ${mult(rc.mode_drivers.ev_ebitda_multiple)}</th></tr></thead>
    <tbody>${row("bear", rc.cases.bear)}${row("base", rc.cases.base)}${row("bull", rc.cases.bull)}</tbody></table>`;
  const base = rc.cases.base, bgap = rc.base_gap_pct;
  let note = `Your base target of ${money(m.base_price)} sits ${spct(-bgap, 1).replace("+", "")} ${bgap < 0 ? "above" : "below"} the simulated median. `;
  note += `To reach it at your mode margin it needs a ${mult(base.implied_multiple_at_mode_margin)} exit multiple against a mode of ${mult(rc.mode_drivers.ev_ebitda_multiple)}; `;
  note += `at your mode multiple it needs a ${pct(base.implied_margin_at_mode_multiple, 2)} margin against a mode of ${pct(rc.mode_drivers.ebitda_margin, 2)}. `;
  note += `Red cells fall outside the range you gave the model, which means the target assumes something the distribution does not.`;
  $("reconNote").textContent = note;
}
function cards(r) {
  const s = r.stats, m = r.meta, rc = r.reconciliation;
  const items = [
    { k: "Median terminal", v: money(s.median_price), s: `${spct(s.median_return)} vs spot ${money(m.current_price)}`, cls: s.median_return >= 0 ? "pos" : "neg" },
    { k: "Median return, annualised", v: spct(s.median_return_annualized), s: `${spct(s.median_return)} over ${m.horizon_years} years`, cls: s.median_return_annualized >= 0 ? "pos" : "neg" },
    { k: "P(at least base)", v: `${pct(s.p_at_least_base)}<small>&plusmn;${(s.se.p_at_least_base * 100).toFixed(2)}pp</small>`, s: `base ${money0(m.base_price)}` },
    { k: "P(at least bear)", v: `${pct(s.p_at_least_bear)}<small>&plusmn;${(s.se.p_at_least_bear * 100).toFixed(2)}pp</small>`, s: `bear ${money0(m.bear_price)}` },
    { k: "P5 to P95", v: `${money0(s.p5)} to ${money0(s.p95)}`, s: "central 90% of paths" },
    { k: "Mode-path price", v: money(rc.mode_path_price), s: `median is ${spct(rc.median_price / rc.mode_path_price - 1)} against it` },
  ];
  $("cards").innerHTML = items.map((c) => `<div class="card ${c.cls || ""}"><div class="k">${esc(c.k)}</div><div class="v">${c.v}</div><div class="s">${esc(c.s)}</div></div>`).join("");
}
function compareStrip(r) {
  const el = $("compareStrip");
  if (!pinned) { el.hidden = true; return; }
  const a = pinned.stats, b = r.stats;
  const d = (x, y, f) => { const diff = y - x; const sign = diff > 0 ? "+" : diff < 0 ? "-" : ""; return `<span class="d ${diff === 0 ? "" : diff > 0 ? "up" : "down"}">${sign}${f(Math.abs(diff))}</span>`; };
  el.innerHTML = `<span class="lbl">Pinned: ${esc(pinned.label)}</span>
    <div class="cells">
      <span>Median <b>${money(a.median_price)}</b> vs <b>${money(b.median_price)}</b>${d(a.median_price, b.median_price, (v) => money(v))}</span>
      <span>P(base) <b>${pct(a.p_at_least_base)}</b> vs <b>${pct(b.p_at_least_base)}</b>${d(a.p_at_least_base, b.p_at_least_base, (v) => (v * 100).toFixed(1) + "pp")}</span>
      <span>P(bear) <b>${pct(a.p_at_least_bear)}</b> vs <b>${pct(b.p_at_least_bear)}</b>${d(a.p_at_least_bear, b.p_at_least_bear, (v) => (v * 100).toFixed(1) + "pp")}</span>
      <span>P5-P95 <b>${money0(a.p5)}-${money0(a.p95)}</b> vs <b>${money0(b.p5)}-${money0(b.p95)}</b></span>
    </div>`;
  el.hidden = false;
}
function render(r, payload) {
  const s = r.stats, m = r.meta;
  $("emptyState").hidden = true; $("output").hidden = false;
  $("resultTitle").textContent = `${m.company_name} (${m.ticker})`;
  $("resultSub").textContent = `${m.n_simulations.toLocaleString()} paths, ${m.horizon_years}-year horizon, seed ${m.random_seed}, spot ${money(m.current_price)}, ${m.elapsed_ms} ms`;
  compareStrip(r); cards(r); reconciliation(r, payload);
  $("cap1").textContent = `Terminal price distribution, ${m.horizon_years}-year, ${m.n_simulations.toLocaleString()} paths`;
  $("chartHist").innerHTML = histChart(r); $("chartBars").innerHTML = barsChart(r);
  $("chartCdf").innerHTML = cdfChart(r); $("chartTornado").innerHTML = tornadoChart(r);
  $("summaryTables").innerHTML = summaryTables(r); $("sensitivityTables").innerHTML = sensitivityTables(r);
  ["shareBtn", "pinBtn", "exportPng", "exportJson"].forEach((id) => { $(id).disabled = false; });
}

/* Export */
function svgToImage(svgEl) {
  return new Promise((resolve, reject) => {
    const xml = new XMLSerializer().serializeToString(svgEl);
    const url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml;charset=utf-8" }));
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Could not rasterise chart")); };
    img.src = url;
  });
}
async function exportPng() {
  if (!lastResult) return;
  const m = lastResult.meta, s = lastResult.stats, scale = 2, gap = 24, titleH = 76;
  const canvas = document.createElement("canvas");
  canvas.width = (W * 2 + gap * 3) * scale; canvas.height = (titleH + H * 2 + gap * 3) * scale;
  const ctx = canvas.getContext("2d"); ctx.scale(scale, scale);
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = C.text; ctx.font = "600 20px IBM Plex Sans, system-ui, sans-serif";
  ctx.fillText(`${m.company_name} (${m.ticker}). Monte Carlo scenario analysis`, gap, 34);
  ctx.fillStyle = C.muted; ctx.font = "12px IBM Plex Mono, ui-monospace, monospace";
  ctx.fillText(`${m.n_simulations.toLocaleString()} paths, ${m.horizon_years}-year horizon, spot ${money(m.current_price)}, median ${money(s.median_price)}, P(at least base) ${pct(s.p_at_least_base)}`, gap, 56);
  const ids = ["chartHist", "chartBars", "chartCdf", "chartTornado"];
  const imgs = await Promise.all(ids.map((id) => svgToImage($(id).querySelector("svg"))));
  imgs.forEach((img, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = gap + col * (W + gap), y = titleH + gap + row * (H + gap);
    ctx.fillStyle = "#ffffff"; ctx.fillRect(x, y, W, H); ctx.drawImage(img, x, y, W, H);
  });
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png"); a.download = `${m.ticker}_monte_carlo.png`; a.click();
}

/* Wiring */
function showError(msg) { const el = $("formError"); el.textContent = msg; el.hidden = false; }
async function run() {
  const btn = $("runBtn"); $("formError").hidden = true;
  let payload;
  try { payload = readForm(); } catch (err) { showError(err.message); return; }
  btn.disabled = true; btn.classList.add("busy"); btn.textContent = "Running";
  try {
    const res = await fetch("/api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await res.json().catch(() => ({ error: `Server returned ${res.status}` }));
    if (!res.ok) throw new Error(data.error || `Server returned ${res.status}`);
    lastResult = data; lastPayload = payload;
    writeUrl(payload); render(data, payload);
  } catch (err) { showError(err.message); }
  finally { btn.disabled = false; btn.classList.remove("busy"); btn.textContent = "Run simulation"; }
}
form.addEventListener("submit", (e) => { e.preventDefault(); run(); });
$("autofillBtn").addEventListener("click", autofill);
form.ticker.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); autofill(); } });
form.current_price.addEventListener("input", recomputeEV);
["low", "mode", "high"].forEach((p) => form[`ev_ebitda_multiple_${p}`].addEventListener("input", () => { multipleEdited = true; }));
[...TRI, ...NORM].forEach((d) => form[`${d}_${TRI.includes(d) ? "mode" : "mean"}`].addEventListener("input", () => refreshChipState(d)));
$("corrReset").addEventListener("click", () => { buildCorrGrid(DEFAULT_CORR); $("corrStatus").textContent = "Reset."; $("corrStatus").className = "hint"; });
$("loadPreset").addEventListener("click", () => { fillForm(PRESET); clearProvenance(); $("lookupMsg").hidden = true; $("formError").hidden = true; });
$("shareBtn").addEventListener("click", async () => {
  if (lastPayload) writeUrl(lastPayload);
  try { await navigator.clipboard.writeText(location.href); $("shareBtn").textContent = "Link copied"; }
  catch (_) { $("shareBtn").textContent = "Copy from address bar"; }
  setTimeout(() => { $("shareBtn").textContent = "Copy link"; }, 1800);
});
$("pinBtn").addEventListener("click", () => {
  if (pinned) { pinned = null; $("pinBtn").textContent = "Pin result"; $("pinBtn").classList.remove("pinned"); }
  else if (lastResult) {
    pinned = { label: `${lastResult.meta.ticker} ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`, stats: lastResult.stats };
    $("pinBtn").textContent = "Unpin"; $("pinBtn").classList.add("pinned");
  }
  if (lastResult) compareStrip(lastResult);
});
$("exportPng").addEventListener("click", () => exportPng().catch((e) => showError(e.message)));
$("exportJson").addEventListener("click", () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify({ inputs: lastPayload, result: lastResult }, null, 2)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${lastResult.meta.ticker}_monte_carlo.json`; a.click(); URL.revokeObjectURL(a.href);
});

/* Boot */
buildCorrGrid(DEFAULT_CORR);
const fromUrl = readUrl();
if (fromUrl) { fillForm(fromUrl); run(); }
else {
  form.n_simulations.value = PRESET.n_simulations; form.horizon_years.value = PRESET.horizon_years; form.random_seed.value = PRESET.random_seed;
  form.ticker.focus();
}
