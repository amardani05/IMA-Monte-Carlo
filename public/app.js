/* IMA Monte Carlo — front end.
   Serializes the assumption form, calls POST /api/simulate, renders SVG panels. */

const PRESET = {
  ticker: "MYRG",
  company_name: "MYR Group",
  current_price: 271.0,
  bear_price: 222.32,
  base_price: 323.01,
  bull_price: 362.32,
  current_revenue: 3510.0,
  current_ebitda_margin: 0.061,
  current_ev: 4310.0,
  current_net_debt: 110.0,
  shares_outstanding: 15.5,
  rev_cagr: [0.057, 0.069, 0.075],
  ebitda_margin: [0.071, 0.075, 0.077],
  ev_ebitda_multiple: [12.4, 17.0, 18.5],
  net_debt_change_pct: [0.0, 0.015],
  share_dilution_pct: [0.01, 0.01],
  n_simulations: 100000,
  horizon_years: 2,
  random_seed: 42,
};

const TRI = ["rev_cagr", "ebitda_margin", "ev_ebitda_multiple"];
const NORM = ["net_debt_change_pct", "share_dilution_pct"];
const SCALARS = [
  "current_price", "bear_price", "base_price", "bull_price",
  "current_revenue", "current_ebitda_margin", "current_ev",
  "current_net_debt", "shares_outstanding",
  "n_simulations", "horizon_years", "random_seed",
];

const C = {
  blue: "#1f6feb", red: "#f85149", amber: "#d29922",
  green: "#3fb950", purple: "#bc8cff", text: "#e6edf3",
  muted: "#9aa7b4", grid: "#262d38", axis: "#3d444d",
};

const $ = (id) => document.getElementById(id);
const form = $("pitchForm");
let lastResult = null;

/* ── formatting ───────────────────────────────────────────────────── */
const money = (v, d = 2) =>
  "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const money0 = (v) => money(v, 0);
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ── form <-> payload ─────────────────────────────────────────────── */
function fillForm(p) {
  form.ticker.value = p.ticker;
  form.company_name.value = p.company_name;
  SCALARS.forEach((k) => { if (form[k]) form[k].value = p[k]; });
  TRI.forEach((k) => {
    form[`${k}_low`].value = p[k][0];
    form[`${k}_mode`].value = p[k][1];
    form[`${k}_high`].value = p[k][2];
  });
  NORM.forEach((k) => {
    form[`${k}_mean`].value = p[k][0];
    form[`${k}_std`].value = p[k][1];
  });
}

function readForm() {
  const num = (name) => {
    const el = form[name];
    const v = el.value.trim();
    if (v === "" || !Number.isFinite(Number(v))) {
      el.classList.add("invalid");
      throw new Error(`"${el.parentElement.childNodes[0].textContent.trim()}" needs a number`);
    }
    el.classList.remove("invalid");
    return Number(v);
  };

  form.querySelectorAll("input").forEach((el) => el.classList.remove("invalid"));

  const payload = {
    ticker: form.ticker.value.trim() || "TICKER",
    company_name: form.company_name.value.trim() || form.ticker.value.trim() || "Untitled",
  };
  SCALARS.forEach((k) => { payload[k] = num(k); });
  TRI.forEach((k) => { payload[k] = [num(`${k}_low`), num(`${k}_mode`), num(`${k}_high`)]; });
  NORM.forEach((k) => { payload[k] = [num(`${k}_mean`), num(`${k}_std`)]; });
  return payload;
}

/* ── SVG helpers ──────────────────────────────────────────────────── */
const W = 560, H = 300;
const PAD = { t: 14, r: 18, b: 40, l: 52 };
const IW = W - PAD.l - PAD.r;
const IH = H - PAD.t - PAD.b;

const frame = (inner) =>
  `<svg viewBox="0 0 ${W} ${H}" role="img" font-family="ui-sans-serif, system-ui, sans-serif">${inner}</svg>`;

function axes({ xTicks, yTicks, xLabel, yLabel }) {
  let s = "";
  yTicks.forEach(({ y, label }) => {
    s += `<line x1="${PAD.l}" y1="${y}" x2="${PAD.l + IW}" y2="${y}" stroke="${C.grid}" stroke-width="1"/>`;
    s += `<text x="${PAD.l - 8}" y="${y + 3.5}" fill="${C.muted}" font-size="10" text-anchor="end">${esc(label)}</text>`;
  });
  xTicks.forEach(({ x, label }) => {
    s += `<text x="${x}" y="${PAD.t + IH + 15}" fill="${C.muted}" font-size="10" text-anchor="middle">${esc(label)}</text>`;
  });
  s += `<line x1="${PAD.l}" y1="${PAD.t + IH}" x2="${PAD.l + IW}" y2="${PAD.t + IH}" stroke="${C.axis}" stroke-width="1"/>`;
  if (xLabel) s += `<text x="${PAD.l + IW / 2}" y="${H - 6}" fill="${C.muted}" font-size="10.5" text-anchor="middle">${esc(xLabel)}</text>`;
  if (yLabel) s += `<text transform="translate(12,${PAD.t + IH / 2}) rotate(-90)" fill="${C.muted}" font-size="10.5" text-anchor="middle">${esc(yLabel)}</text>`;
  return s;
}

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(v);
  return out;
}

/* ── 1. histogram ─────────────────────────────────────────────────── */
function histChart(r) {
  const { edges, density } = r.histogram;
  const m = r.meta, s = r.stats;
  const xMin = edges[0], xMax = edges[edges.length - 1];
  const yMax = Math.max(...density) * 1.12 || 1;
  const sx = (v) => PAD.l + ((v - xMin) / (xMax - xMin)) * IW;
  const sy = (v) => PAD.t + IH - (v / yMax) * IH;

  let g = axes({
    xTicks: niceTicks(xMin, xMax, 6).map((v) => ({ x: sx(v), label: money0(v) })),
    yTicks: niceTicks(0, yMax, 4).map((v) => ({ y: sy(v), label: v.toFixed(3) })),
    xLabel: "Terminal share price ($)",
    yLabel: "Density",
  });

  density.forEach((d, i) => {
    if (d <= 0) return;
    const x = sx(edges[i]), w = Math.max(sx(edges[i + 1]) - x - 0.4, 0.6);
    g += `<rect x="${x.toFixed(2)}" y="${sy(d).toFixed(2)}" width="${w.toFixed(2)}" height="${(PAD.t + IH - sy(d)).toFixed(2)}" fill="${C.blue}" opacity="0.72"/>`;
  });

  const marks = [
    ["Bear", m.bear_price, C.red, "4 3"],
    ["Base", m.base_price, C.amber, "4 3"],
    ["Bull", m.bull_price, C.green, "4 3"],
    ["Current", m.current_price, C.text, ""],
    ["Median", s.median_price, C.purple, "1 3"],
  ];
  marks.forEach(([, v, color, dash]) => {
    if (v < xMin || v > xMax) return;
    g += `<line x1="${sx(v).toFixed(2)}" y1="${PAD.t}" x2="${sx(v).toFixed(2)}" y2="${PAD.t + IH}" stroke="${color}" stroke-width="1.6" ${dash ? `stroke-dasharray="${dash}"` : ""} opacity="0.95"/>`;
  });

  // legend, top-right inside the plot
  const lw = 108, lh = marks.length * 14 + 10;
  const lx = PAD.l + IW - lw - 6, ly = PAD.t + 5;
  g += `<rect x="${lx}" y="${ly}" width="${lw}" height="${lh}" rx="5" fill="#0d1117" opacity="0.85" stroke="${C.grid}"/>`;
  marks.forEach(([label, v, color], i) => {
    const y = ly + 15 + i * 14;
    g += `<line x1="${lx + 8}" y1="${y - 3.5}" x2="${lx + 22}" y2="${y - 3.5}" stroke="${color}" stroke-width="2"/>`;
    g += `<text x="${lx + 27}" y="${y}" fill="${C.muted}" font-size="9.5">${esc(label)} ${money0(v)}</text>`;
  });
  return frame(g);
}

/* ── 2. scenario bars ─────────────────────────────────────────────── */
function barsChart(r) {
  const m = r.meta, s = r.stats;
  const data = [
    { label: `Below bear`, sub: `<${money0(m.bear_price)}`, v: s.below_bear, c: C.red },
    { label: `Bear→Base`, sub: `${money0(m.bear_price)}–${money0(m.base_price)}`, v: s.bear_to_base, c: C.amber },
    { label: `Base→Bull`, sub: `${money0(m.base_price)}–${money0(m.bull_price)}`, v: s.base_to_bull, c: C.blue },
    { label: `Above bull`, sub: `>${money0(m.bull_price)}`, v: s.above_bull, c: C.green },
  ];
  const yMax = Math.max(...data.map((d) => d.v)) * 1.25 || 1;
  const sy = (v) => PAD.t + IH - (v / yMax) * IH;

  let g = axes({
    xTicks: [],
    yTicks: niceTicks(0, yMax, 4).map((v) => ({ y: sy(v), label: pct(v, 0) })),
    yLabel: "Probability",
  });

  const slot = IW / data.length, bw = Math.min(slot * 0.56, 74);
  data.forEach((d, i) => {
    const cx = PAD.l + slot * (i + 0.5);
    const y = sy(d.v);
    g += `<rect x="${(cx - bw / 2).toFixed(2)}" y="${y.toFixed(2)}" width="${bw.toFixed(2)}" height="${(PAD.t + IH - y).toFixed(2)}" rx="3" fill="${d.c}" opacity="0.88"/>`;
    g += `<text x="${cx.toFixed(2)}" y="${(y - 7).toFixed(2)}" fill="${C.text}" font-size="12.5" font-weight="600" text-anchor="middle">${pct(d.v)}</text>`;
    g += `<text x="${cx.toFixed(2)}" y="${PAD.t + IH + 15}" fill="${C.muted}" font-size="10.5" text-anchor="middle">${esc(d.label)}</text>`;
    g += `<text x="${cx.toFixed(2)}" y="${PAD.t + IH + 27}" fill="#6e7b8a" font-size="9.5" text-anchor="middle">${esc(d.sub)}</text>`;
  });
  return frame(g);
}

/* ── 3. CDF ───────────────────────────────────────────────────────── */
function cdfChart(r) {
  const { x, y } = r.cdf, m = r.meta, s = r.stats;
  const xMin = 0, xMax = r.histogram.edges[r.histogram.edges.length - 1];
  const sx = (v) => PAD.l + ((v - xMin) / (xMax - xMin)) * IW;
  const sy = (v) => PAD.t + IH - v * IH;

  let g = axes({
    xTicks: niceTicks(xMin, xMax, 6).map((v) => ({ x: sx(v), label: money0(v) })),
    yTicks: [0, 0.25, 0.5, 0.75, 1].map((v) => ({ y: sy(v), label: pct(v, 0) })),
    xLabel: "Terminal share price ($)",
    yLabel: "Cumulative probability",
  });

  const pts = x
    .map((v, i) => (v <= xMax ? `${sx(v).toFixed(2)},${sy(y[i]).toFixed(2)}` : null))
    .filter(Boolean)
    .join(" ");
  g += `<polyline points="${pts}" fill="none" stroke="${C.blue}" stroke-width="1.8"/>`;

  [["Bear", m.bear_price, C.red, s.p_at_least_bear],
   ["Base", m.base_price, C.amber, s.p_at_least_base],
   ["Bull", m.bull_price, C.green, s.p_at_least_bull]].forEach(([label, v, color, p], i) => {
    if (v < xMin || v > xMax) return;
    const px = sx(v), py = sy(1 - p);
    g += `<line x1="${px.toFixed(2)}" y1="${PAD.t}" x2="${px.toFixed(2)}" y2="${PAD.t + IH}" stroke="${color}" stroke-width="1.4" stroke-dasharray="4 3"/>`;
    g += `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="3" fill="${color}"/>`;
    const anchor = px > PAD.l + IW * 0.62 ? "end" : "start";
    const dx = anchor === "end" ? -8 : 8;
    const ly = Math.min(Math.max(py - 8, PAD.t + 10), PAD.t + IH - 4);
    g += `<text x="${(px + dx).toFixed(2)}" y="${ly.toFixed(2)}" fill="${color}" font-size="10" font-weight="600" text-anchor="${anchor}">P(≥${label}) = ${pct(p)}</text>`;
  });

  if (m.current_price <= xMax) {
    g += `<line x1="${sx(m.current_price).toFixed(2)}" y1="${PAD.t}" x2="${sx(m.current_price).toFixed(2)}" y2="${PAD.t + IH}" stroke="${C.text}" stroke-width="1" opacity="0.45"/>`;
  }
  return frame(g);
}

/* ── 4. tornado ───────────────────────────────────────────────────── */
function tornadoChart(r) {
  const rows = r.tornado.map((d) => ({ ...d, lo: Math.min(d.low, d.high), hi: Math.max(d.low, d.high) }));
  const median = r.stats.median_price;
  const L = { t: 14, r: 46, b: 40, l: 132 };
  const iw = W - L.l - L.r, ih = H - L.t - L.b;
  const xMin = Math.min(...rows.map((d) => d.lo), median);
  const xMax = Math.max(...rows.map((d) => d.hi), median);
  const padX = (xMax - xMin) * 0.08 || 1;
  const lo = xMin - padX, hi = xMax + padX;
  const sx = (v) => L.l + ((v - lo) / (hi - lo)) * iw;

  let g = "";
  niceTicks(lo, hi, 5).forEach((v) => {
    g += `<line x1="${sx(v).toFixed(2)}" y1="${L.t}" x2="${sx(v).toFixed(2)}" y2="${L.t + ih}" stroke="${C.grid}" stroke-width="1"/>`;
    g += `<text x="${sx(v).toFixed(2)}" y="${L.t + ih + 15}" fill="${C.muted}" font-size="10" text-anchor="middle">${money0(v)}</text>`;
  });

  const slot = ih / rows.length, bh = Math.min(slot * 0.5, 22);
  rows.forEach((d, i) => {
    const cy = L.t + slot * (i + 0.5);
    g += `<rect x="${sx(d.lo).toFixed(2)}" y="${(cy - bh / 2).toFixed(2)}" width="${Math.max(sx(d.hi) - sx(d.lo), 1).toFixed(2)}" height="${bh}" rx="3" fill="${C.blue}" opacity="0.75"/>`;
    g += `<text x="${L.l - 9}" y="${(cy + 3.5).toFixed(2)}" fill="${C.muted}" font-size="10.5" text-anchor="end">${esc(d.name)}</text>`;
    g += `<text x="${(sx(d.lo) - 5).toFixed(2)}" y="${(cy + 3.5).toFixed(2)}" fill="${C.red}" font-size="9.5" text-anchor="end">${money0(d.lo)}</text>`;
    g += `<text x="${(sx(d.hi) + 5).toFixed(2)}" y="${(cy + 3.5).toFixed(2)}" fill="${C.green}" font-size="9.5">${money0(d.hi)}</text>`;
  });

  g += `<line x1="${sx(median).toFixed(2)}" y1="${L.t}" x2="${sx(median).toFixed(2)}" y2="${L.t + ih}" stroke="${C.purple}" stroke-width="1.5" stroke-dasharray="2 3"/>`;
  g += `<line x1="${L.l}" y1="${L.t + ih}" x2="${L.l + iw}" y2="${L.t + ih}" stroke="${C.axis}" stroke-width="1"/>`;
  g += `<text x="${L.l + iw / 2}" y="${H - 6}" fill="${C.muted}" font-size="10.5" text-anchor="middle">Median terminal price ($) — bar spans P10 → P90 of the driver</text>`;
  return frame(g);
}

/* ── tables ───────────────────────────────────────────────────────── */
function summaryTables(r) {
  const s = r.stats;
  const dist = [
    ["5th percentile", money(s.p5)], ["25th percentile", money(s.p25)],
    ["Median", money(s.median_price)], ["Mean", money(s.mean_price)],
    ["75th percentile", money(s.p75)], ["95th percentile", money(s.p95)],
    ["Std deviation", money(s.std_price)],
  ];
  const thresh = [
    ["P(≥ Bear target)", pct(s.p_at_least_bear)],
    ["P(≥ Base target)", pct(s.p_at_least_base)],
    ["P(≥ Bull target)", pct(s.p_at_least_bull)],
    ["P(≥ Current price)", "—"],
  ];
  thresh.pop();
  const tbl = (cap, rows) =>
    `<table><caption>${esc(cap)}</caption><tbody>${rows
      .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`)
      .join("")}</tbody></table>`;
  return tbl("Terminal price distribution", dist) + tbl("Cumulative thresholds", thresh);
}

function sensitivityTables(r) {
  return Object.entries(r.sensitivity)
    .map(([name, rows]) => {
      const body = rows
        .map(
          (b) =>
            `<tr><td>${esc(b.bucket)}</td><td>${b.count.toLocaleString()}</td><td>${money(b.median_price)}</td><td>${pct(b.p_above_base)}</td></tr>`
        )
        .join("");
      return `<div class="sens-block"><h3>${esc(name)}</h3><table>
        <thead><tr><th>Driver quintile</th><th>Paths</th><th>Median price</th><th>P(≥ base)</th></tr></thead>
        <tbody>${body}</tbody></table></div>`;
    })
    .join("");
}

/* ── render ───────────────────────────────────────────────────────── */
function render(r) {
  const s = r.stats, m = r.meta;
  $("emptyState").hidden = true;
  $("output").hidden = false;

  $("resultTitle").textContent = `${m.company_name} (${m.ticker})`;
  $("resultSub").textContent =
    `${m.n_simulations.toLocaleString()} paths · ${m.horizon_years}-year horizon · seed ${m.random_seed} · ` +
    `current ${money(m.current_price)} · computed in ${m.elapsed_ms} ms`;

  const upside = s.median_price / m.current_price - 1;
  $("cards").innerHTML = [
    { k: "Median terminal", v: money(s.median_price), s: `${upside >= 0 ? "+" : ""}${pct(upside)} vs current`, cls: upside >= 0 ? "pos" : "neg" },
    { k: "Expected return", v: pct(s.expected_return), s: `mean ${money(s.mean_price)}`, cls: s.expected_return >= 0 ? "pos" : "neg" },
    { k: "P(≥ base target)", v: pct(s.p_at_least_base), s: `base ${money0(m.base_price)}` },
    { k: "P(≥ bear target)", v: pct(s.p_at_least_bear), s: `bear ${money0(m.bear_price)}` },
    { k: "P90 / P10 band", v: `${money0(s.p5)}–${money0(s.p95)}`, s: "5th – 95th percentile" },
  ]
    .map((c) => `<div class="card ${c.cls || ""}"><div class="k">${esc(c.k)}</div><div class="v">${esc(c.v)}</div><div class="s">${esc(c.s)}</div></div>`)
    .join("");

  $("cap1").textContent = `Terminal price distribution — ${m.horizon_years}-year, ${m.n_simulations.toLocaleString()} sims`;
  $("chartHist").innerHTML = histChart(r);
  $("chartBars").innerHTML = barsChart(r);
  $("chartCdf").innerHTML = cdfChart(r);
  $("chartTornado").innerHTML = tornadoChart(r);
  $("summaryTables").innerHTML = summaryTables(r);
  $("sensitivityTables").innerHTML = sensitivityTables(r);
}

/* ── wiring ───────────────────────────────────────────────────────── */
function showError(msg) {
  const el = $("formError");
  el.textContent = msg;
  el.hidden = false;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("runBtn");
  $("formError").hidden = true;

  let payload;
  try {
    payload = readForm();
  } catch (err) {
    showError(err.message);
    return;
  }

  btn.disabled = true;
  btn.classList.add("busy");
  btn.textContent = "Running…";
  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({ error: `Server returned ${res.status}` }));
    if (!res.ok) throw new Error(data.error || `Server returned ${res.status}`);
    lastResult = data;
    $("exportJson").disabled = false;
    render(data);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
    btn.textContent = "Run simulation";
  }
});

$("loadPreset").addEventListener("click", () => {
  fillForm(PRESET);
  $("formError").hidden = true;
});

$("exportJson").addEventListener("click", () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${lastResult.meta.ticker}_monte_carlo.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

fillForm(PRESET);
