/* IMA Monte Carlo front end.
   P10/P50/P90 drivers, SEC auto-fill with peer-panel calibration, thesis mechanisms,
   POST /api/simulate, SVG panels, reconciliation and price-implied expectations,
   share links, correlation editor, PNG export, pin-and-compare, calibration log. */

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
const MECHS = ["cost_model", "backlog", "fcf_bridge", "justified_multiple"];

const C = {
  blue: "#2a63b0", red: "#b1271f", amber: "#9a6300", green: "#1f6d44", purple: "#574ca6",
  text: "#101820", muted: "#64707e", faint: "#8b959f", grid: "#e9edf2", axis: "#c9d1db", panel: "#ffffff",
};

const $ = (id) => document.getElementById(id);
const form = $("pitchForm");
let lastResult = null, lastPayload = null, lastPrefill = null, pinned = null;
let multipleIsPlaceholder = false, multipleEdited = false;

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

/* Triangular quantiles, for converting legacy low/mode/high into P10/P50/P90 */
function triQuantile(u, low, mode, high) {
  const c = (mode - low) / (high - low);
  return u < c ? low + Math.sqrt(u * (high - low) * (mode - low)) : high - Math.sqrt((1 - u) * (high - low) * (high - mode));
}
const triToQ = ([lo, mo, hi]) => [triQuantile(0.1, lo, mo, hi), triQuantile(0.5, lo, mo, hi), triQuantile(0.9, lo, mo, hi)];

/* Correlation grid */
function buildCorrGrid(matrix) {
  const g = $("corrGrid");
  let html = `<div class="h"></div>` + CORR_LABELS.map((l) => `<div class="h">${l}</div>`).join("");
  for (let i = 0; i < 5; i++) {
    html += `<div class="h row">${CORR_LABELS[i]}</div>`;
    for (let j = 0; j < 5; j++) {
      const diag = i === j;
      html += `<input type="number" step="0.05" min="-1" max="1" data-i="${i}" data-j="${j}" value="${round(matrix[i][j], 4)}"
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
  $("corrGrid").querySelectorAll("input").forEach((el) => { const v = parseFloat(el.value); if (Number.isFinite(v)) m[+el.dataset.i][+el.dataset.j] = v; });
  for (let i = 0; i < 5; i++) m[i][i] = 1;
  return m;
}
function corrIsDefault(m) { return m.every((r, i) => r.every((v, j) => Math.abs(v - DEFAULT_CORR[i][j]) < 1e-9)); }

/* Driver specs <-> form */
function setQ(driver, p10, p50, p90) { form[`${driver}_p10`].value = round(p10, 6); form[`${driver}_p50`].value = round(p50, 6); form[`${driver}_p90`].value = round(p90, 6); }
function readQ(driver) { return [form[`${driver}_p10`].value, form[`${driver}_p50`].value, form[`${driver}_p90`].value].map(Number); }
function specFor(driver) {
  const [p10, p50, p90] = readQ(driver);
  if (![p10, p50, p90].every(Number.isFinite)) throw new Error(`${driver.replace(/_/g, " ")} needs P10, P50 and P90.`);
  if (!(p10 < p50 && p50 < p90)) throw new Error(`${driver.replace(/_/g, " ")}: P10 must be below P50, and P50 below P90.`);
  if (driver === "ev_ebitda_multiple") return { type: "log_metalog", p10, p50, p90, lower: 0 };
  return { type: "metalog", p10, p50, p90 };
}
function applySpecs(specs) {
  TRI.forEach((k) => { const s = specs[k]; if (s) setQ(k, s.p10, s.p50, s.p90); });
  NORM.forEach((k) => { const s = specs[k]; if (s) { form[`${k}_mean`].value = s.mean; form[`${k}_std`].value = s.std; } });
}

/* Form <-> payload */
function fillForm(p) {
  form.ticker.value = p.ticker || ""; form.company_name.value = p.company_name || "";
  SCALARS.forEach((k) => { if (form[k] && p[k] !== undefined && p[k] !== null) form[k].value = p[k]; });
  if (p.driver_specs) applySpecs(p.driver_specs);
  else {
    TRI.forEach((k) => { if (p[k]) setQ(k, ...triToQ(p[k])); });
    NORM.forEach((k) => { if (p[k]) { form[`${k}_mean`].value = p[k][0]; form[`${k}_std`].value = p[k][1]; } });
  }
  form.copula.value = p.copula || "gaussian";
  form.copula_df.value = p.copula_df || 5;
  buildCorrGrid(p.correlation_matrix || DEFAULT_CORR);
  if (p.correlation_matrix && !corrIsDefault(p.correlation_matrix)) $("corrDetails").open = true;
  MECHS.forEach((m) => setMech(m, p[m] || null, Boolean(p[m])));
  if (p.dcf_path) { form.dcf_g.value = p.dcf_path.rev_cagr; form.dcf_m.value = p.dcf_path.ebitda_margin; form.dcf_mult.value = p.dcf_path.ev_ebitda_multiple; }
}
function num(name, label) {
  const el = form[name]; const v = el.value.trim();
  if (v === "" || !Number.isFinite(Number(v))) { el.classList.add("invalid"); throw new Error(`"${label || name}" needs a number.`); }
  el.classList.remove("invalid"); return Number(v);
}
function readForm() {
  form.querySelectorAll("input").forEach((el) => el.classList.remove("invalid"));
  const payload = { ticker: form.ticker.value.trim().toUpperCase() || "TICKER",
    company_name: form.company_name.value.trim() || form.ticker.value.trim() || "Untitled" };
  SCALARS.forEach((k) => { payload[k] = num(k, k.replace(/_/g, " ")); });
  const specs = {};
  TRI.forEach((k) => { specs[k] = specFor(k); });
  NORM.forEach((k) => { specs[k] = { type: "normal", mean: num(`${k}_mean`), std: num(`${k}_std`) }; });
  specs.share_dilution_pct.clip_low = -0.10; specs.share_dilution_pct.clip_high = 0.15;
  payload.driver_specs = specs;
  // legacy tuples so the server's required fields validate; the specs override them
  TRI.forEach((k) => { const [a, b, c] = readQ(k); payload[k] = [a - 0.5 * (b - a), b, c + 0.5 * (c - b)]; });
  NORM.forEach((k) => { payload[k] = [specs[k].mean, specs[k].std]; });
  payload.copula = form.copula.value; payload.copula_df = Number(form.copula_df.value) || 5;
  const corr = readCorr(); if (!corrIsDefault(corr)) payload.correlation_matrix = corr;
  MECHS.forEach((m) => { const block = readMech(m); if (block) payload[m] = block; });
  const g = form.dcf_g.value.trim(), m = form.dcf_m.value.trim(), x = form.dcf_mult.value.trim();
  if (g && m && x) payload.dcf_path = { rev_cagr: Number(g), ebitda_margin: Number(m), ev_ebitda_multiple: Number(x) };
  return payload;
}

/* Mechanisms */
const mechEl = (m) => document.querySelector(`details.mech[data-mech="${m}"]`);
function setMech(m, block, on) {
  const d = mechEl(m); const cb = form[`use_${m}`];
  if (block) {
    if (m === "cost_model") {
      const g = block.gross_margin; form.cm_gm_p10.value = g.p10; form.cm_gm_p50.value = g.p50; form.cm_gm_p90.value = g.p90;
      form.cm_fixed.value = block.fixed_cost_m; form.cm_fcg_mean.value = block.fixed_cost_growth?.mean ?? 0.03; form.cm_fcg_std.value = block.fixed_cost_growth?.std ?? 0.05;
    } else if (m === "backlog") {
      form.bl_m.value = block.backlog_m; form.bl_p10.value = block.conversion.p10; form.bl_p50.value = block.conversion.p50; form.bl_p90.value = block.conversion.p90;
    } else if (m === "fcf_bridge") {
      form.fb_fm_p10.value = block.fcf_margin.p10; form.fb_fm_p50.value = block.fcf_margin.p50; form.fb_fm_p90.value = block.fcf_margin.p90;
      form.fb_mult_p10.value = block.ev_fcf_multiple.p10; form.fb_mult_p50.value = block.ev_fcf_multiple.p50; form.fb_mult_p90.value = block.ev_fcf_multiple.p90;
    } else if (m === "justified_multiple") {
      form.jm_roic.value = block.roic ?? ""; form.jm_wacc.value = block.wacc; form.jm_tax.value = block.tax_rate; form.jm_da.value = block.da_share;
      form.jm_lrg.value = block.long_run_growth ?? 0.03; form.jm_fade.value = block.fade ?? 0.5; form.jm_prem.value = block.premium_reversion ?? 0.5;
      d.dataset.currentGrowth = block.current_growth ?? ""; d.dataset.currentMultiple = block.current_multiple ?? "";
    }
  }
  cb.checked = Boolean(on); d.open = Boolean(on);
}
function readMech(m) {
  if (!form[`use_${m}`].checked) return null;
  const q = (a, b, c, type = "metalog", lbl) => {
    const v = [num(a, lbl), num(b, lbl), num(c, lbl)];
    if (!(v[0] < v[1] && v[1] < v[2])) throw new Error(`${lbl}: P10 < P50 < P90 required.`);
    return type === "log_metalog" ? { type, p10: v[0], p50: v[1], p90: v[2], lower: 0 } : { type, p10: v[0], p50: v[1], p90: v[2] };
  };
  if (m === "cost_model") return { gross_margin: q("cm_gm_p10", "cm_gm_p50", "cm_gm_p90", "metalog", "Gross margin"),
    fixed_cost_m: num("cm_fixed", "Fixed cost"), fixed_cost_growth: { type: "normal", mean: num("cm_fcg_mean", "Fixed cost growth"), std: num("cm_fcg_std", "Fixed cost growth std") } };
  if (m === "backlog") return { backlog_m: num("bl_m", "Backlog"), conversion: q("bl_p10", "bl_p50", "bl_p90", "metalog", "Conversion") };
  if (m === "fcf_bridge") return { fcf_margin: q("fb_fm_p10", "fb_fm_p50", "fb_fm_p90", "metalog", "FCF margin"),
    ev_fcf_multiple: q("fb_mult_p10", "fb_mult_p50", "fb_mult_p90", "log_metalog", "EV/FCF") };
  if (m === "justified_multiple") {
    const d = mechEl(m);
    const block = { roic: num("jm_roic", "ROIC"), wacc: num("jm_wacc", "WACC"), tax_rate: num("jm_tax", "Tax rate"), da_share: num("jm_da", "D&A share"),
      long_run_growth: num("jm_lrg", "Long-run growth"), fade: num("jm_fade", "Growth fade"), premium_reversion: num("jm_prem", "Premium reversion") };
    const cg = parseFloat(d.dataset.currentGrowth), cm = parseFloat(d.dataset.currentMultiple);
    block.current_growth = Number.isFinite(cg) ? cg : readQ("rev_cagr")[1];
    block.current_multiple = Number.isFinite(cm) ? cm : readQ("ev_ebitda_multiple")[1];
    return block;
  }
  return null;
}
function applyMechanisms(mech) {
  MECHS.forEach((m) => {
    const d = mechEl(m); const avail = Boolean(mech?.available?.[m]);
    d.classList.toggle("unavailable", !avail);
    form[`use_${m}`].disabled = !avail;
    const note = d.querySelector(".mech-note"); note.textContent = mech?.notes?.[m] || "";
    note.classList.toggle("weak", !avail);
    if (avail && mech[m]) setMech(m, mech[m], false); else setMech(m, null, false);
  });
}

/* URL state */
const b64e = (s) => btoa(unescape(encodeURIComponent(s))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const b64d = (s) => decodeURIComponent(escape(atob(s.replace(/-/g, "+").replace(/_/g, "/"))));
function writeUrl(payload) { try { history.replaceState(null, "", "#s=" + b64e(JSON.stringify(payload))); } catch (_) { /* ignore */ } }
function readUrl() { const m = location.hash.match(/^#s=(.+)$/); if (!m) return null; try { return JSON.parse(b64d(m[1])); } catch (_) { return null; } }

/* Provenance and reference chips */
function clearProvenance() {
  document.querySelectorAll(".src-badge, .calib-note:not(.mech-note)").forEach((el) => el.remove());
  document.querySelectorAll(".needs-input").forEach((el) => el.classList.remove("needs-input"));
  document.querySelectorAll(".refs").forEach((el) => { el.innerHTML = ""; });
  $("sources").hidden = true; $("sourcesList").innerHTML = ""; $("calibBadge").hidden = true;
}
function badge(label, title, tone) { const b = document.createElement("span"); b.className = "src-badge" + (tone ? ` ${tone}` : ""); b.textContent = label; b.title = title; return b; }
function setP50(driver, value) {
  const isTri = TRI.includes(driver);
  if (isTri) {
    const [p10, , p90] = readQ(driver); const lo = Math.min(p10, value - 1e-6), hi = Math.max(p90, value + 1e-6);
    setQ(driver, lo, value, hi);
  } else form[`${driver}_mean`].value = value;
  if (driver === "ev_ebitda_multiple") multipleEdited = true;
  refreshChipState(driver);
}
function refreshChipState(driver) {
  const cur = TRI.includes(driver) ? readQ(driver)[1] : parseFloat(form[`${driver}_mean`].value);
  document.querySelectorAll(`.refs[data-refs="${driver}"] .chip[data-value]`).forEach((ch) => ch.classList.toggle("active", Math.abs(parseFloat(ch.dataset.value) - cur) < 1e-9));
}
function renderRefs(driver, ref, note) {
  const box = document.querySelector(`.refs[data-refs="${driver}"]`); if (!box) return;
  const f = REF_FMT[driver]; let html = ""; const settable = [];
  if (ref) {
    if (ref.current !== undefined && ref.current !== null) settable.push(["Today", ref.current]);
    if (ref.path_end !== undefined && ref.path_end !== null) settable.push(["Reverting path", ref.path_end]);
    if (ref.own?.p50 !== undefined) settable.push(["Own median", ref.own.p50]);
    if (ref.panel?.p50 !== undefined) settable.push(["Peer median", ref.panel.p50]);
    if (ref.hist_median !== undefined) settable.push(["Hist. median", ref.hist_median]);
    if (ref.hist_mean !== undefined) settable.push(["Hist. mean", ref.hist_mean]);
  }
  if (settable.length) {
    html += `<span class="cap">Set P50 to</span>`;
    html += settable.map(([l, v]) => `<button type="button" class="chip" data-value="${v}" title="Set the P50 to ${l.toLowerCase()} ${f(v)}">${l} ${f(v)}</button>`).join("");
  }
  if (ref?.own && ref.own.n) html += `<span class="chip static" title="Company's own filed observations">Own P10/P90 ${f(ref.own.p10)} to ${f(ref.own.p90)}, n=${ref.own.n}</span>`;
  if (ref?.panel && ref.panel.n) html += `<span class="chip static" title="Peer group observations">Peers ${f(ref.panel.p10)} to ${f(ref.panel.p90)}, n=${ref.panel.n}</span>`;
  if (ref?.raw_min !== undefined) html += `<span class="chip static">Filed ${f(ref.raw_min)} to ${f(ref.raw_max)}${ref.n ? `, n=${ref.n}` : ""}</span>`;
  if (ref?.fcf_yield_mean !== undefined) html += `<span class="chip static">FCF yield ${pct(ref.fcf_yield_mean, 1)}/yr</span>`;
  if (note) { const weak = /Generic|Insufficient|Placeholder|Capped|too few|No usable|no peer panel/i.test(note); html += `<p class="calib-note${weak ? " weak" : ""}">${esc(note)}</p>`; }
  box.innerHTML = html;
  box.querySelectorAll(".chip[data-value]").forEach((ch) => ch.addEventListener("click", () => setP50(driver, parseFloat(ch.dataset.value))));
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
  const f = data.fields; lastPrefill = data;
  clearProvenance();
  form.ticker.value = f.ticker; form.company_name.value = f.company_name;
  ["current_revenue", "current_ebitda_margin", "current_ev", "current_net_debt", "shares_outstanding", "horizon_years"]
    .forEach((k) => { if (f[k] !== null && f[k] !== undefined) form[k].value = f[k]; });
  if (f.current_price !== null && f.current_price !== undefined) {
    form.current_price.value = round(f.current_price, 2);
    const pl = document.querySelector('label[data-field="current_price"]');
    pl.prepend(badge("quote", data.provenance.current_price || "quote", ""));
  } else {
    form.current_price.value = "";
  }
  if (data.driver_specs) applySpecs(data.driver_specs);
  else { TRI.forEach((k) => setQ(k, ...triToQ(f[k]))); NORM.forEach((k) => { form[`${k}_mean`].value = f[k][0]; form[`${k}_std`].value = f[k][1]; }); }
  form.copula.value = data.copula || "gaussian"; form.copula_df.value = data.copula_df || 5;
  buildCorrGrid(data.correlation_matrix || DEFAULT_CORR);
  applyProvenance(data.provenance, data.reference || {});
  applyMechanisms(data.mechanisms);
  multipleIsPlaceholder = Boolean(data.needs_price); multipleEdited = false;

  const badgeEl = $("calibBadge");
  if (data.calibration === "v2") {
    const p = data.panel || {};
    badgeEl.className = "calib-badge"; badgeEl.textContent = `Calibration v2: peer panel (${p.basis || "SIC"}, ${(p.peers_used || []).length} names), mean reversion, Student-t copula`;
  } else { badgeEl.className = "calib-badge v1"; badgeEl.textContent = "Calibration v1: own history only. No peer panel is cached for this name; run tools/build_panels.py to add one."; }
  badgeEl.hidden = false;

  const c = data.company;
  const bits = [`<strong>${esc(c.name)}</strong>, CIK ${c.cik}`, `FY${c.latest_fy} EBITDA ${money0(c.ebitda_fy_m)}M (${esc(c.ebitda_basis)})`];
  if (c.current_ev_ebitda) bits.push(`current ${mult(c.current_ev_ebitda)} EV/EBITDA`);
  let html = `<p>${bits.join(" &middot; ")}</p>`;
  if (c.revenue_ttm_m) html += `<p class="fine">Form uses FY${c.latest_fy} revenue so it pairs with the FY margin. ${esc(c.revenue_ttm_label)} revenue is ${money0(c.revenue_ttm_m)}M.</p>`;
  if (data.needs_price) { html += `<p class="warn">No quote came back, so enter the current price yourself. Enterprise value and the multiple update when you do.</p>`; form.current_price.value = ""; document.querySelector('label[data-field="current_price"]').classList.add("needs-input"); }
  html += `<p class="warn">Bear, base and bull are yours to set. They are the DCF output this model exists to test.</p>`;
  html += `<p class="warn">P50s are the reverting path, not today's values. Your view goes in the P50 of the driver, or in a thesis mechanism below.</p>`;
  (data.warnings || []).forEach((w) => { html += `<p class="warn">${esc(w)}</p>`; });
  html += `<p class="fine"><a href="${esc(c.filings_url)}" target="_blank" rel="noopener">Check against the filings on EDGAR</a></p>`;
  showLookup("ok", html);
  ["bear_price", "base_price", "bull_price"].forEach((k) => { form[k].value = ""; form[k].closest("label").classList.add("needs-input"); });
}

function recomputeEV() {
  const price = parseFloat(form.current_price.value), shares = parseFloat(form.shares_outstanding.value), nd = parseFloat(form.current_net_debt.value);
  if (![price, shares, nd].every(Number.isFinite) || price <= 0 || shares <= 0) return;
  const ev = price * shares + nd; form.current_ev.value = round(ev, 1);
  const label = document.querySelector('label[data-field="current_ev"]');
  if (label && !label.querySelector(".src-badge")) label.prepend(badge("derived", "derived: price x shares + net debt", "derived"));
  document.querySelector('label[data-field="current_price"]')?.classList.remove("needs-input");
  const ebitda = parseFloat(form.current_revenue.value) * parseFloat(form.current_ebitda_margin.value);
  if (!Number.isFinite(ebitda) || ebitda <= 0) return;
  const m = ev / ebitda;
  if (multipleIsPlaceholder && !multipleEdited) {
    setQ("ev_ebitda_multiple", m * 0.75, m, m * 1.3);
    renderRefs("ev_ebitda_multiple", { current: round(m, 2) }, `Anchored on the current ${mult(m)}, 0.75x to 1.3x. Your comps view goes in P50.`);
  }
}

async function autofill() {
  const ticker = form.ticker.value.trim().toUpperCase();
  if (!ticker) { showLookup("err", "Enter a ticker first."); return; }
  const btn = $("autofillBtn"); btn.disabled = true; btn.textContent = "Loading";
  showLookup("busy", `Reading ${esc(ticker)} filings from SEC EDGAR and the peer panel.`);
  try {
    // The quote wins. A price left in the field from the previous name used to be sent
    // along and echoed back as this name's price, which is how TDS showed MYRG's $287.
    const params = new URLSearchParams({ horizon: form.horizon_years.value || "2" });
    const res = await fetch(`/api/company/${encodeURIComponent(ticker)}?${params}`);
    const data = await res.json().catch(() => ({ error: `Server returned ${res.status}` }));
    if (!res.ok) throw new Error(data.error || `Server returned ${res.status}`);
    fillFromPrefill(data);
  } catch (err) { showLookup("err", esc(err.message)); }
  finally { btn.disabled = false; btn.textContent = "Auto-fill from SEC"; }
}

/* SVG helpers */
const W = 560, H = 300, PAD = { t: 14, r: 18, b: 40, l: 52 }, IW = W - PAD.l - PAD.r, IH = H - PAD.t - PAD.b;
const frame = (inner, label) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(label)}" font-family="IBM Plex Sans, ui-sans-serif, system-ui, sans-serif">${inner}</svg>`;
function axes({ xTicks, yTicks, xLabel, yLabel }) {
  let s = "";
  yTicks.forEach(({ y, label }) => { s += `<line x1="${PAD.l}" y1="${y}" x2="${PAD.l + IW}" y2="${y}" stroke="${C.grid}"/><text x="${PAD.l - 8}" y="${y + 3.5}" fill="${C.muted}" font-size="10" text-anchor="end">${esc(label)}</text>`; });
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
  let g = axes({ xTicks: niceTicks(xMin, xMax, 6).map((v) => ({ x: sx(v), label: money0(v) })), yTicks: niceTicks(0, yMax, 4).map((v) => ({ y: sy(v), label: v.toFixed(3) })), xLabel: "Terminal share price ($)", yLabel: "Density" });
  density.forEach((d, i) => { if (d <= 0) return; const x = sx(edges[i]), w = Math.max(sx(edges[i + 1]) - x - 0.4, 0.6); g += `<rect x="${x.toFixed(2)}" y="${sy(d).toFixed(2)}" width="${w.toFixed(2)}" height="${(PAD.t + IH - sy(d)).toFixed(2)}" fill="${C.blue}" opacity="0.72"/>`; });
  const marks = [["Bear", m.bear_price, C.red, "4 3"], ["Base", m.base_price, C.amber, "4 3"], ["Bull", m.bull_price, C.green, "4 3"], ["Spot", m.current_price, C.text, ""], ["Median", s.median_price, C.purple, "1 3"]];
  if (r.fcf_bridge) marks.push(["FCF median", r.fcf_bridge.median_price, C.purple, "6 3"]);
  marks.forEach(([, v, color, dash]) => { if (v < xMin || v > xMax) return; g += `<line x1="${sx(v).toFixed(2)}" y1="${PAD.t}" x2="${sx(v).toFixed(2)}" y2="${PAD.t + IH}" stroke="${color}" stroke-width="1.6" ${dash ? `stroke-dasharray="${dash}"` : ""}/>`; });
  const lw = 126, lh = marks.length * 14 + 10, rightSide = sx(s.median_price) < PAD.l + IW / 2, lx = rightSide ? PAD.l + IW - lw - 6 : PAD.l + 6, ly = PAD.t + 5;
  g += `<rect x="${lx}" y="${ly}" width="${lw}" height="${lh}" rx="4" fill="${C.panel}" opacity="0.94" stroke="${C.grid}"/>`;
  marks.forEach(([label, v, color], i) => { const y = ly + 15 + i * 14; g += `<line x1="${lx + 8}" y1="${y - 3.5}" x2="${lx + 22}" y2="${y - 3.5}" stroke="${color}" stroke-width="2"/><text x="${lx + 27}" y="${y}" fill="${C.muted}" font-size="9.5">${esc(label)} ${money0(v)}</text>`; });
  return frame(g, `Distribution of simulated terminal share price. Median ${money(s.median_price)}.`);
}
function barsChart(r) {
  const m = r.meta, s = r.stats;
  const data = [{ label: "Below bear", sub: `under ${money0(m.bear_price)}`, v: s.below_bear, c: C.red }, { label: "Bear to base", sub: `${money0(m.bear_price)} to ${money0(m.base_price)}`, v: s.bear_to_base, c: C.amber }, { label: "Base to bull", sub: `${money0(m.base_price)} to ${money0(m.bull_price)}`, v: s.base_to_bull, c: C.blue }, { label: "Above bull", sub: `over ${money0(m.bull_price)}`, v: s.above_bull, c: C.green }];
  const yMax = Math.max(...data.map((d) => d.v)) * 1.25 || 1, sy = (v) => PAD.t + IH - (v / yMax) * IH;
  let g = axes({ xTicks: [], yTicks: niceTicks(0, yMax, 4).map((v) => ({ y: sy(v), label: pct(v, 0) })), yLabel: "Probability" });
  const slot = IW / data.length, bw = Math.min(slot * 0.56, 74);
  data.forEach((d, i) => { const cx = PAD.l + slot * (i + 0.5), y = sy(d.v); g += `<rect x="${(cx - bw / 2).toFixed(2)}" y="${y.toFixed(2)}" width="${bw.toFixed(2)}" height="${(PAD.t + IH - y).toFixed(2)}" rx="3" fill="${d.c}" opacity="0.88"/><text x="${cx.toFixed(2)}" y="${(y - 7).toFixed(2)}" fill="${C.text}" font-size="12.5" font-weight="600" text-anchor="middle">${pct(d.v)}</text><text x="${cx.toFixed(2)}" y="${PAD.t + IH + 15}" fill="${C.muted}" font-size="10.5" text-anchor="middle">${esc(d.label)}</text><text x="${cx.toFixed(2)}" y="${PAD.t + IH + 27}" fill="${C.faint}" font-size="9.5" text-anchor="middle">${esc(d.sub)}</text>`; });
  return frame(g, `Probability of each scenario band. Below bear ${pct(s.below_bear)}, above bull ${pct(s.above_bull)}.`);
}
function cdfChart(r) {
  const { x, y } = r.cdf, m = r.meta, s = r.stats, xMin = 0, xMax = r.histogram.edges[r.histogram.edges.length - 1];
  const sx = (v) => PAD.l + ((v - xMin) / (xMax - xMin)) * IW, sy = (v) => PAD.t + IH - v * IH;
  let g = axes({ xTicks: niceTicks(xMin, xMax, 6).map((v) => ({ x: sx(v), label: money0(v) })), yTicks: [0, 0.25, 0.5, 0.75, 1].map((v) => ({ y: sy(v), label: pct(v, 0) })), xLabel: "Terminal share price ($)", yLabel: "Cumulative probability" });
  const pts = x.map((v, i) => (v <= xMax ? `${sx(v).toFixed(2)},${sy(y[i]).toFixed(2)}` : null)).filter(Boolean).join(" ");
  g += `<polyline points="${pts}" fill="none" stroke="${C.blue}" stroke-width="1.8"/>`;
  [["Bear", m.bear_price, C.red, s.p_at_least_bear], ["Base", m.base_price, C.amber, s.p_at_least_base], ["Bull", m.bull_price, C.green, s.p_at_least_bull]].forEach(([label, v, color, p]) => {
    if (v < xMin || v > xMax) return; const px = sx(v), py = sy(1 - p);
    g += `<line x1="${px.toFixed(2)}" y1="${PAD.t}" x2="${px.toFixed(2)}" y2="${PAD.t + IH}" stroke="${color}" stroke-width="1.4" stroke-dasharray="4 3"/><circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="3" fill="${color}"/>`;
    const anchor = px > PAD.l + IW * 0.62 ? "end" : "start", dx = anchor === "end" ? -8 : 8, ly = Math.min(Math.max(py - 8, PAD.t + 10), PAD.t + IH - 4);
    g += `<text x="${(px + dx).toFixed(2)}" y="${ly.toFixed(2)}" fill="${color}" font-size="10" font-weight="600" text-anchor="${anchor}">P(at least ${label}) ${pct(p)}</text>`;
  });
  if (m.current_price <= xMax) g += `<line x1="${sx(m.current_price).toFixed(2)}" y1="${PAD.t}" x2="${sx(m.current_price).toFixed(2)}" y2="${PAD.t + IH}" stroke="${C.text}" opacity="0.4"/>`;
  return frame(g, `Cumulative distribution. P(at least base) ${pct(s.p_at_least_base)}.`);
}
function tornadoChart(r) {
  const rows = r.tornado.map((d) => ({ ...d, lo: Math.min(d.low, d.high), hi: Math.max(d.low, d.high) })), median = r.stats.median_price;
  const L = { t: 14, r: 46, b: 40, l: 132 }, iw = W - L.l - L.r, ih = H - L.t - L.b;
  const xMin = Math.min(...rows.map((d) => d.lo), median), xMax = Math.max(...rows.map((d) => d.hi), median), padX = (xMax - xMin) * 0.08 || 1, lo = xMin - padX, hi = xMax + padX, sx = (v) => L.l + ((v - lo) / (hi - lo)) * iw;
  let g = "";
  niceTicks(lo, hi, 5).forEach((v) => { g += `<line x1="${sx(v).toFixed(2)}" y1="${L.t}" x2="${sx(v).toFixed(2)}" y2="${L.t + ih}" stroke="${C.grid}"/><text x="${sx(v).toFixed(2)}" y="${L.t + ih + 15}" fill="${C.muted}" font-size="10" text-anchor="middle">${money0(v)}</text>`; });
  const slot = ih / rows.length, bh = Math.min(slot * 0.5, 22);
  rows.forEach((d, i) => { const cy = L.t + slot * (i + 0.5); g += `<rect x="${sx(d.lo).toFixed(2)}" y="${(cy - bh / 2).toFixed(2)}" width="${Math.max(sx(d.hi) - sx(d.lo), 1).toFixed(2)}" height="${bh}" rx="3" fill="${C.blue}" opacity="0.78"/><text x="${L.l - 9}" y="${(cy + 3.5).toFixed(2)}" fill="${C.text}" font-size="10.5" text-anchor="end">${esc(d.name)}</text><text x="${(sx(d.lo) - 5).toFixed(2)}" y="${(cy + 3.5).toFixed(2)}" fill="${C.red}" font-size="9.5" text-anchor="end">${money0(d.lo)}</text><text x="${(sx(d.hi) + 5).toFixed(2)}" y="${(cy + 3.5).toFixed(2)}" fill="${C.green}" font-size="9.5">${money0(d.hi)}</text>`; });
  g += `<line x1="${sx(median).toFixed(2)}" y1="${L.t}" x2="${sx(median).toFixed(2)}" y2="${L.t + ih}" stroke="${C.purple}" stroke-width="1.5" stroke-dasharray="2 3"/><line x1="${L.l}" y1="${L.t + ih}" x2="${L.l + iw}" y2="${L.t + ih}" stroke="${C.axis}"/><text x="${L.l + iw / 2}" y="${H - 6}" fill="${C.muted}" font-size="10.5" text-anchor="middle">Median terminal price ($). Bar spans the driver's P10 to P90.</text>`;
  const top = rows[rows.length - 1];
  return frame(g, `Driver sensitivity. Widest swing: ${top.name}, ${money0(top.lo)} to ${money0(top.hi)}.`);
}

/* Price path with the terminal distribution: assessor only.
   History on the left; from today, log-space cones whose width grows with the square
   root of time to the P5/P95 and P25/P75 of the simulated horizon; at the horizon the
   terminal density drawn as a profile, coloured by scenario band with each band's
   probability beside it. Y-range is the central mass, not the tails. */
let lastPath = null;
async function fetchPath(ticker) {
  try { const res = await fetch(`/api/prices/${encodeURIComponent(ticker)}?years=3`); return res.ok ? await res.json() : null; }
  catch (_) { return null; }
}
function pathChart(r, path) {
  const m = r.meta, s = r.stats, closes = (path && path.closes) || [];
  const PW = 1120, PH = 560, L = { t: 36, r: 230, b: 40, l: 64 }, iw = PW - L.l - L.r, ih = PH - L.t - L.b;
  const t0 = closes.length ? Date.parse(closes[0][0]) : Date.now() - 365 * 864e5, tSpot = Date.now();
  const tEnd = tSpot + m.horizon_years * 365.25 * 864e5;
  const spot = m.current_price, med = s.median_price;
  // y-range: the central mass and the history, not the tails
  const lo = Math.min(s.p5 * 0.92, m.bear_price * 0.92, ...closes.map((c) => c[1]));
  const hi = Math.max(s.p95 * 1.04, m.bull_price * 1.04, ...closes.map((c) => c[1]));
  const yMin = Math.max(0, lo), yMax = hi;
  const sx = (t) => L.l + ((t - t0) / (tEnd - t0)) * iw, sy = (v) => L.t + ih - ((v - yMin) / (yMax - yMin)) * ih;
  const xs = sx(tSpot), xe = sx(tEnd);
  let g = "";
  // frame, grid, axes
  g += `<rect x="${L.l}" y="${L.t}" width="${iw}" height="${ih}" fill="#fbfcfe" stroke="${C.grid}"/>`;
  niceTicks(yMin, yMax, 6).forEach((v) => { g += `<line x1="${L.l}" y1="${sy(v).toFixed(1)}" x2="${L.l + iw}" y2="${sy(v).toFixed(1)}" stroke="${C.grid}"/><text x="${L.l - 8}" y="${(sy(v) + 3.5).toFixed(1)}" fill="${C.muted}" font-size="11" text-anchor="end">${money0(v)}</text>`; });
  for (let y = new Date(t0).getFullYear(); y <= new Date(tEnd).getFullYear(); y++) {
    const t = Date.parse(`${y}-01-01`); if (t > t0 && t < tEnd) g += `<line x1="${sx(t).toFixed(1)}" y1="${L.t}" x2="${sx(t).toFixed(1)}" y2="${L.t + ih}" stroke="${C.grid}" stroke-dasharray="2 4"/><text x="${sx(t).toFixed(1)}" y="${L.t + ih + 16}" fill="${C.muted}" font-size="11" text-anchor="middle">${y}</text>`;
  }
  // cones in log space: median drifts linearly, dispersion grows with sqrt(time)
  const q = (v, u) => Math.exp(Math.log(spot) + u * Math.log(med / spot) + Math.sqrt(u) * Math.log(v / med));
  const cone = (loQ, hiQ, color, op) => {
    const up = [], dn = [];
    for (let i = 0; i <= 40; i++) { const u = i / 40, t = tSpot + u * (tEnd - tSpot); up.push(`${sx(t).toFixed(1)},${sy(q(hiQ, u)).toFixed(1)}`); dn.unshift(`${sx(t).toFixed(1)},${sy(q(loQ, u)).toFixed(1)}`); }
    return `<polygon points="${up.join(" ")} ${dn.join(" ")}" fill="${color}" opacity="${op}"/>`;
  };
  g += cone(s.p5, s.p95, C.blue, 0.10) + cone(s.p25, s.p75, C.blue, 0.16);
  const medPts = []; for (let i = 0; i <= 40; i++) { const u = i / 40, t = tSpot + u * (tEnd - tSpot); medPts.push(`${sx(t).toFixed(1)},${sy(q(med, u)).toFixed(1)}`); }
  g += `<polyline points="${medPts.join(" ")}" fill="none" stroke="${C.purple}" stroke-width="1.6" stroke-dasharray="5 4"/>`;
  // price history
  if (closes.length) g += `<polyline points="${closes.map((c) => `${sx(Date.parse(c[0])).toFixed(1)},${sy(c[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="${C.text}" stroke-width="1.5"/>`;
  g += `<circle cx="${xs}" cy="${sy(spot).toFixed(1)}" r="3.5" fill="${C.text}"/>`;
  g += `<line x1="${xs}" y1="${L.t}" x2="${xs}" y2="${L.t + ih}" stroke="${C.axis}" stroke-dasharray="3 3"/>`;
  g += `<text x="${xs}" y="${L.t + ih + 16}" fill="${C.text}" font-size="11" font-weight="600" text-anchor="middle">today ${money0(spot)}</text>`;
  g += `<text x="${xe}" y="${L.t + ih + 16}" fill="${C.muted}" font-size="11" text-anchor="middle">+${m.horizon_years}y</text>`;
  // terminal density as a smoothed profile at the horizon, coloured by scenario band
  const edges = r.histogram.edges, raw = r.histogram.density, n = raw.length;
  const dens = raw.map((_, i) => { let a = 0, k = 0; for (let j = -2; j <= 2; j++) { if (raw[i + j] !== undefined) { a += raw[i + j]; k++; } } return a / k; });
  const dMax = Math.max(...dens) || 1, profW = L.r - 88, x0 = xe;
  const bands = [["below bear", 0, m.bear_price, C.red, s.below_bear], ["bear to base", m.bear_price, m.base_price, C.amber, s.bear_to_base],
    ["base to bull", m.base_price, m.bull_price, C.blue, s.base_to_bull], ["above bull", m.bull_price, Infinity, C.green, s.above_bull]];
  bands.forEach(([, bLo, bHi, color]) => {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const c = (edges[i] + edges[i + 1]) / 2; if (c < bLo || c >= bHi || c < yMin || c > yMax) continue;
      pts.push([x0 + (dens[i] / dMax) * profW, sy(c)]);
    }
    if (pts.length < 2) return;
    const yTop = Math.max(sy(Math.min(bHi, yMax)), L.t), yBot = Math.min(sy(Math.max(bLo, yMin)), L.t + ih);
    g += `<path d="M${x0},${yBot.toFixed(1)} L${pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" L")} L${x0},${yTop.toFixed(1)} Z" fill="${color}" opacity="0.55"/>`;
  });
  g += `<line x1="${x0}" y1="${L.t}" x2="${x0}" y2="${L.t + ih}" stroke="${C.axis}"/>`;
  // probability labels beside each band, placed at the band's centre within the visible range
  bands.forEach(([label, bLo, bHi, color, p]) => {
    const a = Math.max(bLo, yMin), b = Math.min(bHi === Infinity ? yMax : bHi, yMax); if (b <= a) return;
    const y = sy((a + b) / 2);
    g += `<text x="${x0 + profW + 10}" y="${(y + 4).toFixed(1)}" fill="${color}" font-size="12" font-weight="600">${pct(p, 0)}</text><text x="${x0 + profW + 44}" y="${(y + 4).toFixed(1)}" fill="${C.muted}" font-size="10.5">${label}</text>`;
  });
  // targets across the fan and profile, labelled on the frame's right edge
  [["Bull", m.bull_price, C.green], ["Base", m.base_price, C.amber], ["Bear", m.bear_price, C.red]].forEach(([lbl, v, col]) => {
    if (v < yMin || v > yMax) return;
    g += `<line x1="${xs}" y1="${sy(v).toFixed(1)}" x2="${x0 + profW}" y2="${sy(v).toFixed(1)}" stroke="${col}" stroke-width="1.1" stroke-dasharray="6 4"/>`;
    g += `<text x="${xs + 6}" y="${(sy(v) - 4).toFixed(1)}" fill="${col}" font-size="10.5" font-weight="600">${lbl} ${money0(v)}</text>`;
  });
  g += `<text x="${x0 + 4}" y="${(sy(med) + 4).toFixed(1)}" fill="${C.purple}" font-size="10.5" font-weight="600">median ${money0(med)}</text>`;
  g += `<text x="${L.l}" y="${L.t - 12}" fill="${C.text}" font-size="13" font-weight="600">${esc(m.ticker)} price, the simulated range to the horizon, and the distribution at ${m.horizon_years} years</text>`;
  g += `<text x="${L.l + iw}" y="${L.t - 12}" fill="${C.muted}" font-size="10.5" text-anchor="end">bands: P5 to P95, P25 to P75; dashed: median path</text>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${PW} ${PH}" width="${PW}" height="${PH}" role="img" aria-label="Price history, simulated range to the horizon and the terminal distribution at the horizon by scenario band." font-family="IBM Plex Sans, ui-sans-serif, system-ui, sans-serif">${g}</svg>`;
}
function svgStringToImage(svg) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" })), img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); }; img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Could not rasterise chart")); }; img.src = url;
  });
}

/* Tables and panels */
const seCell = (p, se) => `<td class="num">${pct(p)}<span class="se">&plusmn;${(se * 100).toFixed(2)}</span></td>`;
function summaryTables(r) {
  const s = r.stats;
  const dist = [["5th percentile", money(s.p5)], ["25th percentile", money(s.p25)], ["Median", money(s.median_price)], ["Mean", money(s.mean_price)], ["75th percentile", money(s.p75)], ["95th percentile", money(s.p95)], ["Standard deviation", money(s.std_price)]];
  const tbl = (cap, rows) => `<table><caption>${esc(cap)}</caption><tbody>${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${esc(v)}</td></tr>`).join("")}</tbody></table>`;
  const thresh = `<table><caption>Cumulative thresholds, with sampling error</caption><tbody>
    <tr><td>P(at least bear)</td>${seCell(s.p_at_least_bear, s.se.p_at_least_bear)}</tr><tr><td>P(at least base)</td>${seCell(s.p_at_least_base, s.se.p_at_least_base)}</tr><tr><td>P(at least bull)</td>${seCell(s.p_at_least_bull, s.se.p_at_least_bull)}</tr>
    <tr><td>Median return, total</td><td class="num">${spct(s.median_return)}</td></tr><tr><td>Median return, annualised</td><td class="num">${spct(s.median_return_annualized)}</td></tr><tr><td>Mean return, annualised</td><td class="num">${spct(s.mean_return_annualized)}</td></tr></tbody></table>`;
  return tbl("Terminal price distribution", dist) + thresh;
}
function sensitivityTables(r) {
  return Object.entries(r.sensitivity).map(([name, rows]) => `<div class="sens-block"><h4>${esc(name)}</h4><table><thead><tr><th>Driver quintile</th><th>Paths</th><th>Median price</th><th>P(at least base)</th></tr></thead><tbody>${rows.map((b) => `<tr><td>${esc(b.bucket)}</td><td class="num">${b.count.toLocaleString()}</td><td class="num">${money(b.median_price)}</td><td class="num">${pct(b.p_above_base)}</td></tr>`).join("")}</tbody></table></div>`).join("");
}
function reconciliation(r, payload) {
  const rc = r.reconciliation, s = r.stats, m = r.meta, dq = r.meta.driver_quantiles;
  const gap = rc.median_price / rc.mode_path_price - 1;
  $("reconSub").textContent = `Mode-path price ${money(rc.mode_path_price)}. Simulated median ${money(rc.median_price)}, ${spct(gap)} against it.`;
  const hiMult = dq.ev_ebitda_multiple.p90, loMult = dq.ev_ebitda_multiple.p10, hiMargin = dq.ebitda_margin.p90, loMargin = dq.ebitda_margin.p10;
  const row = (key, c) => {
    const im = c.implied_multiple_at_mode_margin, ig = c.implied_margin_at_mode_multiple;
    const multOut = im !== null && (im > hiMult || im < loMult), margOut = ig !== null && (ig > hiMargin || ig < loMargin);
    return `<tr class="${key}"><td>${key[0].toUpperCase() + key.slice(1)}</td><td class="num">${money(c.target)}</td><td class="num">${spct(c.vs_spot, 0)}</td>${seCell(c.p_at_least, s.se[`p_at_least_${key}`])}<td class="num${multOut ? " flag" : ""}">${im === null ? "n/a" : mult(im)}</td><td class="num${margOut ? " flag" : ""}">${ig === null ? "n/a" : pct(ig, 2)}</td></tr>`;
  };
  $("reconTable").innerHTML = `<table><thead><tr><th>Case</th><th>Target</th><th>vs spot</th><th>P(at least)</th><th>Multiple needed at P50 margin ${pct(rc.mode_drivers.ebitda_margin, 2)}</th><th>Margin needed at P50 multiple ${mult(rc.mode_drivers.ev_ebitda_multiple)}</th></tr></thead><tbody>${row("bear", rc.cases.bear)}${row("base", rc.cases.base)}${row("bull", rc.cases.bull)}</tbody></table>`;
  const base = rc.cases.base, bgap = rc.base_gap_pct;
  $("reconNote").textContent = `Your base target of ${money(m.base_price)} sits ${pct(Math.abs(bgap), 1)} ${bgap < 0 ? "above" : "below"} the simulated median. It needs a ${mult(base.implied_multiple_at_mode_margin)} exit multiple at your P50 margin, or a ${pct(base.implied_margin_at_mode_multiple, 2)} margin at your P50 multiple. Red cells fall outside the P10 to P90 you gave the model.`;
}
function expectations(r, payload) {
  const pi = r.price_implied, dq = r.meta.driver_quantiles, m = r.meta;
  const rows = [
    ["Revenue CAGR", pi.implied_growth_at_mode_margin_and_multiple, dq.rev_cagr, (v) => pct(v, 1)],
    ["EBITDA margin", pi.implied_margin_at_mode_growth_and_multiple, dq.ebitda_margin, (v) => pct(v, 2)],
    ["Exit multiple", pi.implied_multiple_at_mode_growth_and_margin, dq.ev_ebitda_multiple, (v) => mult(v)],
  ];
  let html = `<table><thead><tr><th>Driver</th><th>Market-implied at spot ${money(m.current_price)}</th><th>Your P50</th><th>Your P10 to P90</th></tr></thead><tbody>`;
  rows.forEach(([name, implied, q, f]) => {
    const out = implied !== null && (implied < q.p10 || implied > q.p90);
    html += `<tr><td>${name}</td><td class="num${out ? " flag" : ""}">${implied === null ? "no solution" : f(implied)}</td><td class="num">${f(q.p50)}</td><td class="num">${f(q.p10)} to ${f(q.p90)}</td></tr>`;
  });
  html += `</tbody></table>`;
  $("expectTable").innerHTML = html;
  let sub = `Each implied value holds the other two drivers at their P50.`;
  let note = `The market-implied column is what spot already assumes. If your P50 is above it, you are paying for a view the price does not hold yet; the probability the thesis beats the market is P(at least spot) in the thresholds table.`;
  if (r.dcf_path) {
    const d = r.dcf_path;
    sub += ` Your three-statement path (${pct(d.drivers.rev_cagr, 1)} growth, ${pct(d.drivers.ebitda_margin, 2)} margin, ${mult(d.drivers.ev_ebitda_multiple)}) lands at ${money(d.price)}, the ${Math.round(d.percentile * 100)}th percentile of the distribution.`;
    note += ` ${Math.round(d.p_at_least * 100)}% of simulated paths finish at or above your three-statement price.`;
  }
  if (r.fcf_bridge) note += ` The FCF bridge's median is ${money(r.fcf_bridge.median_price)}, ${spct(r.fcf_bridge.gap_to_ebitda_bridge)} against the EBITDA bridge; a gap that size is itself a finding about where the value sits.`;
  $("expectSub").textContent = sub; $("expectNote").textContent = note;
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
  if (r.fcf_bridge) items.push({ k: "FCF bridge median", v: money(r.fcf_bridge.median_price), s: `${spct(r.fcf_bridge.gap_to_ebitda_bridge)} vs EBITDA bridge; P(base) ${pct(r.fcf_bridge.p_at_least_base)}`, cls: "alt" });
  if (r.dcf_path) items.push({ k: "Your DCF path", v: money(r.dcf_path.price), s: `${Math.round(r.dcf_path.percentile * 100)}th percentile of the distribution`, cls: "alt" });
  $("cards").innerHTML = items.map((c) => `<div class="card ${c.cls || ""}"><div class="k">${esc(c.k)}</div><div class="v">${c.v}</div><div class="s">${esc(c.s)}</div></div>`).join("");
}
function compareStrip(r) {
  const el = $("compareStrip");
  if (!pinned) { el.hidden = true; return; }
  const a = pinned.stats, b = r.stats;
  const d = (x, y, f) => { const diff = y - x; const sign = diff > 0 ? "+" : diff < 0 ? "-" : ""; return `<span class="d ${diff === 0 ? "" : diff > 0 ? "up" : "down"}">${sign}${f(Math.abs(diff))}</span>`; };
  el.innerHTML = `<span class="lbl">Pinned: ${esc(pinned.label)}</span><div class="cells">
      <span>Median <b>${money(a.median_price)}</b> vs <b>${money(b.median_price)}</b>${d(a.median_price, b.median_price, (v) => money(v))}</span>
      <span>P(base) <b>${pct(a.p_at_least_base)}</b> vs <b>${pct(b.p_at_least_base)}</b>${d(a.p_at_least_base, b.p_at_least_base, (v) => (v * 100).toFixed(1) + "pp")}</span>
      <span>P(bear) <b>${pct(a.p_at_least_bear)}</b> vs <b>${pct(b.p_at_least_bear)}</b>${d(a.p_at_least_bear, b.p_at_least_bear, (v) => (v * 100).toFixed(1) + "pp")}</span>
      <span>P5-P95 <b>${money0(a.p5)}-${money0(a.p95)}</b> vs <b>${money0(b.p5)}-${money0(b.p95)}</b></span></div>`;
  el.hidden = false;
}
function render(r, payload) {
  const m = r.meta;
  $("emptyState").hidden = true; $("output").hidden = false;
  $("resultTitle").textContent = `${m.company_name} (${m.ticker})`;
  const on = Object.entries(m.mechanisms || {}).filter(([, v]) => v).map(([k]) => k.replace(/_/g, " "));
  $("resultSub").textContent = `${m.n_simulations.toLocaleString()} paths, ${m.horizon_years}-year horizon, seed ${m.random_seed}, ${m.copula === "t" ? `t-copula (df ${m.copula_df})` : "Gaussian copula"}, spot ${money(m.current_price)}, ${m.elapsed_ms} ms${on.length ? `, mechanisms: ${on.join(", ")}` : ""}`;
  compareStrip(r); cards(r); reconciliation(r, payload); expectations(r, payload);
  $("cap1").textContent = `Terminal price distribution, ${m.horizon_years}-year, ${m.n_simulations.toLocaleString()} paths`;
  $("chartHist").innerHTML = histChart(r); $("chartBars").innerHTML = barsChart(r); $("chartCdf").innerHTML = cdfChart(r); $("chartTornado").innerHTML = tornadoChart(r);
  $("summaryTables").innerHTML = summaryTables(r); $("sensitivityTables").innerHTML = sensitivityTables(r);
  ["shareBtn", "pinBtn", "exportPng", "exportJson", "logRun", "assessBtn"].forEach((id) => { $(id).disabled = false; });
}

/* Export */
function svgToImage(svgEl) {
  return new Promise((resolve, reject) => {
    const xml = new XMLSerializer().serializeToString(svgEl), url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml;charset=utf-8" })), img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); }; img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Could not rasterise chart")); }; img.src = url;
  });
}
async function exportPng() {
  if (!lastResult) return;
  const m = lastResult.meta, s = lastResult.stats, scale = 2, gap = 24, titleH = 76;
  const canvas = document.createElement("canvas"); canvas.width = (W * 2 + gap * 3) * scale; canvas.height = (titleH + H * 2 + gap * 3) * scale;
  const ctx = canvas.getContext("2d"); ctx.scale(scale, scale); ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = C.text; ctx.font = "600 20px IBM Plex Sans, system-ui, sans-serif"; ctx.fillText(`${m.company_name} (${m.ticker}). Monte Carlo scenario analysis`, gap, 34);
  ctx.fillStyle = C.muted; ctx.font = "12px IBM Plex Mono, ui-monospace, monospace"; ctx.fillText(`${m.n_simulations.toLocaleString()} paths, ${m.horizon_years}-year horizon, spot ${money(m.current_price)}, median ${money(s.median_price)}, P(at least base) ${pct(s.p_at_least_base)}`, gap, 56);
  const imgs = await Promise.all(["chartHist", "chartBars", "chartCdf", "chartTornado"].map((id) => svgToImage($(id).querySelector("svg"))));
  imgs.forEach((img, i) => { const col = i % 2, row = Math.floor(i / 2), x = gap + col * (W + gap), y = titleH + gap + row * (H + gap); ctx.fillStyle = "#ffffff"; ctx.fillRect(x, y, W, H); ctx.drawImage(img, x, y, W, H); });
  const a = document.createElement("a"); a.href = canvas.toDataURL("image/png"); a.download = `${m.ticker}_monte_carlo.png`; a.click();
}

/* Pitch assessor: one image for the deck or the Excel model, plus a tab-separated block to paste */
function assessorRows(r, payload) {
  const s = r.stats, m = r.meta, rc = r.reconciliation, pi = r.price_implied, dq = m.driver_quantiles;
  const rows = [
    ["Company", `${m.company_name} (${m.ticker})`], ["Run date", new Date().toISOString().slice(0, 10)],
    ["Spot", money(m.current_price)], ["Horizon", `${m.horizon_years} years`], ["Paths", m.n_simulations.toLocaleString()],
    ["Mechanisms", Object.entries(m.mechanisms || {}).filter(([, v]) => v).map(([k]) => k.replace(/_/g, " ")).join(", ") || "none"],
    ["", ""],
    ["Bear target", `${money(m.bear_price)} (${spct(m.bear_price / m.current_price - 1, 0)})`], ["Base target", `${money(m.base_price)} (${spct(m.base_price / m.current_price - 1, 0)})`], ["Bull target", `${money(m.bull_price)} (${spct(m.bull_price / m.current_price - 1, 0)})`],
    ["", ""],
    ["Median terminal", `${money(s.median_price)} (${spct(s.median_return)})`], ["Median return, annualised", spct(s.median_return_annualized)],
    ["Mean terminal", money(s.mean_price)], ["P5 to P95", `${money0(s.p5)} to ${money0(s.p95)}`], ["P25 to P75", `${money0(s.p25)} to ${money0(s.p75)}`],
    ["Mode-path price", money(rc.mode_path_price)],
    ["", ""],
    ["P(at least bear)", `${pct(s.p_at_least_bear)} ± ${(s.se.p_at_least_bear * 100).toFixed(2)}pp`],
    ["P(at least base)", `${pct(s.p_at_least_base)} ± ${(s.se.p_at_least_base * 100).toFixed(2)}pp`],
    ["P(at least bull)", `${pct(s.p_at_least_bull)} ± ${(s.se.p_at_least_bull * 100).toFixed(2)}pp`],
    ["P(below bear)", pct(s.below_bear)],
    ["", ""],
    ["Revenue CAGR P10 / P50 / P90", `${pct(dq.rev_cagr.p10, 1)} / ${pct(dq.rev_cagr.p50, 1)} / ${pct(dq.rev_cagr.p90, 1)}`],
    ["EBITDA margin P10 / P50 / P90", `${pct(dq.ebitda_margin.p10, 2)} / ${pct(dq.ebitda_margin.p50, 2)} / ${pct(dq.ebitda_margin.p90, 2)}`],
    ["Exit multiple P10 / P50 / P90", `${mult(dq.ev_ebitda_multiple.p10)} / ${mult(dq.ev_ebitda_multiple.p50)} / ${mult(dq.ev_ebitda_multiple.p90)}`],
    ["", ""],
    ["Market-implied margin at spot", pi.implied_margin_at_mode_growth_and_multiple === null ? "no solution" : pct(pi.implied_margin_at_mode_growth_and_multiple, 2)],
    ["Market-implied growth at spot", pi.implied_growth_at_mode_margin_and_multiple === null ? "no solution" : pct(pi.implied_growth_at_mode_margin_and_multiple, 1)],
    ["Market-implied multiple at spot", pi.implied_multiple_at_mode_growth_and_margin === null ? "no solution" : mult(pi.implied_multiple_at_mode_growth_and_margin)],
    ["Base needs multiple at P50 margin", mult(rc.cases.base.implied_multiple_at_mode_margin)],
    ["Base needs margin at P50 multiple", pct(rc.cases.base.implied_margin_at_mode_multiple, 2)],
  ];
  if (r.fcf_bridge) rows.push(["", ""], ["FCF bridge median", `${money(r.fcf_bridge.median_price)} (${spct(r.fcf_bridge.gap_to_ebitda_bridge)} vs EBITDA bridge)`], ["FCF bridge P(at least base)", pct(r.fcf_bridge.p_at_least_base)]);
  if (r.dcf_path) rows.push(["", ""], ["Three-statement path price", money(r.dcf_path.price)], ["Its percentile in the distribution", `${Math.round(r.dcf_path.percentile * 100)}th`], ["Paths at or above it", pct(r.dcf_path.p_at_least)]);
  return rows;
}
function assessorVerdict(r) {
  const s = r.stats, m = r.meta, rc = r.reconciliation;
  const pb = s.p_at_least_base, gap = rc.base_gap_pct;
  let v;
  if (pb >= 0.5) v = `The base target is at or below the simulated median: ${pct(pb)} of paths reach it.`;
  else if (pb >= 0.3) v = `The base target sits above the median; ${pct(pb)} of paths reach it. Defensible if the thesis moves a driver the model holds at its P50.`;
  else v = `The base target is in the top ${pct(pb, 0)} of outcomes. It needs ${mult(rc.cases.base.implied_multiple_at_mode_margin)} at the P50 margin or a ${pct(rc.cases.base.implied_margin_at_mode_multiple, 2)} margin at the P50 multiple.`;
  v += ` Downside: ${pct(s.below_bear)} of paths finish below the bear target.`;
  if (r.fcf_bridge && Math.abs(r.fcf_bridge.gap_to_ebitda_bridge) > 0.15) v += ` The FCF bridge disagrees with the EBITDA bridge by ${spct(r.fcf_bridge.gap_to_ebitda_bridge)}; say which one the thesis rests on.`;
  return v;
}
async function exportAssessor() {
  if (!lastResult) return;
  const r = lastResult, m = r.meta, rows = assessorRows(r, lastPayload), verdict = assessorVerdict(r);
  // 4:3 slide: 1600 x 1200 logical, rendered at 2x
  const scale = 2, Wc = 1600, Hc = 1200, left = 48, colW = 600, lineH = 21;
  const canvas = document.createElement("canvas"); canvas.width = Wc * scale; canvas.height = Hc * scale;
  const ctx = canvas.getContext("2d"); ctx.scale(scale, scale);
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, Wc, Hc);
  ctx.fillStyle = "#1b4b8a"; ctx.fillRect(0, 0, Wc, 6);
  ctx.fillStyle = C.text; ctx.font = "600 26px IBM Plex Sans, system-ui, sans-serif";
  ctx.fillText(`${m.company_name} (${m.ticker}): pitch assessment`, left, 52);
  // verdict
  ctx.fillStyle = "#e8eff8"; ctx.fillRect(left, 70, Wc - 2 * left, 54);
  ctx.fillStyle = C.text; ctx.font = "500 14px IBM Plex Sans, system-ui, sans-serif";
  const words = verdict.split(" "); let line = "", y = 92; const maxW = Wc - 2 * left - 24;
  for (const w of words) { const t = line ? line + " " + w : w; if (ctx.measureText(t).width > maxW) { ctx.fillText(line, left + 12, y); line = w; y += 19; } else line = t; }
  ctx.fillText(line, left + 12, y);
  // figures, left column
  let ty = 160;
  rows.forEach(([k, v]) => {
    if (k) {
      ctx.fillStyle = C.muted; ctx.font = "12.5px IBM Plex Sans, system-ui, sans-serif"; ctx.fillText(k, left, ty);
      ctx.fillStyle = C.text; ctx.font = "600 12.5px IBM Plex Mono, ui-monospace, monospace"; ctx.textAlign = "right"; ctx.fillText(v, left + colW, ty); ctx.textAlign = "left";
      ctx.fillStyle = C.grid; ctx.fillRect(left, ty + 6, colW, 1);
    }
    ty += lineH;
  });
  // right column: the path chart over the probability bars
  const cx = left + colW + 40, cw = Wc - cx - left;
  const pathSvg = pathChart(r, lastPath), pathH = cw * 560 / 1120;
  const [pathImg, barsImg] = await Promise.all([svgStringToImage(pathSvg), svgToImage($("chartBars").querySelector("svg"))]);
  ctx.drawImage(pathImg, cx, 150, cw, pathH);
  const barsW = cw * 0.72, barsH = barsW * H / W;
  ctx.drawImage(barsImg, cx + (cw - barsW) / 2, 150 + pathH + 20, barsW, barsH);
  ctx.fillStyle = C.faint; ctx.font = "11px IBM Plex Mono, ui-monospace, monospace";
  ctx.fillText(`Model output, not investment advice. ${new Date().toISOString().slice(0, 10)}.`, left, Hc - 18);
  const a = document.createElement("a"); a.href = canvas.toDataURL("image/png"); a.download = `${m.ticker}_pitch_assessment.png`; a.click();
  const tsv = rows.filter(([k]) => k).map(([k, v]) => `${k}\t${v}`).join("\n") + `\nVerdict\t${verdict}`;
  try { await navigator.clipboard.writeText(tsv); $("assessBtn").textContent = "Image saved, table copied"; }
  catch (_) { $("assessBtn").textContent = "Image saved"; }
  setTimeout(() => { $("assessBtn").textContent = "Pitch assessor"; }, 2200);
}

/* Wiring */
function showError(msg) { const el = $("formError"); el.textContent = msg; el.hidden = false; }
async function run() {
  const btn = $("runBtn"); $("formError").hidden = true;
  let payload; try { payload = readForm(); } catch (err) { showError(err.message); return; }
  btn.disabled = true; btn.classList.add("busy"); btn.textContent = "Running";
  try {
    const res = await fetch("/api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await res.json().catch(() => ({ error: `Server returned ${res.status}` }));
    if (!res.ok) throw new Error(data.error || `Server returned ${res.status}`);
    lastResult = data; lastPayload = payload; writeUrl(payload); render(data, payload);
  } catch (err) { showError(err.message); }
  finally { btn.disabled = false; btn.classList.remove("busy"); btn.textContent = "Run simulation"; }
}
form.addEventListener("submit", (e) => { e.preventDefault(); run(); });
$("autofillBtn").addEventListener("click", autofill);
form.ticker.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); autofill(); } });
form.current_price.addEventListener("input", recomputeEV);
["p10", "p50", "p90"].forEach((p) => form[`ev_ebitda_multiple_${p}`].addEventListener("input", () => { multipleEdited = true; }));
TRI.forEach((d) => form[`${d}_p50`].addEventListener("input", () => refreshChipState(d)));
NORM.forEach((d) => form[`${d}_mean`].addEventListener("input", () => refreshChipState(d)));
MECHS.forEach((m) => { const d = mechEl(m); form[`use_${m}`].addEventListener("change", (e) => { d.open = e.target.checked; }); });
$("corrReset").addEventListener("click", () => { buildCorrGrid(DEFAULT_CORR); $("corrStatus").textContent = "Reset."; $("corrStatus").className = "hint"; });
$("loadPreset").addEventListener("click", () => { fillForm(PRESET); clearProvenance(); applyMechanisms(null); $("lookupMsg").hidden = true; $("formError").hidden = true; });
$("shareBtn").addEventListener("click", async () => {
  if (lastPayload) writeUrl(lastPayload);
  try { await navigator.clipboard.writeText(location.href); $("shareBtn").textContent = "Link copied"; } catch (_) { $("shareBtn").textContent = "Copy from address bar"; }
  setTimeout(() => { $("shareBtn").textContent = "Copy link"; }, 1800);
});
$("pinBtn").addEventListener("click", () => {
  if (pinned) { pinned = null; $("pinBtn").textContent = "Pin result"; $("pinBtn").classList.remove("pinned"); }
  else if (lastResult) { pinned = { label: `${lastResult.meta.ticker} ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`, stats: lastResult.stats }; $("pinBtn").textContent = "Unpin"; $("pinBtn").classList.add("pinned"); }
  if (lastResult) compareStrip(lastResult);
});
$("exportPng").addEventListener("click", () => exportPng().catch((e) => showError(e.message)));
$("assessBtn").addEventListener("click", async () => { lastPath = await fetchPath(lastResult.meta.ticker); exportAssessor().catch((e) => showError(e.message)); });
$("exportJson").addEventListener("click", () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify({ inputs: lastPayload, result: lastResult }, null, 2)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${lastResult.meta.ticker}_monte_carlo.json`; a.click(); URL.revokeObjectURL(a.href);
});
$("logRun").addEventListener("click", async () => {
  if (!lastResult || !lastPayload) return;
  const s = lastResult.stats, m = lastResult.meta;
  const entry = { ticker: m.ticker, as_of: new Date().toISOString().slice(0, 10), horizon_years: m.horizon_years, spot: m.current_price,
    targets: { bear: m.bear_price, base: m.base_price, bull: m.bull_price }, driver_specs: lastPayload.driver_specs,
    mechanisms: m.mechanisms, median: s.median_price, p5: s.p5, p95: s.p95 };
  try {
    const res = await fetch("/api/calibration-log", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(entry) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Server returned ${res.status}`);
    $("logRun").textContent = "Logged"; setTimeout(() => { $("logRun").textContent = "Log this run"; }, 1800);
  } catch (err) { showError(err.message); }
});

/* Boot */
buildCorrGrid(DEFAULT_CORR); applyMechanisms(null);
const fromUrl = readUrl();
if (fromUrl) { fillForm(fromUrl); run(); }
else { form.n_simulations.value = PRESET.n_simulations; form.horizon_years.value = PRESET.horizon_years; form.random_seed.value = PRESET.random_seed; form.copula.value = "t"; form.ticker.focus(); }
