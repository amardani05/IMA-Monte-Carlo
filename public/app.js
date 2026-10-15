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
const MECHS = ["backlog", "fcf_bridge", "justified_multiple", "cost_model"];
const MECH_LABEL = { backlog: "backlog", fcf_bridge: "FCF", justified_multiple: "ROIC and EVA", cost_model: "cost structure" };
/* What an added driver starts from when the filings propose nothing: stated assumptions, each named in its note */
const EXTRA_DEFAULT = {
  backlog: { conversion: { type: "metalog", p10: 0.55, p50: 0.70, p90: 0.85 }, note: "No remaining-performance-obligation figure from the filings. Enter backlog from the 10-K. Conversion 0.55 / 0.70 / 0.85 is an assumption, not a filed figure." },
  fcf_bridge: { note: "No usable cash-flow history from the filings. Enter the FCF margin and EV/FCF you underwrite." },
  justified_multiple: { fade: { type: "metalog", p10: 0.25, p50: 0.5, p90: 0.75 }, tax_rate: 0.25, da_share: 0.15, long_run_growth: 0.03, premium_reversion: 0.25,
    note: "No filed return on capital to start from. Enter the ROIC and WACC you underwrite; tax 25% and D&A 15% of EBITDA are placeholders." },
  cost_model: { fixed_cost_growth: { type: "normal", mean: 0.03, std: 0.03 }, note: "Cost of revenue is not tagged in enough years to split the margin. Enter gross margin and fixed cost by hand." },
};

const C = {
  blue: "#2a63b0", red: "#b1271f", amber: "#9a6300", green: "#1f6d44", purple: "#574ca6",
  text: "#101820", text2: "#364150", muted: "#64707e", faint: "#8b959f", grid: "#e9edf2", axis: "#c9d1db", panel: "#ffffff",
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

/* Added drivers: the four mechanisms, each one a group of P10/P50/P90 drivers beside the base five.
   A group is on when its fieldset is shown; the payload carries only the groups that are on. */
let lastExtra = null;
const mechEl = (m) => document.querySelector(`fieldset.extra[data-mech="${m}"]`);
const mechOn = (m) => !mechEl(m).hidden;
const setVal = (name, v) => { form[name].value = v === null || v === undefined || v === "" ? "" : round(Number(v), 6); };
function setTriple(prefix, spec) {
  ["p10", "p50", "p90"].forEach((k) => setVal(`${prefix}_${k}`, spec ? spec[k] : null));
}
function showMech(m, on) {
  mechEl(m).hidden = !on;
  document.querySelector(`#addDriver [data-add="${m}"]`).hidden = on;
  const n = MECHS.filter(mechOn).length;
  $("addedHead").hidden = n === 0;
  $("stackHint").hidden = n < 2;
  $("addDriver").querySelector(".cap").textContent = n ? "Add another" : "Add a driver";
  $("addDriver").hidden = n === MECHS.length;
}
function setMech(m, block, on) {
  if (block) {
    if (m === "cost_model") {
      setTriple("cm_gm", block.gross_margin); setVal("cm_fixed", block.fixed_cost_m);
      setVal("cm_fcg_mean", block.fixed_cost_growth?.mean ?? 0.03); setVal("cm_fcg_std", block.fixed_cost_growth?.std ?? 0.03);
    } else if (m === "backlog") {
      setVal("bl_m", block.backlog_m); setTriple("bl", block.conversion);
    } else if (m === "fcf_bridge") {
      setTriple("fb_fm", block.fcf_margin); setTriple("fb_mult", block.ev_fcf_multiple);
    } else if (m === "justified_multiple") {
      // a number is a point input (P50 only); an object is a drawn driver
      ["roic", "wacc", "fade"].forEach((k) => { const v = block[k]; setTriple(`jm_${k}`, v !== null && typeof v === "object" ? v : (v === null || v === undefined ? null : { p50: v })); });
      setVal("jm_tax", block.tax_rate); setVal("jm_da", block.da_share); setVal("jm_lrg", block.long_run_growth ?? 0.03); setVal("jm_prem", block.premium_reversion ?? 0.25);
      const d = mechEl(m);
      d.dataset.currentGrowth = block.current_growth ?? ""; d.dataset.currentMultiple = block.current_multiple ?? "";
      d.dataset.investedCapital = block.invested_capital_m ?? "";
      updateEva();
    }
  }
  mechEl(m).querySelectorAll("input").forEach((el) => { if (el.value !== "") el.closest("label")?.classList.remove("needs-input"); });
  showMech(m, Boolean(on));
}
/* P10/P50/P90 inputs to a spec. With allowPoint, a P50 alone is a number. */
function readTriple(prefix, label, { kind = "metalog", allowPoint = false, optional = false } = {}) {
  const els = ["p10", "p50", "p90"].map((k) => form[`${prefix}_${k}`]), raw = els.map((el) => el.value.trim());
  els.forEach((el) => el.classList.remove("invalid"));
  if (raw.every((v) => v === "")) {
    if (optional) return undefined;
    els[1].classList.add("invalid"); throw new Error(`${label} needs ${allowPoint ? "a P50, or a P10, P50 and P90" : "a P10, P50 and P90"}.`);
  }
  if (allowPoint && raw[0] === "" && raw[2] === "") {
    const v = Number(raw[1]); if (!Number.isFinite(v)) { els[1].classList.add("invalid"); throw new Error(`${label} P50 needs a number.`); }
    return v;
  }
  const v = raw.map((x) => (x === "" ? NaN : Number(x)));
  if (!v.every(Number.isFinite)) { els.forEach((el, i) => { if (!Number.isFinite(v[i])) el.classList.add("invalid"); }); throw new Error(`${label} needs a P10, P50 and P90${allowPoint ? ", or a P50 alone" : ""}.`); }
  if (!(v[0] < v[1] && v[1] < v[2])) { els.forEach((el) => el.classList.add("invalid")); throw new Error(`${label}: P10 must be below P50, and P50 below P90.`); }
  const spec = { type: kind, p10: v[0], p50: v[1], p90: v[2] };
  if (kind === "log_metalog") spec.lower = 0;
  return spec;
}
function readMech(m) {
  if (!mechOn(m)) return null;
  if (m === "cost_model") return { gross_margin: readTriple("cm_gm", "Gross margin"), fixed_cost_m: num("cm_fixed", "Fixed cost ($M)"),
    fixed_cost_growth: { type: "normal", mean: num("cm_fcg_mean", "Fixed cost growth mean"), std: num("cm_fcg_std", "Fixed cost growth std") } };
  if (m === "backlog") return { backlog_m: num("bl_m", "Backlog ($M)"), conversion: readTriple("bl", "Backlog conversion") };
  if (m === "fcf_bridge") return { fcf_margin: readTriple("fb_fm", "FCF margin"), ev_fcf_multiple: readTriple("fb_mult", "EV/FCF", { kind: "log_metalog" }) };
  if (m === "justified_multiple") {
    const d = mechEl(m);
    const block = { roic: readTriple("jm_roic", "ROIC", { allowPoint: true }), wacc: readTriple("jm_wacc", "WACC", { allowPoint: true }) };
    const fade = readTriple("jm_fade", "Growth fade", { allowPoint: true, optional: true });
    if (fade !== undefined) block.fade = fade;
    Object.assign(block, { tax_rate: num("jm_tax", "Tax rate"), da_share: num("jm_da", "D&A / EBITDA"),
      long_run_growth: num("jm_lrg", "Long-run growth"), premium_reversion: num("jm_prem", "Premium reversion") });
    const cg = parseFloat(d.dataset.currentGrowth), cm = parseFloat(d.dataset.currentMultiple), ic = parseFloat(d.dataset.investedCapital);
    block.current_growth = Number.isFinite(cg) ? cg : readQ("rev_cagr")[1];
    block.current_multiple = Number.isFinite(cm) ? cm : readQ("ev_ebitda_multiple")[1];
    if (Number.isFinite(ic)) block.invested_capital_m = ic;
    return block;
  }
  return null;
}
const smoney0 = (v) => (v < 0 ? "-" : "+") + money0(Math.abs(v));
function updateEva() {
  const d = mechEl("justified_multiple"), el = $("evaLine");
  const r = parseFloat(form.jm_roic_p50.value), w = parseFloat(form.jm_wacc_p50.value), ic = parseFloat(d.dataset.investedCapital);
  const filed = lastExtra?.justified_multiple;
  let txt;
  if (Number.isFinite(r) && Number.isFinite(w)) {
    txt = `At P50: ROIC ${pct(r, 1)} less WACC ${pct(w, 1)} is a ${spct(r - w, 1)} spread`;
    txt += Number.isFinite(ic) ? `, EVA ${smoney0((r - w) * ic)}M a year on ${money0(ic)}M invested capital.` : ".";
  } else txt = "Enter a ROIC P50 to see the spread over WACC and the EVA.";
  if (filed && filed.filed_roic !== null && filed.filed_roic !== undefined) {
    txt += ` Filed: ROIC ${pct(filed.filed_roic, 1)}${filed.eva_m !== null && filed.eva_m !== undefined ? `, EVA ${smoney0(filed.eva_m)}M` : ""}.`;
  }
  el.textContent = txt;
  el.classList.toggle("neg", Number.isFinite(r) && Number.isFinite(w) && r < w);
}
/* Auto-fill proposals: fill every group's inputs, leave them off, and mark what only the analyst can supply */
function applyExtraDrivers(x) {
  lastExtra = x || null;
  MECHS.forEach((m) => {
    const d = mechEl(m), g = x?.[m] || null, base = EXTRA_DEFAULT[m];
    d.querySelectorAll("input").forEach((el) => { el.value = ""; el.classList.remove("invalid"); el.closest("label")?.classList.remove("needs-input"); });
    d.querySelectorAll(".src-badge").forEach((b) => b.remove());
    const block = { ...base };
    if (g) Object.entries(g).forEach(([k, v]) => { if (v !== null && v !== undefined) block[k] = v; });
    if (m === "justified_multiple" && !g?.invested_capital_m) block.invested_capital_m = null;
    setMech(m, block, false);
    const filed = Boolean(g?.filed);
    const note = d.querySelector(".mech-note[data-note]"); note.textContent = (g?.note) || base.note; note.classList.toggle("weak", !filed);
    d.querySelectorAll("input").forEach((el) => {
      const optional = /^jm_(roic|wacc|fade)_p(10|90)$/.test(el.name);    // a P50 alone is a valid point input
      if (el.value === "" && !optional) el.closest("label")?.classList.add("needs-input");
    });
    const chip = document.querySelector(`#addDriver [data-add="${m}"]`);
    chip.classList.toggle("unfiled", !filed); chip.title = filed ? "Proposed from the filings; you can edit every value" : "Not in the filings: offered for you to fill in";
    chip.setAttribute("aria-label", `Add ${MECH_LABEL[m]}${filed ? "" : ", by hand"}`);
    if (m === "backlog" && filed && g.backlog_m) document.querySelector('label[data-field="bl_m"]').prepend(badge("SEC", g.note || "remaining performance obligation", ""));
    if (m === "justified_multiple") {
      const wn = $("jmWidthNote"); wn.textContent = g?.width_note || ""; wn.hidden = !g?.width_note;
      const rl = document.querySelector('label[data-field="jm_roic"]');
      if (g && g.roic) rl.prepend(badge("derived", "ROIC = NOPAT / invested capital from the latest 10-K; P10 and P90 are a stated width", "derived"));
      updateEva();
    }
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
  applyExtraDrivers(data.extra_drivers || null);
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
/* Dense panels for the assessor and the sensitivity chart: crisp hairlines, small monospace labels,
   numbers beside every mark. Light theme throughout. */
const SANS = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif", MONO = "IBM Plex Mono, Menlo, Consolas, monospace";
const fx = (v) => String(Math.round(v * 10) / 10);
function tx(x, y, s, o = {}) {
  const { size = 10.5, fill = C.text, anchor = "start", weight = 400, font = MONO, ls = 0 } = o;
  return `<text x="${fx(x)}" y="${fx(y)}" fill="${fill}" font-family="${font}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}"${ls ? ` letter-spacing="${ls}"` : ""}>${esc(s)}</text>`;
}
const hl = (x1, x2, y, stroke, extra = "") => `<line x1="${fx(x1)}" y1="${fx(y)}" x2="${fx(x2)}" y2="${fx(y)}" stroke="${stroke}" shape-rendering="crispEdges"${extra}/>`;
const vl = (x, y1, y2, stroke, extra = "") => `<line x1="${fx(x)}" y1="${fx(y1)}" x2="${fx(x)}" y2="${fx(y2)}" stroke="${stroke}" shape-rendering="crispEdges"${extra}/>`;
const box = (x, y, w, h, fill, extra = "") => `<rect x="${fx(x)}" y="${fx(y)}" width="${fx(Math.max(w, 0))}" height="${fx(Math.max(h, 0))}" fill="${fill}"${extra}/>`;
const HEAD = 21;
function panel(w, h, title, right) {
  let g = box(0, 0, w, h, "#ffffff", ` stroke="${C.axis}" shape-rendering="crispEdges"`);
  g += box(0.5, 0.5, w - 1, HEAD - 1, "#f1f4f8") + hl(0, w, HEAD - 0.5, C.axis);
  g += tx(9, 14.5, title.toUpperCase(), { size: 10, weight: 600, ls: 0.6, fill: "#1b4b8a" });
  if (right) g += tx(w - 9, 14.5, right, { size: 9.5, fill: C.muted, anchor: "end" });
  return g;
}
const tickDigits = (ticks) => (ticks.length > 1 && ticks[1] - ticks[0] < 0.01 ? 1 : 0);
/* Ticks on a 1, 2, 2.5, 5 ladder, as many as fit under `count`: denser than niceTicks */
function denseTicks(min, max, count = 6) {
  const span = max - min || 1, mag = 10 ** Math.floor(Math.log10(span / count));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((st) => span / st <= count) || 10 * mag, out = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + 1e-9; v += step) out.push(Math.round(v / step) * step || 0);
  return out;
}

/* Probability per bin, coloured by scenario band, with the cumulative line on the right axis */
function histPanel(r, w, h) {
  const m = r.meta, s = r.stats, e = r.histogram.edges, dens = r.histogram.density, dfcf = r.histogram.density_fcf;
  // the axis runs to P98 so a long right tail does not squeeze the body; what lies beyond is stated
  const i98 = r.cdf.y.findIndex((y) => y >= 0.98), x98 = i98 >= 0 ? r.cdf.x[i98] : e[e.length - 1];
  const bw0 = e[1] - e[0], nVis = Math.min(dens.length, Math.ceil((x98 - e[0]) / bw0));
  const k = nVis > 64 ? 2 : 1, nb = Math.ceil(nVis / k), bins = [];
  for (let i = 0; i < nb && (i + 1) * k < e.length; i++) {
    let p = 0, q = 0;
    for (let j = 0; j < k; j++) { p += dens[i * k + j] * bw0; if (dfcf) q += dfcf[i * k + j] * bw0; }
    bins.push({ lo: e[i * k], hi: e[(i + 1) * k], p, q });
  }
  const xMin = e[0], xMax = e[Math.min(bins.length * k, e.length - 1)];
  const beyond = 1 - (r.cdf.y[r.cdf.x.findIndex((x) => x > xMax)] ?? 1);
  const L = { l: 48, r: 44, t: HEAD + 30, b: 30 }, iw = w - L.l - L.r, ih = h - L.t - L.b;
  const sx = (v) => L.l + ((v - xMin) / (xMax - xMin)) * iw;
  const yMax = Math.max(...bins.map((b) => Math.max(b.p, b.q))) * 1.12 || 0.01;
  const sy = (v) => L.t + ih - (v / yMax) * ih, sc = (v) => L.t + ih - v * ih;
  const bandOf = (v) => (v < m.bear_price ? C.red : v < m.base_price ? C.amber : v < m.bull_price ? C.blue : C.green);
  let g = panel(w, h, "Terminal price probability", `${m.horizon_years}-year horizon, ${m.n_simulations.toLocaleString()} paths, ${money(bw0 * k, bw0 * k < 10 ? 2 : 0)} bins`);
  const yt = denseTicks(0, yMax, 5), yd = tickDigits(yt);
  yt.forEach((v) => { g += hl(L.l, L.l + iw, sy(v), C.grid) + tx(L.l - 6, sy(v) + 3.5, pct(v, yd), { size: 9.5, fill: C.muted, anchor: "end" }); });
  [0, 0.25, 0.5, 0.75, 1].forEach((v) => { g += tx(L.l + iw + 6, sc(v) + 3.5, pct(v, 0), { size: 9.5, fill: C.faint }); });
  denseTicks(xMin, xMax, 10).forEach((v) => { g += vl(sx(v), L.t, L.t + ih, "#f1f3f7") + tx(sx(v), L.t + ih + 13, money0(v), { size: 9.5, fill: C.muted, anchor: "middle" }); });
  bins.forEach((b) => {
    if (b.p <= 0) return;
    const x = sx(b.lo) + 0.4, bw = Math.max(sx(b.hi) - sx(b.lo) - 0.8, 0.6);
    g += box(x, sy(b.p), bw, L.t + ih - sy(b.p), bandOf((b.lo + b.hi) / 2), ' opacity="0.82"');
  });
  if (dfcf) {
    let d = `M${fx(sx(bins[0].lo))},${fx(sy(0))}`;
    bins.forEach((b) => { d += ` V${fx(sy(b.q))} H${fx(sx(b.hi))}`; });
    g += `<path d="${d} V${fx(sy(0))}" fill="none" stroke="${C.purple}" stroke-width="1.3"/>`;
  }
  const pts = r.cdf.x.map((v, i) => (v <= xMax ? `${fx(sx(v))},${fx(sc(r.cdf.y[i]))}` : null)).filter(Boolean).join(" ");
  g += `<polyline points="${pts}" fill="none" stroke="${C.text}" stroke-width="1.1" opacity="0.75"/>`;
  g += hl(L.l, L.l + iw, L.t + ih, C.axis) + vl(L.l, L.t, L.t + ih, C.axis) + vl(L.l + iw, L.t, L.t + ih, C.axis);
  // the four scenario bands as a ruler above the plot, each with its probability
  const bands = [["below bear", 0, m.bear_price, C.red, s.below_bear], ["bear to base", m.bear_price, m.base_price, C.amber, s.bear_to_base],
    ["base to bull", m.base_price, m.bull_price, C.blue, s.base_to_bull], ["above bull", m.bull_price, Infinity, C.green, s.above_bull]];
  bands.forEach(([label, lo, hi, color, p]) => {
    const a = Math.max(lo, xMin), b = Math.min(hi, xMax);
    const x1 = b > a ? sx(a) : (lo >= xMax ? L.l + iw - 2 : L.l), x2 = b > a ? sx(b) : x1 + 2;
    g += box(x1, HEAD + 22, x2 - x1, 3, color);
    const full = `${label} ${pct(p, 1)}`, short = pct(p, 0), room = x2 - x1;
    const txt = room > full.length * 5.7 + 6 ? full : room > short.length * 5.7 + 2 ? short : "";
    if (txt) g += tx((x1 + x2) / 2, HEAD + 17, txt, { size: 9.5, fill: color, anchor: "middle", weight: 600 });
  });
  // targets, spot and medians; labels staggered so close marks stay legible
  const marks = [["BEAR", m.bear_price, C.red, "4 3"], ["BASE", m.base_price, C.amber, "4 3"], ["BULL", m.bull_price, C.green, "4 3"],
    ["SPOT", m.current_price, C.text, ""], ["MEDIAN", s.median_price, C.purple, "1.5 2"]];
  if (r.fcf_bridge) marks.push(["FCF MED", r.fcf_bridge.median_price, C.purple, "6 2"]);
  const rows = [], shown = marks.filter(([, v]) => v >= xMin && v <= xMax).sort((a, b) => a[1] - b[1]);
  shown.forEach(([, v, color, dash]) => { g += vl(sx(v), L.t, L.t + ih, color, ` stroke-width="1.2"${dash ? ` stroke-dasharray="${dash}"` : ""}`); });
  shown.forEach(([label, v, color]) => {
    const x = sx(v), txt = `${label} ${money0(v)}`, wTxt = txt.length * 5.45 + 8;
    const x0 = x + wTxt > L.l + iw ? x - wTxt : x;                  // near the right edge the tag sits left of its line
    let row = rows.findIndex((end) => end < x0 - 1); if (row < 0) { row = rows.length; rows.push(0); }
    rows[row] = x0 + wTxt;
    const y = L.t + 11 + row * 12;
    g += box(x0 + 1, y - 9, wTxt - 1, 12, "#ffffff", ` stroke="${color}" stroke-width="0.6"`) + tx(x0 + 4, y, txt, { size: 9, fill: color, weight: 600 });
  });
  // legend
  let lx = L.l;
  const ly = h - 8;
  g += tx(lx, ly, beyond > 0.0005 ? `TERMINAL PRICE ($); ${pct(beyond, 1)} OF PATHS ABOVE ${money0(xMax)}` : "TERMINAL PRICE ($)", { size: 9, fill: C.faint, ls: 0.4 });
  lx = L.l + iw;
  const leg = [["cumulative, right axis", (x) => hl(x, x + 14, ly - 3, C.text, ' stroke-width="1.2"')]];
  if (dfcf) leg.unshift(["FCF bridge", (x) => hl(x, x + 14, ly - 3, C.purple, ' stroke-width="1.4"')]);
  leg.unshift(["share of paths per bin, left axis", (x) => box(x, ly - 7, 9, 8, C.blue, ' opacity="0.82"')]);
  leg.reverse().forEach(([label, mark]) => { const tw = label.length * 5.4; g += tx(lx, ly, label, { size: 9, fill: C.muted, anchor: "end" }); lx -= tw + 4; g += mark(lx - 14); lx -= 26; });
  return g;
}

/* Each driver held at its P10, then its P90, on the same paths; bars from the run's own median (and P(base)) */
const DRV_FMT = {
  rev_cagr: (v) => pct(v, 1), ebitda_margin: (v) => pct(v, 2), gross_margin: (v) => pct(v, 1), fc_growth: (v) => spct(v, 1),
  ev_ebitda: (v, u) => (u === "ratio" ? v.toFixed(2) + "x" : mult(v)), net_debt_chg: (v) => spct(v, 1), dilution: (v) => spct(v, 1),
  backlog_conv: (v) => pct(v, 0), roic: (v) => pct(v, 1), wacc: (v) => pct(v, 1), fade: (v) => v.toFixed(2), fcf_margin: (v) => pct(v, 1), ev_fcf: (v) => mult(v),
};
const fmtDrv = (row, v) => (DRV_FMT[row.key] || ((x) => x.toFixed(3)))(v, row.unit);
const T_LO = "#9fbadc", T_HI = "#1b4b8a";
function tornadoPanel(r, w, h, { metric = "median", bare = false } = {}) {
  const sw = r.swing || {};
  const secs = [["ebitda", "EBITDA bridge"], ["fcf", "FCF bridge"]].filter(([b]) => sw[b]).map(([b, label]) => ({ b, label, ...sw[b] }));
  const compact = w < 800, multi = secs.length > 1, metrics = compact ? [metric] : ["median", "p_base"];
  const top = (bare ? 4 : HEAD) + 18, bottom = 22;
  const n = secs.reduce((a, sec) => a + sec.rows.length + (multi ? 1 : 0), 0) || 1;
  const rowH = Math.min(24, (h - top - bottom) / n);
  // columns: name, P10, P90, then for each metric a bar area and its swing
  const cols = compact ? { name: 8, bars: [[150, w - 64]], swing: [w - 8] }
    : { name: 9, p10: 226, p90: 290, bars: [[306, 634], [714, 958]], swing: [690, w - 9] };
  const acc = { median: (row, end) => row[`median_${end}`], p_base: (row, end) => row[`p_base_${end}`] };
  const centre = { median: (sec) => sec.median, p_base: (sec) => sec.p_base };
  const scales = metrics.map((mt, i) => {
    const vals = secs.flatMap((sec) => [centre[mt](sec), ...sec.rows.flatMap((row) => [acc[mt](row, "lo"), acc[mt](row, "hi")])]);
    let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi - lo) * 0.16 || (mt === "median" ? 1 : 0.01);
    lo -= pad; hi += pad; if (mt === "p_base") { lo = Math.max(lo, -0.02); hi = Math.min(hi, 1.02); }
    const [a, b] = cols.bars[i];
    return { mt, lo, hi, a, b, sx: (v) => a + ((v - lo) / (hi - lo)) * (b - a) };
  });
  const fmtV = { median: (v) => money0(v), p_base: (v) => pct(v, 0) };
  const fmtS = { median: (d) => (d >= 0 ? "+" : "-") + money0(Math.abs(d)), p_base: (d) => (d >= 0 ? "+" : "-") + (Math.abs(d) * 100).toFixed(1) + "pp" };
  let g = bare ? "" : panel(w, h, "Driver sensitivity", "each driver held at its P10, then its P90, the rest as drawn; bars start at the run's median and P(base)");
  // column heads
  const hy = top - 6;
  g += tx(cols.name, hy, "DRIVER", { size: 9, fill: C.faint, ls: 0.4 });
  if (!compact) { g += tx(cols.p10, hy, "P10", { size: 9, fill: C.faint, anchor: "end" }) + tx(cols.p90, hy, "P90", { size: 9, fill: C.faint, anchor: "end" }); }
  scales.forEach((sc, i) => {
    const head = sc.mt === "median" ? "MEDIAN PRICE AT P10 | P90" : "P(AT LEAST BASE) AT P10 | P90";
    g += tx((sc.a + sc.b) / 2, hy, head, { size: 9, fill: C.faint, anchor: "middle", ls: 0.4 }) + tx(cols.swing[i], hy, "SWING", { size: 9, fill: C.faint, anchor: "end" });
    denseTicks(sc.lo, sc.hi, compact ? 6 : 5).forEach((v) => {
      if (sc.sx(v) < sc.a - 0.5 || sc.sx(v) > sc.b + 0.5) return;
      g += vl(sc.sx(v), top, h - bottom, "#eef1f5") + tx(sc.sx(v), h - bottom + 12, fmtV[sc.mt](v), { size: 9, fill: C.muted, anchor: "middle" });
    });
  });
  g += hl(4, w - 4, top - 0.5, C.axis) + hl(4, w - 4, h - bottom + 0.5, C.axis);
  let y = top;
  secs.forEach((sec) => {
    if (multi) {
      g += box(1, y, w - 2, rowH, "#f6f8fb");
      g += tx(cols.name, y + rowH * 0.7, `${sec.label.toUpperCase()}: MEDIAN ${money0(sec.median)}, P(BASE) ${pct(sec.p_base, 1)}`, { size: 9, weight: 600, fill: C.text2, ls: 0.3 });
      y += rowH;
    }
    const y0 = y;
    sec.rows.forEach((row, i) => {
      const cy = y + rowH / 2, bh = Math.max(Math.min(rowH * 0.6, 14), 3), fs = Math.min(10.5, rowH * 0.62);
      if (i % 2 === 1) g += box(1, y, w - 2, rowH, "#fafbfd");
      const tip = `${row.name} at P10 ${fmtDrv(row, row.lo)}: median ${money(row.median_lo)}, P(base) ${pct(row.p_base_lo)}. At P90 ${fmtDrv(row, row.hi)}: median ${money(row.median_hi)}, P(base) ${pct(row.p_base_hi)}.`;
      g += `<g><title>${esc(tip)}</title>`;
      g += tx(cols.name, cy + fs * 0.36, row.name, { size: fs, fill: C.text });
      if (!compact) {
        g += tx(cols.p10, cy + fs * 0.36, fmtDrv(row, row.lo), { size: fs, fill: C.muted, anchor: "end" });
        g += tx(cols.p90, cy + fs * 0.36, fmtDrv(row, row.hi), { size: fs, fill: C.muted, anchor: "end" });
      }
      scales.forEach((sc, j) => {
        const c = sc.sx(centre[sc.mt](sec)), vLo = acc[sc.mt](row, "lo"), vHi = acc[sc.mt](row, "hi"), xl = sc.sx(vLo), xh = sc.sx(vHi);
        g += box(Math.min(c, xl), cy - bh / 2, Math.abs(xl - c), bh, T_LO) + box(Math.min(c, xh), cy - bh / 2, Math.abs(xh - c), bh, T_HI, ' opacity="0.92"');
        const lab = (x, v) => (x < c ? tx(x - 3, cy + 3.2, fmtV[sc.mt](v), { size: 9, fill: C.muted, anchor: "end" }) : tx(x + 3, cy + 3.2, fmtV[sc.mt](v), { size: 9, fill: C.muted }));
        if (rowH >= 11) g += lab(xl, vLo) + lab(xh, vHi);
        const d = vHi - vLo;
        g += tx(cols.swing[j], cy + fs * 0.36, fmtS[sc.mt](d), { size: fs, weight: 600, fill: d >= 0 ? C.green : C.red, anchor: "end" });
      });
      g += "</g>";
      y += rowH;
    });
    scales.forEach((sc) => { g += vl(sc.sx(centre[sc.mt](sec)), y0 - 1, y + 1, C.text, ' stroke-width="1"'); });
  });
  // legend
  const ly = h - 7;
  g += box(cols.name, ly - 7, 9, 8, T_LO) + tx(cols.name + 13, ly, "at P10", { size: 9, fill: C.muted });
  g += box(cols.name + 58, ly - 7, 9, 8, T_HI) + tx(cols.name + 71, ly, "at P90", { size: 9, fill: C.muted });
  return g;
}
function swingChart(r, metric) {
  if (!r.swing || !r.swing.ebitda) return frame("", "No sensitivity in this result.");
  const rr = { ...r, swing: { ebitda: r.swing.ebitda } };
  const top = rr.swing.ebitda.rows[0];
  return frame(tornadoPanel(rr, W, H, { metric, bare: true }), `Driver sensitivity. Widest swing in the median: ${top ? top.name : "none"}.`);
}

/* Price history and the simulated distributions, the bottom band of the assessor.
   Two years of weekly closes end in a dot at the last close. From a vertical baseline there,
   smoothed densities at quarters of the horizon grow to the right on the history's price axis
   and on one probability-per-dollar scale, so the near horizon is narrow and tall and the far
   one broad and flat. Densities come from the summary (`horizon_densities`); the intermediate
   horizons are interpolated from the terminal run until the engine simulates those dates. */
const pathCache = {};
const RAMP = ["#0d3266", "#1f5aa6", "#4a86cc", "#8fb9e6"];     // dark (nearest) to light (horizon)
function weeklyCloses(closes) {
  const out = []; let last = null;
  closes.forEach(([d, c]) => {
    const wk = Math.floor((Date.parse(d) / 864e5 + 3) / 7);    // weeks starting Monday
    if (wk === last) out[out.length - 1] = [d, c]; else { out.push([d, c]); last = wk; }
  });
  return out;
}
async function fetchPath(ticker) {
  if (pathCache[ticker] !== undefined) return pathCache[ticker];
  try {
    const res = await fetch(`/api/prices/${encodeURIComponent(ticker)}?years=2`);
    const data = res.ok ? await res.json() : null;
    pathCache[ticker] = data ? { ...data, closes: weeklyCloses(data.closes || []) } : null;
  } catch (_) { pathCache[ticker] = null; }
  return pathCache[ticker];
}
function pathPanel(r, path, PW, PH) {
  const m = r.meta, s = r.stats, hd = r.horizon_densities, closes = (path && path.closes) || [];
  const legendW = 262, densW = 210;
  const L = { t: HEAD + 16, r: legendW + densW, b: 26, l: 58 }, ih = PH - L.t - L.b;
  const xb = PW - L.r;                                              // the baseline: last close
  const t0 = closes.length ? Date.parse(closes[0][0]) : Date.now() - 2 * 365.25 * 864e5;
  const tLast = closes.length ? Date.parse(closes[closes.length - 1][0]) : Date.now();
  const lastClose = closes.length ? closes[closes.length - 1][1] : m.current_price;
  // price range: the history, the targets and the bulk of the horizon distribution
  const lo = Math.min(s.p5 * 0.92, m.bear_price * 0.92, ...closes.map((c) => c[1]));
  const hi = Math.max(s.p95 * 1.04, m.bull_price * 1.04, ...closes.map((c) => c[1]));
  const yMin = Math.max(0, lo), yMax = hi;
  const sx = (t) => L.l + ((t - t0) / Math.max(tLast - t0, 1)) * (xb - L.l), sy = (v) => L.t + ih - ((v - yMin) / (yMax - yMin)) * ih;
  const labels = hd ? hd.horizons.map((h) => h.label) : [];
  let g = panel(PW, PH, `${m.ticker} price history and the simulated distribution at ${labels.join(", ") || `${m.horizon_years} years`}`,
    closes.length ? `weekly closes to ${closes[closes.length - 1][0]}; curves on one probability-per-dollar scale` : "no price history available; curves on one probability-per-dollar scale");
  const clip = `ppclip-${String(m.ticker).replace(/[^A-Za-z0-9]/g, "")}`;
  g += `<defs><clipPath id="${clip}"><rect x="${L.l}" y="${L.t}" width="${fx(xb + densW - L.l)}" height="${fx(ih)}"/></clipPath></defs>`;
  g += box(L.l, L.t, xb - L.l, ih, "#fbfcfe", ` stroke="${C.grid}" shape-rendering="crispEdges"`);
  denseTicks(yMin, yMax, 7).forEach((v) => {
    g += hl(L.l, xb + densW, sy(v), C.grid) + tx(L.l - 7, sy(v) + 3.5, money0(v), { size: 10, fill: C.muted, anchor: "end" });
  });
  // history time axis: quarters, labelled by year
  for (let y = new Date(t0).getFullYear(); y <= new Date(tLast).getFullYear(); y++) {
    [0, 3, 6, 9].forEach((mo) => {
      const t = Date.UTC(y, mo, 1); if (t <= t0 || t >= tLast) return;
      const x = sx(t);
      g += vl(x, L.t, L.t + ih, mo ? "#f0f2f6" : C.grid, mo ? "" : ' stroke-dasharray="2 4"');
      if (!mo && x < xb - 70) g += tx(x, L.t + ih + 14, String(y), { size: 10, fill: C.muted, anchor: "middle" });
    });
  }
  // the curves, longest horizon first so the near ones sit on top
  let dMax = 1;
  if (hd) {
    const grid = hd.grid;
    dMax = Math.max(...hd.horizons.map((h) => Math.max(...h.density))) || 1;
    let curves = "";
    hd.horizons.map((h, i) => [h, RAMP[Math.min(i, RAMP.length - 1)]]).reverse().forEach(([h, color]) => {
      const pts = [];
      grid.forEach((v, j) => { if (v >= yMin - (yMax - yMin) * 0.02 && v <= yMax + (yMax - yMin) * 0.02) pts.push([xb + (h.density[j] / dMax) * densW, sy(v)]); });
      if (pts.length < 2) return;
      const d = `M${fx(xb)},${fx(pts[0][1])} L${pts.map(([x, y]) => `${fx(x)},${fx(y)}`).join(" L")} L${fx(xb)},${fx(pts[pts.length - 1][1])} Z`;
      curves += `<path d="${d}" fill="${color}" fill-opacity="0.16" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>`;
    });
    g += `<g clip-path="url(#${clip})">${curves}</g>`;
  }
  // targets across the history and the curves, labelled on the left
  const targets = [["Bull", m.bull_price, C.green], ["Base", m.base_price, C.amber], ["Bear", m.bear_price, C.red]].filter(([, v]) => v >= yMin && v <= yMax);
  targets.forEach(([, v, col]) => { g += `<line x1="${fx(L.l)}" y1="${fx(sy(v))}" x2="${fx(xb + densW)}" y2="${fx(sy(v))}" stroke="${col}" stroke-width="1.1" stroke-dasharray="6 4"/>`; });
  // history and the last close, then the target labels on top of both
  if (closes.length) g += `<polyline points="${closes.map((c) => `${fx(sx(Date.parse(c[0])))},${fx(sy(c[1]))}`).join(" ")}" fill="none" stroke="${C.text}" stroke-width="1.4" stroke-linejoin="round"/>`;
  targets.forEach(([lbl, v, col]) => {
    const t = `${lbl} ${money0(v)}`;
    g += box(L.l + 3, sy(v) - 14, t.length * 6.1 + 8, 13, "#fbfcfe", ` stroke="${col}" stroke-width="0.6"`) + tx(L.l + 7, sy(v) - 4.5, t, { size: 10, fill: col, weight: 600 });
  });
  g += vl(xb, L.t, L.t + ih, C.text, ' stroke-width="1"');
  g += `<circle cx="${fx(xb)}" cy="${fx(sy(lastClose))}" r="3.6" fill="${C.text}"/>`;
  g += tx(xb, L.t + ih + 14, `last close ${money(lastClose)}`, { size: 10, fill: C.text, anchor: "middle", weight: 600 });
  if (Math.abs(lastClose - m.current_price) / m.current_price > 0.002) g += tx(xb + 6, L.t + ih - 6, `run spot ${money(m.current_price)}`, { size: 9, fill: C.muted });
  // each horizon's median as a tick just left of the baseline
  if (hd) hd.horizons.forEach((h, i) => {
    if (h.median < yMin || h.median > yMax) return;
    g += `<line x1="${fx(xb - 16)}" y1="${fx(sy(h.median))}" x2="${fx(xb - 2)}" y2="${fx(sy(h.median))}" stroke="${RAMP[Math.min(i, RAMP.length - 1)]}" stroke-width="2.2"/>`;
  });
  // legend in the gutter: horizon, median, P(at least base)
  const gx = xb + densW + 18, gw = PW - gx - 10;
  let ly = L.t + 6;
  g += tx(gx, ly + 8, "HORIZON", { size: 9, fill: C.faint, ls: 0.4 }) + tx(gx + gw * 0.6, ly + 8, "MEDIAN", { size: 9, fill: C.faint, anchor: "end", ls: 0.4 })
    + tx(gx + gw, ly + 8, "P(BASE)", { size: 9, fill: C.faint, anchor: "end", ls: 0.4 });
  ly += 14; g += hl(gx, gx + gw, ly, C.axis);
  if (hd) hd.horizons.forEach((h, i) => {
    const color = RAMP[Math.min(i, RAMP.length - 1)], y = ly + 18 + i * 20;
    g += box(gx, y - 9, 12, 11, color, ' fill-opacity="0.16"') + `<rect x="${fx(gx)}" y="${fx(y - 9)}" width="12" height="11" fill="none" stroke="${color}" stroke-width="1.5"/>`;
    g += tx(gx + 19, y, h.label, { size: 11, fill: C.text, weight: 600 });
    g += tx(gx + gw * 0.6, y, money0(h.median), { size: 11, fill: C.text, anchor: "end" });
    g += tx(gx + gw, y, pct(h.p_at_least_base, 0), { size: 11, fill: C.text, anchor: "end" });
  });
  const ny = ly + 18 + (hd ? hd.horizons.length : 0) * 20 + 6;
  g += hl(gx, gx + gw, ny - 8, C.grid);
  const notes = [`base target ${money0(m.base_price)}; ticks at the baseline are the medians`];
  if (hd && hd.note) notes.unshift(hd.note);
  else if (!hd) notes.unshift("no distribution to draw: the median price is zero");
  notes.flatMap((n) => [...wrapText(n, `9.5px ${MONO}`, gw), ""]).forEach((line, j) => { if (line) g += tx(gx, ny + 6 + j * 13, line, { size: 9.5, fill: C.muted }); });
  return g;
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
  const on = Object.entries(m.mechanisms || {}).filter(([, v]) => v).map(([k]) => MECH_LABEL[k] || k.replace(/_/g, " "));
  $("resultSub").textContent = `${m.n_simulations.toLocaleString()} paths, ${m.horizon_years}-year horizon, seed ${m.random_seed}, ${m.copula === "t" ? `t-copula (df ${m.copula_df})` : "Gaussian copula"}, spot ${money(m.current_price)}, ${m.elapsed_ms} ms${on.length ? `, added drivers: ${on.join(", ")}` : ""}`;
  compareStrip(r); cards(r); reconciliation(r, payload); expectations(r, payload);
  $("cap1").textContent = `Terminal price distribution, ${m.horizon_years}-year, ${m.n_simulations.toLocaleString()} paths`;
  $("chartHist").innerHTML = histChart(r); $("chartBars").innerHTML = barsChart(r); $("chartCdf").innerHTML = cdfChart(r); $("chartTornado").innerHTML = swingChart(r, $("tornadoMetric").value);
  $("summaryTables").innerHTML = summaryTables(r); $("sensitivityTables").innerHTML = sensitivityTables(r);
  ["shareBtn", "pinBtn", "exportPng", "exportJson", "logRun", "assessBtn"].forEach((id) => { $(id).disabled = false; });
  if (activeTab() === "assessor") renderAssessor().catch((e) => showError(e.message));
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

/* Pitch assessor: one 4:3 image for the deck or the Excel model, plus a tab-separated block to paste.
   Figures on the left; the probability histogram and the driver sensitivity on the right; the price
   path with the terminal distribution along the bottom. Shown in its own tab and saved as a PNG. */
const triple = (q, f) => `${f(q.p10)} / ${f(q.p50)} / ${f(q.p90)}`;
const qOrValue = (q, f) => (q.value !== undefined ? `${f(q.value)} (point)` : triple(q, f));
function assessorFigures(r) {
  const s = r.stats, m = r.meta, rc = r.reconciliation, pi = r.price_implied, dq = m.driver_quantiles, xq = m.extra_quantiles || {};
  const added = Object.entries(m.mechanisms || {}).filter(([, v]) => v).map(([k]) => MECH_LABEL[k] || k.replace(/_/g, " "));
  const sd = (q) => `${spct(q.p50, 1)}, P10 to P90 ${spct(q.p10, 1)} to ${spct(q.p90, 1)}`;
  const items = [
    { h: "Run" },
    { k: "Run date, horizon", v: `${new Date().toISOString().slice(0, 10)}, ${m.horizon_years} years` },
    { k: "Spot, paths", v: `${money(m.current_price)}, ${m.n_simulations.toLocaleString()}` },
    { k: "Added drivers", v: added.join(", ") || "none" },
    { h: "Targets" },
    { k: "Bear target", v: `${money(m.bear_price)} (${spct(m.bear_price / m.current_price - 1, 0)})`, c: C.red },
    { k: "Base target", v: `${money(m.base_price)} (${spct(m.base_price / m.current_price - 1, 0)})`, c: C.amber },
    { k: "Bull target", v: `${money(m.bull_price)} (${spct(m.bull_price / m.current_price - 1, 0)})`, c: C.green },
    { h: "Distribution" },
    { k: "Median terminal", v: `${money(s.median_price)} (${spct(s.median_return)}, ${spct(s.median_return_annualized)}/yr)` },
    { k: "Mean terminal", v: money(s.mean_price) },
    { k: "P5 / P25 / P75 / P95", v: `${money0(s.p5)} / ${money0(s.p25)} / ${money0(s.p75)} / ${money0(s.p95)}` },
    { k: "Mode-path price", v: money(rc.mode_path_price) },
    { h: "Probability" },
    { k: "P(at least bear)", v: `${pct(s.p_at_least_bear)} ± ${(s.se.p_at_least_bear * 100).toFixed(2)}pp` },
    { k: "P(at least base)", v: `${pct(s.p_at_least_base)} ± ${(s.se.p_at_least_base * 100).toFixed(2)}pp` },
    { k: "P(at least bull)", v: `${pct(s.p_at_least_bull)} ± ${(s.se.p_at_least_bull * 100).toFixed(2)}pp` },
    { k: "P(below bear)", v: pct(s.below_bear) },
    { h: "Drivers, P10 / P50 / P90" },
    { k: "Revenue CAGR", v: triple(dq.rev_cagr, (v) => pct(v, 1)) },
    m.mechanisms?.cost_model ? null : { k: "EBITDA margin", v: triple(dq.ebitda_margin, (v) => pct(v, 2)) },
    { k: m.mechanisms?.justified_multiple ? "Exit multiple, dispersion only" : "Exit multiple", v: triple(dq.ev_ebitda_multiple, (v) => mult(v)) },
    { k: "Net debt change, % of EV", v: sd(dq.net_debt_change_pct) },
    { k: "Share dilution", v: sd(dq.share_dilution_pct) },
  ];
  if (xq.backlog_conversion) items.push({ k: `Backlog conversion, on ${money0(xq.backlog_m.value)}M`, v: triple(xq.backlog_conversion, (v) => pct(v, 0)) });
  if (xq.gross_margin) items.push({ k: "Gross margin", v: triple(xq.gross_margin, (v) => pct(v, 1)) },
    { k: "Fixed cost, growth", v: `${money0(xq.fixed_cost_m.value)}M, ${xq.fixed_cost_growth ? `${spct(xq.fixed_cost_growth.p50, 1)}/yr` : "flat"}` });
  if (xq.fcf_margin) items.push({ k: "FCF margin", v: triple(xq.fcf_margin, (v) => pct(v, 1)) }, { k: "EV/FCF exit multiple", v: triple(xq.ev_fcf_multiple, (v) => mult(v)) });
  if (xq.roic) {
    items.push({ k: "ROIC", v: qOrValue(xq.roic, (v) => pct(v, 1)) }, { k: "WACC", v: qOrValue(xq.wacc, (v) => pct(v, 1)) }, { k: "Growth fade", v: qOrValue(xq.fade, (v) => v.toFixed(2)) });
    const e = m.eva;
    if (e) items.push({ k: "EVA at P50", v: e.eva_m !== null ? `${smoney0(e.eva_m)}M (spread ${spct(e.spread, 1)})` : `spread ${spct(e.spread, 1)}`, c: e.spread < 0 ? C.red : undefined });
  }
  const na = (v, f) => (v === null || v === undefined ? "no solution" : f(v));
  items.push(
    { h: "Market-implied at spot" },
    { k: "Margin, at P50 growth and multiple", v: na(pi.implied_margin_at_mode_growth_and_multiple, (v) => pct(v, 2)) },
    { k: "Growth, at P50 margin and multiple", v: na(pi.implied_growth_at_mode_margin_and_multiple, (v) => pct(v, 1)) },
    { k: "Multiple, at P50 growth and margin", v: na(pi.implied_multiple_at_mode_growth_and_margin, (v) => mult(v)) },
    { k: "Base needs, multiple at P50 margin", v: na(rc.cases.base.implied_multiple_at_mode_margin, (v) => mult(v)) },
    { k: "Base needs, margin at P50 multiple", v: na(rc.cases.base.implied_margin_at_mode_multiple, (v) => pct(v, 2)) },
  );
  if (r.fcf_bridge) items.push({ h: "FCF bridge" },
    { k: "FCF bridge median", v: `${money(r.fcf_bridge.median_price)} (${spct(r.fcf_bridge.gap_to_ebitda_bridge)} vs EBITDA)`, c: C.purple },
    { k: "FCF bridge P(at least base)", v: pct(r.fcf_bridge.p_at_least_base) });
  if (r.dcf_path) items.push({ h: "Three-statement path" },
    { k: "Path price, percentile", v: `${money(r.dcf_path.price)}, ${Math.round(r.dcf_path.percentile * 100)}th` },
    { k: "Paths at or above it", v: pct(r.dcf_path.p_at_least) });
  return items.filter(Boolean);
}
function assessorRows(r) {
  const m = r.meta, rows = [["Company", `${m.company_name} (${m.ticker})`]];
  assessorFigures(r).forEach((it) => { rows.push(it.h ? ["", ""] : [it.k, it.v]); });
  const sw = r.swing || {};
  [["ebitda", "EBITDA bridge"], ["fcf", "FCF bridge"]].forEach(([b, label]) => {
    if (!sw[b]) return;
    rows.push(["", ""], [`Sensitivity, ${label}`, `median ${money(sw[b].median)}, P(base) ${pct(sw[b].p_base)}`]);
    sw[b].rows.forEach((row) => rows.push([`${row.name} P10 to P90 (${fmtDrv(row, row.lo)} to ${fmtDrv(row, row.hi)})`,
      `median ${money0(row.median_lo)} to ${money0(row.median_hi)}; P(base) ${pct(row.p_base_lo)} to ${pct(row.p_base_hi)}`]));
  });
  return rows;
}
function figuresPanel(r, w, h) {
  const items = assessorFigures(r), nh = items.filter((i) => i.h).length, nr = items.length - nh;
  const headH = 17, rowH = Math.min(16.5, (h - HEAD - 6 - nh * headH) / nr), fs = Math.min(11, rowH * 0.74);
  let g = panel(w, h, "Key figures", ""), y = HEAD + 4, zebra = 0;
  items.forEach((it) => {
    if (it.h) { g += tx(9, y + 12.5, it.h.toUpperCase(), { size: 9, weight: 600, fill: C.muted, ls: 0.6 }) + hl(9, w - 9, y + headH - 1.5, C.grid); y += headH; zebra = 0; return; }
    if (zebra++ % 2 === 1) g += box(1, y, w - 2, rowH, "#f8fafc");
    g += tx(9, y + rowH * 0.7, it.k, { size: fs, fill: it.c || C.text2, font: SANS, weight: it.c ? 600 : 400 });
    g += tx(w - 9, y + rowH * 0.7, it.v, { size: fs, weight: 500, anchor: "end" });
    y += rowH;
  });
  return g;
}
function assessorVerdict(r) {
  const s = r.stats, rc = r.reconciliation, sw = r.swing && r.swing.ebitda;
  const pb = s.p_at_least_base;
  let v;
  if (pb >= 0.5) v = `The base target is at or below the simulated median: ${pct(pb)} of paths reach it.`;
  else if (pb >= 0.3) v = `The base target sits above the median; ${pct(pb)} of paths reach it. Defensible if the thesis moves a driver the model holds at its P50.`;
  else v = `The base target is in the top ${pct(pb, 0)} of outcomes. It needs ${mult(rc.cases.base.implied_multiple_at_mode_margin)} at the P50 margin or a ${pct(rc.cases.base.implied_margin_at_mode_multiple, 2)} margin at the P50 multiple.`;
  v += ` Downside: ${pct(s.below_bear)} of paths finish below the bear target.`;
  if (sw && sw.rows.length) {
    const top = sw.rows[0];
    const who = top.key === "ev_ebitda" ? "The exit multiple" : top.name;
    v += ` ${who} moves the median most: ${money0(top.median_lo)} at its P10 to ${money0(top.median_hi)} at its P90.`;
  }
  if (r.fcf_bridge && Math.abs(r.fcf_bridge.gap_to_ebitda_bridge) > 0.15) v += ` The FCF bridge disagrees with the EBITDA bridge by ${spct(r.fcf_bridge.gap_to_ebitda_bridge)}; say which one the thesis rests on.`;
  return v;
}
let measureCtx = null;
function wrapText(text, font, maxW) {
  measureCtx = measureCtx || document.createElement("canvas").getContext("2d");
  measureCtx.font = font;
  const lines = []; let line = "";
  text.split(" ").forEach((w) => { const t = line ? `${line} ${w}` : w; if (measureCtx.measureText(t).width > maxW && line) { lines.push(line); line = w; } else line = t; });
  if (line) lines.push(line);
  return lines;
}
const AW = 1600, AH = 1200;
function assessorSvg(r, path, fontCss = "") {
  const m = r.meta, left = 48, inner = AW - 2 * left;
  const verdict = wrapText(assessorVerdict(r), "500 14px IBM Plex Sans, Helvetica Neue, Arial, sans-serif", inner - 28);
  const vH = 16 + verdict.length * 19;
  const top = 72 + vH + 14, bottomEdge = AH - 34, gap = 14;
  // the middle band grows with the figures table, the path chart keeps at least 380
  const items = assessorFigures(r), nh = items.filter((i) => i.h).length, nr = items.length - nh;
  const mid = Math.min(Math.max(HEAD + 6 + nh * 17 + nr * 15, 520), bottomEdge - top - gap - 380);
  const pathH = bottomEdge - top - mid - gap;
  const figW = 452, rx = left + figW + gap, rw = AW - left - rx;
  const swRows = Object.values(r.swing || {}).reduce((a, b) => a + b.rows.length, 0) + (Object.keys(r.swing || {}).length > 1 ? 2 : 0);
  const tornH = Math.min(Math.max(HEAD + 40 + swRows * 18, 220), mid - 230 - gap);
  const histH = mid - tornH - gap;
  const at = (x, y, body) => `<g transform="translate(${x},${y})">${body}</g>`;
  let g = box(0, 0, AW, AH, "#ffffff") + box(0, 0, AW, 6, "#1b4b8a");
  g += tx(left, 50, `${m.company_name} (${m.ticker}): pitch assessment`, { size: 26, weight: 600, font: SANS });
  g += box(left, 70, inner, vH, "#e8eff8") + box(left, 70, 3, vH, "#1b4b8a");
  verdict.forEach((line, i) => { g += tx(left + 16, 70 + 22 + i * 19, line, { size: 14, weight: 500, font: SANS }); });
  g += at(left, top, figuresPanel(r, figW, mid));
  g += at(rx, top, histPanel(r, rw, histH));
  g += at(rx, top + histH + gap, tornadoPanel(r, rw, tornH));
  g += at(left, top + mid + gap, pathPanel(r, path, inner, pathH));
  g += tx(left, AH - 14, `Model output, not investment advice. ${new Date().toISOString().slice(0, 10)}. Every figure is filed with the SEC, a closing price, arithmetic over those, or simulation output.`, { size: 10.5, fill: C.faint });
  const style = fontCss ? `<defs><style>${fontCss}</style></defs>` : "";
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${AW} ${AH}" width="${AW}" height="${AH}" role="img" aria-label="${esc(`${m.company_name} pitch assessment`)}">${style}${g}</svg>`;
}

/* The PNG carries the page's own fonts: an SVG drawn as an image cannot reach web fonts, so the
   latin IBM Plex files are inlined as data URIs once per session. */
let fontCssPromise = null;
function embeddedFontCss() {
  if (!fontCssPromise) {
    fontCssPromise = (async () => {
      try {
        const link = document.querySelector('link[href*="fonts.googleapis.com/css2"]');
        if (!link) return "";
        const css = await (await fetch(link.href)).text();
        const blocks = (css.match(/@font-face\s*{[^}]*}/g) || []).filter((b) => /unicode-range:[^;]*U\+0000-00FF/i.test(b));
        const out = await Promise.all(blocks.map(async (b) => {
          const url = (b.match(/url\(([^)]+)\)/) || [])[1];
          if (!url) return "";
          const bytes = new Uint8Array(await (await fetch(url.replace(/['"]/g, ""))).arrayBuffer());
          let bin = ""; for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
          return b.replace(/url\([^)]+\)/, `url(data:font/woff2;base64,${btoa(bin)})`);
        }));
        return out.join("\n");
      } catch (_) { return ""; }
    })();
  }
  return fontCssPromise;
}
let assessorZoom = false;
async function renderAssessor() {
  if (!lastResult) return;
  const r = lastResult, path = await fetchPath(r.meta.ticker);
  if (r !== lastResult) return;            // a newer run arrived while the closes were loading
  $("assessFrame").innerHTML = assessorSvg(r, path);
  $("assessFrame").classList.toggle("actual", assessorZoom);
}
async function saveAssessorPng() {
  if (!lastResult) return;
  const r = lastResult, path = await fetchPath(r.meta.ticker), css = await embeddedFontCss();
  const img = await svgStringToImage(assessorSvg(r, path, css));
  const scale = 2, canvas = document.createElement("canvas"); canvas.width = AW * scale; canvas.height = AH * scale;
  const ctx = canvas.getContext("2d"); ctx.scale(scale, scale); ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, AW, AH); ctx.drawImage(img, 0, 0, AW, AH);
  const a = document.createElement("a"); a.href = canvas.toDataURL("image/png"); a.download = `${r.meta.ticker}_pitch_assessment.png`; a.click();
}
async function copyAssessorTable() {
  if (!lastResult) return;
  const tsv = assessorRows(lastResult).map(([k, v]) => (k ? `${k}\t${v}` : "")).join("\n") + `\n\nVerdict\t${assessorVerdict(lastResult)}`;
  const btn = $("assessCopy");
  try { await navigator.clipboard.writeText(tsv); btn.textContent = "Table copied"; } catch (_) { btn.textContent = "Clipboard blocked"; }
  setTimeout(() => { btn.textContent = "Copy table"; }, 2000);
}
function selectTab(name) {
  document.querySelectorAll(".tabs .tab").forEach((t) => { const on = t.dataset.tab === name; t.classList.toggle("active", on); t.setAttribute("aria-selected", String(on)); });
  $("analysisView").hidden = name !== "analysis";
  $("assessorView").hidden = name !== "assessor";
  if (name === "assessor") renderAssessor().catch((e) => showError(e.message));
}
const activeTab = () => (document.querySelector(".tabs .tab.active") || {}).dataset?.tab || "analysis";

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
MECHS.forEach((m) => {
  const d = mechEl(m);
  d.querySelector("[data-remove]").addEventListener("click", () => showMech(m, false));
  document.querySelector(`#addDriver [data-add="${m}"]`).addEventListener("click", () => { showMech(m, true); d.querySelector("input")?.focus(); });
  d.addEventListener("input", (e) => { e.target.closest("label")?.classList.remove("needs-input"); if (m === "justified_multiple") updateEva(); });
});
$("corrReset").addEventListener("click", () => { buildCorrGrid(DEFAULT_CORR); $("corrStatus").textContent = "Reset."; $("corrStatus").className = "hint"; });
$("loadPreset").addEventListener("click", () => { clearProvenance(); applyExtraDrivers(null); fillForm(PRESET); $("lookupMsg").hidden = true; $("formError").hidden = true; });
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
$("assessBtn").addEventListener("click", () => { selectTab("assessor"); $("results").scrollIntoView({ block: "start" }); });
document.querySelectorAll(".tabs .tab").forEach((t) => t.addEventListener("click", () => selectTab(t.dataset.tab)));
$("assessPng").addEventListener("click", () => saveAssessorPng().catch((e) => showError(e.message)));
$("assessCopy").addEventListener("click", () => copyAssessorTable());
$("assessZoom").addEventListener("click", () => {
  assessorZoom = !assessorZoom; $("assessFrame").classList.toggle("actual", assessorZoom);
  $("assessZoom").textContent = assessorZoom ? "Fit to width" : "Actual size";
});
$("tornadoMetric").addEventListener("change", () => { if (lastResult) $("chartTornado").innerHTML = swingChart(lastResult, $("tornadoMetric").value); });
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
buildCorrGrid(DEFAULT_CORR); applyExtraDrivers(null);
const fromUrl = readUrl();
if (fromUrl) { fillForm(fromUrl); run(); }
else { form.n_simulations.value = PRESET.n_simulations; form.horizon_years.value = PRESET.horizon_years; form.random_seed.value = PRESET.random_seed; form.copula.value = "t"; form.ticker.focus(); }
