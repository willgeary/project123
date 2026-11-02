// Builds causality.pptx from causal_deck_data.json (written by prepare.py).
// Run: NODE_PATH=<dir with pptxgenjs> node build.js <path to pptx skill apply_theme.js>
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.argv[2]);

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "causal_deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "causality.pptx");
const n = (x) => Math.round(x).toLocaleString("en-US");
const pts = (x, d = 1) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
const iv = (r, d = 1) => `[${pts(r.lo89, d)}, ${pts(r.hi89, d)}]`;
const AP = D.approaches, LM = D.landmark;

// Same theme and layouts as sandbox/deck/build.js, so the two decks read as a set.
const THEME = {
  name: "Outreach causal analysis",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B0B0B", lt1: "FFFFFF", dk2: "1C2B3A", lt2: "EEF2F6",
    accent1: "2A78D6", accent2: "EB6834", accent3: "52514E", accent4: "1BAF7A",
    accent5: "EDA100", accent6: "E87BA4", hlink: "2A78D6", folHlink: "6E4BB3",
  },
};
// Blue = estimated effect, gray = negative control (true effect 0), orange = the selection/bias story.
const EFFECT = "2A78D6", CONTROL = "52514E", BIAS = "EB6834", BASE = "B4B2A9", GRID = "E3E6EA";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Did Meridian's outreach work?";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const footer = { text: { text: "Vi Labs · Meridian outreach · causal analysis (sandbox)", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent3, margin: 0 } } };
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontFace: "Cambria", fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.6, fontSize: 20, color: "CADCFC", valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.7, h: 1.0, fontSize: 18, color: "CADCFC", valign: "top" }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: "CADCFC" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontFace: "Cambria", fontSize: 36, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    footer,
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent3 },
});

const content = (section, title) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
};
const section = (title, body) => {
  pres.addSection({ title });
  const s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: title });
  s.addText(title, { placeholder: "title" });
  s.addText(body, { placeholder: "body" });
  return s;
};
const text = (s, t, o) => s.addText(t, { isTextBox: true, fontSize: 16, color: C.text1, valign: "top", margin: 0, ...o });
const card = (s, x, y, w, h, name) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name });
const stat = (s, x, y, w, big, label, color, name) => {
  card(s, x, y, w, 2.0, name);
  text(s, big, { x: x + 0.3, y: y + 0.25, w: w - 0.6, h: 0.95, fontSize: 44, bold: true, color: color || C.text2, fontFace: "Cambria", valign: "middle" });
  text(s, label, { x: x + 0.3, y: y + 1.22, w: w - 0.6, h: 0.7, fontSize: 14, color: C.accent3 });
};
const bullets = (lines) => lines.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < lines.length - 1 } }));
const chartBase = (title) => ({
  showTitle: !!title, title, titleFontSize: 14, titleColor: "1C2B3A", titleFontFace: "+mn-lt",
  catAxisLabelColor: "52514E", valAxisLabelColor: "52514E", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false,
  showLegend: true, legendPos: "t", legendFontSize: 11, legendColor: "52514E", legendFontFace: "+mn-lt",
  catAxisTitleColor: "52514E", valAxisTitleColor: "52514E", catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
  dataLabelFontSize: 11, dataLabelColor: "1C2B3A", dataLabelFontFace: "+mn-lt",
});
const node = (s, x, y, w, h, title, sub, fill, color, name) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: fill }, objectName: name });
  text(s, [{ text: title, options: { bold: true, fontSize: 16, color, breakLine: true } }, { text: sub, options: { fontSize: 12, color } }], { x: x + 0.15, y, w: w - 0.3, h, align: "center", valign: "middle" });
};
const arrow = (s, x, y, w, h, opts = {}) =>
  s.addShape(pres.shapes.LINE, { x, y, w, h, line: { color: opts.color || CONTROL, width: 1.75, endArrowType: "triangle", dashType: opts.dash || "solid" }, flipH: !!opts.flipH, objectName: opts.name });

const enr = AP.C.enroll_180d, surg = AP.C.surgery_12m, cal = LM.calibrated_rd;

// 1. Title
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
s.addText("Did Meridian's outreach work?", { placeholder: "title" });
s.addText([
  { text: `It raised enrollment by about ${enr.mean.toFixed(1)} points.`, options: { breakLine: true } },
  { text: "Its effect on surgery can't be detected from the historical data: that needs the pilot." },
], { placeholder: "body" });
s.addNotes("This deck estimates the causal effect of Meridian's 2024-2025 outreach waves on program enrollment and on lumbar spine surgery, using only historical data. Outreach wasn't randomized, so every estimate rests on emulating the trial Meridian didn't run.");

// 2. Bottom line
s = content("Overview", "The bottom line");
stat(s, 0.6, 1.6, 3.85, pts(enr.mean), `points of enrollment within 180 days, 89% interval ${iv(enr)}`, EFFECT, "stat-enroll");
stat(s, 4.74, 1.6, 3.85, pts(cal.mean, 2), `points of surgery within 12 months, best-controlled estimate ${iv(cal, 2)}`, null, "stat-surgery");
stat(s, 8.88, 1.6, 3.85, `${Math.round(LM.p_protective * 100)}%`, "posterior probability that outreach reduces surgery: a coin flip", null, "stat-p");
card(s, 0.6, 3.95, 12.13, 2.75, "card-meaning");
text(s, [
  { text: "What this means for Meridian", options: { bold: true, fontSize: 20, color: C.text2, breakLine: true } },
  ...bullets([
    `Outreach works at the first step: contacted patients enrolled ${Math.round(D.enroll_baseline_pct)}% → ${Math.round(D.enroll_baseline_pct + enr.mean)}% of the time.`,
    "Whether that enrollment prevents surgery can't be read from history. Meridian chose who to contact using information that isn't in the data, and the plausible effect is smaller than the remaining uncertainty.",
    "The randomized holdout in the pilot, followed for 12 months, is the only way to measure the surgery effect.",
  ]),
], { x: 0.9, y: 4.15, w: 11.5, h: 2.45, paraSpaceAfter: 6, fontSize: 15 });
s.addNotes("Enrollment is the first link in the chain and it's well measured. The surgery effect is the one that matters financially, and history can only bound it.");

// 3. Why this is hard
s = content("Overview", "Why the simple comparison misleads");
node(s, 4.9, 1.55, 3.55, 1.05, "What Meridian's team saw", "Referral notes, scheduling: not in the data", "FBE3D6", "7A2E0E", "node-unseen");
node(s, 0.9, 3.9, 2.9, 1.05, "Contacted", "Meridian's choice", "DCE9F8", "0C447C", "node-contact");
node(s, 5.2, 3.9, 2.9, 1.05, "Enrolled", "PT or weight management", "DCE9F8", "0C447C", "node-enroll");
node(s, 9.5, 3.9, 2.9, 1.05, "Surgery", "Within 12 months", "DCE9F8", "0C447C", "node-surgery");
arrow(s, 3.85, 4.42, 1.3, 0, { color: EFFECT, name: "arrow-contact-enroll" });
arrow(s, 8.15, 4.42, 1.3, 0, { color: EFFECT, name: "arrow-enroll-surgery" });
arrow(s, 2.35, 2.6, 2.5, 1.25, { color: BIAS, dash: "dash", flipH: true, name: "arrow-unseen-contact" });
arrow(s, 8.5, 2.6, 2.45, 1.25, { color: BIAS, dash: "dash", name: "arrow-unseen-surgery" });
text(s, "Back door: patients closer to surgery were more likely to be contacted", { x: 0.6, y: 2.75, w: 3.6, h: 0.6, fontSize: 12, color: "993C1D" });
text(s, [
  { text: "Raw comparison: ", options: { bold: true } },
  { text: `contacted patients had surgery ${D.raw["1"].surgery_12m.toFixed(1)}% of the time vs ${D.raw["0"].surgery_12m.toFixed(1)}% for everyone else. That isn't outreach causing surgery. It's the dashed path: Meridian contacted sicker patients.` },
], { x: 0.6, y: 5.45, w: 12.13, h: 1.0, fontSize: 15 });
s.addNotes("Blue arrows are the effect we want. The dashed orange path is confounding: whatever made Meridian's team pick a patient also made surgery more likely. Adjusting for the 37 covariates closes the parts of that path we can see.");

// 4. Design
s = content("Overview", "Emulate the trial Meridian didn't run, once per wave");
const tx = 0.9, tw = 11.5, day = (d) => tx + (d / 365) * tw;
const bar = (label, d0, d1, y, fill, color, name) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: day(d0), y, w: day(d1) - day(d0), h: 0.5, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
  text(s, label, { x: day(d0) + 0.12, y, w: Math.max(day(d1) - day(d0) - 0.2, 3.4), h: 0.5, fontSize: 13, color, valign: "middle", bold: true });
};
bar("Contacts (days 8–27 in practice)", 0, 92, 1.65, "2A78D6", "FFFFFF", "bar-contact");
bar("Enrollment counted (days 0–180)", 0, 180, 2.3, "9CC3EF", "0C447C", "bar-enroll");
bar("Surgery while covered (days 0–365)", 0, 365, 2.95, "D7E5F5", "0C447C", "bar-surgery");
[0, 92, 180, 365].forEach((d) => text(s, `day ${d}`, { x: day(d) - 0.4, y: 3.5, w: 0.8, h: 0.3, fontSize: 11, color: C.accent3, align: "center" }));
text(s, "Time zero = wave start for both groups, so nobody has to stay surgery-free to count as uncontacted.", { x: 0.6, y: 3.85, w: 12.13, h: 0.4, fontSize: 13, color: C.accent3 });
const th = (t, al = "center") => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13, align: al } });
s.addTable([
  [th("Wave", "left"), th("Eligible"), th("Contacted"), th("Share contacted")],
  ...D.waves.map((w) => [{ text: w.wave, options: { bold: true, color: C.text2 } }, { text: n(w.eligible), options: { align: "center" } }, { text: n(w.contacted), options: { align: "center" } }, { text: `${(w.share * 100).toFixed(0)}%`, options: { align: "center" } }]),
], { x: 0.6, y: 4.4, w: 6.4, colW: [1.6, 1.6, 1.6, 1.6], fontSize: 13, color: C.text1, rowH: 0.42, valign: "middle", border: { type: "solid", pt: 0.5, color: "D9DEE4" }, margin: [0.04, 0.1, 0.04, 0.1] });
card(s, 7.3, 4.4, 5.43, 2.3, "card-eligibility");
text(s, [
  { text: "Eligible at wave start", options: { bold: true, color: C.text2, breakLine: true } },
  ...bullets(["Covered, back-pain history, no surgery yet", "No surgical consult in the last 90 days", "Never opted out of outreach", "Adjusted for 37 covariates: the risk-model features plus earlier outreach"]),
], { x: 7.55, y: 4.55, w: 5.0, h: 2.05, fontSize: 13, paraSpaceAfter: 3 });
s.addNotes("Three waves have a full year of follow-up. Patients can appear in more than one wave; refitting on each patient's first wave only gives the same answer.");

// 5. Balance
s = content("Overview", "Weighting makes the groups comparable on what we see");
s.addChart(pres.charts.BAR, [
  { name: "Before weighting", labels: D.balance.map((b) => b.label), values: D.balance.map((b) => Math.abs(b.raw)) },
  { name: "After weighting", labels: D.balance.map((b) => b.label), values: D.balance.map((b) => Math.abs(b.weighted)) },
], {
  ...chartBase("Imbalance between contacted and uncontacted (|standardized mean difference|)"), x: 0.6, y: 1.5, w: 8.0, h: 5.3, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 50,
  chartColors: [BASE, EFFECT], valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisLabelFormatCode: "0.0", catAxisOrientation: "maxMin", showValue: true, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd",
});
card(s, 8.9, 1.6, 3.83, 5.1, "card-balance");
text(s, [
  { text: "Who Meridian contacted", options: { bold: true, color: C.text2, breakLine: true } },
  { text: `A model of who got contacted reaches AUC ${D.propensity.auc.toFixed(2)}. Meridian mostly avoided re-contacting earlier-wave patients, and favored patients with recent imaging, consults and MRIs.`, options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "After weighting", options: { bold: true, color: C.text2, breakLine: true } },
  { text: `Every covariate is within ${D.propensity.max_abs_smd_weighted.toFixed(2)} (0.10 is the usual flag), and the comparison group keeps an effective ${n(D.propensity.ess_control)} patients.`, options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "The catch", options: { bold: true, color: BIAS, breakLine: true } },
  { text: "Balance on what we see says nothing about what Meridian's team saw." },
], { x: 9.15, y: 1.85, w: 3.35, h: 4.7, fontSize: 13 });

// 6-7. Enrollment
section("Enrollment", "The first link in the chain: does contact get patients into a program?");
s = content("Enrollment", `Outreach raises enrollment by about ${Math.round(enr.mean)} points`);
const enrRows = [["A. Outcome model", AP.A.enroll_180d], ["A, first wave per patient", AP.A_first_wave.enroll_180d], ["B. Similar-patient strata", AP.B.enroll_180d], ["C. Doubly robust", AP.C.enroll_180d]];
s.addChart(pres.charts.BAR, [{ name: "Enrollment effect", labels: enrRows.map(([l, r]) => `${l}  ${iv(r)}`), values: enrRows.map(([, r]) => r.mean) }], {
  ...chartBase("Effect on 180-day enrollment, percentage points (89% interval)"), showLegend: false, x: 0.6, y: 1.5, w: 6.6, h: 3.0, barDir: "bar", barGapWidthPct: 45,
  chartColors: [EFFECT], valAxisMinVal: 0, valAxisMaxVal: 14, valAxisLabelFormatCode: "0", catAxisOrientation: "maxMin", showValue: true, dataLabelFormatCode: "+0.0", dataLabelPosition: "outEnd",
});
const P = D.posterior.enroll_aipw;
s.addChart(pres.charts.BAR, [{ name: "Posterior", labels: P.mid.map((m) => m.toFixed(1)), values: P.share }], {
  ...chartBase("Posterior distribution of the effect (doubly robust)"), showLegend: false, x: 7.5, y: 1.5, w: 5.23, h: 3.0, barDir: "col", barGapWidthPct: 15,
  chartColors: [EFFECT], catAxisLabelFrequency: 5, valAxisLabelFormatCode: "0", showValAxisTitle: true, valAxisTitle: "% of draws", showCatAxisTitle: true, catAxisTitle: "Percentage points",
});
const H = D.heterogeneity;
s.addChart(pres.charts.BAR, [{ name: "Enrollment effect", labels: H.enroll.map((r, i) => `Q${r.quintile}: ${H.baseline[i].toFixed(1)}% risk`), values: H.enroll.map((r) => r.mean) }], {
  ...chartBase("By baseline surgery-risk quintile (hierarchical model)"), showLegend: false, x: 0.6, y: 4.65, w: 6.6, h: 2.2, barDir: "col", barGapWidthPct: 45,
  chartColors: [EFFECT], valAxisMinVal: 0, valAxisMaxVal: 16, valAxisLabelFormatCode: "0", showValue: true, dataLabelFormatCode: "+0.0", dataLabelPosition: "outEnd",
});
card(s, 7.5, 4.75, 5.23, 2.05, "card-enroll");
text(s, [
  { text: "Consistent everywhere", options: { bold: true, color: C.text2, breakLine: true } },
  { text: `Every approach lands at +11 to +12 points, from a ${D.enroll_baseline_pct.toFixed(0)}% baseline. The lift is about the same at every risk level, slightly lower at the top. The risk score decides who's worth contacting; outreach works about equally on everyone.` },
], { x: 7.75, y: 4.9, w: 4.75, h: 1.8, fontSize: 13 });
s.addNotes("Enrollment counts any PT or weight-management enrollment within 180 days of wave start, before any surgery. The negative controls (next section) show some residual selection, but it's small relative to an 11-point effect.");

// 8-12. Surgery
section("Surgery", "The link that matters financially: does outreach prevent surgery?");
s = content("Surgery", "Every naive estimate comes with a matching bias");
const naive = [["A. Outcome model", AP.A], ["B. Similar-patient strata", AP.B], ["C. Doubly robust", AP.C]];
s.addChart(pres.charts.BAR, [
  { name: "Surgery within 12 months (the estimate)", labels: naive.map(([l]) => l), values: naive.map(([, a]) => a.surgery_12m.mean) },
  { name: "Surgery within 30 days (negative control: should be 0)", labels: naive.map(([l]) => l), values: naive.map(([, a]) => a.nc_surgery_30d.mean) },
], {
  ...chartBase("'Effect' of contact, percentage points"), x: 0.6, y: 1.5, w: 7.6, h: 5.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
  chartColors: [EFFECT, CONTROL], valAxisMinVal: 0, valAxisMaxVal: 0.8, valAxisLabelFormatCode: "0.0", showValue: true, dataLabelFormatCode: "+0.00", dataLabelPosition: "outEnd",
});
card(s, 8.5, 1.6, 4.23, 5.1, "card-naive");
text(s, [
  { text: "Taken at face value", options: { bold: true, color: C.text2, breakLine: true } },
  { text: `outreach raises surgery by ${AP.C.surgery_12m.mean.toFixed(2)} points (doubly robust, ${iv(AP.C.surgery_12m, 2)}).`, options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "But look at the gray bars", options: { bold: true, color: BIAS, breakLine: true } },
  { text: "Contacted patients also had more surgeries in the first 30 days, before any program could act. Outreach can't cause those. They were already in the pipeline.", options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "Approach B is the most biased", options: { bold: true, color: C.text2, breakLine: true } },
  { text: "Ten strata per wave leave imbalance inside each stratum. The doubly robust estimate agrees with A, not B." },
], { x: 8.75, y: 1.85, w: 3.75, h: 4.7, fontSize: 13 });
s.addNotes("A negative control is an outcome the treatment can't plausibly affect. Any 'effect' on it measures bias. Non-spine acute visits before and after the wave were also checked; those show no meaningful difference.");

s = content("Surgery", "Restart the clock at day 30 and correct for what's left");
text(s, [
  { text: "Landmark design", options: { bold: true, color: C.text2, breakLine: true } },
  ...bullets([
    "Drop the 179 surgeries already in the pipeline (days 0–30)",
    `Days 30–90 as a control window: too soon for a program to change a surgical decision. Risk ratio ${LM.nc_30_90.rr.mean.toFixed(2)}`,
    `Days 90–365 as the main window. Risk ratio ${LM.main_90_365.rr.mean.toFixed(2)}`,
    `Corrected = main ÷ control: ${LM.calibrated_rr.mean.toFixed(2)} ${`[${LM.calibrated_rr.lo89.toFixed(2)}, ${LM.calibrated_rr.hi89.toFixed(2)}]`}`,
  ]),
], { x: 0.6, y: 1.55, w: 5.2, h: 3.3, fontSize: 14, paraSpaceAfter: 6 });
const PC = D.posterior.surgery_calibrated;
s.addChart(pres.charts.BAR, [{ name: "Posterior", labels: PC.mid.map((m) => m.toFixed(1)), values: PC.share }], {
  ...chartBase("Posterior: corrected effect on surgery, days 90–365 (percentage points)"), showLegend: false, x: 6.1, y: 1.5, w: 6.63, h: 3.6, barDir: "col", barGapWidthPct: 15,
  chartColors: [EFFECT], catAxisLabelFrequency: 5, valAxisLabelFormatCode: "0", showValAxisTitle: true, valAxisTitle: "% of draws", showCatAxisTitle: true, catAxisTitle: "Below 0 = outreach prevents surgery",
});
stat(s, 0.6, 4.95, 3.85, pts(cal.mean, 2), `points, 89% interval ${iv(cal, 2)}`, null, "stat-cal");
stat(s, 4.74, 4.95, 3.85, `${Math.round(LM.p_protective * 100)}%`, "posterior probability outreach is protective", null, "stat-protective");
stat(s, 8.88, 4.95, 3.85, LM.e_value.toFixed(2), "E-value: a modest unmeasured confounder could explain the uncorrected estimate", BIAS, "stat-evalue");
s.addNotes("The correction assumes the remaining selection acts as a constant risk ratio over time, so the days 30-90 window measures the bias in days 90-365. After correction the estimate is centered at zero with an interval wide enough to include a meaningful benefit or a small harm.");

s = content("Surgery", "Where the apparent 'effect' sits: the highest-risk patients");
s.addChart(pres.charts.BAR, [{ name: "Surgery effect", labels: H.surgery.map((r, i) => `Q${r.quintile}: ${H.baseline[i].toFixed(1)}%`), values: H.surgery.map((r) => r.mean) }], {
  ...chartBase("Uncorrected effect on 12-month surgery by baseline-risk quintile (percentage points)"), showLegend: false, x: 0.6, y: 1.5, w: 7.6, h: 5.3, barDir: "col", barGapWidthPct: 45,
  chartColors: [EFFECT], valAxisMinVal: 0, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.0", showValue: true, dataLabelFormatCode: "+0.00", dataLabelPosition: "outEnd",
  showCatAxisTitle: true, catAxisTitle: "Quintile: baseline 12-month surgery risk",
});
card(s, 8.5, 1.6, 4.23, 5.1, "card-quintile");
text(s, [
  { text: "Concentrated in Q4–Q5", options: { bold: true, color: C.text2, breakLine: true } },
  { text: `Q4 ${pts(H.surgery[3].mean, 2)} ${iv(H.surgery[3], 2)}, Q5 ${pts(H.surgery[4].mean, 2)} ${iv(H.surgery[4], 2)}. Q1–Q3 are close to zero.`, options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "Why this points to selection", options: { bold: true, color: BIAS, breakLine: true } },
  { text: "That's where clinical information outside the data, like a booked procedure, would matter most. It isn't evidence that outreach causes surgery in high-risk patients." },
], { x: 8.75, y: 1.85, w: 3.75, h: 4.7, fontSize: 13 });
s.addNotes("Partially pooled treatment effects by quintile of predicted untreated risk. The hierarchical model was refit with target_accept 0.95: 0 divergences.");

s = content("Surgery", "Three other explanations, checked and ruled out");
const ruled = [
  ["Contacts triggered by new visits", "If Meridian called patients after a new encounter mid-wave, adjusting at wave start would miss it.", "Ruled out: contacted and uncontacted patients had near-identical activity before the contact date (median 54 vs 56 days since the last visit)."],
  ["A hidden confounder in the extract", "Cancelled consults, recent spine activity, denied claims, recent orthopedic visits.", "Ruled out: all differ by under 0.06 SMD between groups. The selection is on something outside the data."],
  ["PCP-level contact rates as an instrument", "Some PCPs' patients were contacted more often (F ≈ 172).", "Rejected: it tracks case mix (correlates −0.24 with tenure) and would give an interval of about ±40 points."],
];
ruled.forEach(([title, what, verdict], i) => {
  const x = 0.6 + i * 4.14;
  card(s, x, 1.6, 3.85, 4.9, `card-ruled-${i}`);
  text(s, [
    { text: title, options: { bold: true, fontSize: 17, color: C.text2, breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: what, options: { color: C.accent3, breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: verdict, options: { color: C.text1 } },
  ], { x: x + 0.25, y: 1.85, w: 3.35, h: 4.45, fontSize: 14 });
});

s = content("Surgery", "History can't resolve an effect this small");
stat(s, 0.6, 1.6, 3.85, "0.1–0.3", "points: the plausible surgery reduction (≈10-point enrollment lift × ~30% risk reduction × ~3% risk)", EFFECT, "stat-plausible");
stat(s, 4.74, 1.6, 3.85, `±${((cal.hi89 - cal.lo89) / 2).toFixed(1)}`, "points: half-width of the best-controlled 89% interval", BIAS, "stat-uncertainty");
stat(s, 8.88, 1.6, 3.85, pts(LM.wald.mean), `points per enrollment (Wald ratio ${iv(LM.wald)}): too wide, and still biased`, null, "stat-wald");
card(s, 0.6, 3.95, 12.13, 2.75, "card-power");
text(s, [
  { text: "What it would take", options: { bold: true, fontSize: 20, color: C.text2, breakLine: true } },
  ...bullets([
    "The best-controlled interval is several times wider than any realistic effect, and it still leans on an untestable assumption about the remaining selection.",
    "A randomized holdout removes the selection entirely. Enrollment is measurable in one quarter; surgery needs the holdout kept for 12 months and pooled across quarters.",
    "Use the 11.5-point enrollment lift in the value-of-the-list calculation now, and treat the surgery reduction as an assumption until the pilot reads out.",
  ]),
], { x: 0.9, y: 4.15, w: 11.5, h: 2.45, paraSpaceAfter: 6, fontSize: 15 });

// 13. Recommendation
pres.addSection({ title: "Recommendation" });
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Recommendation" });
s.addText("Bank the enrollment lift; let the pilot answer surgery", { placeholder: "title" });
s.addText([
  { text: `Outreach raises enrollment by about ${enr.mean.toFixed(1)} points, consistently and at every risk level.`, options: { bullet: true, breakLine: true } },
  { text: "Its effect on surgery is undetectable from history: Meridian's targeting used information outside the data.", options: { bullet: true, breakLine: true } },
  { text: "Run the 80/20 randomized holdout, keep it for 12 months, and measure surgery directly.", options: { bullet: true } },
], { placeholder: "body" });

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
