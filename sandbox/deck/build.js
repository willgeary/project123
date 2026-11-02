// Builds model_comparison.pptx from deck_data.json (written by prepare.py).
// Run: NODE_PATH=<dir with pptxgenjs> node build.js <path to pptx skill apply_theme.js>
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.argv[2]);

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(__dirname, "model_comparison.pptx");
const M = D.metrics;
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const n = (x) => Math.round(x).toLocaleString("en-US");

const THEME = {
  name: "Surgery risk comparison",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B0B0B", lt1: "FFFFFF", dk2: "1C2B3A", lt2: "EEF2F6",
    accent1: "2A78D6", accent2: "EB6834", accent3: "52514E", accent4: "1BAF7A",
    accent5: "EDA100", accent6: "E87BA4", hlink: "2A78D6", folHlink: "6E4BB3",
  },
};
// Validated categorical pair (dataviz reference palette, slots 1-2) plus ink for "observed".
const FREQ = "2A78D6", BAYES = "EB6834", OBS = "52514E", BASE = "B4B2A9", GRID = "E3E6EA";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Frequentist vs Bayesian surgery-risk models";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const footer = { text: { text: "Vi Labs · Meridian lumbar surgery risk · sandbox comparison", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent3, margin: 0 } } };
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.6, fontFace: "Cambria", fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.1, w: 11.7, h: 1.2, fontSize: 20, color: "CADCFC", valign: "top" }, text: "" } },
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
const text = (s, t, o) => s.addText(t, { isTextBox: true, fontSize: 16, color: C.text1, valign: "top", margin: 0, ...o });
const card = (s, x, y, w, h, name) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name });
const stat = (s, x, y, w, big, label, color, name) => {
  card(s, x, y, w, 2.0, name);
  text(s, big, { x: x + 0.3, y: y + 0.25, w: w - 0.6, h: 0.95, fontSize: 48, bold: true, color: color || C.text2, fontFace: "Cambria", valign: "middle" });
  text(s, label, { x: x + 0.3, y: y + 1.25, w: w - 0.6, h: 0.65, fontSize: 14, color: C.accent3 });
};
const chartBase = (title) => ({
  showTitle: !!title, title, titleFontSize: 14, titleColor: "1C2B3A", titleFontFace: "+mn-lt",
  catAxisLabelColor: "52514E", valAxisLabelColor: "52514E", catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: GRID, size: 0.5 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false,
  showLegend: true, legendPos: "t", legendFontSize: 11, legendColor: "52514E", legendFontFace: "+mn-lt",
  catAxisTitleColor: "52514E", valAxisTitleColor: "52514E", catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
});

// 1. Title
pres.addSection({ title: "Overview" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Overview" });
s.addText("Frequentist vs Bayesian surgery-risk models", { placeholder: "title" });
s.addText("Same accuracy, same list. The difference is how each says how sure it is.", { placeholder: "body" });
s.addNotes("A comparison of two ways to estimate each patient's 12-month risk of lumbar spine surgery while covered by Meridian. Both use the same features, the same training data (2024-07 to 2025-04) and the same out-of-time test (2025-07).");

// 2. Bottom line
s = content("Overview", "The bottom line");
const fm = M.frequentist, bm = M.bayesian, cap = D.capacity;
const SIM = Object.fromEntries(D.coverage.simulation.map((r) => [r.method, r]));
const COVERAGE_BULLET = `Neither interval is better calibrated: in a simulation with known true risk, 89% intervals covered ${pct(SIM["bayesian N(0, 0.25)"].coverage)} (Bayesian) vs ${pct(SIM["frequentist bootstrap"].coverage)} (bootstrap).`;
stat(s, 0.6, 1.6, 3.85, fm.auc.toFixed(2), `AUC for both on the 2025-07 test (${fm.auc.toFixed(3)} vs ${bm.auc.toFixed(3)})`, null, "stat-auc");
stat(s, 4.74, 1.6, 3.85, pct(D.list_overlap), `of the top-${n(cap)} contact list is the same patients under both models`, null, "stat-overlap");
stat(s, 8.88, 1.6, 3.85, pct(fm.mean_predicted), `predicted vs ${pct(fm.observed_rate)} observed in 2025-07, after adding time since first visit (was +30%)`, null, "stat-calibrated");
card(s, 0.6, 3.95, 12.13, 2.75, "card-verdict");
text(s, [
  { text: "What differs is uncertainty, not accuracy", options: { bold: true, fontSize: 20, color: C.text2, breakLine: true } },
  { text: "Both models give an interval around every prediction: a bootstrap confidence interval for the frequentist model, a posterior credible interval for the Bayesian one.", options: { bullet: true, breakLine: true } },
  { text: `The Bayesian model also answers "how sure are we this patient belongs on the list?": ${n(D.list_certainty.sure_in)} patients are near-certain, ${n(D.list_certainty.uncertain)} are a coin flip.`, options: { bullet: true, breakLine: true } },
  { text: COVERAGE_BULLET, options: { bullet: true, breakLine: true } },
  { text: "Recommendation: score with the frequentist model and use the Bayesian one to report uncertainty.", options: { bullet: true } },
], { x: 0.9, y: 4.15, w: 11.5, h: 2.45, paraSpaceAfter: 6, fontSize: 15 });
s.addNotes("Accuracy is identical and the contact list is nearly identical. Both models over-predicted 2025-07 by about 30% until we added time since the first back pain visit, capped at 12 months or more. With it, both are calibrated.");

// 3. Setup
s = content("Overview", "Two methods, identical inputs");
card(s, 0.6, 1.6, 5.9, 3.0, "card-freq");
card(s, 6.83, 1.6, 5.9, 3.0, "card-bayes");
s.addShape(pres.shapes.OVAL, { x: 0.9, y: 1.9, w: 0.45, h: 0.45, fill: { color: FREQ }, line: { color: FREQ }, objectName: "dot-freq" });
s.addShape(pres.shapes.OVAL, { x: 7.13, y: 1.9, w: 0.45, h: 0.45, fill: { color: BAYES }, line: { color: BAYES }, objectName: "dot-bayes" });
text(s, "Frequentist", { x: 1.5, y: 1.88, w: 4.5, h: 0.5, fontSize: 22, bold: true, color: C.text2, valign: "middle" });
text(s, "Bayesian", { x: 7.73, y: 1.88, w: 4.5, h: 0.5, fontSize: 22, bold: true, color: C.text2, valign: "middle" });
const freqLines = ["scikit-learn L2 logistic regression (C = 1)", "One best-fit set of coefficients", "Intervals from 200 patient-level bootstrap refits", `Fits in ${fm.fit_seconds.toFixed(2)} s`];
const bayesLines = ["PyMC logistic regression, nutpie NUTS sampler", "N(0, 0.25) priors chosen by a prior predictive check", "4,000 posterior draws of every coefficient", `Fits in ${bm.fit_seconds.toFixed(0)} s, 0 divergences, r-hat ≤ 1.003`];
const bullets = (lines) => lines.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < lines.length - 1 } }));
text(s, bullets(freqLines), { x: 0.9, y: 2.65, w: 5.3, h: 1.85, paraSpaceAfter: 8 });
text(s, bullets(bayesLines), { x: 7.13, y: 2.65, w: 5.3, h: 1.85, paraSpaceAfter: 8 });
text(s, `Shared: ${D.n_features} standardized features (recent activity, BMI, age, conditions, and time since first back pain visit capped at 12m+), trained on 60,594 snapshots (2024-07 to 2025-04, 1,878 surgeries), tested out-of-time on ${n(D.n_valid)} patients in 2025-07. Target: surgery while still covered within 12 months.`,
  { x: 0.6, y: 4.9, w: 12.13, h: 0.8, fontSize: 14, color: C.accent3 });
s.addNotes("C = 1 in scikit-learn is mathematically the same as an N(0, 1) prior. With that prior, the Bayesian posterior mode matches the scikit-learn coefficients to within 0.017. The Bayesian fit uses a tighter N(0, 0.25) prior because N(0, 1) implies 27% of patients have a better-than-even chance of surgery, which is implausible.");

// 4. Accuracy section
pres.addSection({ title: "Accuracy" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Accuracy" });
s.addText("Accuracy", { placeholder: "title" });
s.addText("Out-of-time test on 2025-07: discrimination, calibration, and how many surgeries the list catches", { placeholder: "body" });

// 5. Metrics table
s = content("Accuracy", "Same accuracy on every metric");
const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 14 } });
const row = (label, f, b, note) => [
  { text: label, options: { bold: true, color: C.text2 } }, { text: f, options: { align: "center" } }, { text: b, options: { align: "center" } }, { text: note, options: { color: C.accent3, fontSize: 12 } },
];
s.addTable([
  [hdr("Metric"), { ...hdr("Frequentist"), options: { ...hdr("").options, align: "center" } }, { ...hdr("Bayesian"), options: { ...hdr("").options, align: "center" } }, hdr("What it means")],
  row("AUC", fm.auc.toFixed(3), bm.auc.toFixed(3), "Chance a surgery patient is ranked above a non-surgery patient"),
  row("Brier score", fm.brier.toFixed(4), bm.brier.toFixed(4), "Mean squared error of the probabilities (lower is better)"),
  row("Log loss", fm.log_loss.toFixed(4), bm.log_loss.toFixed(4), "Penalizes confident wrong predictions (lower is better)"),
  row("Calibration slope", fm.calibration_slope.toFixed(2), bm.calibration_slope.toFixed(2), "1.00 = spread of risk is right; slightly below 1 is mild over-confidence"),
  row("Mean predicted / observed", `${pct(fm.mean_predicted)} / ${pct(fm.observed_rate)}`, `${pct(bm.mean_predicted)} / ${pct(bm.observed_rate)}`, "Calibrated in both (was 30% too high without the tenure feature)"),
  row(`Surgeries caught in top ${n(cap)}`, pct(fm[`top_${cap}_capture`]), pct(bm[`top_${cap}_capture`]), `Top ${n(cap)} is ${pct(cap / D.n_valid, 0)} of patients, about Meridian's quarterly capacity`),
  row("Fit time", `${fm.fit_seconds.toFixed(2)} s`, `${bm.fit_seconds.toFixed(0)} s`, "Matters for automated retraining"),
], { x: 0.6, y: 1.6, w: 12.13, colW: [3.0, 1.7, 1.7, 5.73], fontSize: 14, color: C.text1, rowH: 0.6, valign: "middle", border: { type: "solid", pt: 0.5, color: "D9DEE4" }, margin: [0.05, 0.12, 0.05, 0.12] });
s.addNotes("Differences between the two models are in the third or fourth decimal place, well inside sampling noise for 431 surgeries in the test period.");

// 6. Calibration
s = content("Accuracy", "Both models are calibrated out-of-time");
const deciles = D.calibration.frequentist.map((r) => `D${r.decile}`);
s.addChart(pres.charts.LINE, [
  { name: "Frequentist predicted", labels: deciles, values: D.calibration.frequentist.map((r) => r.predicted * 100) },
  { name: "Bayesian predicted", labels: deciles, values: D.calibration.bayesian.map((r) => r.predicted * 100) },
  { name: "Observed", labels: deciles, values: D.calibration.frequentist.map((r) => r.observed * 100) },
], {
  ...chartBase("Predicted vs observed surgery rate by risk decile, 2025-07 (%)"),
  x: 0.6, y: 1.5, w: 8.2, h: 5.3, chartColors: [FREQ, BAYES, OBS], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 7,
  valAxisLabelFormatCode: "0.0", showCatAxisTitle: true, catAxisTitle: "Risk decile (D10 = highest predicted risk)",
});
card(s, 9.1, 1.6, 3.63, 5.1, "card-calibration");
text(s, [
  { text: "The two predicted lines overlap", options: { bold: true, color: C.text2, breakLine: true } },
  { text: "The models agree decile by decile.", options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "Both track observed", options: { bold: true, color: C.text2, breakLine: true } },
  { text: `Overall ${pct(fm.mean_predicted)} predicted vs ${pct(fm.observed_rate)} observed, even though risk fell from 3.1% in training to 2.2%.`, options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "Slight misses at the bottom", options: { bold: true, color: C.text2, breakLine: true } },
  { text: "The lowest deciles run a little below observed (0.4% vs 0.7%). That doesn't matter for a top-of-list decision." },
], { x: 9.35, y: 1.85, w: 3.13, h: 4.7, fontSize: 14 });
s.addNotes("Before the tenure feature, every decile sat about 30% above observed. Now the top decile is 8.4% predicted against 8.3% observed.");

// 7. Gains
s = content("Accuracy", `A ${n(cap)}-patient list catches about ${pct(fm[`top_${cap}_capture`], 0)} of surgeries`);
s.addChart(pres.charts.LINE, [
  { name: "Frequentist", labels: D.gains.share_contacted.map((x) => `${x}%`), values: D.gains.frequentist.map((x) => x * 100) },
  { name: "Bayesian", labels: D.gains.share_contacted.map((x) => `${x}%`), values: D.gains.bayesian.map((x) => x * 100) },
  { name: "Random list", labels: D.gains.share_contacted.map((x) => `${x}%`), values: D.gains.share_contacted },
], {
  ...chartBase("Share of 2025-07 surgeries captured vs share of patients contacted (%)"),
  x: 0.6, y: 1.5, w: 8.2, h: 5.3, chartColors: [FREQ, BAYES, BASE], lineSize: 2, lineDataSymbol: "none",
  valAxisMaxVal: 100, valAxisLabelFormatCode: "0", showCatAxisTitle: true, catAxisTitle: "Patients contacted, highest risk first", catAxisLabelFrequency: 2,
});
stat(s, 9.1, 1.6, 3.63, pct(fm[`top_${cap}_capture`], 0), `of surgeries land in the top ${n(cap)}, for either model`, null, "stat-capture");
stat(s, 9.1, 3.9, 3.63, `${(fm[`top_${cap}_capture`] / (cap / D.n_valid)).toFixed(1)}×`, `a random list's catch. The two models differ by ${Math.round((bm[`top_${cap}_capture`] - fm[`top_${cap}_capture`]) * D.metrics.frequentist.observed_rate * D.n_valid)} surgeries`, null, "stat-lift");
s.addNotes(`At Meridian's capacity of about ${n(cap)} contacts a quarter (12.5% of eligible patients), both models capture over three times as many surgeries as a random list. Adding the tenure feature raised capture from 36% to ${pct(fm[`top_${cap}_capture`], 0)}.`);

// 7b. The missing feature
s = content("Accuracy", "The 30% over-prediction was a missing feature");
const dates = Object.keys(D.drift.without_tenure);
s.addChart(pres.charts.LINE, [
  { name: "Without time since first visit", labels: dates, values: dates.map((d) => D.drift.without_tenure[d]) },
  { name: "With it, capped at 12m+", labels: dates, values: dates.map((d) => D.drift.with_tenure[d]) },
  { name: "Perfect calibration", labels: dates, values: dates.map(() => 1) },
], {
  ...chartBase("Predicted ÷ observed by index date (2025-07 is out-of-time)"),
  x: 0.6, y: 1.5, w: 8.2, h: 5.3, chartColors: [BASE, FREQ, OBS], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 7,
  valAxisMinVal: 0.8, valAxisMaxVal: 1.4, valAxisLabelFormatCode: "0.0", showCatAxisTitle: true, catAxisTitle: "Index date",
});
card(s, 9.1, 1.6, 3.63, 5.1, "card-drift");
text(s, [
  { text: "What went wrong", options: { bold: true, color: C.text2, breakLine: true } },
  { text: `Patients who reach a year without surgery are far less likely to have one. Without that feature, the model over-predicted them ${D.drift.without_tenure_12m_plus_ratio.toFixed(1)}×.`, options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "Why it grew over time", options: { bold: true, color: C.text2, breakLine: true } },
  { text: "The cohort is closed, so it ages: patients 12+ months out were 19% of training rows and 48% of the 2025-07 test.", options: { breakLine: true } },
  { text: " ", options: { fontSize: 8, breakLine: true } },
  { text: "Why capped", options: { bold: true, color: C.text2, breakLine: true } },
  { text: "Every 2026-07 patient is 12m+, a band training covers well." },
], { x: 9.35, y: 1.85, w: 3.13, h: 4.7, fontSize: 13 });
s.addNotes("This is the same for both methods: it's a feature problem, not a frequentist vs Bayesian one. We checked reporting lag first: about 70% of EHR surgeries match a paid claim in every quarter through 2026-Q2, so the drop in surgeries is real.");

// 8. Examples section
pres.addSection({ title: "Real examples" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Real examples" });
s.addText("Five real patients", { placeholder: "title" });
s.addText("Point estimates, intervals, and the full distribution behind each prediction (2025-07 test period)", { placeholder: "body" });

// 9. Example table
s = content("Real examples", "Point estimates agree; intervals differ");
const E = D.examples;
const profile = (e) => `${e.age}${e.sex}, BMI ${e.bmi}${e.prior_consult ? ", prior consult" : ""}${e.injection_365d ? ", injection" : ""}${e.imaging_180d ? ", recent imaging" : ""}, ${e.lbp_visits_365d} LBP visits/yr`;
const ci = (c) => `${pct(c[0])} – ${pct(c[1])}`;
const th = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13, align: "center" } });
s.addTable([
  [{ ...th("Situation"), options: { ...th("").options, align: "left" } }, { ...th("Profile"), options: { ...th("").options, align: "left" } }, th("Outcome"), th("Frequentist (89% CI)"), th("Bayesian (89% CrI)"), th("P(on list)")],
  ...E.map((e) => [
    { text: e.label, options: { bold: true, color: C.text2 } },
    { text: profile(e), options: { fontSize: 12 } },
    { text: e.outcome, options: { align: "center", bold: e.outcome === "Surgery" } },
    { text: [{ text: pct(e.freq_point), options: { bold: true, breakLine: true } }, { text: ci(e.freq_ci), options: { fontSize: 11, color: C.accent3 } }], options: { align: "center" } },
    { text: [{ text: pct(e.bayes_mean), options: { bold: true, breakLine: true } }, { text: ci(e.bayes_ci), options: { fontSize: 11, color: C.accent3 } }], options: { align: "center" } },
    { text: e.p_in_top.toFixed(2), options: { align: "center", bold: true } },
  ]),
], { x: 0.6, y: 1.5, w: 12.13, colW: [2.35, 4.08, 1.25, 1.6, 1.6, 1.25], fontSize: 13, color: C.text1, rowH: [0.5, 0.85, 0.85, 0.85, 0.85, 0.85], valign: "middle", border: { type: "solid", pt: 0.5, color: "D9DEE4" }, margin: [0.04, 0.1, 0.04, 0.1] });
text(s, `P(on list) is Bayesian only: the share of posterior draws in which the patient ranks in the top ${n(cap)}. The cut-off sits at ${pct(D.cutoff)} predicted risk. The frequentist interval comes from 200 bootstrap refits.`,
  { x: 0.6, y: 6.35, w: 12.13, h: 0.55, fontSize: 12, color: C.accent3 });
s.addNotes("Both models agree on every point estimate to within half a percentage point. The cut-off patient is the interesting row: a 47% chance of making the list means the data cannot really tell whether this patient belongs on it. The missed surgery is a reminder that a 0.7% risk is not zero: some surgeries happen to patients who look low-risk.");

// 10-11. Distributions
const histChart = (slide, e, x, y, w, h) => {
  const mid = e.hist_edges.slice(0, -1).map((a, i) => (a + e.hist_edges[i + 1]) / 2);
  const labels = mid.map((m) => pct(m, m < 0.1 ? 1 : 0));
  slide.addChart(pres.charts.BAR, [
    { name: "Frequentist bootstrap", labels, values: e.freq_hist.map((v) => v * 100) },
    { name: "Bayesian posterior", labels, values: e.bayes_hist.map((v) => v * 100) },
  ], {
    ...chartBase(e.label), x, y, w, h, barDir: "col", barGrouping: "clustered", barGapWidthPct: 40,
    chartColors: [FREQ, BAYES], catAxisLabelFrequency: 6, valAxisLabelFormatCode: "0", showValAxisTitle: true, valAxisTitle: "% of draws",
  });
};
s = content("Real examples", "Each prediction is a distribution, not a point");
histChart(s, E[0], 0.6, 1.5, 4.0, 4.5);
histChart(s, E[2], 4.67, 1.5, 4.0, 4.5);
histChart(s, E[3], 8.73, 1.5, 4.0, 4.5);
text(s, "Distribution of each patient's 12-month risk across 200 bootstrap refits (blue) and 4,000 posterior draws (orange). The x-axis is the predicted risk.",
  { x: 0.6, y: 6.2, w: 12.13, h: 0.6, fontSize: 14, color: C.accent3 });
s.addNotes("Both models put the bulk of each distribution in the same place. The Bayesian posterior is a little narrower because its prior pulls weak coefficients toward zero. The bootstrap also resamples patients, which adds the variation between patient samples.");

s = content("Real examples", "Same story for the patients the list gets wrong");
histChart(s, E[1], 0.6, 1.5, 5.9, 4.5);
histChart(s, E[4], 6.83, 1.5, 5.9, 4.5);
text(s, `A confident ${pct(E[1].bayes_mean, 0)} risk still means most such patients don't have surgery. A tight interval around ${pct(E[4].bayes_mean)} still allows the surgery we missed. Intervals describe the risk, not the outcome.`,
  { x: 0.6, y: 6.2, w: 12.13, h: 0.6, fontSize: 14, color: C.accent3 });
s.addNotes("Neither patient is a model error in the usual sense. The intervals are narrow, so the models are confident about the risk. The outcome is still a coin flip weighted by that risk.");

// 12. Do frequentist models have intervals?
s = content("Real examples", "Yes, the frequentist model has intervals too");
card(s, 0.6, 1.6, 5.9, 3.3, "card-freq-ci");
card(s, 6.83, 1.6, 5.9, 3.3, "card-bayes-ci");
text(s, [
  { text: "Frequentist confidence interval", options: { bold: true, fontSize: 20, color: FREQ, breakLine: true } },
  { text: "Built from the coefficients' sampling uncertainty, either by bootstrap (used here) or by the delta method on the standard errors.", options: { bullet: true, breakLine: true } },
  { text: '"If we refit on many datasets like this one, 89% of intervals would contain the true risk."', options: { bullet: true, breakLine: true } },
  { text: `Median width here: ${pct(D.interval_width.freq_median)}`, options: { bullet: true } },
], { x: 0.9, y: 1.85, w: 5.3, h: 2.9, paraSpaceAfter: 8 });
text(s, [
  { text: "Bayesian credible interval", options: { bold: true, fontSize: 20, color: BAYES, breakLine: true } },
  { text: "Read directly off the posterior: prior plus data.", options: { bullet: true, breakLine: true } },
  { text: '"Given the data and prior, there is an 89% probability this patient\'s risk is in this range."', options: { bullet: true, breakLine: true } },
  { text: `Median width here: ${pct(D.interval_width.bayes_median)}. It's narrower because the prior shrinks weak coefficients.`, options: { bullet: true } },
], { x: 7.13, y: 1.85, w: 5.3, h: 2.9, paraSpaceAfter: 8 });
text(s, "What scikit-learn doesn't give out of the box is the interval: predict_proba returns only a point. The uncertainty still exists; you have to compute it.",
  { x: 0.6, y: 5.2, w: 12.13, h: 0.5, fontSize: 14, color: C.accent3 });
s.addNotes("This answers the question of whether the frequentist model has confidence intervals. It does, but scikit-learn does not compute them, so we bootstrapped. The two intervals mean different things: a confidence interval is about the procedure, a credible interval is a probability statement about this patient. For ranking and capacity decisions the practical difference is small.");

// 13. PPC section
pres.addSection({ title: "Predictive checks" });
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Predictive checks" });
s.addText("Posterior predictive checks", { placeholder: "title" });
s.addText("Simulate outcomes from each model and compare with what actually happened in 2025-07", { placeholder: "body" });

// 14. PPC charts
s = content("Predictive checks", "Simulated surgery counts match what happened");
const ppcChart = (key, title, x) => {
  const P = D.ppc[key];
  const labels = P.edges.slice(0, -1).map((a, i) => `${Math.round((a + P.edges[i + 1]) / 2)}`);
  s.addChart(pres.charts.BAR, [
    { name: "Frequentist (bootstrap)", labels, values: P.freq.map((v) => v * 100) },
    { name: "Bayesian (posterior)", labels, values: P.bayes.map((v) => v * 100) },
  ], {
    ...chartBase(title), x, y: 1.5, w: 5.9, h: 3.9, barDir: "col", barGrouping: "clustered", barGapWidthPct: 40, chartColors: [FREQ, BAYES],
    catAxisLabelFrequency: 3, valAxisLabelFormatCode: "0", showValAxisTitle: true, valAxisTitle: "% of simulations", showCatAxisTitle: true, catAxisTitle: "Simulated number of surgeries",
  });
};
ppcChart("top", `Surgeries among the top ${n(cap)} patients`, 0.6);
ppcChart("all", `Surgeries among all ${n(D.n_valid)} patients`, 6.83);
const lowNote = (P) => (P.observed >= P.bayes_interval[0] && P.observed <= P.bayes_interval[1] && P.observed >= P.freq_interval[0] && P.observed <= P.freq_interval[1] ? "Inside both ranges." : "Outside at least one range.");
const pi = (iv) => `${Math.round(iv[0])}–${Math.round(iv[1])}`;
card(s, 0.6, 5.55, 5.9, 1.25, "card-ppc-top");
card(s, 6.83, 5.55, 5.9, 1.25, "card-ppc-all");
text(s, [{ text: `Observed: ${D.ppc.top.observed}`, options: { bold: true, fontSize: 20, color: C.text2, breakLine: true } }, { text: `Bayesian 89% range ${pi(D.ppc.top.bayes_interval)}, frequentist ${pi(D.ppc.top.freq_interval)}. ${lowNote(D.ppc.top)}` }],
  { x: 0.85, y: 5.68, w: 5.4, h: 1.0, fontSize: 14 });
text(s, [{ text: `Observed: ${D.ppc.all.observed}`, options: { bold: true, fontSize: 20, color: C.text2, breakLine: true } }, { text: `Bayesian 89% range ${pi(D.ppc.all.bayes_interval)}, frequentist ${pi(D.ppc.all.freq_interval)}. ${lowNote(D.ppc.all)}` }],
  { x: 7.08, y: 5.68, w: 5.4, h: 1.0, fontSize: 14 });
s.addNotes("Both checks now pass: the observed counts sit near the middle of both models' simulated ranges. Before the tenure feature was added, no simulation from either model got as low as the observed count.");

// 14b. Interval coverage
s = content("Predictive checks", "Bayesian intervals are not better calibrated");
const simOrder = ["frequentist bootstrap", "bayesian N(0, 0.25)", "bayesian N(0, 1)"];
s.addChart(pres.charts.BAR, [{ name: "Coverage", labels: ["Bootstrap", "Bayesian N(0, 0.25)", "Bayesian N(0, 1)"], values: simOrder.map((m) => SIM[m].coverage * 100) }], {
  ...chartBase("Simulation: share of patients whose 89% interval contains their true risk (%)"), showLegend: false,
  x: 0.6, y: 1.5, w: 5.9, h: 4.1, barDir: "col", barGapWidthPct: 60, chartColors: [FREQ, BAYES, "F5A57F"], valAxisMinVal: 0, valAxisMaxVal: 100,
  valAxisLabelFormatCode: "0", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 12, dataLabelColor: "1C2B3A", dataLabelFontFace: "+mn-lt",
});
const covHdr = (t, al = "center") => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 12, align: al } });
const realCols = ["frequentist bootstrap", "bayesian N(0, 0.25)", "plug-in binomial (no parameter uncertainty)"];
s.addTable([
  [covHdr("2025-07 patient groups", "left"), covHdr("Bootstrap"), covHdr("Bayesian"), covHdr("No parameter uncertainty")],
  ...D.coverage.real.map((r) => [{ text: r.groups, options: { bold: true, color: C.text2 } }, ...realCols.map((c) => ({ text: pct(r[c], 0), options: { align: "center" } }))]),
], { x: 6.83, y: 1.65, w: 5.9, colW: [2.3, 1.1, 1.1, 1.4], fontSize: 13, color: C.text1, rowH: 0.55, valign: "middle", border: { type: "solid", pt: 0.5, color: "D9DEE4" }, margin: [0.04, 0.08, 0.04, 0.08] });
text(s, "Real data: share of groups whose observed surgery count falls in the 89% predictive interval.", { x: 6.83, y: 4.55, w: 5.9, h: 0.6, fontSize: 12, color: C.accent3 });
card(s, 0.6, 5.75, 12.13, 1.05, "card-coverage");
text(s, `Simulation (10 replicates): both hit about 89%, and the 1.5-point gap is within replicate noise (each run ranges ${pct(SIM["frequentist bootstrap"].coverage_min, 0)}–${pct(SIM["frequentist bootstrap"].coverage_max, 0)}). Real data: all three columns are identical, because parameter uncertainty is tiny next to outcome noise. The miss in risk-sorted groups is model error that no interval represents.`,
  { x: 0.85, y: 5.85, w: 11.6, h: 0.85, fontSize: 13, valign: "middle" });
s.addNotes("Test 1 simulates outcomes from a known true model, refits each method 10 times and checks whether each patient's 89% interval contains the true risk. Test 2 uses real 2025-07 outcomes: a single patient's outcome can't check an interval, but a group's surgery count can. Small random groups over-cover because counts are discrete. The 79% for risk-sorted groups of 1,000 reflects the bottom deciles, where predicted risk runs about 0.3 points below observed. That's misspecification, which neither interval includes.");

// 15. List certainty
s = content("Predictive checks", "What only the Bayesian model adds: list certainty");
stat(s, 0.6, 1.6, 3.85, n(D.list_certainty.sure_in), `patients make the top ${n(cap)} in over 90% of posterior draws`, BAYES, "stat-sure");
stat(s, 4.74, 1.6, 3.85, n(D.list_certainty.uncertain), "patients make it in 10–90% of draws: effectively interchangeable", BAYES, "stat-uncertain");
stat(s, 8.88, 1.6, 3.85, pct(D.cutoff), `predicted-risk cut-off for the top ${n(cap)} in 2025-07`, null, "stat-cutoff");
card(s, 0.6, 3.95, 12.13, 2.75, "card-use");
text(s, [
  { text: "How to use it", options: { bold: true, fontSize: 20, color: C.text2, breakLine: true } },
  { text: "Contact the near-certain group every quarter.", options: { bullet: true, breakLine: true } },
  { text: "The uncertain group is the natural pool for the pilot's randomized holdout: the data can't separate these patients, so randomizing among them costs almost nothing.", options: { bullet: true, breakLine: true } },
  { text: "Report the value of the list as a range, carrying uncertainty in the risk, the enrollment lift and the surgery cost.", options: { bullet: true } },
], { x: 0.9, y: 4.2, w: 11.5, h: 2.35, paraSpaceAfter: 8 });
s.addNotes("P(on list) comes from re-ranking all patients under each of 1,000 posterior draws. A frequentist bootstrap can approximate this, but the Bayesian posterior gives it directly.");

// 16. Recommendation
pres.addSection({ title: "Recommendation" });
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Recommendation" });
s.addText("Score with frequentist, report uncertainty with Bayesian", { placeholder: "title" });
s.addText([
  { text: "Same accuracy at 1/1000th of the fit time. Simpler to automate.", options: { bullet: true, breakLine: true } },
  { text: `Time since first visit (capped at 12m+) fixed the shared 30% over-prediction. The 2026-07 list is low-risk: ${n(D.score.above_breakeven)} patients clear the 1.7% break-even.`, options: { bullet: true, breakLine: true } },
  { text: "Bayesian intervals are no better calibrated. Bayesian earns its place in list certainty, the pilot pool, and a future model across clients.", options: { bullet: true } },
], { placeholder: "body" });
s.addNotes("Recommendation and next steps.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
