# Causal: what did Meridian's outreach do to enrollment and surgery?

Outreach wasn't randomized, so every estimate here rests on emulating the trial Meridian didn't run. Every approach is Bayesian (PyMC + nutpie, or closed-form Beta posteriors), reports 89% intervals, and is checked against **negative controls**: outcomes outreach can't affect, so any non-zero "effect" on them measures bias.

## Bottom line

| Question | Answer |
|---|---|
| Does outreach raise enrollment? | **Yes, by about 11.5 points** within 180 days (25% → 37% among contacted patients). Consistent across all approaches and every risk quintile. |
| Does outreach reduce surgery? | **Not detectably.** The best-controlled estimate is **+0.05 points [−0.42, +0.61]** (risk ratio 1.03 [0.78, 1.33]); P(protective) = 0.45. |
| Why not a firmer answer? | Contacted patients were selected on information outside the data: they had more surgeries in the first 30 days, before outreach could act. Plus the plausible true effect (~0.1–0.3 points) is smaller than the remaining uncertainty. |

This matches what the pilot design predicted. The historical data credibly estimates the enrollment lift and only bounds the surgery effect. The surgery effect needs the randomized holdout, followed for 12 months.

## Design: one emulated trial per wave

| | |
|---|---|
| Waves | 2024-07, 2025-01, 2025-07 (all have 12 months of follow-up) |
| Eligible at wave start *t* | Same population as the risk model (covered, LBP history, no surgery, no consult in the last 90 days), never opted out before *t* |
| Treatment | Any contact attempt within 92 days of *t* (intention to treat). 7,642 contacted vs 38,804 not |
| Time zero | *t* for both arms, so nobody has to survive surgery-free to count as untreated |
| Covariates | The 36 risk-model features + prior outreach in earlier waves |
| Outcomes | `enroll_180d` (PT or WM, before any surgery), `surgery_12m` (while covered) |
| Negative controls | `nc_surgery_30d` (surgery within 30 days, too soon for outreach), `nc_acute_12m` / `nc_acute_pre` (non-spine acute visits after / before *t*) |

**Who got contacted.** A Bayesian propensity model reaches AUC 0.69. The strongest driver is that Meridian rarely re-contacted earlier-wave patients (SMD −0.62). Contacted patients were also somewhat more clinically active: recent imaging (+0.19), prior consult (+0.13), MRI (+0.12). Weighting brings every covariate within 0.01 SMD, and the effective control sample stays at 26,389. Overlap is good.

## Approaches

| File | Approach | Enrollment ATT | Surgery ATT | Negative control (surgery ≤30d) |
|---|---|---|---|---|
| `a_gcomp.py` | A. Bayesian outcome model + g-computation | +11.9 [11.0, 12.7] | +0.38 [−0.01, 0.75] | +0.11 [−0.03, 0.27] |
| | A, one row per patient (first wave only) | +11.0 [9.9, 12.2] | +0.38 [−0.15, 0.93] | |
| `b_subclass.py` | B. Propensity strata (10 per wave) + Beta-Binomial | +11.5 [10.5, 12.5] | +0.65 [0.25, 1.06] | **+0.40 [0.21, 0.58]** |
| `c_aipw.py` | C. Doubly robust (AIPW), Bayesian bootstrap | +11.5 [10.6, 12.4] | +0.33 [−0.02, 0.69] | +0.13 [−0.02, 0.29] |
| `e_landmark.py` | E. Landmark at day 30, surgery days 90–365 | | +0.19 [−0.09, 0.49] | +0.08 [−0.12, 0.29] (days 30–90) |
| | E, negative-control calibrated | | **+0.05 [−0.42, 0.61]** | |

All values are percentage points, posterior mean [89% interval], as the effect on the contacted (ATT). `estimates.png` plots them.

**How to read it:**
- **The naive answer is "outreach increases surgery".** Approaches A–C all give a positive surgery effect. The negative control shows why: the contacted group also has more surgeries in the first 30 days, which outreach can't cause. Meridian selected patients who were already closer to surgery, using information outside the extract (likely referral notes or scheduled procedures).
- **B is the most biased.** Ten strata leave residual imbalance, so its negative control is +0.40. C, which stays unbiased if either model is right, agrees with A, not B.
- **E removes most of the remaining selection.** It drops the 179 surgeries already in the pipeline (the first 30 days) and measures what's left with a days 30–90 window. Then it divides the main-window risk ratio by the control-window risk ratio. This assumes the remaining selection acts as a constant risk ratio over time.
- **E-value 1.45.** An unmeasured confounder associated with both contact and surgery by a risk ratio of 1.45 would explain away the uncalibrated landmark estimate. That's a modest confounder, so the result is fragile.

**Who it works for (`d_heterogeneity.py`, hierarchical by baseline-risk quintile).**

| Quintile | Baseline 12-month risk | Enrollment ATT | Surgery ATT |
|---|---|---|---|
| Q1 | 0.6% | +13.1 [11.6, 14.9] | +0.12 [−0.05, 0.36] |
| Q2 | 1.2% | +12.0 [10.6, 13.4] | +0.13 [−0.18, 0.44] |
| Q3 | 1.9% | +11.9 [10.6, 13.4] | +0.09 [−0.36, 0.47] |
| Q4 | 3.0% | +12.1 [10.7, 13.6] | +0.70 [0.08, 1.47] |
| Q5 | 7.0% | +10.5 [8.6, 12.2] | +0.75 [−0.27, 1.79] |

- **Enrollment lift is flat across risk,** slightly lower at the top. For the list, the risk score decides *who is worth contacting*; outreach works about equally on everyone.
- **The positive surgery "effect" concentrates in Q4–Q5.** That's where selection on unseen clinical information would be strongest. It's not evidence that outreach causes surgery in high-risk patients.

**Effect of enrolling on surgery.** Dividing the surgery ATT by the enrollment ATT (a Wald ratio; per draw, uncalibrated) gives +2.8 points [−0.2, +6.1]. It inherits the selection bias above and is too wide to use. It's reported only to show why the historical data can't estimate the program effect.

## Considered and rejected

- **PCP-preference instrument.** Contact rates vary by PCP more than chance, and the first stage is strong (F ≈ 172). But the leave-one-out PCP rate correlates −0.24 with tenure, so it reflects case mix, which breaks the exclusion restriction. And with a 0.06 instrument–treatment correlation, the implied standard error on the surgery effect is about ±40 points.
- **Contacts triggered by in-wave encounters** (time-varying confounding). Ruled out: contacted and uncontacted patients had near-identical encounter activity between wave start and the contact date (median 54 vs 56 days since the last one).
- **Hidden confounders in the extract.** Cancelled or no-show consults, recent spine activity, denied claims and recent orthopedic claims all differ by less than 0.06 SMD between arms. The selection is on something outside the data.

## Caveats

- **Patients appear in up to three waves**, so pooled intervals are slightly too narrow. The first-wave-only refit gives the same answer with a wider interval.
- **Negative-control calibration assumes the remaining selection is a constant risk ratio** across the days 30–90 and 90–365 windows.
- **`enroll_180d` counts any enrollment**, not just the program offered.

## Run

```bash
uv run python sandbox/causal/data.py                         # build the trial dataset (cached)
uv run --group bayes python sandbox/causal/propensity.py     # propensity model, balance
uv run --group bayes python sandbox/causal/a_gcomp.py        # ~6 min
uv run python sandbox/causal/b_subclass.py
uv run python sandbox/causal/c_aipw.py
uv run --group bayes python sandbox/causal/d_heterogeneity.py
uv run --group bayes python sandbox/causal/e_landmark.py
cd sandbox/causal && uv run python compare.py
```
