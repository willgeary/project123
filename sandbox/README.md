# Sandbox: frequentist vs Bayesian surgery-risk model

**Target:** P(lumbar spine surgery while still covered, within 12 months | eligible at t). Leaving coverage is a competing event. The same features, standardization and out-of-time split are used for both attempts.

| File | What it does |
|---|---|
| `panel.py` | Builds the snapshot panel and features (recent activity, BMI, age, conditions, and time since first LBP visit capped at 12m+). Results are cached in `.cache/`. `python panel.py` rebuilds and checks them |
| `evaluate.py` | Shared metrics (AUC, Brier, calibration, top-2,500 capture). `python evaluate.py` runs a self-check |
| `frequentist/fit.py` | sklearn L2 logistic regression (C = 1) with patient-clustered bootstrap CIs. Saves the bootstrap draws |
| `bayesian/fit.py` | PyMC + nutpie logistic regression: prior predictive check, MAP check, NUTS, diagnostics, PPC |
| `compare.py` | Side-by-side results |
| `interval_coverage/test.py` | Do the 89% intervals cover 89%? Simulation with known truth, plus group counts on real 2025-07 outcomes |
| `deck/` | `prepare.py` + `build.js` produce `model_comparison.pptx` |

```bash
uv run python sandbox/panel.py
uv run python sandbox/frequentist/fit.py
uv run --group bayes python sandbox/bayesian/fit.py
cd sandbox && uv run python compare.py && cd ..
uv run --group bayes python sandbox/interval_coverage/test.py   # ~10 min
uv run --group bayes python sandbox/deck/prepare.py
```

## Data

| Split | Index dates | Rows | Surgeries | Rate |
|---|---|---|---|---|
| train | 2024-07 → 2025-04 (quarterly) | 60,594 | 1,878 | 3.10% |
| valid | 2025-07 (out-of-time) | 19,986 | 431 | 2.16% |
| score | 2026-07-01 (no label) | 16,010 | – | – |

## Results

| Validation 2025-07 | Frequentist | Bayesian |
|---|---|---|
| AUC | 0.733 | 0.732 |
| Brier | 0.0203 | 0.0203 |
| Mean predicted vs observed | 2.16% vs 2.16% | 2.18% vs 2.16% |
| Calibration slope | 0.96 | 0.97 |
| Top-2,500 capture of surgeries | 41.8% | 41.5% |
| Fit time | 0.04 s | 46 s |

- **The two give the same model.** Coefficients correlate at 0.998, with a maximum difference of 0.053 on the log-odds scale. On 2026-07, the score ranks correlate at 0.998 and the top-2,500 lists share 2,453 patients (98.1%). With an N(0, 1) prior, the MAP estimate reproduces sklearn's C = 1 coefficients to within 0.006.
- **The prior predictive check is the one place the Bayesian workflow changed a decision.** C = 1 is equivalent to an N(0, 1) prior, which implies 28% of patients have a better-than-even chance of surgery. N(0, 0.25) brings that to 3%. With 1,878 events the data overrides either prior.
- **The Bayesian model's addition is uncertainty about the list itself.** In 2026-07, 1,414 patients make the top 2,500 with probability > 0.9, and 2,385 sit between 0.1 and 0.9.

## The missing feature: time since first LBP visit

The first version left this feature out. It worried that the first visit is left-truncated, and that the 2026-07 population sits entirely beyond the training range. Both models then over-predicted 2025-07 by 30% (2.80% vs 2.16%), and the gap grew steadily across index dates (predicted ÷ observed 0.88 → 1.30).

**Diagnosis.** Patients 12+ months past their first visit have a 1.0% surgery rate, but the model predicted 2.6% for them: 2.5–2.7× too high in both training and validation. The cohort is closed, so it ages: those patients were 19% of training rows but 48% of the 2025-07 test. Reporting lag was ruled out: about 70% of EHR surgeries match a paid claim in every quarter through 2026-Q2.

**Fix.** Add the feature as bands capped at 12m+ (`tenure_6_12m`, `tenure_12m_plus`).
- 99.9% of the 2026-07 population is 12m+, a band with 11.7k training rows, so there's no extrapolation.
- Left-truncation only blurs the <6m and 6–12m bands.

| | Without | With capped bands |
|---|---|---|
| Valid predicted vs observed | 2.80% vs 2.16% | 2.16% vs 2.16% |
| Predicted ÷ observed by index date | 0.88 → 1.30 | 0.97–1.03 |
| AUC | 0.703 | 0.733 |
| Top-2,500 capture | 35.7% | 41.8% |
| 2026-07 patients above the 1.7% break-even | 9,112 | 919 |

Training only on the latest quarter helps a little but doesn't fix it (1.16×). Neither would intercept recalibration, because the error depends on tenure.

**Remaining caveat.** Half the 2026-07 population is 24+ months out, which no 12-month label can reach. A 6-month-horizon check from 2026-01 would test whether risk keeps falling past 24 months.

## Interval coverage: are Bayesian intervals better calibrated?

No. `interval_coverage/test.py` runs two tests.

**1. Simulation with known truth.** The fitted frequentist model is treated as the true risk. Surgeries are simulated for the real training patients, each method is refit, and we check how often each 2025-07 patient's 89% interval contains their true risk (10 replicates).

| Method | Coverage (target 89%) | Range across replicates | Median width |
|---|---|---|---|
| Frequentist bootstrap (100 refits) | 88.0% | 72–95% | 0.6 pts |
| Bayesian N(0, 0.25) | 89.5% | 74–96% | 0.6 pts |
| Bayesian N(0, 1) | 89.0% | 75–96% | 0.6 pts |

The 1.5-point gap is within replicate noise. Coverage varies widely between replicates because every patient's interval shares the same coefficient error: one unlucky fit shifts them all.

**2. Real 2025-07 outcomes.** A single patient's outcome can't check an interval, but a group's surgery count can. For each group we build an 89% predictive interval for the count and check whether the observed count falls inside.

| Groups | Bootstrap | Bayesian | Plug-in binomial (no parameter uncertainty) |
|---|---|---|---|
| random, size 1,000 | 100% | 100% | 100% |
| random, size 200 | 96% | 96% | 96% |
| risk-sorted, size 1,000 | 79% | 79% | 79% |
| risk-sorted, size 200 | 91% | 91% | 90% |

- **All three columns are identical.** Parameter uncertainty is negligible next to the noise in outcomes.
- **Small random groups over-cover** because counts are discrete.
- **The miss in risk-sorted groups of 1,000** comes from the bottom deciles, where predicted risk runs about 0.3 points below observed. That's model misspecification, which neither interval represents.

## Outreach is not a feature

No outreach variable is a model input. That was deliberate. Outreach records Meridian's targeting decision, not the patient's clinical state: it's non-random, so its coefficient would mix selection with any real effect. It would also change meaning as soon as Vi's list drives outreach (a feedback loop), and it doesn't transfer to other clients.

One feature is downstream of outreach: `prior_program` (enrolled in PT or WM in the last 12 months). Its coefficient is +0.24, meaning enrollees have higher risk, which is selection rather than a program effect. It's kept because it describes care received, but it shouldn't be read causally.

## Verdict

For **ranking and scoring**, use the frequentist model. It gives the same answer in 1/1000th of the time, has fewer dependencies and is simpler to automate.

Keep the Bayesian version for:
- **List-membership uncertainty.** The roughly 2,400 interchangeable patients near the cut-off are a natural pool for the pilot's randomization.
- **Uncertainty propagation** into the value of the list.
- **A hierarchical model across clients** later.
