# Applied Data Science Take-Home

Start with [`client-brief.md`](client-brief.md) — the client, their problem, and what
they asked for. Then read [`assignment.md`](assignment.md) — what to produce, the
submission contract, and how it is evaluated.

## Setup

```bash
uv sync
```

## Run

```bash
uv run main.py train   # ~10 s: fits on every labeled snapshot, writes artifacts/model.pkl
uv run main.py score   # ~6 s: scores 2026-07-01, writes out/
uv run python -m assignment.evaluation   # ~15 s: out-of-time evaluation: report, charts and PDF deck in out/evaluation/
uv run python -m assignment.list_depth   # ~2 s, after score: "how far down the list" PDF deck in out/list_depth/
uv run python -m assignment.presentation   # ~10 s, after score: the final 29-page presentation, out/presentation/final.pdf
```

| Output | Contents |
|---|---|
| `out/targets.csv` | Patients to **contact**, in priority order: `patient_token, program_recommended, score` plus `rank, tier, expected_net_value, allowed_channels, prior_consult, prior_enrollment_365d`. `score` is P(surgery while covered within 12 months). |
| `out/ranking.csv` | Every eligible patient with rank, tier and `arm` (`contact`, `holdout`, `none`). **`arm == "holdout"` is the do-not-contact list:** Meridian must suppress these patients from all outreach, its own campaigns included, for 12 months. |
| `out/sensitivity.csv` | Break-even risk and list length across the range of every economic assumption. |

Every judgment call (surgery cost, contact cost, surgery reduction, learning-band size, eligibility windows, holdout shares) lives in [`assignment/assumptions.py`](assignment/assumptions.py). Change one and re-run `score`; no retrain needed.

## Test and lint

```bash
uv run pytest
uv run ruff check && uv run ruff format --check
```

## Layout

- `main.py` — the two contract entry points, `train` and `score`. The deck modules (`evaluation`, `list_depth`, `presentation`) are run separately and write to their own `out/` folders.
- `assignment/` — the pipeline: `assumptions.py`, `features.py` (snapshots and features), `model.py`, `targeting.py` (eligibility, program, tiers, holdout, sensitivity), `evaluation.py` (out-of-time metrics, charts, report, PDF deck), `list_depth.py` (how-far-down deck), `presentation.py` (the final presentation), `slides.py` (shared slide helpers).
- `tests/` — tests.
- `data/` — the extract.
- `docs/` — [`data-profile.md`](docs/data-profile.md), [`assumptions.md`](docs/assumptions.md) (sources for every assumption), [`plan.md`](docs/plan.md).
- `exploratory/` — EDA notebook on the surgery target.
- `sandbox/` — the experiments behind each decision: frequentist vs Bayesian comparison, interval-coverage test, causal analysis of past outreach, explainability, and two decks (`sandbox/deck/`, `sandbox/causal/deck/`). Optional dependencies: `uv sync --group bayes`.

---

# Methodology

## The question, reframed

Meridian asked for a ranked list, a program per patient, a list length, and a way to know whether it worked. The quantity that serves all four is **expected surgeries avoided per contact**:

> P(surgery while covered within 12 months) × enrollment lift from contact × surgery reduction per enrollment

That splits the problem into three parts with very different evidence behind them:

| Component | Source | Confidence |
|---|---|---|
| Surgery risk | A predictive model | High: calibrated out-of-time |
| Enrollment lift | A causal estimate from past campaigns | Moderate: 11.5 points, consistent across five methods |
| Surgery reduction | A literature assumption (30%) | Low: the data can't test it |

The list is ranked by the first. The third drives how long it is. The pilot is designed around that gap.

## Data decisions that shape everything

These come from [`docs/data-profile.md`](docs/data-profile.md).

- **Two sources for surgery.** EHR `Spine Surgery` events (`Performed`, or null status, since null-status surgeries match paid claims at the same rate) **or** a paid surgery claim. Claims add 66 surgeries the EHR misses. `Scheduled` surgeries match claims only 2% of the time, so they don't count.
- **Leaving coverage is a competing event, not censoring.** Meridian's files stop at `coverage_end` while claims continue. Meridian only pays for surgeries while a patient is attributed, so the label is *surgery before coverage ends*: a plain logistic regression, with no censoring weights. That gives 2.87%; Kaplan–Meier, the "risk if they stayed" view, gives 3.02%.
- **The cohort is closed and ages.** Nobody joins after 2025-04, and the whole 2026-07 population is 12+ months past their first observed back-pain visit.
- **Status codes are unreliable** outside surgery: cancelled imaging matches paid claims as often as performed imaging.
- **Excluded from treatment:** 1,489 enrollments that happen after surgery (post-op PT).

## Modeling

- **Unit and target.** One row per eligible patient per quarterly index date. That's 2024-07 to 2025-07: five snapshots, 80,580 rows, 2,309 surgeries.
  - **Eligible:** covered on the date, back-pain history, no surgery yet, no surgical consult in the last 90 days (already late-stage).
- **Features.** 36 in total, from the 12 months before each date:
  - recent back-pain activity, imaging, MRI and injections,
  - a prior surgical consult,
  - BMI bands (from claims Z68 codes; two-thirds of patients have none),
  - age, sex, chronic conditions, utilization, prior program enrollment,
  - **time since first back-pain visit, capped at 12 months or more.**
- **Model.** L2 logistic regression (scikit-learn, C = 1) on standardized features. Calibration matters more than ranking accuracy, because the list length depends on actual probabilities.
- **Outreach history is deliberately not a feature.** It records Meridian's targeting decision, not the patient's condition, and its meaning would change once Vi's list drives outreach.
- **Frequentist vs Bayesian** (`sandbox/`). A PyMC logistic regression with priors chosen by a prior predictive check gives the same model:

  | | Frequentist | Bayesian |
  |---|---|---|
  | AUC | 0.733 | 0.732 |
  | Coefficient correlation | 0.998 (shared) | |
  | Overlap of the top-2,500 lists | 98.1% (shared) | |
  | Fit time | 0.04 s | 46 s |

  In a coverage test with known true risk, Bayesian 89% intervals covered 89.5% and bootstrap intervals 88.0%, within noise. On real outcomes, parameter uncertainty is negligible next to outcome noise. **The frequentist model ships.** Bayesian methods are kept where uncertainty drives a decision: the causal estimates, and a future multi-client hierarchical model.

## Evaluation

Train on 2024-07 to 2025-04, test **out-of-time** on 2025-07: 19,986 patients, 431 surgeries. `assignment/evaluation.py` reproduces all of it, with bootstrap 95% intervals, a confusion matrix, predicted-vs-observed and gains charts, and five real patients: [`out/evaluation/report.md`](out/evaluation/report.md), and as a 9-slide deck, [`out/evaluation/surgery_risk_model_evaluation.pdf`](out/evaluation/surgery_risk_model_evaluation.pdf).

| Metric | Value |
|---|---|
| AUC | 0.733 |
| Predicted vs observed risk | **2.16% vs 2.16%** |
| Calibration slope | 0.96 |
| Share of surgeries in the top 2,500 (evaluation list size) | 41.8% (vs 12.5% for a random list) |

**The calibration fix that mattered.** The first version left out time since first visit, worried about extrapolation. It over-predicted 2025-07 by 30%, and the error grew steadily across index dates. Patients 12+ months out run at 1.0% observed vs 2.6% predicted, and they grow from 19% of training rows to 48% of the test.
- **Fix:** adding the feature *capped* at 12m+ made predicted ÷ observed 0.97–1.03 at every index date and raised AUC from 0.70 to 0.73.
- **Ruled out:** a reporting lag. EHR-to-claims match rates are steady through 2026-Q2.

## Did past outreach work? (`sandbox/causal/`)

One emulated trial per wave (2024-07, 2025-01, 2025-07):
- **Time zero:** wave start, for both arms.
- **Treatment:** any contact within the wave. 7,642 contacted vs 38,804 not.
- **Adjustment:** 37 covariates.
- **Methods:** five Bayesian approaches (outcome model, propensity strata, doubly robust, hierarchical by risk quintile, landmark design), each checked against **negative controls**: outcomes outreach can't affect.

**Results:**
- **Enrollment:** +11.5 points [10.6, 12.4] within 180 days (25% → 37%). It's consistent across methods and flat across risk quintiles.
- **Surgery:** the naive estimates say outreach *increases* surgery. But contacted patients also had more surgeries in the first **30 days**, before outreach could act: Meridian selected on information outside the data.
  - The landmark design restarts the clock at day 30 and corrects by a days 30–90 control window. It gives **+0.05 points [−0.42, +0.61]**, a 45% probability of being protective.
  - Ruled out: contacts triggered by new visits, measured confounders, and a PCP-preference instrument.

So the historical data credibly measures the enrollment lift and only bounds the surgery effect.

## The list

"Next quarter" here means Meridian's next outreach wave: past campaigns ran every six months (July and January), so the list is scored on 2026-07-01 for the wave that follows.

1. **Eligibility.** 14,329 patients at 2026-07-01, after these exclusions:
   - **recent enrollees:** anyone enrolled in the last 90 days (1,675), probably mid-course,
   - **no usable channel:** 7 patients with every channel opted out, unsubscribed, invalid or bounced.

   Contacts use only each patient's unblocked channels. Patients with a consult 90–365 days ago stay in, labeled.
2. **Ranking.** Lift and surgery reduction are constant across patients, so expected value is proportional to risk, and the list is ranked by risk.
3. **Program.** A rule: **weight management** for BMI ≥ 30 or an obesity diagnosis, **PT** otherwise, including patients with no BMI on file. The data can't choose the program per patient, and past enrollment by program offered isn't causal.
4. **How far down.** Contact while expected savings exceed the cost of contact:

   > risk × 0.115 × 0.30 × $20,000 > $10

   That puts the break-even risk at **1.45%**, and **1,368 patients clear it**. Expected value peaks there: those contacts are expected to prevent about 0.9 surgeries per wave, $6.8k net of contact costs. The surgery reduction moves the list more than anything the data measures:

   | Scenario | Patients above break-even |
   |---|---|
   | Base case | 1,368 |
   | 0.10 reduction | 98 |
   | 0.50 reduction | 4,745 |
   | $5 per contact | 6,636 |

   `out/sensitivity.csv` has the full table, and [`out/list_depth/how_far_down.pdf`](out/list_depth/how_far_down.pdf) presents it. So the answer is a decision rule plus a measurement design, not one number. The list is set by economics, not capacity: 1,644 patients to contact this wave, about 5,700 contact attempts at past rates (3.5 per patient). Past waves reached 2,189–3,183 patients; if Meridian's capacity is lower, shrink the learning band first.

## Pilot design: how Meridian will know it worked

| Tier | Who | Randomization | Contacted / held out |
|---|---|---|---|
| 1 | Above break-even (1,368) | 80/20 holdout | 1,094 / 274 |
| 2 | Learning band: the next 1,100 below break-even | 50/50 | 550 / 550 |

- **Assignment.** Randomized within tier × program with a fixed seed, recorded in `out/ranking.csv` (`arm`). Holdout patients get usual care, and **Meridian's campaign must skip them** for 12 months.
- **Primary outcome.** Pre-surgery enrollment in the recommended program within 90 days of first contact. Analyzed by assignment, stratified by tier. Detectable effect: **5.2 points**, against 11.5 measured historically. Tier 2 also tests whether the lift holds below the cut-off, for about $0.6k of expected value ($5.5k of contacts).
- **Secondary outcomes.** Time to enrollment, surgical consults, and surgery while covered at 12 months, with the holdout kept for a year.
- **What it can't do.** Detecting a realistic 0.3-point drop in surgery needs about 34,000 patients per arm. The pilot rules out large surgery effects. The 30% surgery reduction stays a literature assumption, refined only by pooling across quarters and clients.

**In plain language for Meridian:** *"We'll set aside a random group of similar patients and not call them. If the people we call enroll more, the outreach worked. We'll know that within a quarter. Whether enrollment prevents surgery takes a year, and more patients than one quarter gives us."*

## Reuse and engineering

- **Configuration, not code.** All client-specific values sit in `assumptions.py`, and code lists and windows are constants in `features.py`. A new client changes those, not the logic.
- **Reusable as is:** the snapshot design, the label, the model, the decision rule and the pilot design.
- **Reworked per client:** the feature mapping to their encounter data, and the causal estimate of the enrollment lift.
- **Scale.** Pandas over one extract runs in seconds. The snapshot builder is per index date, so it parallelizes or moves to SQL directly.
- **Next across clients:** a hierarchical model with client-level intercepts, so a new client borrows strength from earlier ones.

## Caveats

- **"Earlier in the journey"** means before a surgical consult and on recent escalation. The entire 2026-07 population is already 12+ months past their first observed visit, and new onset barely occurs in this closed cohort.
- **Half the scoring population is 24+ months out**, beyond any training data (a 12-month label caps follow-up at 24 months). Their risks may still run slightly high.
- **The 30% surgery reduction is an assumption.** The list length depends on it more than on anything measured.
- **The enrollment lift comes from non-randomized history.** Negative controls show residual selection that is small relative to an 11.5-point effect.
- **Assumed equal across programs:** the lift and the surgery reduction are assumed the same for PT and weight management.
- **The surgery cost** ($20k blended) is from dated published Medicare figures, not Meridian's contracted rates.

## Where each evaluation criterion is addressed

| Criterion | Where |
|---|---|
| Study design | [Pilot design](#pilot-design-how-meridian-will-know-it-worked): randomized holdout and learning band, primary and secondary outcomes, detectable effect. Why history can't answer the surgery question: [Did past outreach work?](#did-past-outreach-work-sandboxcausal) |
| Commercial reasoning | [The list](#the-list): ranked by expected surgeries avoided, cut off where savings beat contact cost, with the sensitivity table |
| Presentation | `out/presentation/final.pdf` (29 pages for Meridian: cover, problem, decision rule, worked examples, plan, six sections), built by `assignment/presentation.py`; outline in `out/presentation/outline.md` |
| Robust methodology | [Modeling](#modeling) and [Evaluation](#evaluation): a calibrated risk model, because the cut-off uses actual probabilities |
| Generalizable approach | [Reuse and engineering](#reuse-and-engineering): what carries over to a new client and what's reworked |
| Architecture and engineering | Two CLI commands, all assumptions in one file, outputs regenerated in seconds; [Reuse and engineering](#reuse-and-engineering) |
| Code and pipeline quality | 15 tests (`uv run pytest`), ruff lint and format, a clean-checkout run of `train` and `score` |
| Communication | This README, `docs/` (data profile, assumption sources, plan), and the decks |

## Tools used

**How AI was used.** I directed the analysis and made the key decisions: the target and censoring treatment, shipping the frequentist model, the eligibility exclusions, the holdout and learning-band design, and the presentation's structure. Claude Code, an AI coding agent, wrote most of the code, analyses, docs and decks under that direction, and I reviewed the results as we went.

- **Claude Code** (Anthropic, Claude Opus 5.5) as a pair programmer throughout:
  - data profiling, analysis design and code,
  - the causal analysis and the frequentist vs Bayesian comparison,
  - the decks and the docs.

  Claude Code skills used:
  - `pymc-modeling` for the Bayesian models,
  - `pptx` for the decks,
  - `dataviz` for chart design,
  - `ponytail` for simplicity reviews of the pipeline.

  A Claude subagent built `sandbox/explainability/`. Claude Code's web search found the published Medicare surgery-cost figures behind `SURGERY_COST` (sources in `docs/assumptions.md`).
- **Python:** pandas, scikit-learn, NumPy, SciPy, matplotlib, Jupyter. Optionally PyMC 6.3, nutpie and ArviZ 1.3 for the Bayesian work.
- **`uv`** for environments and dependencies; **ruff** for linting and formatting.
- **Decks:** the three decks in `out/` are drawn with matplotlib. The two exploratory decks in `sandbox/` were built with pptxgenjs (Node).
- **Rendering checks:** Microsoft PowerPoint and PyMuPDF rendered every slide so it could be checked visually; python-pptx was used to read deck structure.
