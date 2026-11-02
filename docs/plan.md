# Plan: the ranked outreach list

How `uv run main.py train` and `uv run main.py score` produce next quarter's list: who to contact, in what order, with which program, and how far down to go. Sections 1–4 use counts from the sandbox model (trained on 2024-07 to 2025-04). **Built (2026-10-27):** the final model trains on all labeled data and gives tier 1 = 1,368 (1,094 contacted / 274 held out) and tier 2 = 1,100 (550 / 550), with a pooled detectable effect of 5.2 points. The 2,500 is now only the evaluation list size; capacity is not a targeting input. Current values are in `out/sensitivity.csv`.

## 1. Who is eligible

Score the population on **2026-07-01**:
- covered on that date,
- back-pain history,
- no surgery yet (EHR `Performed` or null-status surgery, or a paid surgery claim),
- no surgical consult in the last 90 days (already late-stage),
- **not enrolled in PT or weight management in the last 90 days**, since they're probably mid-course. That excludes 1,675 patients. Older enrollees stay eligible and are flagged: repeat enrollment is common, with a median gap of about 200 days.

That leaves about **14,300 eligible patients**.

**Channels.** A patient can be contacted only on channels they haven't opted out of or unsubscribed from, and that haven't returned an invalid number or bounced. 360 of the top 2,500 have at least one blocked channel. Patients with every channel blocked are dropped (7).

**Prior surgical consult.** Patients with a consult 90–365 days ago stay in and are labeled: 351 of the top 2,500 (14%). Keeping them serves Meridian's cost goal. The label lets Meridian treat them differently if "earlier in the journey" matters more.

## 2. How the list is ranked

Rank by **expected surgeries avoided per contact**:

> P(surgery while covered in 12 months) × enrollment lift × surgery reduction per enrollment

- **P(surgery):** the frequentist L2 logistic regression from `sandbox/frequentist/`, with 36 features including time since first visit capped at 12m+. It's calibrated out-of-time: 2.16% predicted vs 2.16% observed on 2025-07, AUC 0.733.
- **Enrollment lift:** +11.5 points [10.6, 12.4], the doubly robust estimate from `sandbox/causal/`. It's roughly flat across risk quintiles.
- **Surgery reduction per enrollment:** an assumption (base case 30%), the same for every patient.

The lift and the reduction are constant, so **the ranking is risk order**. `score` is P(surgery); expected value goes in its own column.

**Assumption:** the same lift for PT and weight management. The historical data can't separate them.

## 3. Which program

A rule, not a model:
- **Weight management** if the latest BMI (Z68 code) is 30 or more, or there's an obesity diagnosis (E66).
- **PT** otherwise, including the two-thirds of patients with no BMI on file.

In the top 2,500 that's 734 weight management and 1,766 PT.

The data can't choose the program per patient. Past enrollment was higher when weight management was offered (43% vs 35%), but that isn't causal. If Meridian wants to know, the pilot can randomize PT vs weight management among patients eligible for both.

## 4. How far down

Contact a patient while their expected surgery savings exceed the cost of contacting them:

> P(surgery) × 0.115 × surgery reduction × $20,000 > cost per contact

| Surgery reduction per enrollment | Break-even risk ($10 contact) | Patients above it |
|---|---|---|
| 10% | 4.35% | 96 |
| 20% | 2.17% | 434 |
| **30% (base case)** | **1.45%** | **1,218** |
| 50% | 0.87% | 4,392 |
| 30%, $5 contact | 0.72% | 6,296 |

The list runs from about 100 to 6,000+ patients depending on an assumption the data can't test. So the answer is a decision rule plus a measurement design:

| Tier | Who | Arm | Contacted / held out |
|---|---|---|---|
| **1** | Above the base-case break-even (~1,200) | 80/20 randomized holdout | ~975 / ~245 |
| **2** | Learning band: the next `LEARNING_BAND_SIZE` (1,100) below break-even | 50/50 randomized | ~550 / ~550 |
| — | Below the learning band | Not contacted this wave | |

- **Randomization** is balanced within tier and program, with a fixed seed.
- **Holdout patients** get usual care. **Meridian's own campaign must skip them** for 12 months.
- **What it measures.** Enrollment within 90 days of first contact (before any surgery), analyzed by tier with a stratified estimate.
  - Tier 1 alone can detect a lift of about 9 points.
  - Both tiers pooled can detect about **5.4 points**, against the 11.5 measured historically.
  - Tier 2 also tests whether the lift holds below the cut-off. Its size is set for learning, not by capacity, and it costs about $0.6k of expected value in the base case.
- **Capacity is not an input.** The list is set by economics: tier 1 plus the learning band, 1,644 contacts in the built version. If Meridian's capacity (past waves: 2,189–3,183 patients) is lower, shrink the learning band first, then tier 1 from the bottom.
- **What it can't measure.** Detecting a 0.3-point drop in surgery from a 2% base needs about 34,000 patients per arm. The pilot rules out large surgery effects and tracks surgery over 12 months. **The 30% reduction stays a literature assumption**, refined only by pooling across quarters and clients.

## 5. Assumptions live in one file: `assignment/assumptions.py`

Every judgment call a stakeholder might challenge or want to change lives in `assignment/assumptions.py`. Each one has a value, a sensitivity range and a source, and nowhere else in the code hardcodes it. `score` reads the ranges to write the sensitivity table (`out/sensitivity.csv`). Changing an assumption and re-running `score` is the whole workflow; no retraining.

| Group | Name | Value | Range | Source |
|---|---|---|---|---|
| **Economics** | `SURGERY_COST` | $20,000 | $10k–$40k | Medicare 90-day episodes, fusion/decompression blend (`docs/assumptions.md`) |
| | `CONTACT_COST` | $10 | $5–$10 | Vi call: amortized cost per contacted patient |
| | `SURGERY_REDUCTION_PER_ENROLLMENT` | 0.30 | 0.10–0.50 | Literature assumption, not testable in this data |
| | `ENROLLMENT_LIFT` | 0.115 | 0.106–0.124 (89% interval) | Estimated: doubly robust ATT, `sandbox/causal/c_aipw.py` |
| | `EVALUATION_LIST_SIZE` | 2,500 | | **Evaluation only:** fixed list size for precision / recall / accuracy, about the size of past waves. Not used for targeting |
| | `SAME_EFFECT_ACROSS_PROGRAMS` | True | | Vi call: all enrollments count equally; history can't separate PT from WM |
| **Eligibility** | `LATE_STAGE_CONSULT_DAYS` | 90 | | Consult this recent = already late-stage |
| | `RECENT_ENROLLMENT_EXCLUSION_DAYS` | 90 | | Agreed: probably mid-course |
| | `INCLUDE_PRIOR_CONSULT` | True (labeled) | | Agreed |
| | `BLOCKING_RESULTS` | Opted Out, Unsubscribed, Invalid Number, Bounced | | Channel can't be used |
| **Program** | `WM_BMI_THRESHOLD` | 30 | | Obesity cut-off; any obesity diagnosis (E66, the same codes as the `obesity_dx` feature) also qualifies |
| **Pilot** | `TIER1_HOLDOUT_SHARE` | 0.20 | | Agreed: Meridian accepts a holdout |
| | `LEARNING_BAND_SIZE` | 1,100 | | Tier 2: patients just below break-even, randomized 50/50 |
| | `TIER2_HOLDOUT_SHARE` | 0.50 | | Learning band below the cut-off |
| | `BASELINE_ENROLLMENT` | 0.25 | | Uncontacted enrollment rate (`sandbox/causal`), for the detectable effect |
| | `ALPHA`, `POWER` | 0.05, 0.80 | | For the reported detectable effect |
| | `RANDOM_SEED` | fixed | | Reproducible assignment |

**What doesn't go in it.** Model and target definitions stay as constants in the modules that use them: the 12-month horizon and lookback, the tenure bands capped at 12m+, the surgery CPT codes and the null-status rule, the clinical code lists, and L2 strength C = 1. Changing any of these changes what the model predicts, so it needs a retrain and a fresh out-of-time check, not just a re-score. Each is documented where it's defined.

**`docs/assumptions.md` is out of date.** It still uses the pre-causal enrollment lift of 0.10, the 1.7% break-even and the "919 above break-even" figure. It gets rewritten to point at `assumptions.py` for current values and to keep the sourcing narrative.

## 6. What gets built

The frequentist model moves into `assignment/`. The Bayesian model stays in `sandbox/` as the documented comparison.

- `assignment/`: `assumptions.py` (section 5), the snapshot and feature builder (from `sandbox/panel.py`), the model, the eligibility and channel rules, the program rule, the decision rule and randomization. All of them read their values from `assumptions.py`.
- **`main.py train`** fits on all labeled snapshots (2024-07 to 2025-07) and writes the model and standardization to `artifacts/`.
- **`main.py score`** scores 2026-07-01 and writes:
  - **`out/targets.csv`**: patients to **contact** only, in rank order. Columns: `patient_token, program_recommended, score` (required; `score` is P(surgery)), plus `rank, tier, expected_net_value, allowed_channels, prior_consult, prior_enrollment_365d`.
  - **`out/ranking.csv`**: every eligible patient with rank, tier and arm. `arm == "holdout"` is the do-not-contact list for Meridian's own campaigns and the comparison group for measurement. (A separate `holdout.csv` was dropped as redundant.)
  - **`out/sensitivity.csv`**: the break-even, the uncapped count above it, and the tier sizes, across every assumption range in `assumptions.py`.
- **Tests:**
  - eligibility exclusions, including recent enrollees and all-channels-blocked,
  - the program rule,
  - the decision rule and tier assignment,
  - no holdout patient appears in `targets.csv`,
  - the output passes `validate_targets`,
  - changing an assumption (e.g. `SURGERY_REDUCTION_PER_ENROLLMENT`) moves the break-even and tier boundaries as expected.

## README caveats

- **"Earlier in the journey"** here means before a surgical consult and on recent escalation, not near onset. Every 2026-07 patient is already 12+ months past their first observed visit.
- **Half the scoring population is 24+ months past their first visit**, beyond any training data, so their risks may run slightly high.
- **The 30% surgery reduction is an assumption**, and the depth of the list depends on it more than on anything the data measures.
- **The surgery readout** needs the holdout kept for 12 months, and even then only bounds the effect.
- **The enrollment lift comes from non-randomized history.** The negative controls show some residual selection, small relative to an 11.5-point effect.
