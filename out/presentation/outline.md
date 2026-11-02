# Final presentation: outline

**Built:** `uv run python -m assignment.presentation` writes `out/presentation/final.pdf` (29 pages: cover, problem, decision rule, two worked examples, plan, then six sections each opened by a divider). Open questions were resolved with their proposals: "next wave" wording, "how to measure" as its own section 5, the causal result in the main flow, truncated tokens on screen.

**Audience:** Meridian stakeholders who will run the outreach campaign. They're not familiar with machine learning, and they're unsure how Vi helps with enrollment targets or cost savings. Vi panelists role-play them.
**Slot:** 20–25 minutes, then about 20 minutes of methodology discussion with Vi's DS team (backup slides).
**Deliverable it answers:** *A ranked outreach list: which patients to contact next wave, in priority order, with a recommended program for each, and a defensible answer to how far down the list to go.*

**The answer, in one breath:** *Contact the 1,368 patients whose risk of back surgery makes outreach worth its cost, starting with the highest risk. Offer weight management to the 409 with obesity and PT to the rest. Contact another 1,100 just below the cut-off as a randomized test. Hold out a random 20% to prove it works.*

## Running order

| # | Section | Slides | Minutes |
|---|---|---|---|
| 0 | Opening: cover, the problem, the decision rule, two worked examples, the plan | 5 | 5 |
| 1 | Who is eligible (divider + 1) | 2 | 2 |
| 2 | How the list is ranked: model, signals, drivers, how it works, two example patients, evaluation, the list (divider + 8) | 9 | 10 |
| 3 | Which program (divider + 1) | 2 | 2 |
| 4 | How many to contact (divider + 3) | 4 | 4 |
| 5 | How to measure: why history can't settle surgery, and the plan for each wave (divider + 3) | 4 | 4 |
| 6 | Assumptions (divider + 2) | 3 | 2 |
| | **Total** | **29** | **29** |

The plan slide lists the six sections; each section opens with a divider (large violet number, title, one line on what it answers). Both read `SECTIONS` in `assignment/presentation.py`, so the agenda and the dividers can't drift apart.

Every slide below lists its **headline** (the one message), the **content**, the **source** (existing output, or NEW if it has to be built), and **notes** for the speaker.

---

## 0. Opening

**0.1 The problem and the opportunity.** Headline: *"Targeted outreach can avoid expensive back surgeries"* (follows the cover: "Preventing back surgery with targeted outreach")
- Content: three panels, then the ask.
  - **The problem:** lumbar spine surgeries in the past year among covered patients (694, about $13.9M at $20k each). Meridian pays the full cost under its value-based contract.
  - **What's been tried:** outreach reached 2,189–3,183 patients per wave, a median 288 days after the first back-pain visit, and was never measured: there was no holdout.
  - **The opportunity:** reach the patients most likely to need surgery before they see a surgeon, offer the right program, and measure it.
  - **The ask, on one line:** a ranked list, a program per patient, how far down, how we'll know.
- Source: computed from the extract (`presentation.problem`).

**0.2 The decision rule.** Headline: *"The expected savings of an outreach must beat its cost"*
- Content: the decision rule alone, large and centered under a "Decision rule" label: P(surgery) × enrollment lift × surgery reduction × surgery cost > outreach cost. The terms are defined on the section 4 rule slide; the next slide works two patients through it.

**0.3 Two worked examples.** Headline: *"Two patients: one worth an outreach, one not"*
- Content: two hypothetical patients run through the rule. Patient A (5% risk): $34.50 expected savings > $10, reach out. Patient B (1% risk): $6.90 < $10, skip. Footnote: each 1% of risk is worth $6.90, so the break-even is 1.45%.

**0.4 The plan.** Headline: *"The plan"*
- Content: the six sections, numbered 01–06, each with a one-line description. They map one-to-one to the divider slides.


## 1. Who is eligible

**1.1 From 30,000 patients to 14,329 we can reach.** Headline: *"We score patients who are still on a path to surgery, and still reachable"*
- Content: funnel chart, with a plain-language note beside each bar on the right (e.g. "Covered on 2026-07-01" → "Still a Meridian member on the day the list is built"). The slide doesn't mention tiers, which aren't defined until section 4.

  | Step | Patients |
  |---|---|
  | All patients in the extract | 30,000 |
  | Covered on 2026-07-01 | 22,200 |
  | With back-pain history | 17,391 |
  | No surgery yet | 16,070 |
  | No surgical consult in the last 90 days (already late-stage) | 16,010 |
  | Not enrolled in PT/WM in the last 90 days (probably mid-course) | 14,335 |
  | At least one usable channel | **14,329** |

- Source: **NEW** funnel chart (numbers computed; see the script note at the end).
- Notes (formerly slide 1.2):
  - **Each step is an assumption Meridian can change** (section 6).
  - **Channels:** opted-out, unsubscribed, invalid and bounced channels are never used. 2,119 of the 2,468 listed patients are reachable on all three channels. Past campaigns contacted 531 patients after they opted out; the list prevents that.
  - **Labeled, not excluded:**
    - **Prior surgical consult (90–365 days ago):** stay in, flagged on the list.
    - **Enrolled in the last 12 months (but not the last 90 days):** stay in, flagged on the list.
  - **"Earlier in the journey"** means before a surgical consult. Every patient is already 12+ months past their first back-pain visit.
- Source for notes: `out/ranking.csv` (`allowed_channels`, `prior_consult`, `prior_enrollment_365d`); `docs/data-profile.md`.

## 2. How the list is ranked

**2.1 What the model predicts.** Headline: *"For each patient: the chance of back surgery in the next 12 months"*
- Content:
  - **Estimand** across the top: Risk = P( lumbar spine surgery within 12 months, while covered | signals before the list date ).
  - **The model:** logistic regression; each signal gets a learned weight, the weighted sum becomes a probability. Fit on quarterly patient snapshots, 2024-07 to 2025-07.
  - **The target:** 1 if lumbar spine surgery (EHR or paid claim) in the 12 months after the list date, before leaving Meridian; otherwise 0.
  - **The signals, high level:** 36 from the 12 months before the list date (back-pain care, spine diagnoses, BMI and obesity, chronic conditions, age and sex); defined on the next slide.
  - **Example ranks:**

    | Rank | Risk |
    |---|---|
    | 1 | 14.5% |
    | 1,368 (the cut-off) | 1.45% |
    | 2,468 (the end of the learning band) | 1.15% |
    | Population average | 0.86% |

- Source: **NEW** (text); `out/ranking.csv`.
- Notes:
  - "A 14.5% patient is about 17 times as likely as the average to have surgery this year."
  - Leaving coverage counts as "no surgery", because Meridian only pays while a patient is attributed.

**2.1b Two patients through the model.** Headline: *"Two patients through the model"*
- Content: two hypothetical patients scored by the trained model (`artifacts/model.pkl`), each card listing their signals, the predicted 12-month risk, rank on the current list, and whether it clears the break-even. Patient A (58, BMI 36 with obesity diagnosis, surgical consult 6 months ago, recent visits, MRI, sciatica): about 7%, top 20, on the list. Patient B (72, BMI 23, no consult, one visit 8 months ago, first visit 12m+ ago): about 0.3%, near the bottom, not contacted. Defined in `EXAMPLE_PATIENTS`; numbers recompute on each build.

**2.2 What it looks at.** Headline: *"36 signals from the past 12 months of each patient's records"*
- Content: two tables listing all 36 signals by name with a plain-language description: **back pain and care use** (19: consults, visits, imaging, MRI, injections, PT, ED, prescriptions, earlier enrollment, spine diagnoses) and **who the patient is** (17: age, sex, BMI bands, obesity, chronic conditions). Built from `FEATURE_TABLES` in `assignment/presentation.py`, which asserts it covers `features.FEATURES` exactly.

**2.2b Biggest drivers of risk.** Headline: *"Biggest drivers of risk"*
- Content: a full-width bar chart of the biggest drivers in percentage points of risk:
  - prior consult **+2.8**,
  - BMI 30+ **+0.5**,
  - recent imaging **+0.4**,
  - injection **+0.4**,
  - 12+ months since first visit **−2.5**,
  - each extra 10 years of age **−0.25**.
- Source: `assignment/features.py`; `sandbox/explainability/effects.csv` → **NEW** simplified bar chart.
- Notes:
  - **These are associations, not causes:** weight loss won't necessarily lower risk by the BMI effect.
  - **Outreach history is deliberately not a signal:** it reflects Meridian's past choices, not the patient's condition.

**2.3 How it works.** Headline: *"How it works: a weighted checklist"*
- Content:
  - **Diagram:** 36 signals per patient → × weights learned from past patients → = a score (higher means riskier) → chance of surgery in the next 12 months.
  - **Logistic regression:** the score is the log-odds of surgery, so chance = 1 / (1 + exp(−score)).
  - **Trained** on 80,580 patient snapshots taken quarterly from 2024-07 to 2025-07.
- Source: `assignment/model.py`.

**2.4 Better than chance.** Headline: *"On a quarter it never saw, the model beats a random list by 3.3×"*
- Content: three stat cards, each against random chance.

  | Measure | Model | Random |
  |---|---|---|
  | AUC | **0.73** | 0.50 |
  | Surgeries caught in the top 2,500 | **42%** | 12.5% |
  | Surgery rate on the list (precision) | **7.2%** | 2.2% |

  Plus the gains curve.
- Source: `out/evaluation/metrics.csv`, `out/evaluation/gains.png`; `out/evaluation/surgery_risk_model_evaluation.pdf` slides 2 and 7.
- Notes:
  - **The test:** refit on 2024-07 to 2025-04, scored on the 2025-07 quarter the model never saw.
  - **The 2,500 is only the evaluation list size,** not the recommended contact list.
  - **Keep the full metrics table** (recall, F1, accuracy, intervals) for backup. Accuracy is misleading at a 2% surgery rate.

**2.5 Its probabilities are honest.** Headline: *"Predicted 2.16%, observed 2.16%, in every risk band"*
- Content: the predicted vs observed by decile chart.
- Source: `out/evaluation/calibration.png`; evaluation deck slide 6.
- Notes:
  - **Why it matters to Meridian:** the cut-off in section 4 uses the actual probabilities, not just the order.
  - **The fix that got it here:** adding time since first visit, without which risk ran 30% too high. Backup.

**2.6 What the list looks like.** Headline: *"Every patient comes with a rank, a program and the channels to use"*
- Content: the first 10 rows of `out/targets.csv` (tokens shortened), with rank, patient, risk, program, channels and a prior-consult flag.
- Source: `out/targets.csv`; **NEW** formatted table.
- Notes: this is the file Meridian's team works from.

## 3. Which program

**3.1 A simple, explainable rule.** Headline: *"Weight management where obesity is a factor; PT for everyone else"*
- Content: the rule (BMI ≥ 30 or an obesity diagnosis → WM). Split by tier:

  | Tier | Weight management | PT |
  |---|---|---|
  | 1 | 409 | 959 |
  | 2 | 310 | 790 |

- **One line on the limits** (formerly slide 3.2): two-thirds of patients have no BMI on file and default to PT, and history can't pick the program per patient. The pilot could randomize PT vs WM among patients eligible for both.
- Source: `out/ranking.csv`; `assignment/targeting.py::recommend_program`; `docs/data-profile.md`.
- Notes:
  - BMI 30+ carries about twice the surgery risk of BMI under 25.
  - Past enrollment was higher when WM was offered (43% vs 35%), but that isn't a fair comparison.

## 4. How many to contact

**4.1 The rule.** Headline: *"Contact a patient when the surgery cost avoided beats the cost of the call"*
- Content: risk × 11.5% enrollment lift × 30% surgery reduction × $20,000 > $10, so the **break-even risk is 1.45%**. Short source per input.
- Source: `out/list_depth/how_far_down.pdf` slide 3.
- Notes: the 11.5% lift is measured from Meridian's past campaigns (`sandbox/causal/`).

**4.2 Where the list is worth most.** Headline: *"Value peaks at 1,368 patients; beyond that each contact costs more than it saves"*
- Content: cumulative expected net value curve.
  - **Peak:** $8.6k.
  - **Tier-1 contacts:** about 0.9 surgeries prevented per wave, $6.8k net.
- Source: `out/list_depth/how_far_down.pdf` slide 4.
- Notes: **be candid that the economics are thin per wave.** The value grows with repeated waves, and with Meridian's real costs.

**4.3 The tiers.** Headline: *"Contact 1,368, test 1,100 more, hold out a random share of each"*
- Content: tier table (contacted / held out, program mix, prior consult).
- Source: `out/list_depth/how_far_down.pdf` slide 7.

## 5. How to measure

**5.1 Why history can't answer the surgery question.** Headline: *"Past outreach went to patients already closer to surgery, so it can't tell us what outreach prevents"*
- Content, left: a **DAG**.
  - **Nodes:**
    - **Outreach** and **Enrollment** (solid),
    - **Surgery** (solid),
    - **"What Meridian's team saw"** (dashed, *unobserved*: referral notes, scheduled procedures, clinical judgment),
    - **"What we can see"** (solid: the 37 covariates).
  - **Arrows:**
    - **The causal path we want:** Outreach → Enrollment → Surgery, in blue.
    - **"What Meridian's team saw" → Outreach and → Surgery:** dashed orange, labeled **backdoor path: Outreach ← unseen information → Surgery**.
    - **"What we can see" → Outreach and → Surgery:** gray, labeled "adjusted for".
- Content, right: three evidence cards.
  1. **The raw comparison misleads:** contacted patients had surgery **3.6%** of the time vs **2.7%** for uncontacted. Taken at face value, outreach *causes* surgery.
  2. **A negative control flags the bias:** contacted patients also had more surgeries in the **first 30 days** (0.58% vs 0.35%), before any program could act. Outreach can't cause those. They were selected because they were already in the pipeline.
  3. **Adjusting for what we see isn't enough:** after controlling for 37 covariates and correcting with the negative control, the effect is **+0.05 points [−0.42, +0.61]**, a 45% chance it's protective. That's indistinguishable from zero, and wider than any realistic effect (0.1–0.3 points).
- Takeaway line: *"We can estimate the enrollment lift (+11.5 points). We can't estimate the surgery effect: the backdoor is open, and the effect is small."*
- Source: `sandbox/causal/README.md`, `sandbox/causal/estimates.csv`, `sandbox/causal/deck/causal_deck_data.json` (raw rates); **NEW** DAG drawing.
- Notes:
  - **Two terms, defined in one line each:**
    - **Confounder:** something that affects both who got contacted and who had surgery.
    - **Backdoor path:** a non-causal route from outreach to surgery that runs through a confounder.
  - **Why enrollment can be estimated when surgery can't:** the leftover bias is small next to an 11.5-point enrollment effect, but as large as the plausible surgery effect.

**5.2 How we'll know it worked.** Headline: *"A random holdout cuts the backdoor: we'll know about enrollment in a quarter, surgery in a year"*
- Content:
  - **The same DAG, with a "Randomization (coin flip)" node → Outreach.** The arrows from "What Meridian's team saw" and "What we can see" into Outreach are crossed out, so the backdoor is closed. Any difference between contacted and held-out patients is then caused by outreach.
  - **Readout timeline:** wave start → **90 days:** enrollment, detecting a 5.2-point lift (11.5 measured historically) → **12 months:** surgery, which only rules out large effects (a 0.3-point drop needs about 34,000 patients per arm).
- Source: README pilot section; **NEW** DAG variant and timeline.
- Notes:
  - **The one operational ask:** Meridian's own campaigns must skip holdout patients for 12 months, or the backdoor reopens.
  - **Plain-language version:** *"We'll set aside a random group of similar patients and not call them. If the people we call enroll more, the outreach worked."*

**5.3 Measure every wave the same way.** Headline: *"Measure every wave the same way"*
- Content: a table of the four questions each wave answers: what to compare, when to read out, and what one wave can detect.
  - **Does contact raise enrollment?** Contacted vs held out, at 90 days; one wave detects a 5.2-point lift (11.5 measured historically).
  - **Is the cut-off in the right place?** Lift in the learning band vs tier 1, at 90 days.
  - **Does it prevent surgery?** Surgery while covered, at 12 months; one wave sees only large effects, so pool waves.
  - **Which program works better?** Optional: randomize PT vs weight management among patients eligible for both.
- Three panels beneath: keep the holdout clean (skip it for 12 months; compare by assignment), randomize every wave (within tier and program, fixed seed, `out/ranking.csv`), feed results back (replace the assumed lift and surgery reduction in `assumptions.py`, re-score).
- Source: `list_depth.summary` (detectable effects); **NEW** slide.

## 6. Assumptions

**6.1 What the answer depends on.** Headline: *"One assumption moves the list more than anything we measured"*
- Content: the sensitivity chart only (patients above break-even vs surgery reduction, at $5 and $10 per contact). The low/high table moves to backup.
- Source: `out/list_depth/how_far_down.pdf` slide 5.
- Notes:
  - **The callback:** the surgery reduction is the assumption slide 5.1 explained we can't measure from history.
  - **The range:** 98 to 6,636 patients across the assumption ranges.

**6.2 Every assumption in one place.** Headline: *"Each judgment call has a value, a source, and a way to change it"*
- Content: table grouped by economics, eligibility, program and pilot, with a confidence column:
  - **Measured:** the enrollment lift.
  - **Published:** the surgery cost.
  - **Agreed:** the recent-enrollment exclusion, keeping prior-consult patients, the holdout.
  - **Assumed:** the 30% surgery reduction.
- **What Meridian needs to confirm:** capacity (the list asks for 1,644 patients, about 5,700 contact attempts), contact cost, contracted surgery cost.
- Source: `assignment/assumptions.py`; `docs/assumptions.md`; `out/list_depth/how_far_down.pdf` slide 8.
- Notes: changing any of these and re-running `score` regenerates the list in seconds.

The deck ends on the assumptions table; there is no closing slide.

## Backup slides, for the DS discussion

| Topic | Source |
|---|---|
| Out-of-time evaluation, full: metrics table with recall, F1, accuracy and 95% intervals; confusion matrix; five real patients | `out/evaluation/surgery_risk_model_evaluation.pdf` |
| Full feature list and coefficients | `assignment/features.py`, `sandbox/frequentist/coefficients.csv` |
| Better than Meridian's own past targeting? (not yet measured) | **NEW** if wanted: compare how well past contact choices predicted surgery |
| Did past outreach work? All five causal methods, the negative controls, and what was ruled out | `sandbox/causal/deck/causality.pptx`, `sandbox/causal/estimates.png` |
| Every assumption at its low and high value (moved from 6.1) | `out/list_depth/how_far_down.pdf` slide 6 |
| Frequentist vs Bayesian; interval coverage | `sandbox/deck/model_comparison.pptx`, `sandbox/README.md` |
| The missing-feature fix (time since first visit) | `sandbox/deck/model_comparison.pptx` slide 8 |
| Feature effects and two-way sensitivity | `sandbox/explainability/` |
| Data quality findings | `docs/data-profile.md` |
| Reuse across clients; engineering | README "Reuse and engineering" |

## Build plan

- **Format:** a Python script, `assignment/presentation.py`, writing `out/presentation/final.pdf`. It reuses `assignment/slides.py`, the evaluation and list-depth chart functions, and `out/` CSVs, so every number is computed, not typed.
- **New pieces:**
  - the eligibility funnel (1.1),
  - the formatted top-10 list (2.6),
  - the risk-drivers bar chart (2.2),
  - the signals → score → probability diagram (2.3),
  - the three better-than-chance cards (2.4),
  - the DAG with the backdoor path (5.1) and its randomized variant (5.4),
  - the readout timeline (4.5),
  - the assumptions table with a confidence column (5.2).
- **Reused as is:** calibration and gains charts, the value curve, the sensitivity chart, and the tier table.

## Open questions

1. **"Next quarter" or "next wave"?** The brief says quarter; Meridian's campaigns run every six months. Proposal: say "next wave", noting it's the next campaign.
2. **Resolved:** "How to measure" is its own section 5: the causal DAGs (5.1, 5.2), then the per-wave measurement plan (5.3). The enrollment-lift method slides were removed; the method is in `sandbox/causal/README.md`.
3. **Resolved:** the causal result now appears in the main flow as slide 5.1 (the DAG and the backdoor path). Methods detail stays in backup.
4. **Slide 2.6 shows real patient tokens,** truncated. Synthetic data, but confirm that's acceptable for the room.
