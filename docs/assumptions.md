# Assumptions and constraints

Where each assumption comes from. **Current values live in `assignment/assumptions.py`**, which is the only place the code reads them. This doc keeps the sourcing and the reasoning.

## Cost per surgery

No cost data is in the assets, so this is an estimate from published Medicare figures.

| Procedure (CPT) | Share of surgeries in extract | Assumed 90-day episode cost |
|---|---|---|
| Fusion (22630, 22633) | 51% (1,261) | $30,000 |
| Decompression (63030, 63047) | 49% (1,193) | $12,000 |
| **Blended** | | **≈ $21,000, rounded to $20,000** |

- **Fusion:** Medicare 90-day payments for a single-level posterior interbody fusion averaged about $25.5k in 2018 data ([Spine, 2018](https://www.ovid.com/jnls/spinejournal/abstract/10.1097/brs.0000000000002283~ninety-day-reimbursements-for-primary-single-level-posterior?redirectionsource=fulltextview)). Multilevel fusions are about $36k ([Jain et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29369155)). Rounded up to $30k for inflation.
- **Decompression:** facility payment for a single-level laminectomy (63047) is about $7.7k in an outpatient setting ([source, low reliability](https://mymedicineadvisor.com/health/laminectomy-cost-recovery-success/)). Professional fees and post-op care bring the episode to an assumed $12k.
- **`SURGERY_COST` = $20,000, sensitivity range $10k–$40k.** The source data is dated and Meridian's actual contracted rates are unknown.
- **Simplification:** one blended cost is used rather than a cost per procedure, because the model predicts surgery, not which procedure.

## Enrollments

**All enrollments count equally**, per the call. A PT enrollment and a WM enrollment have the same value, and there's no weighting by program, sessions attended or completion. The pilot's primary metric counts any pre-surgery enrollment in the recommended program.

## Outreach cost and wave size

| Channel | Cost | In historical data? |
|---|---|---|
| Phone | Most expensive (staff time) | Yes, 66% of attempts |
| SMS | Low | Yes, 22% |
| Email | Free | Yes, 12% |
| Direct mail | $1 per piece | **No.** This is a new channel with no historical response data. |

- **`CONTACT_COST`:** amortized to **$5–$10** across all attempts. The base case is $10, the conservative end.
- **`EVALUATION_LIST_SIZE` = 2,500, used only for evaluation metrics** (precision, recall, accuracy at a fixed list size). It is **not** used to decide how many patients to contact. It's about the size of past outreach waves. The waves ran every six months (July and January), so this is per wave, not per quarter. It's also what Meridian *did* contact, not necessarily the most it could, and the latest wave was the smallest. **Confirm with Meridian.**

| Wave | Patients | Attempts | Phone calls | Length |
|---|---|---|---|---|
| 2024-07 | 2,966 | 10,321 | 6,818 | 52 days |
| 2025-01 | 3,183 | 11,076 | 7,290 | 54 days |
| 2025-07 | 2,737 | 9,529 | 6,275 | 58 days |
| 2026-01 | 2,189 | 7,538 | 4,994 | 64 days |

Each patient got about 3.5 attempts, 2.3 of them phone calls. That's about 86 calls per day during a wave, so phone staffing is the binding constraint.

## Enrollment lift and surgery reduction

| Assumption | Value | Source |
|---|---|---|
| `ENROLLMENT_LIFT` | 0.115 (89%: 0.106–0.124) | **Estimated.** Doubly robust effect of contact on 180-day enrollment across the 2024-07, 2025-01 and 2025-07 waves (`sandbox/causal/c_aipw.py`). All five causal approaches agree, and it's roughly flat across risk quintiles. |
| `SURGERY_REDUCTION_PER_ENROLLMENT` | 0.30 (range 0.10–0.50) | **Literature assumption.** The historical data can't estimate it: Meridian's targeting used information outside the data, and the best-controlled surgery estimate is +0.05 points [−0.42, +0.61]. |
| `BASELINE_ENROLLMENT` | 0.25 | Enrollment rate of the contacted patients had they not been contacted (`sandbox/causal/a_gcomp.py`). Only used for the pilot's detectable effect. |

PT and weight management are assumed to have the same lift and the same surgery reduction. The historical data can't separate them.

## How far down the list

Contacting a patient pays for itself when the surgery cost it is expected to avoid exceeds the contact cost:

> P(surgery) × `ENROLLMENT_LIFT` × `SURGERY_REDUCTION_PER_ENROLLMENT` × `SURGERY_COST` > `CONTACT_COST`

With the base case, the break-even risk is **1.45%**. `main.py score` writes this table to `out/sensitivity.csv` from the ranges in `assignment/assumptions.py`. It's computed on the 14,329 eligible 2026-07 patients:

| Assumption at | Break-even risk | Tier 1: patients above it |
|---|---|---|
| **Base case** | **1.45%** | **1,368** |
| Surgery cost $10k / $40k | 2.90% / 0.72% | 266 / 6,636 |
| Contact cost $5 | 0.72% | 6,636 |
| Surgery reduction 0.10 / 0.50 | 4.35% / 0.87% | 98 / 4,745 |
| Enrollment lift 0.106 / 0.124 | 1.57% / 1.34% | 1,096 / 1,654 |

Tier 2 is always the next `LEARNING_BAND_SIZE` = 1,100 patients below the cut-off, randomized 50/50.

- **The surgery reduction moves the list most**, and it's the one input the data can't test.
- **The list is set by economics, not capacity.** Tier 1 is everyone above break-even (1,368 in the base case), and tier 2 is a fixed learning band. Together that's 1,644 contacts. If Meridian's real capacity is lower, shrink the learning band first, then tier 1 from the bottom. Capacity is to confirm with Meridian (past waves reached 2,189–3,183 patients).
- **The earlier model without the tenure feature** put 9,112 patients above break-even. Its risks ran about 30% too high overall, and 2.6× too high for patients 12+ months past their first visit.
- **Caveat:** half the 2026-07 population is 24+ months past their first visit, beyond any training data. If risk keeps falling past 24 months, these risks are still slightly too high.

## Eligibility, program and pilot

| Assumption | Value | Why |
|---|---|---|
| `LATE_STAGE_CONSULT_DAYS` | 90 | A consult this recent means already late-stage. Also defines the training population, so changing it needs a retrain. |
| `RECENT_ENROLLMENT_EXCLUSION_DAYS` | 90 | Probably mid-course (agreed). Older enrollees stay in, flagged. |
| `INCLUDE_PRIOR_CONSULT` | True | Consults 90–365 days ago stay in, labeled (agreed). |
| `BLOCKING_RESULTS` | Opted Out, Unsubscribed, Invalid Number, Bounced | That channel can't be used. Patients with every channel blocked are dropped. |
| `WM_BMI_THRESHOLD` | 30 | Weight management for BMI ≥ 30 or any obesity diagnosis (E66); PT otherwise. |
| `TIER1_HOLDOUT_SHARE` | 0.20 | Meridian accepts a holdout. Meridian's own campaign must skip these patients. |
| `TIER2_HOLDOUT_SHARE` | 0.50 | Learning band below the cut-off: tests whether the lift holds there. |
| `LEARNING_BAND_SIZE` | 1,100 | Patients just below break-even, randomized 50/50: tests whether the lift holds below the cut-off. About $0.6k of expected value in the base case. |
| `ALPHA`, `POWER` | 0.05, 0.80 | For the reported detectable effect: 5.2 points pooled across tiers. |
| `RANDOM_SEED` | 2026 | Reproducible assignment. |

## Censoring

Notes only; not modelled in depth.

- **What it is:** Meridian's files stop at `coverage_end`. 15% of snapshots leave coverage before their 12-month horizon ends.
- **Current handling:** Kaplan–Meier or censoring weights (IPCW) treat leaving as unrelated to surgery risk. That gives 3.02%, against 3.36% for a complete-case rate, so the choice moves calibration by about 10% relative.
- **Why that assumption may not hold:** we don't know why coverage ends.
  - **Death** is a competing risk. Patients who die can't have surgery, and KM treats them as if they still could, so it slightly overstates risk.
  - **Moving away or switching plans** is probably close to random.
  - **Losing attribution** could be linked to how often the patient uses care.
- **Cheap check available:** Vi's claims continue after `coverage_end`. Comparing surgery claims among recent leavers with the KM-predicted rate would show whether leavers are more or less likely to have surgery. If the gap is large, IPCW by age and stage is the next step.
- **Leaving coverage could instead be treated as a competing event.** Meridian only pays while a patient is attributed, so a surgery after they leave costs Meridian nothing. The quantity the list should maximise is P(surgery *while still covered* in the next 12 months). That treats leaving as a competing event, not censoring.
  - The estimate is simply surgeries during coverage ÷ all eligible snapshots: 2,310 / 80,580 = **2.87%**, against 3.02% from KM. Leavers without surgery count as zeros.
  - Leaving coverage needs no special handling: a plain logistic regression on every snapshot, with "surgery before coverage end, within 12 months" as the label, estimates this directly.
  - KM (3.02%) answers a different question: the risk if the patient stayed covered. It's the right clinical number but not what Meridian pays for.
- **Decision:** the target is surgery during coverage, with leaving treated as a competing event, fitted with a plain logistic regression. Report KM alongside it in the EDA as the clinical comparison. Dropping censored patients (the 3.36% complete-case rate) is wrong under both framings.
