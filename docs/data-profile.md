# Data Profile

Profile of the five extract files (window 2023-07-01 → 2026-06-30, extracted 2026-07-01). On the surface the data is clean: no orphan `patient_token`s, no duplicate primary keys, and every date parses. The issues that matter only show up when you compare the files with each other.

## Overview

| File | Rows | Patients | Date range | Notable nulls |
|---|---|---|---|---|
| patients | 30,000 | 30,000 | `coverage_start` 2023-07-01 → 2025-04-30 | `coverage_end` 74%, `pcp_id` 9.2%, `zip3` 1.1%, `sex` 0.5% |
| encounters | 249,559 | 28,067 | 2023-07-01 → 2026-06-30 | `spine_level` 72%, `facility_id` 14%, `event_status` 8%, `event_code` 4% |
| outreach | 38,464 | 11,075 | 2024-07-11 → 2026-03-13 | `message` 2.1%, `attempt_result` 1.0% |
| enrollments | 22,345 | 10,042 | 2023-07-09 → 2026-06-29 | none |
| claims | 1,752,450 | 29,999 | 2023-07-01 → 2026-06-30 | `place_of_service` / `provider_specialty` 26% (all NDC pharmacy lines, as expected), `code_system` 2% |

Category distributions:

- **Encounters:** LBP Encounter 171,900 · Imaging 38,061 · PT Referral 21,946 · Injection 9,797 · Surgical Consult 5,293 · Spine Surgery 2,562. Each event code belongs to exactly one category.
- **Outreach channels:** Phone 25,377 · SMS 8,473 · Email 4,614.
- **Outreach programs:** PT 26,900 · WM 11,564. Each creative belongs to exactly one program: PT = Back in Motion, Stay in the Game, Stand Tall; WM = Fresh Start, More Moments.
- **Enrollments:** PT 18,914 · WM 3,431.
- **Claim status:** PAID 68% · DISPENSED 23% · DENIED 6% · REJECTED 2.3% · REVERSED 0.8%.

## Issues that shape the model, most important first

### 1. Meridian's files and Vi's claims stop at different points

Encounters, outreach and enrollments have **zero** rows outside `[coverage_start, coverage_end]`. The only exception is 593 encounters and 25 enrollments shortly before `coverage_start`. Claims are not bounded by coverage:

- 7,308 of the 7,800 patients who left coverage have claims afterwards: 152k lines, median 209 days after `coverage_end`.
- 6,148 patients have claims before their `coverage_start` (83k lines).

**Implication:** limit feature and outcome windows to the coverage period. Otherwise, because only one source keeps going after disenrollment, the difference in coverage leaks into the model. Leaving coverage is also a censoring event: Meridian only bears the cost while a patient is attributed.

### 2. The population is a closed low back pain cohort

- 28,067 of 30,000 patients (94%) have an encounter.
- No coverage starts after 2025-04-30. 23,441 patients start on 2023-07-01, so their earlier history is cut off.
- Patients' first encounters fall from about 3–4k per quarter (2023-Q4 → 2025-Q1) to about 150 per quarter from 2025-Q3 onward.

**Implication:** next quarter's list is drawn from patients Meridian already knows. "Earlier in the patient journey" has to mean earlier on the LBP → imaging → injection → surgical consult path, not first contact.

### 3. Outreach ran in four waves, late in the journey

| Wave | Attempts (PT / WM) |
|---|---|
| 2024-07 → 09 | 7,181 / 3,140 |
| 2025-01 → 03 | 7,913 / 3,163 |
| 2025-07 → 09 | 6,700 / 2,829 |
| 2026-01 → 03 | 5,106 / 2,432 |

- Patients were first contacted a median of **289 days after their first LBP encounter** (IQR 168–439). This is the gap the client wants Vi to close.
- Attempts per patient: median 3, max 9. No patient was ever offered both programs.
- **621 patients were contacted after their first performed surgery** (2,143 attempts).
- **531 patients were contacted on a channel after opting out or unsubscribing on it** (641 attempts). This is a compliance issue to raise with Meridian, and opted-out channels are a hard exclusion for the list.
- 410 contacted patients have no LBP encounter at all.
- Each channel has its own result codes, so the data is internally consistent. Phone: Connected / Voicemail / No Answer / Declined / Invalid Number. SMS: Delivered / Failed / Opted Out. Email: Delivered / Opened / Bounced / Unsubscribed.

**Implication:** the waves, plus the about 19k patients never contacted, give a workable structure for estimating the campaign's effect. The campaigns were not randomized, so who got contacted is confounded with their clinical trajectory.

### 4. Most enrollments have nothing to do with the campaigns

- 10,042 patients enrolled; only 5,785 of them were ever contacted.
- 3,768 enrollments happened *before* that program's first contact for the patient.
- **1,489 enrollments come after the patient's first surgery**, most likely post-op PT. They must not count as "conservative care treatment".
- Patients enroll repeatedly: 1–7 enrollments each, with a median gap of 203 days between same-program enrollments.

**Implication:** treatment is "first enrollment before surgery", and enrolling on their own is a strong competing path into the programs.

### 5. Surgery must be defined from both sources

- 2,556 patients have a Spine Surgery event (any status); 2,046 have one marked `Performed`. Six patients have two surgery events.
- Paid surgery CPT claims (22630, 22633, 63030, 63047) cover only **about 73%** of EHR-performed surgeries. The rate is steady across quarters, so this isn't a reporting lag.
- 66 patients have a paid surgery claim and no EHR surgery event.

**Implication:** outcome = EHR surgery that is `Performed` or has a null status, OR a paid surgery claim, before `coverage_end`. Null-status surgeries match paid claims like Performed ones (64% vs 74% of patients, dated 0–2 days apart). `Scheduled` and `Cancelled` surgeries match about 2%, so they are excluded. EHR and claim CPT codes disagree about 75% of the time, so match on patient and date, not code.

### 6. Encounter statuses are unreliable

- 20,376 events are still `Scheduled` even though their dates have passed. This holds in every quarter, not just recent ones.
- 19,894 (8%) have no status, including 229 surgeries. Another 147 surgeries are `Scheduled` and 96 are `Cancelled`. For non-surgery events, status carries no signal: cancelled imaging matches a paid claim as often as performed imaging (about 25%).
- 113 rows repeat the same patient, date, category and code.
- 9,988 events have no code. These are spread across all categories, so `event_category` is the field to rely on.

### 7. PT happens outside the program

6,466 patients have Physical Therapy claims but never enrolled in the PT program. 98.7% of PT-enrolled patients do have PT claims. Treating "not enrolled" as "no PT" would be wrong. Prior PT is a feature; it doesn't define a control group.

### 8. The claims feed has gaps

- 35,022 lines have no `code_system`. The vocabulary can be inferred from the code format (ICD-10 alphanumeric, CPT 5-digit, HCPCS letter + 4 digits, NDC 11-digit).
- Only 9 of the 15 provider specialties in the data dictionary appear. Neurosurgery, Rheumatology, Anesthesiology, Cardiology, Nephrology and Endocrinology are missing.
- 2,960 lines repeat the same patient, date, code, specialty and status.
- About 9% of lines are DENIED, REJECTED or REVERSED and should not count as care received.
- One patient has no claims at all.
- 117 distinct codes. That's a small, curated vocabulary, so features can be hand-picked groupings.

### 9. Demographics have some oddities

- `birth_year` = 1900 for 63 patients: a placeholder, so treat as missing.
- 1961 has 4,367 patients, about four times its neighbours. Their claim rates, encounter rates and surgery rates match the adjacent years, so it isn't obviously a default value, but it is suspicious.
- 3,410 patients are under 65 in 2026, which fits Medicare disability coverage. Their surgery rate is higher (about 13% for birth years 1964–65 versus about 8–9% for 1955–63).
- Sex is `U` for 322 patients and missing for 143.

### 10. Weight-management targeting is feasible

19,632 patients have a BMI code (Z68.x) and 9,651 have an obesity diagnosis (E66.x). BMI category can be derived for most patients, which supports a rule for which program to recommend.

### Minor

- SMS and email timestamps are spread evenly across all 24 hours, while phone calls happen only from 9:00 to 16:59. The data dictionary's "local time" can't be checked for the automated channels.
- Encounter volume rises from 411 in 2023-07 to about 10k per month in early 2025, then falls to about 6k by 2026-06. This fits a closed cohort that ages and leaves coverage, not onboarding: all 40 facilities first appear in 2023-Q3.
