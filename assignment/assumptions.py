"""Every judgment call a stakeholder might challenge, in one place.

Change a value and re-run `main.py score`; no retrain needed, except for `LATE_STAGE_CONSULT_DAYS`,
which also defines the training population. Sources are in docs/assumptions.md.
"""

# Economics
SURGERY_COST = 20_000  # Medicare 90-day episode, fusion/decompression blend
CONTACT_COST = 10  # amortized cost per contacted patient (Vi call: $5-$10)
SURGERY_REDUCTION_PER_ENROLLMENT = 0.30  # literature assumption; not testable in this data
ENROLLMENT_LIFT = 0.115  # estimated: doubly robust ATT, sandbox/causal/c_aipw.py (89%: 0.106-0.124)
# Same lift and surgery reduction for PT and weight management (Vi call: all enrollments count equally).

# Sensitivity ranges swept into out/sensitivity.csv: (low, high).
SENSITIVITY = {
    "SURGERY_COST": (10_000, 40_000),
    "CONTACT_COST": (5, 10),
    "SURGERY_REDUCTION_PER_ENROLLMENT": (0.10, 0.50),
    "ENROLLMENT_LIFT": (0.106, 0.124),
}

# Eligibility
LATE_STAGE_CONSULT_DAYS = 90  # consult this recent = already late-stage; also defines the training population
RECENT_ENROLLMENT_EXCLUSION_DAYS = 90  # probably mid-course
INCLUDE_PRIOR_CONSULT = True  # consults 90-365 days ago stay in, labeled
BLOCKING_RESULTS = ("Opted Out", "Unsubscribed", "Invalid Number", "Bounced")  # channel can't be used
CHANNELS = ("Phone", "SMS", "Email")

# Program
WM_BMI_THRESHOLD = 30  # or any obesity diagnosis (E66)
PROGRAMS = {"PT": "physical_therapy", "WM": "weight_management"}

# Pilot
LEARNING_BAND_SIZE = 1_100  # patients just below break-even, randomized 50/50: tests whether the lift holds there
TIER1_HOLDOUT_SHARE = 0.20
TIER2_HOLDOUT_SHARE = 0.50
BASELINE_ENROLLMENT = 0.25  # uncontacted enrollment rate, sandbox/causal (for the detectable effect)
ALPHA, POWER = 0.05, 0.80
RANDOM_SEED = 2026

# Evaluation only: fixed list size for precision / recall / accuracy, about the size of past outreach waves
# (2,189-3,183 patients each). Not used to decide how many patients to contact.
EVALUATION_LIST_SIZE = 2_500
