"""Patient snapshots: one row per eligible patient at an index date t.

Label: surgery while still covered within 12 months of t (leaving coverage counts as 0, a competing event).
Features: the 12 months before t, mostly recency windows, plus time since first back-pain visit capped at 12m+.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from assignment.assumptions import LATE_STAGE_CONSULT_DAYS

DATA_START = pd.Timestamp("2023-07-01")
EXTRACT = pd.Timestamp("2026-07-01")
HORIZON = LOOKBACK = pd.Timedelta(days=365)
OPEN_ENDED = pd.Timestamp("2099-12-31")  # coverage with no end date: still covered at the extract

SURGERY_CPT = ["22630", "22633", "63030", "63047"]
PT_CPT = ["97110", "97140", "97530"]
OBESITY_CODES = ["E66.9", "E66.01"]
# Z68.1 = BMI <= 19.9, Z68.2x/3x = 20-39 by unit, Z68.41/42 = 40-44.9 / 45-49.9 (midpoints).
Z68_BMI = {"Z68.1": 19, "Z68.41": 42, "Z68.42": 47} | {f"Z68.{n}": n for n in range(20, 40)}
CONDITIONS = {
    "diabetes": ["E11.9"],
    "hypertension": ["I10"],
    "heart_disease": ["I25.10"],
    "copd": ["J44.9"],
    "ckd": ["N18.3"],
    "depression_anxiety": ["F32.9", "F41.1"],
    "smoker": ["F17.210"],
    "osteoporosis": ["M81.0"],
}
SPINE_DX = {
    "radiculopathy": ["M54.16", "M54.17"],
    "stenosis": ["M48.061", "M48.062"],
    "spondylolisthesis": ["M43.16"],
    "disc_degeneration": ["M51.36"],
}
FEATURES = [
    "age", "under_65", "male", "sex_unknown",
    "bmi_25_29", "bmi_30_34", "bmi_35_plus", "no_bmi", "obesity_dx",
    "prior_consult", "log_lbp_90d", "log_lbp_365d", "days_since_last_lbp",
    "imaging_180d", "mri_365d", "injection_365d", "pt_referral_365d",
    "log_pt_visits", "log_ortho_pain_visits", "log_ed_visits", "log_rx_fills",
    "prior_program", "tenure_6_12m", "tenure_12m_plus",
    *CONDITIONS, *SPINE_DX,
]  # fmt: skip


def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the five extract files and derive each patient's first surgery date."""
    patients = pd.read_csv(
        data_dir / "meridian_health_patients.csv", dtype={"zip3": str}, parse_dates=["coverage_start", "coverage_end"]
    )
    patients["birth_year"] = patients.birth_year.mask(patients.birth_year == 1900)  # placeholder value
    patients["coverage_end"] = patients.coverage_end.fillna(OPEN_ENDED)
    encounters = pd.read_csv(data_dir / "meridian_health_encounters.csv", parse_dates=["event_date"])
    claims = pd.read_csv(
        data_dir / "data_web_claims.csv.gz",
        dtype={"code": str},
        usecols=["patient_token", "service_date", "code", "claim_status", "place_of_service", "provider_specialty"],
        parse_dates=["service_date"],
    )
    claims = claims[claims.claim_status.isin(["PAID", "DISPENSED"])]
    # Null-status surgeries match paid claims like Performed ones; Scheduled ones behave like Cancelled.
    ehr_surgery = (encounters.event_category == "Spine Surgery") & (
        encounters.event_status.eq("Performed") | encounters.event_status.isna()
    )
    surgery_dates = pd.concat(
        [
            encounters.loc[ehr_surgery, ["patient_token", "event_date"]],
            claims.loc[claims.code.isin(SURGERY_CPT), ["patient_token", "service_date"]].rename(
                columns={"service_date": "event_date"}
            ),
        ]
    )
    return {
        "patients": patients.set_index("patient_token"),
        # Status carries no signal outside surgery; only explicit Cancelled / No Show are dropped.
        "occurred": encounters[~encounters.event_status.isin(["Cancelled", "No Show"])],
        "claims": claims,
        "enrollments": pd.read_csv(data_dir / "meridian_health_enrollments.csv", parse_dates=["enrollment_date"]),
        "outreach": pd.read_csv(data_dir / "meridian_health_outreach.csv", parse_dates=["timestamp"]),
        "first_surgery": surgery_dates.groupby("patient_token").event_date.min(),
    }


def eligibility(d: dict[str, pd.DataFrame], t: pd.Timestamp) -> dict[str, pd.Series]:
    """Cumulative eligibility masks over all patients, in order: each step keeps patients passing it and every step before."""
    p, occ, first_surgery = d["patients"], d["occurred"], d["first_surgery"]
    first_lbp = occ[occ.event_category == "LBP Encounter"].groupby("patient_token").event_date.min()
    consults = occ[
        (occ.event_category == "Surgical Consult")
        & occ.event_date.between(t - pd.Timedelta(days=LATE_STAGE_CONSULT_DAYS), t, inclusive="left")
    ]
    rules = {
        "Covered on the index date": (p.coverage_start <= t) & (p.coverage_end > t),
        "With back-pain history": first_lbp.reindex(p.index) < t,
        "No surgery yet": ~(first_surgery.reindex(p.index) < t),
        f"No surgical consult in the last {LATE_STAGE_CONSULT_DAYS} days": ~p.index.isin(consults.patient_token),
    }
    steps, keep = {}, pd.Series(True, index=p.index)
    for name, rule in rules.items():
        keep = keep & rule
        steps[name] = keep
    return steps


def snapshot(d: dict[str, pd.DataFrame], t: pd.Timestamp) -> pd.DataFrame:
    """Eligible patients at t with features, raw BMI, and the label `y` when 12 months of follow-up exist."""
    p, occ, clm_all, first_surgery = d["patients"], d["occurred"], d["claims"], d["first_surgery"]
    lbp_all = occ[occ.event_category == "LBP Encounter"]
    first_lbp = lbp_all.groupby("patient_token").event_date.min()

    def in_window(df: pd.DataFrame, col: str, days: int) -> pd.DataFrame:
        return df[df[col].between(t - pd.Timedelta(days=days), t, inclusive="left")]

    p = p[list(eligibility(d, t).values())[-1]]

    enc = in_window(occ, "event_date", LOOKBACK.days)
    clm = in_window(clm_all, "service_date", LOOKBACK.days)
    codes = pd.concat(
        [enc[["patient_token", "event_code"]].rename(columns={"event_code": "code"}), clm[["patient_token", "code"]]]
    )
    has_code = lambda values: p.index.isin(codes.loc[codes.code.isin(values), "patient_token"])
    has_event = lambda df, category: p.index.isin(df.loc[df.event_category == category, "patient_token"])
    count = lambda df: np.log1p(df.groupby("patient_token").size().reindex(p.index, fill_value=0))

    bmi = (
        clm[clm.code.isin(Z68_BMI)]
        .sort_values("service_date")
        .groupby("patient_token")
        .code.last()
        .map(Z68_BMI)
        .reindex(p.index)
    )
    lbp = enc[enc.event_category == "LBP Encounter"]
    last_lbp = (t - lbp_all[lbp_all.event_date < t].groupby("patient_token").event_date.max()).dt.days
    # Capped at 12m+ so the scoring population (all 12m+, half 24m+) stays inside the training range.
    # The first visit is the first in the extract (left-truncated), which only blurs the <6m / 6-12m bands.
    months_since_first_lbp = (t - first_lbp.reindex(p.index)).dt.days / 30.4
    age = t.year - p.birth_year
    age = age.fillna(age.median())

    out = pd.DataFrame(
        {
            "index_date": t,
            "bmi": bmi,
            "age": age,
            "under_65": age < 65,
            "male": p.sex.eq("M"),
            "sex_unknown": ~p.sex.isin(["F", "M"]),
            "bmi_25_29": bmi.between(25, 29.99),
            "bmi_30_34": bmi.between(30, 34.99),
            "bmi_35_plus": bmi >= 35,
            "no_bmi": bmi.isna(),
            "obesity_dx": has_code(OBESITY_CODES),
            # Recent consults are ineligible, so this is a consult 90-365 days ago.
            "prior_consult": has_event(enc, "Surgical Consult"),
            "log_lbp_90d": count(in_window(lbp, "event_date", 90)),
            "log_lbp_365d": count(lbp),
            "days_since_last_lbp": last_lbp.reindex(p.index).clip(upper=730) / 365,
            "imaging_180d": has_event(in_window(enc, "event_date", 180), "Imaging"),
            "mri_365d": has_code(["72148"]),
            "injection_365d": has_event(enc, "Injection"),
            "pt_referral_365d": has_event(enc, "PT Referral"),
            "log_pt_visits": count(clm[clm.code.isin(PT_CPT)]),
            "log_ortho_pain_visits": count(clm[clm.provider_specialty.isin(["Orthopedic Surgery", "Pain Management"])]),
            "log_ed_visits": count(clm[clm.place_of_service == "Emergency Department"]),
            "log_rx_fills": count(clm[clm.claim_status == "DISPENSED"]),
            "prior_program": p.index.isin(in_window(d["enrollments"], "enrollment_date", 365).patient_token),
            "tenure_6_12m": months_since_first_lbp.between(6, 12, inclusive="left"),
            "tenure_12m_plus": months_since_first_lbp >= 12,
            **{k: has_code(v) for k, v in (CONDITIONS | SPINE_DX).items()},
        },
        index=p.index,
    )
    if t + HORIZON <= EXTRACT:
        surgery = first_surgery.reindex(p.index)
        out["y"] = ((surgery >= t) & (surgery < t + HORIZON) & (surgery <= p.coverage_end)).astype(int)
    out[FEATURES] = out[FEATURES].astype(float)
    return out.rename_axis("patient_token").reset_index()


def training_panel(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Quarterly snapshots with a full lookback and a full 12-month label: 2024-07 to 2025-07."""
    dates = pd.date_range(DATA_START + LOOKBACK, EXTRACT - HORIZON, freq="QS")
    return pd.concat([snapshot(d, t) for t in dates], ignore_index=True)
