"""Shared patient-snapshot panel for the surgery-risk comparison.

One row per (patient, index date t) for eligible patients:
    label   surgery while still covered, within (t, t + 12 months]
            leaving coverage is a competing event (counts as 0), not censoring
    features computed from the 12 months before t, mostly recency-style windows,
             plus time since the first LBP visit capped at 12m+

Splits: train = 2024-07..2025-04, validation = 2025-07 (out-of-time),
score = 2026-07-01 (no label, the population we'd actually rank).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
CACHE = Path(__file__).resolve().parent / ".cache" / "panel.pkl"
EXTRACT = pd.Timestamp("2026-07-01")
TRAIN_DATES = pd.date_range("2024-07-01", "2025-04-01", freq="QS")
VALID_DATE = pd.Timestamp("2025-07-01")
SCORE_DATE = EXTRACT
HORIZON = pd.Timedelta(days=365)
LOOKBACK = pd.Timedelta(days=365)

SURGERY_CPT = ["22630", "22633", "63030", "63047"]
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
Z68_BMI = {"Z68.1": 19, "Z68.41": 42, "Z68.42": 47} | {f"Z68.{n}": n for n in range(20, 40)}
PT_CPT = ["97110", "97140", "97530"]

FEATURES = [
    "age",
    "under_65",
    "male",
    "sex_unknown",
    "bmi_25_29",
    "bmi_30_34",
    "bmi_35_plus",
    "no_bmi",
    "obesity_dx",
    "prior_consult",
    "log_lbp_90d",
    "log_lbp_365d",
    "days_since_last_lbp",
    "imaging_180d",
    "mri_365d",
    "injection_365d",
    "pt_referral_365d",
    "log_pt_visits",
    "log_ortho_pain_visits",
    "log_ed_visits",
    "log_rx_fills",
    "prior_program",
    "tenure_6_12m",
    "tenure_12m_plus",
    *CONDITIONS,
    *SPINE_DX,
]


def load() -> dict[str, pd.DataFrame]:
    patients = pd.read_csv(
        DATA / "meridian_health_patients.csv", dtype={"zip3": str}, parse_dates=["coverage_start", "coverage_end"]
    )
    patients["birth_year"] = patients.birth_year.mask(patients.birth_year == 1900)  # placeholder value
    # Open-ended coverage: still covered at the extract, so eligible for the 2026-07-01 scoring snapshot.
    patients["coverage_end"] = patients.coverage_end.fillna(pd.Timestamp("2099-12-31"))
    encounters = pd.read_csv(DATA / "meridian_health_encounters.csv", parse_dates=["event_date"])
    claims = pd.read_csv(
        DATA / "data_web_claims.csv.gz",
        dtype={"code": str},
        usecols=["patient_token", "service_date", "code", "claim_status", "place_of_service", "provider_specialty"],
        parse_dates=["service_date"],
    )
    claims = claims[claims.claim_status.isin(["PAID", "DISPENSED"])]
    enrollments = pd.read_csv(DATA / "meridian_health_enrollments.csv", parse_dates=["enrollment_date"])

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
        "enrollments": enrollments,
        "first_surgery": surgery_dates.groupby("patient_token").event_date.min(),
    }


def snapshot(d: dict[str, pd.DataFrame], t: pd.Timestamp) -> pd.DataFrame:
    p, occ, clm_all, first_surgery = d["patients"], d["occurred"], d["claims"], d["first_surgery"]
    lbp_all = occ[occ.event_category == "LBP Encounter"]
    first_lbp = lbp_all.groupby("patient_token").event_date.min()

    in_window = lambda df, col, days: df[df[col].between(t - pd.Timedelta(days=days), t, inclusive="left")]
    recent_consult = in_window(occ[occ.event_category == "Surgical Consult"], "event_date", 90).patient_token
    p = p[(p.coverage_start <= t) & (p.coverage_end > t)]
    p = p[
        (first_lbp.reindex(p.index) < t).to_numpy()
        & ~(first_surgery.reindex(p.index) < t).to_numpy()
        & ~p.index.isin(recent_consult)
    ]

    enc = in_window(occ, "event_date", 365)
    clm = in_window(clm_all, "service_date", 365)
    codes = pd.concat([enc[["patient_token", "event_code"]].rename(columns={"event_code": "code"}), clm[["patient_token", "code"]]])
    has_code = lambda values: p.index.isin(codes.loc[codes.code.isin(values), "patient_token"])
    has_event = lambda df, category: p.index.isin(df.loc[df.event_category == category, "patient_token"])
    count = lambda df: np.log1p(df.groupby("patient_token").size().reindex(p.index, fill_value=0))

    bmi = clm[clm.code.isin(Z68_BMI)].sort_values("service_date").groupby("patient_token").code.last().map(Z68_BMI).reindex(p.index)
    lbp = enc[enc.event_category == "LBP Encounter"]
    last_lbp = (t - lbp_all[lbp_all.event_date < t].groupby("patient_token").event_date.max()).dt.days
    # Capped at 12m+: risk drops sharply once a patient passes a year without surgery, and capping keeps the
    # 2026-07 population (all 12m+, half 24m+) inside a band the training data covers. The first visit is the
    # first one in the extract (left-truncated), which only blurs the <6m / 6-12m bands.
    months_since_first_lbp = (t - first_lbp.reindex(p.index)).dt.days / 30.4
    age = t.year - p.birth_year
    age = age.fillna(age.median())

    out = pd.DataFrame(
        {
            "index_date": t,
            "age": age,
            "under_65": age < 65,
            "male": p.sex.eq("M"),
            "sex_unknown": ~p.sex.isin(["F", "M"]),
            "bmi_25_29": bmi.between(25, 29.99),
            "bmi_30_34": bmi.between(30, 34.99),
            "bmi_35_plus": bmi >= 35,
            "no_bmi": bmi.isna(),
            "obesity_dx": has_code(["E66.9", "E66.01"]),
            # Consults in the last 90 days are excluded, so this is a consult 90-365 days ago.
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
            **{k: has_code(v) for k, v in CONDITIONS.items()},
            **{k: has_code(v) for k, v in SPINE_DX.items()},
        },
        index=p.index,
    )
    if t + HORIZON <= EXTRACT:
        surgery = first_surgery.reindex(p.index)
        out["y"] = ((surgery >= t) & (surgery < t + HORIZON) & (surgery <= p.coverage_end)).astype(int)
    out[FEATURES] = out[FEATURES].astype(float)
    return out.rename_axis("patient_token").reset_index()


def build_panel(refresh: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pickle.loads(CACHE.read_bytes())
    d = load()
    panel = pd.concat([snapshot(d, t) for t in [*TRAIN_DATES, VALID_DATE, SCORE_DATE]], ignore_index=True)
    panel["split"] = np.select(
        [panel.index_date.isin(TRAIN_DATES), panel.index_date == VALID_DATE], ["train", "valid"], "score"
    )
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps(panel))
    return panel


def design(panel: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray | None, pd.DataFrame]]:
    """Standardized X per split, using train means/SDs, so both attempts see identical inputs and
    a coefficient means "change in log-odds per 1 SD" for every feature."""
    train = panel[panel.split == "train"]
    mu, sd = train[FEATURES].mean(), train[FEATURES].std().replace(0, 1)
    out = {}
    for split, df in panel.groupby("split"):
        X = ((df[FEATURES] - mu) / sd).to_numpy()
        y = df.y.to_numpy().astype(int) if split != "score" else None
        out[split] = (X, y, df[["patient_token", "index_date"]].reset_index(drop=True))
    return out


if __name__ == "__main__":
    panel = build_panel(refresh=True)
    print(panel.groupby("split").agg(rows=("patient_token", "size"), patients=("patient_token", "nunique"), surgeries=("y", "sum"), rate=("y", "mean")))
    assert panel[FEATURES].notna().all().all(), panel[FEATURES].isna().sum()[lambda s: s > 0]
    assert panel.loc[panel.split != "score", "y"].notna().all()
    print(panel.groupby("split")[["days_since_last_lbp", "log_lbp_90d", "prior_consult"]].mean().round(3))
