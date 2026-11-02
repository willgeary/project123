"""Target-trial emulation dataset: one emulated trial per outreach wave.

    Eligible at wave start t   covered, LBP history, no surgery yet, no consult in the last 90 days (the same
                               population as the risk model), and never opted out or unsubscribed before t
    Treatment                  any contact attempt in the wave, i.e. within 92 days of t (intention to treat)
    Time zero                  t for both arms, so nobody has to "survive" to be counted as untreated
    Covariates                 the 36 risk-model features at t, plus prior outreach in earlier waves
    Outcomes                   enroll_180d   enrolled in PT or WM within 180 days, before any surgery
                               surgery_12m   surgery while covered within 12 months (the risk-model label)
    Negative controls          outcomes outreach can't plausibly move; a "treatment effect" on them is bias
                               nc_acute_pre  non-spine acute visit in the 90 days BEFORE t (placebo, pre-treatment)
                               nc_acute_12m  non-spine acute visit in the 12 months after t
                               nc_surgery_30d surgery within 30 days of t, before a program could act

Waves with 12 months of follow-up: 2024-07, 2025-01, 2025-07. Patients can appear in more than one wave.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from panel import DATA, FEATURES, build_panel, load  # noqa: E402

CACHE = HERE.parent / ".cache" / "causal.pkl"
WAVES = pd.to_datetime(["2024-07-01", "2025-01-01", "2025-07-01"])
WAVE_DAYS = 92
ACUTE_NON_SPINE = ["R07.9", "J06.9", "N39.0", "R10.9", "R51.9"]  # chest pain, URI, UTI, abdominal pain, headache
COVARIATES = [*FEATURES, "prior_outreach"]


def build(refresh: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        return pickle.loads(CACHE.read_bytes())
    panel = build_panel()
    d = load()
    outreach = pd.read_csv(DATA / "meridian_health_outreach.csv", parse_dates=["timestamp"])
    opt_out = outreach[outreach.attempt_result.isin(["Opted Out", "Unsubscribed"])].groupby("patient_token").timestamp.min()
    enrollments, claims, first_surgery = d["enrollments"], d["claims"], d["first_surgery"]
    acute = claims[claims.code.isin(ACUTE_NON_SPINE)]
    coverage_end = d["patients"].coverage_end

    trials = []
    for t in WAVES:
        s = panel[panel.index_date == t].copy()
        s = s[~(s.patient_token.map(opt_out) < t)]
        in_wave = outreach[(outreach.timestamp >= t) & (outreach.timestamp < t + pd.Timedelta(days=WAVE_DAYS))]
        s["treated"] = s.patient_token.isin(in_wave.patient_token).astype(int)
        s["program_offered"] = s.patient_token.map(in_wave.groupby("patient_token").program.first())
        s["prior_outreach"] = s.patient_token.isin(outreach.loc[outreach.timestamp < t, "patient_token"]).astype(float)

        surgery = s.patient_token.map(first_surgery)
        e = enrollments[(enrollments.enrollment_date >= t) & (enrollments.enrollment_date < t + pd.Timedelta(days=180))]
        e = e[~(e.patient_token.map(first_surgery) <= e.enrollment_date)]
        s["enroll_180d"] = s.patient_token.isin(e.patient_token).astype(int)
        s["surgery_12m"] = s.y.astype(int)

        end = np.minimum(s.patient_token.map(coverage_end), t + pd.Timedelta(days=365))
        a = acute.merge(s[["patient_token"]].assign(end=end.to_numpy()), on="patient_token")
        s["nc_acute_12m"] = s.patient_token.isin(a.loc[(a.service_date >= t) & (a.service_date < a.end), "patient_token"]).astype(int)
        s["nc_acute_pre"] = s.patient_token.isin(acute.loc[acute.service_date.between(t - pd.Timedelta(days=90), t, inclusive="left"), "patient_token"]).astype(int)
        s["nc_surgery_30d"] = ((surgery >= t) & (surgery < t + pd.Timedelta(days=30))).astype(int)
        trials.append(s)

    df = pd.concat(trials, ignore_index=True)
    df["wave"] = df.index_date.dt.strftime("%Y-%m")
    CACHE.write_bytes(pickle.dumps(df))
    return df


def design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Standardized covariates and integer wave index."""
    X = df[COVARIATES].astype(float)
    X = ((X - X.mean()) / X.std().replace(0, 1)).to_numpy()
    wave_idx = pd.Categorical(df.wave).codes
    return X, wave_idx


if __name__ == "__main__":
    df = build(refresh=True)
    print(df.groupby(["wave", "treated"]).agg(n=("y", "size"), enroll_180d=("enroll_180d", "mean"), surgery_12m=("surgery_12m", "mean"),
                                                nc_acute_12m=("nc_acute_12m", "mean"), nc_acute_pre=("nc_acute_pre", "mean"), nc_surgery_30d=("nc_surgery_30d", "mean")).round(4))
    print("rows", len(df), "patients", df.patient_token.nunique(), "treated", int(df.treated.sum()), "program offered:", df.program_offered.value_counts().to_dict())
    assert df[COVARIATES].notna().all().all()
