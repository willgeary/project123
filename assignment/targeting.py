"""Turn risk scores into the outreach list: eligibility, program, ranking, tiers, randomized holdout."""

from math import sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

from assignment import assumptions as A

ECONOMICS = ("SURGERY_COST", "CONTACT_COST", "SURGERY_REDUCTION_PER_ENROLLMENT", "ENROLLMENT_LIFT")


def economics(**overrides) -> dict:
    """The economic assumptions, with optional overrides (sensitivity analysis, tests)."""
    return {k: overrides.get(k, getattr(A, k)) for k in ECONOMICS}


def value_per_risk(e: dict) -> float:
    """Dollars of surgery avoided per contact, per unit of surgery risk."""
    return e["ENROLLMENT_LIFT"] * e["SURGERY_REDUCTION_PER_ENROLLMENT"] * e["SURGERY_COST"]


def break_even(e: dict) -> float:
    """Surgery risk at which a contact pays for itself."""
    return e["CONTACT_COST"] / value_per_risk(e)


def tier_sizes(score: pd.Series, e: dict) -> tuple[int, int]:
    """Tier 1: everyone above break-even. Tier 2: the learning band just below it."""
    tier1 = int((score > break_even(e)).sum())
    return tier1, min(A.LEARNING_BAND_SIZE, len(score) - tier1)


def recent_enrollees(enrollments: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """Tokens of patients enrolled in a program within the exclusion window before t."""
    recent = enrollments.enrollment_date.between(
        t - pd.Timedelta(days=A.RECENT_ENROLLMENT_EXCLUSION_DAYS), t, inclusive="left"
    )
    return enrollments.loc[recent, "patient_token"]


def allowed_channels(tokens: pd.Series, outreach: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """Channels each patient can still be contacted on, as "Phone|SMS|Email"; "" if none."""
    blocked = (
        outreach[outreach.attempt_result.isin(A.BLOCKING_RESULTS) & (outreach.timestamp < t)]
        .groupby("patient_token")
        .channel.agg(set)
    )
    usable = blocked.apply(lambda b: "|".join(c for c in A.CHANNELS if c not in b))
    return tokens.map(usable).fillna("|".join(A.CHANNELS))


def apply_exclusions(
    df: pd.DataFrame, enrollments: pd.DataFrame, outreach: pd.DataFrame, t: pd.Timestamp
) -> pd.DataFrame:
    """Drop recent enrollees and patients with no usable channel; add `allowed_channels`."""
    allowed = allowed_channels(df.patient_token, outreach, t)
    keep = ~df.patient_token.isin(recent_enrollees(enrollments, t)) & allowed.ne("")
    if not A.INCLUDE_PRIOR_CONSULT:
        keep &= df.prior_consult.eq(0)
    return df.assign(allowed_channels=allowed)[keep]


def recommend_program(df: pd.DataFrame) -> pd.Series:
    """Weight management for BMI >= threshold or an obesity diagnosis; PT otherwise (including no BMI on file)."""
    wm = df.bmi.ge(A.WM_BMI_THRESHOLD) | df.obesity_dx.eq(1)
    return pd.Series(np.where(wm, A.PROGRAMS["WM"], A.PROGRAMS["PT"]), index=df.index)


def assign(df: pd.DataFrame, e: dict | None = None) -> pd.DataFrame:
    """Rank by risk, set tiers, and randomize the holdout within each tier x program stratum."""
    e = e or economics()
    out = df.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    tier1, tier2 = tier_sizes(out.score, e)
    out["rank"] = np.arange(1, len(out) + 1)
    out["tier"] = np.select([out["rank"] <= tier1, out["rank"] <= tier1 + tier2], [1, 2], 0)
    out["expected_net_value"] = out.score * value_per_risk(e) - e["CONTACT_COST"]
    out["arm"] = "none"
    rng = np.random.default_rng(A.RANDOM_SEED)
    share = {1: A.TIER1_HOLDOUT_SHARE, 2: A.TIER2_HOLDOUT_SHARE}
    for (tier, _), stratum in out[out.tier > 0].groupby(["tier", "program_recommended"]):
        held = rng.permutation(stratum.index)[: round(share[tier] * len(stratum))]
        out.loc[stratum.index, "arm"] = "contact"
        out.loc[held, "arm"] = "holdout"
    return out


def detectable_effect(n_contact: int, n_holdout: int) -> float:
    """Smallest enrollment lift the pilot can detect (two-sided, normal approximation)."""
    z = NormalDist().inv_cdf(1 - A.ALPHA / 2) + NormalDist().inv_cdf(A.POWER)
    p = A.BASELINE_ENROLLMENT
    return z * sqrt(p * (1 - p) * (1 / n_contact + 1 / n_holdout))


def sensitivity(score: pd.Series) -> pd.DataFrame:
    """Break-even and tier sizes with each economic assumption at its low and high value."""
    rows = [("base", None, economics())]
    rows += [(name, v, economics(**{name: v})) for name, bounds in A.SENSITIVITY.items() for v in bounds]
    return pd.DataFrame(
        [
            {
                "assumption": n,
                "value": v,
                "break_even": break_even(e),
                "above_break_even": int((score > break_even(e)).sum()),
                **dict(zip(("tier1", "tier2"), tier_sizes(score, e))),
            }
            for n, v, e in rows
        ]
    )
