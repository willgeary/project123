import numpy as np
import pandas as pd

from assignment import assumptions as A
from assignment.targeting import apply_exclusions, assign, break_even, economics, recommend_program
from assignment.utils import validate_targets

T = pd.Timestamp("2026-07-01")
DAY = pd.Timedelta(days=1)


def patients(n: int, **cols) -> pd.DataFrame:
    return pd.DataFrame({"patient_token": [f"p{i}" for i in range(n)], "prior_consult": 0.0, **cols})


def test_exclusions_drop_recent_enrollees_and_unreachable_patients():
    """Recent enrollees and all-channels-blocked patients are dropped; blocks after t are ignored."""
    enrollments = pd.DataFrame({"patient_token": ["p0", "p1"], "enrollment_date": [T - 10 * DAY, T - 200 * DAY]})
    outreach = pd.DataFrame(
        {
            "patient_token": ["p2", "p2", "p2", "p3", "p4"],
            "channel": ["Phone", "SMS", "Email", "SMS", "Phone"],
            "attempt_result": ["Opted Out", "Opted Out", "Bounced", "Unsubscribed", "Opted Out"],
            "timestamp": [T - DAY, T - DAY, T - DAY, T - DAY, T + DAY],  # p4's opt-out is after t
        }
    )
    out = apply_exclusions(patients(5), enrollments, outreach, T).set_index("patient_token")
    assert list(out.index) == ["p1", "p3", "p4"]  # p0 enrolled 10 days ago, p2 has no channel left
    assert out.allowed_channels.to_dict() == {"p1": "Phone|SMS|Email", "p3": "Phone|Email", "p4": "Phone|SMS|Email"}


def test_program_rule():
    """Weight management for BMI >= 30 or an obesity diagnosis; PT otherwise, including no BMI."""
    df = pd.DataFrame({"bmi": [32, 25, np.nan, np.nan], "obesity_dx": [0, 0, 1, 0]})
    assert list(recommend_program(df)) == [A.PROGRAMS[k] for k in ("WM", "PT", "WM", "PT")]


def test_tiers_and_holdout():
    """Tiers split at break-even, holdouts are randomized at the set share, and no holdout is contacted."""
    e = economics()
    df = patients(5_000, score=np.linspace(0.05, 0.0001, 5_000), program_recommended=A.PROGRAMS["PT"])
    out = assign(df, e)
    tier1, tier2 = out[out.tier == 1], out[out.tier == 2]
    assert (tier1.score > break_even(e)).all() and (tier2.score <= break_even(e)).all()
    assert len(tier2) == A.LEARNING_BAND_SIZE and tier2["rank"].min() == len(tier1) + 1
    assert out.loc[out.tier == 0, "arm"].eq("none").all()
    assert round(A.TIER1_HOLDOUT_SHARE * len(tier1)) == (tier1.arm == "holdout").sum()
    contact = out[out.arm == "contact"]
    assert not set(contact.patient_token) & set(out.loc[out.arm == "holdout", "patient_token"])
    validate_targets(contact)
    assert out.arm.equals(assign(df, e).arm)  # fixed seed


def test_assumptions_move_the_cut_off():
    """A larger surgery reduction lowers the break-even risk and grows tier 1."""
    df = patients(300, score=np.linspace(0.05, 0.001, 300), program_recommended=A.PROGRAMS["PT"])
    low, high = (economics(SURGERY_REDUCTION_PER_ENROLLMENT=r) for r in (0.1, 0.5))
    assert break_even(low) > break_even(high)
    assert (assign(df, low).tier == 1).sum() < (assign(df, high).tier == 1).sum()
