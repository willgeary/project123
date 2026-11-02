import pandas as pd

from assignment.features import eligibility

T = pd.Timestamp("2026-07-01")
DAY = pd.Timedelta(days=1)


def test_eligibility_steps_are_cumulative():
    """Each eligibility rule removes the one patient that fails it, and later steps keep earlier removals."""
    patients = pd.DataFrame(
        {"coverage_start": T - 400 * DAY, "coverage_end": [T + DAY, T - DAY, T + DAY, T + DAY, T + DAY]},
        index=pd.Index(["ok", "uncovered", "no_lbp", "operated", "late_stage"], name="patient_token"),
    )
    occurred = pd.DataFrame(
        {
            "patient_token": ["ok", "uncovered", "operated", "late_stage", "late_stage"],
            "event_category": ["LBP Encounter"] * 4 + ["Surgical Consult"],
            "event_date": [T - 100 * DAY] * 4 + [T - 10 * DAY],
        }
    )
    d = {"patients": patients, "occurred": occurred, "first_surgery": pd.Series({"operated": T - 5 * DAY})}
    counts = [int(m.sum()) for m in eligibility(d, T).values()]
    assert counts == [4, 3, 2, 1]
    assert list(eligibility(d, T).values())[-1].loc["ok"]
