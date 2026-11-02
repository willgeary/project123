"""Collect the causal results into causal_deck_data.json for build.js.

Run after every approach script:  uv run python sandbox/causal/deck/prepare.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CAUSAL = HERE.parent
sys.path.insert(0, str(CAUSAL))
from data import build  # noqa: E402

load = lambda name: json.loads((CAUSAL / f"{name}.json").read_text())
A, B, C, D, E = (load(n) for n in ["a_gcomp", "b_subclass", "c_aipw", "d_heterogeneity", "e_landmark"])
ps = load("propensity_summary")
balance = pd.read_csv(CAUSAL / "balance.csv", index_col=0)
c_draws = np.load(CAUSAL / "c_aipw_draws.npz")
e_draws = np.load(CAUSAL / "e_landmark_draws.npz")
df = build()


def hist(draws: np.ndarray, lo: float, hi: float, bins: int = 30, scale: float = 100) -> dict:
    edges = np.linspace(lo, hi, bins + 1)
    counts = np.histogram(draws * scale, edges)[0] / len(draws)
    return {"mid": ((edges[:-1] + edges[1:]) / 2).round(3).tolist(), "share": (counts * 100).round(3).tolist()}


waves = (
    df.groupby("wave")
    .agg(eligible=("treated", "size"), contacted=("treated", "sum"))
    .assign(share=lambda x: x.contacted / x.eligible)
    .reset_index()
    .to_dict("records")
)
raw = df.groupby("treated")[["enroll_180d", "surgery_12m", "nc_surgery_30d"]].mean().mul(100).round(2).to_dict("index")

top_imbalance = balance.reindex(balance.smd_raw.abs().sort_values(ascending=False).index).head(8)
labels = {
    "prior_outreach": "Contacted in an earlier wave",
    "imaging_180d": "Imaging in last 180 days",
    "prior_consult": "Prior surgical consult",
    "mri_365d": "MRI in last 12 months",
    "tenure_12m_plus": "12+ months since first visit",
    "log_lbp_365d": "Back-pain visits, last 12 months",
    "age": "Age",
    "log_rx_fills": "Pharmacy fills",
}

out = {
    "waves": waves,
    "n_treated": ps["n_treated"],
    "n_control": ps["n_control"],
    "propensity": ps,
    "raw": {str(k): v for k, v in raw.items()},
    "balance": [{"label": labels.get(i, i), "raw": float(r.smd_raw), "weighted": float(r.smd_weighted)} for i, r in top_imbalance.iterrows()],
    "approaches": {
        "A": {o: A[o]["att_rd_pts"] for o in ["enroll_180d", "surgery_12m", "nc_surgery_30d", "nc_acute_12m", "nc_acute_pre"]},
        "A_first_wave": {o: A[o]["first_wave_only_att_rd_pts"] for o in ["enroll_180d", "surgery_12m"]},
        "B": {o: B[o]["att_rd_pts"] for o in B},
        "C": {o: C[o]["att_rd_pts"] for o in C},
    },
    "enroll_baseline_pct": A["enroll_180d"]["untreated_risk_among_treated_pct"],
    "surgery_baseline_pct": A["surgery_12m"]["untreated_risk_among_treated_pct"],
    "heterogeneity": {
        "baseline": D["quintile_baseline_risk_pct"],
        "enroll": D["enroll_180d"]["by_quintile"],
        "surgery": D["surgery_12m"]["by_quintile"],
    },
    "landmark": {
        "nc_30_90": E["surgery_30_90"],
        "main_90_365": E["surgery_90_365"],
        "calibrated_rr": E["calibrated_rr_90_365"],
        "calibrated_rd": E["calibrated_rd_90_365_pts"],
        "p_protective": E["p_calibrated_protective"],
        "e_value": E["e_value_main_rr"],
        "wald": E["enrollment_effect_on_surgery_pts"],
    },
    "posterior": {
        "enroll_aipw": hist(c_draws["enroll_180d"], 9, 14.5),
        "surgery_aipw": hist(c_draws["surgery_12m"], -0.6, 1.2),
        "surgery_calibrated": hist(e_draws["calibrated_rd"], -1.0, 1.4),
    },
}
(HERE / "causal_deck_data.json").write_text(json.dumps(out, indent=1))
print("wrote causal_deck_data.json;", {k: round(v["mean"], 2) for k, v in out["approaches"]["C"].items()})
