"""Approach A: Bayesian outcome regression + g-computation.

For each outcome: logit P(Y) = alpha_wave + X beta + tau * treated, then average P(Y | do(T=1)) - P(Y | do(T=0))
over the contacted patients (ATT, "what did outreach do for the people Meridian contacted") and over everyone
(ATE). The same model is fit to the negative controls: their "effect" should be ~0 if adjustment worked.
Also refit on one row per patient (first eligible wave) to check that repeated patients don't drive the result.

Run: uv run --group bayes python sandbox/causal/a_gcomp.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import build, design  # noqa: E402
from models import fit_logit, standardize, summarize  # noqa: E402

OUTCOMES = ["enroll_180d", "surgery_12m", "nc_acute_pre", "nc_acute_12m", "nc_surgery_30d"]
df = build()
X, wave_idx = design(df)
T = df.treated.to_numpy().astype(float)
treated_rows = np.flatnonzero(T == 1)
all_rows = np.arange(len(df))

results, saved = {}, {}
for outcome in OUTCOMES:
    y = df[outcome].to_numpy()
    draws = fit_logit(y, X, wave_idx, treated=T, seed=100 + OUTCOMES.index(outcome))
    att, ate = standardize(draws, X, wave_idx, treated_rows), standardize(draws, X, wave_idx, all_rows)
    results[outcome] = {
        "att_rd_pts": summarize(att["rd"]),
        "ate_rd_pts": summarize(ate["rd"]),
        "att_rr": summarize(att["rr"], scale=1),
        "untreated_risk_among_treated_pct": float(att["p0"].mean() * 100),
        "odds_ratio": summarize(np.exp(draws["tau"]), scale=1),
        "divergences": int(draws["divergences"]),
        "tau_rhat": float(draws["tau_rhat"].max()),
    }
    saved[outcome] = draws
    r = results[outcome]["att_rd_pts"]
    print(f"{outcome:<16} ATT {r['mean']:+.2f} pts [{r['lo89']:+.2f}, {r['hi89']:+.2f}]  P(<0)={r['p_below_0']:.2f}  "
          f"baseline {results[outcome]['untreated_risk_among_treated_pct']:.2f}%  div={results[outcome]['divergences']}  rhat={results[outcome]['tau_rhat']:.3f}", flush=True)

# Robustness: one row per patient, first eligible wave only.
first = ~df.duplicated("patient_token")
Xf, wf, Tf = X[first.to_numpy()], wave_idx[first.to_numpy()], T[first.to_numpy()]
for outcome in ["enroll_180d", "surgery_12m"]:
    draws = fit_logit(df.loc[first, outcome].to_numpy(), Xf, wf, treated=Tf, seed=7)
    att = standardize(draws, Xf, wf, np.flatnonzero(Tf == 1))
    results[outcome]["first_wave_only_att_rd_pts"] = summarize(att["rd"])
    r = results[outcome]["first_wave_only_att_rd_pts"]
    print(f"{outcome:<16} first-wave-only ATT {r['mean']:+.2f} pts [{r['lo89']:+.2f}, {r['hi89']:+.2f}]  (n={int(first.sum()):,})", flush=True)

np.savez(HERE / "a_gcomp_draws.npz", **{f"{o}__{k}": v for o, d in saved.items() for k, v in d.items() if k in ("alpha", "beta", "tau")})
(HERE / "a_gcomp.json").write_text(json.dumps(results, indent=2))
