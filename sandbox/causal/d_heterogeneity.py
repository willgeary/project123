"""Approach D: does the effect differ by baseline risk? Hierarchical treatment effects by risk quintile.

Quintiles of each patient's predicted surgery risk *without* contact (posterior-mean outcome model from approach A).
tau_k = mu_tau + sigma_tau * z_k (non-centered, partial pooling), fit for enrollment and surgery. This is the
question that matters for targeting: if outreach only moves enrollment or surgery in the high-risk quintiles,
that's where the list should concentrate.

Run after a_gcomp.py:  uv run --group bayes python sandbox/causal/d_heterogeneity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import build, design  # noqa: E402
from models import fit_logit, standardize, summarize  # noqa: E402

N_GROUPS = 5
df = build()
X, wave_idx = design(df)
T = df.treated.to_numpy().astype(float)
A = np.load(HERE / "a_gcomp_draws.npz")
baseline = expit(A["surgery_12m__alpha"].mean(axis=0)[wave_idx] + X @ A["surgery_12m__beta"].mean(axis=0))
group = pd.qcut(baseline, N_GROUPS, labels=False)

results = {"quintile_baseline_risk_pct": [float(baseline[group == k].mean() * 100) for k in range(N_GROUPS)]}
for o in ["enroll_180d", "surgery_12m"]:
    draws = fit_logit(df[o].to_numpy(), X, wave_idx, treated=T, group_idx=group, seed=31, target_accept=0.95)  # hierarchical funnel: fewer divergences
    rows = []
    for k in range(N_GROUPS):
        att = standardize(draws, X, wave_idx, np.flatnonzero((T == 1) & (group == k)), group_idx=group)
        rows.append({"quintile": k + 1, **summarize(att["rd"]), "rr_mean": float(att["rr"].mean())})
    results[o] = {
        "by_quintile": rows,
        "sigma_tau": summarize(draws["sigma_tau"], scale=1),
        "divergences": int(draws["divergences"]),
        "tau_rhat": float(draws["tau_rhat"].max()),
    }
    print(f"\n{o} (divergences {results[o]['divergences']}, max r-hat {results[o]['tau_rhat']:.3f}, "
          f"sigma_tau {results[o]['sigma_tau']['mean']:.2f} [{results[o]['sigma_tau']['lo89']:.2f}, {results[o]['sigma_tau']['hi89']:.2f}])")
    for r, b in zip(rows, results["quintile_baseline_risk_pct"]):
        print(f"  Q{r['quintile']} baseline {b:.2f}%: ATT {r['mean']:+.2f} pts [{r['lo89']:+.2f}, {r['hi89']:+.2f}]  P(<0)={r['p_below_0']:.2f}", flush=True)
(HERE / "d_heterogeneity.json").write_text(json.dumps(results, indent=2))
