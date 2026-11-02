"""Approach C: doubly robust (augmented IPW) ATT with a Bayesian bootstrap.

ATT_AIPW = sum_i w_i [ T_i (Y_i - m0_i) - (1 - T_i) e_i / (1 - e_i) (Y_i - m0_i) ] / sum_i w_i T_i
    m0  outcome-model prediction under no contact (posterior draw from approach A)
    e   propensity (posterior draw from propensity.py)
    w   Dirichlet(1, ..., 1) weights: Rubin's Bayesian bootstrap over patients
Consistent if either the outcome model or the propensity model is right, so it arbitrates between A and B.

Run after propensity.py and a_gcomp.py:  uv run python sandbox/causal/c_aipw.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import build, design  # noqa: E402
from models import summarize  # noqa: E402

OUTCOMES = ["enroll_180d", "surgery_12m", "nc_acute_pre", "nc_acute_12m", "nc_surgery_30d"]
N_DRAWS = 1000
rng = np.random.default_rng(23)
df = build()
X, wave_idx = design(df)
T = df.treated.to_numpy().astype(float)
ps = np.load(HERE / "propensity.npz")
outcome_draws = np.load(HERE / "a_gcomp_draws.npz")

results, saved = {}, {}
for o in OUTCOMES:
    Y = df[o].to_numpy().astype(float)
    a, b = outcome_draws[f"{o}__alpha"], outcome_draws[f"{o}__beta"]
    att = np.empty(N_DRAWS)
    for j in range(N_DRAWS):
        k, m = rng.integers(len(a)), rng.integers(len(ps["alpha"]))
        m0 = expit(a[k][wave_idx] + X @ b[k])
        e = expit(ps["alpha"][m][wave_idx] + X @ ps["beta"][m])
        w = rng.gamma(1.0, size=len(Y))
        resid = Y - m0
        att[j] = np.sum(w * (T * resid - (1 - T) * e / (1 - e) * resid)) / np.sum(w * T)
    results[o] = {"att_rd_pts": summarize(att)}
    saved[o] = att
    r = results[o]["att_rd_pts"]
    print(f"{o:<16} ATT {r['mean']:+.2f} pts [{r['lo89']:+.2f}, {r['hi89']:+.2f}]  P(<0)={r['p_below_0']:.2f}", flush=True)
np.savez(HERE / "c_aipw_draws.npz", **saved)
(HERE / "c_aipw.json").write_text(json.dumps(results, indent=2))
