"""Approach B: propensity subclassification with Beta-Binomial strata (the "compare similar patients" approach).

Within each wave, patients are cut into 10 strata of similar propensity to be contacted. Inside a stratum,
contacted and uncontacted patients look alike on all 37 covariates, so each stratum is a small comparison:
    surgery rate | contacted   ~ Beta(1 + events, 1 + non-events)
    surgery rate | uncontacted ~ Beta(1 + events, 1 + non-events)
The ATT is the treated-weighted average of the stratum differences. Each Monte Carlo draw also redraws the
propensity coefficients from their posterior, so strata membership uncertainty is carried through.
No MCMC needed: the Beta posteriors are closed form.

Run after propensity.py:  uv run python sandbox/causal/b_subclass.py
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
from models import summarize  # noqa: E402

OUTCOMES = ["enroll_180d", "surgery_12m", "nc_acute_pre", "nc_acute_12m", "nc_surgery_30d"]
N_STRATA, N_DRAWS = 10, 2000
rng = np.random.default_rng(11)
df = build()
X, wave_idx = design(df)
T = df.treated.to_numpy()
ps = np.load(HERE / "propensity.npz")
Y = {o: df[o].to_numpy() for o in OUTCOMES}

att = {o: np.empty(N_DRAWS) for o in OUTCOMES}
for j in range(N_DRAWS):
    k = rng.integers(len(ps["alpha"]))
    e = expit(ps["alpha"][k][wave_idx] + X @ ps["beta"][k])
    # Strata within wave; rank-based so every stratum has ~equal size.
    stratum = pd.Series(e).groupby(wave_idx).transform(lambda s: pd.qcut(s.rank(method="first"), N_STRATA, labels=False)).to_numpy()
    cell = wave_idx * N_STRATA + stratum
    n1 = np.bincount(cell, weights=T, minlength=3 * N_STRATA)
    n0 = np.bincount(cell, weights=1 - T, minlength=3 * N_STRATA)
    for o in OUTCOMES:
        y1 = np.bincount(cell, weights=T * Y[o], minlength=3 * N_STRATA)
        y0 = np.bincount(cell, weights=(1 - T) * Y[o], minlength=3 * N_STRATA)
        p1 = rng.beta(1 + y1, 1 + n1 - y1)
        p0 = rng.beta(1 + y0, 1 + n0 - y0)
        att[o][j] = np.sum(n1 * (p1 - p0)) / n1.sum()

results = {o: {"att_rd_pts": summarize(att[o])} for o in OUTCOMES}
for o in OUTCOMES:
    r = results[o]["att_rd_pts"]
    print(f"{o:<16} ATT {r['mean']:+.2f} pts [{r['lo89']:+.2f}, {r['hi89']:+.2f}]  P(<0)={r['p_below_0']:.2f}")
np.savez(HERE / "b_subclass_draws.npz", **att)
(HERE / "b_subclass.json").write_text(json.dumps(results, indent=2))
