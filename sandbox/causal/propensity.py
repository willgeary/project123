"""Who got contacted? Bayesian propensity model, overlap, and covariate balance.

Writes propensity.npz (posterior draws of the propensity coefficients) for the subclassification and doubly
robust approaches. Run: uv run --group bayes python sandbox/causal/propensity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import COVARIATES, build, design  # noqa: E402
from models import fit_logit  # noqa: E402

df = build()
X, wave_idx = design(df)
T = df.treated.to_numpy()

draws = fit_logit(T, X, wave_idx, seed=1)
np.savez(HERE / "propensity.npz", alpha=draws["alpha"], beta=draws["beta"])
e = expit(draws["alpha"].mean(axis=0)[wave_idx] + X @ draws["beta"].mean(axis=0))
df["e"] = e
print(f"divergences {int(draws['divergences'])}; propensity AUC {roc_auc_score(T, e):.3f} (0.5 = contact unrelated to covariates)")
print("propensity range, treated:", np.percentile(e[T == 1], [1, 50, 99]).round(3), " untreated:", np.percentile(e[T == 0], [1, 50, 99]).round(3))

# Balance: standardized mean differences, raw vs ATT-weighted (controls weighted by e / (1 - e)).
w = np.where(T == 1, 1.0, e / (1 - e))
raw = df[COVARIATES].astype(float)


def smd(weights: np.ndarray) -> pd.Series:
    m1 = np.average(raw[T == 1], axis=0, weights=weights[T == 1])
    m0 = np.average(raw[T == 0], axis=0, weights=weights[T == 0])
    sd = np.sqrt((raw[T == 1].var() + raw[T == 0].var()) / 2).replace(0, 1)
    return pd.Series((m1 - m0) / sd.to_numpy(), index=COVARIATES)


balance = pd.DataFrame({"smd_raw": smd(np.ones(len(df))), "smd_weighted": smd(w)})
balance["abs_raw"] = balance.smd_raw.abs()
print("\nlargest raw imbalances (|SMD| > 0.1 is the usual flag):")
print(balance.sort_values("abs_raw", ascending=False).head(8)[["smd_raw", "smd_weighted"]].round(3).to_string())
print(f"max |SMD| raw {balance.smd_raw.abs().max():.3f} -> weighted {balance.smd_weighted.abs().max():.3f}")
ess_control = w[T == 0].sum() ** 2 / (w[T == 0] ** 2).sum()
print(f"effective control sample size after weighting: {ess_control:,.0f} of {(T == 0).sum():,}")

balance.drop(columns="abs_raw").to_csv(HERE / "balance.csv")
(HERE / "propensity_summary.json").write_text(
    json.dumps(
        {
            "auc": float(roc_auc_score(T, e)),
            "max_abs_smd_raw": float(balance.smd_raw.abs().max()),
            "max_abs_smd_weighted": float(balance.smd_weighted.abs().max()),
            "ess_control": float(ess_control),
            "n_treated": int(T.sum()),
            "n_control": int((T == 0).sum()),
        },
        indent=2,
    )
)
