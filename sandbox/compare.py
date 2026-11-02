"""Side-by-side comparison of the two attempts. Run after both fit.py scripts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import CAPACITY

HERE = Path(__file__).resolve().parent
F, B = HERE / "frequentist", HERE / "bayesian"

metrics = pd.DataFrame({"frequentist": json.loads((F / "metrics.json").read_text()), "bayesian": json.loads((B / "metrics.json").read_text())})
keys = ["auc", "brier", "log_loss", "observed_rate", "mean_predicted", "calibration_intercept", "calibration_slope", f"top_{CAPACITY}_capture", f"top_{CAPACITY}_rate", "top_decile_capture", "fit_seconds"]
print("== Validation (2025-07, out-of-time)")
print(metrics.loc[keys].astype(float).round(4).to_string())

fc = pd.read_csv(F / "coefficients.csv").set_index("feature").coef
bc = pd.read_csv(B / "coefficients.csv").set_index("param")["mean"]
bc = bc[bc.index.str.startswith("beta[")].rename(lambda s: s[5:-1])
both = pd.DataFrame({"frequentist_C1": fc, "bayesian_sd0.25": bc}).dropna()
print("\n== Coefficients")
print(f"correlation {both.corr().iloc[0, 1]:.4f}, max |diff| {(both.iloc[:, 0] - both.iloc[:, 1]).abs().max():.3f}")
print(both.assign(diff=both.iloc[:, 1] - both.iloc[:, 0]).sort_values("diff", key=abs, ascending=False).head(5).round(3).to_string())

fp, bp = pd.read_csv(F / "predictions.csv"), pd.read_csv(B / "predictions.csv")
score = fp[fp.split == "score"].merge(bp[bp.split == "score"], on="patient_token", suffixes=("_f", "_b"))
top_f = set(score.nlargest(CAPACITY, "p_f").patient_token)
top_b = set(score.nlargest(CAPACITY, "p_b").patient_token)
print(f"\n== 2026-07 scoring population ({len(score):,} patients)")
print(f"Spearman rank correlation of risk scores: {score.p_f.rank().corr(score.p_b.rank()):.4f}")
print(f"top-{CAPACITY} lists overlap: {len(top_f & top_b):,} of {CAPACITY:,} ({len(top_f & top_b) / CAPACITY:.1%})")
print(f"mean risk in top {CAPACITY}: frequentist {score.nlargest(CAPACITY, 'p_f').p_f.mean():.2%}, bayesian {score.nlargest(CAPACITY, 'p_b').p_b.mean():.2%}")
cut = score.p_b.nlargest(CAPACITY).iloc[-1]
straddle = ((score.p_low < cut) & (score.p_high > cut)).sum()
print(f"Bayesian: {straddle:,} patients have an 89% interval that straddles the top-{CAPACITY} cut-off ({cut:.2%})")
print(f"Bayesian: P(on list) > 0.9 for {(score.p_in_top > 0.9).sum():,}, 0.1-0.9 for {score.p_in_top.between(0.1, 0.9).sum():,}")
