"""Approach E: landmark design, negative-control calibration, and the enrollment -> surgery ratio.

1. Landmark at day 30. Approaches A-C show contacted patients have more surgeries in the first 30 days, which
   outreach can't cause: Meridian selected on something outside the data (likely surgeries already in the
   pipeline). Restart the clock at t + 30 among patients still covered and surgery-free, then split follow-up:
       negative-control window  surgery in days 30-90, too soon for a program to change a surgical decision
       main window              surgery in days 90-365
2. Negative-control calibration. If the selection that remains acts as a constant risk ratio over time, the bias
   in the main window equals the "effect" in the control window, so per posterior draw:
       calibrated RR = RR_main / RR_control
3. Enrollment -> surgery among patients who enroll because they were contacted (Wald ratio, per draw):
       effect of enrolling = ATT on surgery / ATT on enrollment
   Valid only if contact is as-good-as-random given covariates and acts on surgery only through enrollment.
4. E-value for the main-window RR: how strong an unmeasured confounder would need to be to explain it away.

Run after a_gcomp.py and c_aipw.py:  uv run --group bayes python sandbox/causal/e_landmark.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from data import build, design  # noqa: E402
from models import fit_logit, standardize, summarize  # noqa: E402
from panel import load  # noqa: E402

df = build()
d = load()
surgery = df.patient_token.map(d["first_surgery"])
coverage_end = df.patient_token.map(d["patients"].coverage_end)
t = df.index_date
days = (surgery - t).dt.days

landmark = ~(days < 30) & (coverage_end > t + pd.Timedelta(days=30))
covered_until = np.minimum(coverage_end, t + pd.Timedelta(days=365))
df["surgery_30_90"] = ((days >= 30) & (days < 90) & (surgery <= covered_until)).astype(int)
df["surgery_90_365"] = ((days >= 90) & (days < 365) & (surgery <= covered_until)).astype(int)

X, wave_idx = design(df)
L = landmark.to_numpy()
XL, wL, TL = X[L], wave_idx[L], df.treated.to_numpy()[L].astype(float)
treated_rows = np.flatnonzero(TL == 1)
print(f"landmark population: {L.sum():,} of {len(df):,} (dropped {int((days < 30).sum())} surgeries in the first 30 days)")

results, rr = {}, {}
for o in ["surgery_30_90", "surgery_90_365"]:
    draws = fit_logit(df.loc[L, o].to_numpy(), XL, wL, treated=TL, seed=41 if o == "surgery_30_90" else 42)
    att = standardize(draws, XL, wL, treated_rows)
    rr[o] = att["rr"]
    results[o] = {"att_rd_pts": summarize(att["rd"]), "rr": summarize(att["rr"], scale=1), "baseline_pct": float(att["p0"].mean() * 100),
                  "divergences": int(draws["divergences"]), "tau_rhat": float(draws["tau_rhat"].max())}
    r = results[o]
    print(f"{o:<15} ATT {r['att_rd_pts']['mean']:+.2f} pts [{r['att_rd_pts']['lo89']:+.2f}, {r['att_rd_pts']['hi89']:+.2f}]  "
          f"RR {r['rr']['mean']:.2f} [{r['rr']['lo89']:.2f}, {r['rr']['hi89']:.2f}]  baseline {r['baseline_pct']:.2f}%  div={r['divergences']}", flush=True)

calibrated = rr["surgery_90_365"] / rr["surgery_30_90"]
base = results["surgery_90_365"]["baseline_pct"] / 100
results["calibrated_rr_90_365"] = summarize(calibrated, scale=1)
results["calibrated_rd_90_365_pts"] = summarize(base * (calibrated - 1))
results["p_calibrated_protective"] = float((calibrated < 1).mean())
r = results["calibrated_rr_90_365"]
print(f"NC-calibrated RR (days 90-365) {r['mean']:.2f} [{r['lo89']:.2f}, {r['hi89']:.2f}];  "
      f"risk difference {results['calibrated_rd_90_365_pts']['mean']:+.2f} pts [{results['calibrated_rd_90_365_pts']['lo89']:+.2f}, {results['calibrated_rd_90_365_pts']['hi89']:+.2f}];  "
      f"P(protective) {results['p_calibrated_protective']:.2f}")

rr_main = results["surgery_90_365"]["rr"]["mean"]
results["e_value_main_rr"] = float(rr_main + np.sqrt(rr_main * (rr_main - 1))) if rr_main >= 1 else float(1 / rr_main + np.sqrt(1 / rr_main * (1 / rr_main - 1)))
print(f"E-value for main-window RR {rr_main:.2f}: {results['e_value_main_rr']:.2f}")

# Wald ratio using the doubly robust draws (independent posteriors, paired at random).
C = np.load(HERE / "c_aipw_draws.npz")
rng = np.random.default_rng(5)
ratio = rng.permutation(C["surgery_12m"]) / rng.permutation(C["enroll_180d"])
results["enrollment_effect_on_surgery_pts"] = summarize(ratio)
r = results["enrollment_effect_on_surgery_pts"]
print(f"effect of enrolling on 12-month surgery (Wald, uncalibrated): {r['mean']:+.1f} pts [{r['lo89']:+.1f}, {r['hi89']:+.1f}]")
np.savez(HERE / "e_landmark_draws.npz", rr_30_90=rr["surgery_30_90"], rr_90_365=rr["surgery_90_365"], calibrated_rr=calibrated, calibrated_rd=base * (calibrated - 1), wald=ratio)
(HERE / "e_landmark.json").write_text(json.dumps(results, indent=2))
