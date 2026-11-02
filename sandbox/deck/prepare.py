"""Compute everything the comparison deck shows, write deck_data.json for build.js.

Run after both fit.py scripts:  uv run --group bayes python sandbox/deck/prepare.py
"""

import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent
sys.path.insert(0, str(SANDBOX))
from evaluate import CAPACITY, calibration_table  # noqa: E402
from panel import FEATURES, build_panel, design  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

rng = np.random.default_rng(7)
panel = build_panel()
X_valid, y_valid, valid_ids = design(panel)["valid"]
raw_valid = panel[panel.split == "valid"].reset_index(drop=True)

# Parameter draws: 200 patient-clustered bootstrap refits vs 4,000 posterior draws.
boot = np.load(SANDBOX / "frequentist" / "bootstrap.npz")
post = az.from_zarr(SANDBOX / "bayesian" / "trace.zarr").posterior
a_b, b_b = post["alpha"].values.reshape(-1), post["beta"].values.reshape(-1, X_valid.shape[1])
f_draws = expit(boot["intercept"][:, None] + boot["coef"] @ X_valid.T)  # 200 x n
b_draws = expit(a_b[:, None] + b_b @ X_valid.T)  # 4000 x n

fp = pd.read_csv(SANDBOX / "frequentist" / "predictions.csv").query("split == 'valid'").reset_index(drop=True)
bp = pd.read_csv(SANDBOX / "bayesian" / "predictions.csv").query("split == 'valid'").reset_index(drop=True)
assert (fp.patient_token == valid_ids.patient_token).all() and (bp.patient_token == valid_ids.patient_token).all()
p_f, p_b = fp.p.to_numpy(), bp.p.to_numpy()

metrics = {name: json.loads((SANDBOX / name / "metrics.json").read_text()) for name in ["frequentist", "bayesian"]}

calibration = {
    name: calibration_table(y_valid, p).reset_index().to_dict("records") for name, p in [("frequentist", p_f), ("bayesian", p_b)]
}


def gains(p: np.ndarray) -> list[float]:
    order = np.argsort(-p)
    captured = np.cumsum(y_valid[order]) / y_valid.sum()
    return [float(captured[max(int(len(p) * s / 100) - 1, 0)]) for s in range(5, 101, 5)]


gain_curves = {"share_contacted": list(range(5, 101, 5)), "frequentist": gains(p_f), "bayesian": gains(p_b)}

# P(in top-CAPACITY) on the validation population, per posterior draw ranking.
thin = rng.choice(len(b_draws), 1000, replace=False)
in_top = np.zeros(len(p_b))
for draw in b_draws[thin]:
    in_top[np.argpartition(-draw, CAPACITY)[:CAPACITY]] += 1
in_top /= len(thin)
cutoff = float(np.sort(p_b)[-CAPACITY])

# Example patients: one per situation a stakeholder would ask about.
rank_b = pd.Series(p_b).rank(ascending=False).to_numpy()
picks = {
    "Top of the list, had surgery": int(np.argmax(np.where(y_valid == 1, p_b, -1))),
    "High risk, no surgery": int(np.argmax(np.where((y_valid == 0) & (rank_b > 20), p_b, -1))),
    "On the cut-off line": int(np.argmin(np.where(y_valid == 0, np.abs(p_b - cutoff), np.inf))),
    "Typical patient": int(np.argmin(np.where(y_valid == 0, np.abs(p_b - np.median(p_b)), np.inf))),
    "Missed: low risk, had surgery": int(np.argmin(np.where(y_valid == 1, p_b, np.inf))),
}
bins = np.linspace(0, 1, 41)  # rescaled per patient below


def bmi_band(r: pd.Series) -> str:
    if r.no_bmi:
        return "not recorded"
    return "35+" if r.bmi_35_plus else "30-34" if r.bmi_30_34 else "25-29" if r.bmi_25_29 else "<25"


examples = []
for label, i in picks.items():
    r = raw_valid.iloc[i]
    fd, bd = f_draws[:, i], b_draws[:, i]
    hi = max(np.percentile(bd, 99.5), np.percentile(fd, 99.5)) * 1.05
    lo = min(np.percentile(bd, 0.5), np.percentile(fd, 0.5)) * 0.95
    edges = np.linspace(lo, hi, 25)
    examples.append(
        {
            "label": label,
            "token": r.patient_token[:8],
            "age": int(r.age),
            "sex": "M" if r.male else "U" if r.sex_unknown else "F",
            "bmi": bmi_band(r),
            "prior_consult": bool(r.prior_consult),
            "imaging_180d": bool(r.imaging_180d),
            "injection_365d": bool(r.injection_365d),
            "lbp_visits_365d": int(round(np.expm1(r.log_lbp_365d))),
            "months_since_last_lbp": round(float(r.days_since_last_lbp) * 12, 1),
            "outcome": "Surgery" if y_valid[i] else "No surgery",
            "rank": int(rank_b[i]),
            "freq_point": float(p_f[i]),
            "freq_ci": [float(np.percentile(fd, 5.5)), float(np.percentile(fd, 94.5))],
            "bayes_mean": float(p_b[i]),
            "bayes_ci": [float(np.percentile(bd, 5.5)), float(np.percentile(bd, 94.5))],
            "p_in_top": float(in_top[i]),
            "hist_edges": edges.tolist(),
            "freq_hist": (np.histogram(fd, edges)[0] / len(fd)).tolist(),
            "bayes_hist": (np.histogram(bd, edges)[0] / len(bd)).tolist(),
        }
    )

# Posterior predictive: surgeries among the top-CAPACITY list, and in the whole validation population.
top = np.argsort(-p_b)[:CAPACITY]
sim_top_b = rng.binomial(1, b_draws[thin][:, top]).sum(axis=1)
sim_top_f = rng.binomial(1, f_draws[:, top]).sum(axis=1)
sim_all_b = rng.binomial(1, b_draws[thin]).sum(axis=1)
sim_all_f = rng.binomial(1, f_draws).sum(axis=1)


def hist(values: np.ndarray, edges: np.ndarray) -> list[float]:
    return (np.histogram(values, edges)[0] / len(values)).tolist()


edges_top = np.arange(130, 238, 6)
edges_all = np.arange(340, 550, 10)
ppc = {
    "top": {
        "edges": edges_top.tolist(),
        "bayes": hist(sim_top_b, edges_top),
        "freq": hist(sim_top_f, edges_top),
        "observed": int(y_valid[top].sum()),
        "bayes_interval": np.percentile(sim_top_b, [5.5, 94.5]).tolist(),
        "freq_interval": np.percentile(sim_top_f, [5.5, 94.5]).tolist(),
        "bayes_share_at_or_below": float((sim_top_b <= y_valid[top if "top" == "top" else slice(None)].sum()).mean()),
        "freq_share_at_or_below": float((sim_top_f <= y_valid[top if "top" == "top" else slice(None)].sum()).mean()),
    },
    "all": {
        "edges": edges_all.tolist(),
        "bayes": hist(sim_all_b, edges_all),
        "freq": hist(sim_all_f, edges_all),
        "observed": int(y_valid.sum()),
        "bayes_interval": np.percentile(sim_all_b, [5.5, 94.5]).tolist(),
        "freq_interval": np.percentile(sim_all_f, [5.5, 94.5]).tolist(),
        "bayes_share_at_or_below": float((sim_all_b <= y_valid[top if "all" == "top" else slice(None)].sum()).mean()),
        "freq_share_at_or_below": float((sim_all_f <= y_valid[top if "all" == "top" else slice(None)].sum()).mean()),
    },
}

interval_width = {
    "freq_median": float(np.median(np.percentile(f_draws, 94.5, axis=0) - np.percentile(f_draws, 5.5, axis=0))),
    "bayes_median": float(np.median(np.percentile(b_draws, 94.5, axis=0) - np.percentile(b_draws, 5.5, axis=0))),
}
# Before/after the tenure feature: predicted / observed by index date, frequentist, refit both ways.
labeled = panel[panel.split != "score"].reset_index(drop=True)
train_mask = (labeled.split == "train").to_numpy()
drift = {}
for name, feats in [("without_tenure", [f for f in FEATURES if not f.startswith("tenure_")]), ("with_tenure", FEATURES)]:
    tr = labeled[train_mask]
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
    model = LogisticRegression(C=1.0, max_iter=3000).fit(((tr[feats] - mu) / sd).to_numpy(), tr.y.astype(int))
    p = model.predict_proba(((labeled[feats] - mu) / sd).to_numpy())[:, 1]
    g = pd.DataFrame({"d": labeled.index_date.dt.strftime("%Y-%m"), "y": labeled.y, "p": p}).groupby("d")
    drift[name] = (g.p.mean() / g.y.mean()).round(3).to_dict()
    tenure12 = labeled.tenure_12m_plus.to_numpy() == 1
    valid = ~train_mask
    drift[name + "_12m_plus_ratio"] = float(p[valid & tenure12].mean() / labeled.y[valid & tenure12].mean())

coverage_dir = SANDBOX / "interval_coverage"
coverage = {
    "simulation": pd.read_csv(coverage_dir / "simulation_coverage.csv").to_dict("records"),
    "real": pd.read_csv(coverage_dir / "real_group_coverage.csv").to_dict("records"),
}
score_all = pd.read_csv(SANDBOX / "frequentist" / "predictions.csv").query("split == 'score'").merge(
    pd.read_csv(SANDBOX / "bayesian" / "predictions.csv").query("split == 'score'"), on="patient_token", suffixes=("_f", "_b")
)
score_f = score_all.p_f
list_overlap = len(set(score_all.nlargest(CAPACITY, "p_f").patient_token) & set(score_all.nlargest(CAPACITY, "p_b").patient_token)) / CAPACITY

out = {
    "n_features": len(FEATURES),
    "list_overlap": list_overlap,
    "drift": drift,
    "coverage": coverage,
    "score": {"above_breakeven": int((score_f > 0.017).sum()), "mean": float(score_f.mean())},
    "capacity": CAPACITY,
    "n_valid": int(len(y_valid)),
    "cutoff": cutoff,
    "metrics": metrics,
    "calibration": calibration,
    "gains": gain_curves,
    "examples": examples,
    "ppc": ppc,
    "interval_width": interval_width,
    "list_certainty": {
        "sure_in": int((in_top > 0.9).sum()),
        "uncertain": int(((in_top >= 0.1) & (in_top <= 0.9)).sum()),
    },
}
(HERE / "deck_data.json").write_text(json.dumps(out, indent=1))
for e in examples:
    print(
        f"{e['label']:<32} rank {e['rank']:>6}  freq {e['freq_point']:.2%} [{e['freq_ci'][0]:.2%}, {e['freq_ci'][1]:.2%}]"
        f"  bayes {e['bayes_mean']:.2%} [{e['bayes_ci'][0]:.2%}, {e['bayes_ci'][1]:.2%}]  P(top)={e['p_in_top']:.2f}  {e['outcome']}"
    )
print("ppc top:", ppc["top"]["observed"], ppc["top"]["bayes_interval"], ppc["top"]["freq_interval"])
print("ppc all:", ppc["all"]["observed"], ppc["all"]["bayes_interval"], ppc["all"]["freq_interval"])
print("interval widths:", interval_width, "cutoff", round(cutoff, 4), out["list_certainty"])
