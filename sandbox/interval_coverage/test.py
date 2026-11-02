"""Do the intervals cover what they claim to (89%)? Bayesian posterior vs frequentist bootstrap.

Test 1, simulation, where the true risk is known. Treat the fitted frequentist model as the truth, simulate
    surgeries for the real training patients, refit every method, and check how often each patient's 89%
    interval contains their true risk on the 2025-07 population. This tests the risk intervals directly; real
    outcomes can't, because a patient's true risk is never observed.
Test 2, real 2025-07 outcomes. A single patient's outcome can't check an interval, but a group's surgery count can.
    For groups of patients, build each method's 89% predictive interval for the count (parameter uncertainty +
    Bernoulli noise) and check how often the observed count falls inside. A plug-in binomial that ignores
    parameter uncertainty is the baseline.

Run after both fit.py scripts:  uv run --group bayes python sandbox/interval_coverage/test.py
"""

import json
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent
sys.path.insert(0, str(SANDBOX))
from panel import FEATURES, build_panel, design  # noqa: E402

LEVEL, LO, HI = 0.89, 5.5, 94.5
REPLICATES, BOOT = 10, 100
rng = np.random.default_rng(2026)
splits = design(build_panel())
X_train, y_train, train_ids = splits["train"]
X_valid, y_valid, _ = splits["valid"]
patients = train_ids.patient_token.to_numpy()
rows_by_patient = pd.Series(np.arange(len(patients))).groupby(patients).apply(np.array)
unique_patients = rows_by_patient.index.to_numpy()


def bootstrap_draws(y: np.ndarray, n_boot: int = BOOT) -> np.ndarray:
    """Patient-clustered bootstrap, as in frequentist/fit.py. Returns risk draws on X_valid (n_boot x n)."""
    out = []
    for _ in range(n_boot):
        idx = np.concatenate(rows_by_patient.loc[rng.choice(unique_patients, len(unique_patients))].to_numpy())
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X_train[idx], y[idx])
        out.append(m.predict_proba(X_valid)[:, 1])
    return np.array(out)


def bayes_draws(y: np.ndarray, beta_sd: float, n_keep: int = 1000) -> np.ndarray:
    with pm.Model(coords={"feature": FEATURES}):
        alpha = pm.Normal("alpha", logit(y.mean()), 1.0)
        beta = pm.Normal("beta", 0, beta_sd, dims="feature")
        pm.Bernoulli("y", logit_p=alpha + pm.math.dot(X_train, beta), observed=y)
        idata = pm.sample(draws=500, tune=500, chains=4, nuts_sampler="nutpie", random_seed=int(rng.integers(1e6)), progressbar=False)
    a = idata.posterior["alpha"].values.reshape(-1)
    b = idata.posterior["beta"].values.reshape(-1, len(FEATURES))
    keep = rng.choice(len(a), min(n_keep, len(a)), replace=False)
    return expit(a[keep, None] + b[keep] @ X_valid.T)


def coverage(draws: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    lo, hi = np.percentile(draws, [LO, HI], axis=0)
    inside = (truth >= lo) & (truth <= hi)
    return {"coverage": float(inside.mean()), "median_width": float(np.median(hi - lo)), "median_relative_width": float(np.median((hi - lo) / truth))}


# %% Test 1: simulation with a known true risk
freq = pd.read_csv(SANDBOX / "frequentist" / "coefficients.csv").set_index("feature").coef.reindex(FEATURES).to_numpy()
true_intercept = json.loads((SANDBOX / "frequentist" / "metrics.json").read_text())["intercept"]
p_true_train = expit(true_intercept + X_train @ freq)
p_true_valid = expit(true_intercept + X_valid @ freq)

methods = {"frequentist bootstrap": lambda y: bootstrap_draws(y), "bayesian N(0, 0.25)": lambda y: bayes_draws(y, 0.25), "bayesian N(0, 1)": lambda y: bayes_draws(y, 1.0)}
sim_rows = []
for r in range(REPLICATES):
    y_sim = rng.binomial(1, p_true_train)
    for name, fn in methods.items():
        start = time.perf_counter()
        res = coverage(fn(y_sim), p_true_valid)
        sim_rows.append({"replicate": r, "method": name, **res, "seconds": time.perf_counter() - start})
    print(f"replicate {r + 1}/{REPLICATES} done", flush=True)
sim = pd.DataFrame(sim_rows)
sim_summary = sim.groupby("method").agg(
    coverage=("coverage", "mean"), coverage_min=("coverage", "min"), coverage_max=("coverage", "max"),
    median_width=("median_width", "mean"), relative_width=("median_relative_width", "mean"), seconds=("seconds", "mean"),
)
print(f"\nTest 1: share of patients whose {LEVEL:.0%} interval contains their true risk (target {LEVEL:.0%}), {REPLICATES} replicates")
print(sim_summary.round(3).to_string())

# %% Test 2: real 2025-07 outcomes, group-level predictive intervals
boot = np.load(SANDBOX / "frequentist" / "bootstrap.npz")
post = az.from_zarr(SANDBOX / "bayesian" / "trace.zarr").posterior
f_draws = expit(boot["intercept"][:, None] + boot["coef"] @ X_valid.T)
a_b, b_b = post["alpha"].values.reshape(-1), post["beta"].values.reshape(-1, len(FEATURES))
keep = rng.choice(len(a_b), 1000, replace=False)
b_draws = expit(a_b[keep, None] + b_b[keep] @ X_valid.T)
p_point = pd.read_csv(SANDBOX / "frequentist" / "predictions.csv").query("split == 'valid'").p.to_numpy()


def count_interval(draws: np.ndarray, members: np.ndarray, n_sims: int = 4000) -> tuple[float, float]:
    """Predictive interval for the group's surgery count: pick a parameter draw, then simulate outcomes."""
    picks = rng.integers(0, len(draws), n_sims)
    counts = rng.binomial(1, draws[picks][:, members]).sum(axis=1)
    return tuple(np.percentile(counts, [LO, HI]))


real_rows = []
order = np.argsort(p_point)
for scheme, size in [("risk-sorted", 200), ("risk-sorted", 1000), ("random", 200), ("random", 1000)]:
    idx = order if scheme == "risk-sorted" else rng.permutation(len(p_point))
    groups = [idx[i : i + size] for i in range(0, len(idx) - size + 1, size)]
    for name, draws in [("frequentist bootstrap", f_draws), ("bayesian N(0, 0.25)", b_draws), ("plug-in binomial (no parameter uncertainty)", p_point[None, :])]:
        hits = [lo <= y_valid[g].sum() <= hi for g in groups for lo, hi in [count_interval(draws, g)]]
        real_rows.append({"groups": f"{scheme}, size {size}", "n_groups": len(groups), "method": name, "coverage": float(np.mean(hits))})
real = pd.DataFrame(real_rows).pivot(index="groups", columns="method", values="coverage")
print(f"\nTest 2: share of 2025-07 patient groups whose observed surgery count falls in the {LEVEL:.0%} predictive interval")
print(real.round(3).to_string())

sim_summary.to_csv(HERE / "simulation_coverage.csv")
sim.to_csv(HERE / "simulation_replicates.csv", index=False)
real.to_csv(HERE / "real_group_coverage.csv")
