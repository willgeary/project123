# %% [markdown]
# # Attempt 2: Bayesian logistic regression (PyMC + nutpie)
#
# Same features, same standardization, same train/validation split as attempt 1. Steps:
# 1. **Prior predictive check.** Is the N(0, 1) prior implied by sklearn's C = 1 plausible on the probability scale?
# 2. **MAP check.** With an N(0, 1) prior, the posterior mode should reproduce sklearn's coefficients.
# 3. **NUTS fit** with the prior chosen in step 1, then diagnostics and a posterior predictive check.
# 4. **What only the posterior gives:** per-patient risk intervals, and P(patient makes the top-2,500 list).

# %%
import json
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import CAPACITY, calibration_table, metrics  # noqa: E402
from panel import FEATURES, build_panel, design  # noqa: E402

panel = build_panel()
splits = design(panel)
X_train, y_train, _ = splits["train"]
X_valid, y_valid, valid_ids = splits["valid"]
X_score, _, score_ids = splits["score"]
BASE_RATE = y_train.mean()
rng = np.random.default_rng(42)


def build_model(beta_sd: float) -> pm.Model:
    with pm.Model(coords={"feature": FEATURES}) as model:
        X = pm.Data("X", X_train)
        alpha = pm.Normal("alpha", logit(BASE_RATE), 1.0)
        beta = pm.Normal("beta", 0, beta_sd, dims="feature")
        pm.Bernoulli("y", logit_p=alpha + pm.math.dot(X, beta), observed=y_train, shape=X.shape[0])
    return model


# %% [markdown]
# ## 1. Prior predictive check
#
# The question: what spread of patient risks does each prior imply *before* seeing outcomes? With k = 36
# standardized features, the prior SD of the linear predictor is about √k × σ ≈ 6σ.

# %%
prior_rows = []
for beta_sd in [1.0, 0.5, 0.25]:
    a = rng.normal(logit(BASE_RATE), 1.0, 200)
    b = rng.normal(0, beta_sd, (200, len(FEATURES)))
    p = expit(a[:, None] + b @ X_train[rng.choice(len(X_train), 5_000, replace=False)].T)
    prior_rows.append(
        {
            "beta_sd": beta_sd,
            "median_patient_risk": np.median(p),
            "share_patients_above_50pct": (p > 0.5).mean(),
            "share_patients_below_0.1pct": (p < 0.001).mean(),
            "p95_of_cohort_mean_risk": np.percentile(p.mean(axis=1), 95),
        }
    )
prior_check = pd.DataFrame(prior_rows)
print(prior_check.round(4).to_string(index=False))

# %% [markdown]
# N(0, 1) implies that a large share of patients have a better-than-even chance of surgery within a year,
# and that cohort-wide risk could plausibly be 20%+. That's implausible for a 3% outcome. N(0, 0.25) keeps
# individual risks in a believable range while still allowing strong effects (±2 SD ≈ odds ratio 1.65 per
# SD of a feature). It's chosen for the NUTS fit; the MAP check below uses N(0, 1) to match sklearn exactly.

# %%
BETA_SD = 0.25

# %% [markdown]
# ## 2. MAP with the sklearn-equivalent prior

# %%
freq = pd.read_csv(HERE.parent / "frequentist" / "coefficients.csv").set_index("feature")
with build_model(beta_sd=1.0):
    map_est = pm.find_MAP(progressbar=False)
map_vs_sklearn = pd.DataFrame({"map_sd1": map_est["beta"], "sklearn": freq.coef.reindex(FEATURES).to_numpy()}, index=FEATURES)
print("max |MAP - sklearn| over coefficients:", np.abs(map_vs_sklearn.map_sd1 - map_vs_sklearn.sklearn).max().round(4))

# %% [markdown]
# ## 3. NUTS fit

# %%
model = build_model(BETA_SD)
start = time.perf_counter()
with model:
    idata = pm.sample(draws=1000, tune=1000, chains=4, nuts_sampler="nutpie", random_seed=42, progressbar=False)
fit_seconds = time.perf_counter() - start
idata.to_zarr(HERE / "trace.zarr", mode="w")  # save before any post-processing

divergences = int(idata.sample_stats["diverging"].sum().item())
summary = az.summary(idata, var_names=["alpha", "beta"])
print(f"sampling took {fit_seconds:.0f}s, divergences: {divergences}")
print(f"max r_hat {summary.r_hat.max():.3f}, min ess_bulk {summary.ess_bulk.min():.0f}, min ess_tail {summary.ess_tail.min():.0f}")
assert divergences == 0 and summary.r_hat.max() < 1.01 and summary.ess_bulk.min() > 400, "convergence check failed"

# %% [markdown]
# ## Posterior predictive check (on the training data)
#
# This is computed with numpy from thinned draws instead of `pm.sample_posterior_predictive`: 4,000 draws ×
# 60k patients is ~240M Bernoulli samples, and 500 draws answer the same question.

# %%
post = idata.posterior
alpha_draws = post["alpha"].values.reshape(-1)
beta_draws = post["beta"].values.reshape(-1, len(FEATURES))
thin = rng.choice(len(alpha_draws), 500, replace=False)
a_t, b_t = alpha_draws[thin], beta_draws[thin]

p_train_draws = expit(a_t[:, None] + b_t @ X_train.T)
simulated_surgeries = rng.binomial(1, p_train_draws).sum(axis=1)
ppc_low, ppc_high = np.percentile(simulated_surgeries, [5.5, 94.5])
print(f"observed train surgeries {y_train.sum()}, posterior predictive 89% interval [{ppc_low:.0f}, {ppc_high:.0f}]")
p_train_mean = p_train_draws.mean(axis=0)
print(calibration_table(y_train, p_train_mean).round(4))

# %% [markdown]
# ## Validation (2025-07, out-of-time) and the scoring population (2026-07)

# %%
p_valid_draws = expit(a_t[:, None] + b_t @ X_valid.T)
p_score_draws = expit(a_t[:, None] + b_t @ X_score.T)
p_valid = p_valid_draws.mean(axis=0)
p_score = p_score_draws.mean(axis=0)
m = metrics(y_valid, p_valid) | {"fit_seconds": fit_seconds, "beta_sd": BETA_SD, "divergences": divergences}
print(json.dumps(m, indent=2))
print(calibration_table(y_valid, p_valid).round(4))

# %% [markdown]
# ## 4. What only the posterior gives
#
# - **Per-patient 89% interval** for 12-month risk.
# - **P(top-2,500).** For each posterior draw, rank the 2026-07 population and record who makes the top
#   2,500 (Meridian's contact capacity). Averaging over draws gives each patient's probability of being on
#   the list. Patients near the cut-off are the ones whose inclusion is a coin flip.

# %%
lo, hi = np.percentile(p_score_draws, [5.5, 94.5], axis=0)
in_top = np.zeros(len(p_score))
for draw in p_score_draws:
    in_top[np.argpartition(-draw, CAPACITY)[:CAPACITY]] += 1
in_top /= len(p_score_draws)
print(f"score population: median 89% interval width {np.median(hi - lo):.2%} (relative {np.median((hi - lo) / p_score):.0%} of the risk)")
print(f"P(top {CAPACITY}) > 0.9: {(in_top > 0.9).sum()}, between 0.1 and 0.9: {((in_top >= 0.1) & (in_top <= 0.9)).sum()}, < 0.1: {(in_top < 0.1).sum()}")

# %%
coef = summary.reset_index(names="param")
coef.to_csv(HERE / "coefficients.csv", index=False)
prior_check.to_csv(HERE / "prior_predictive.csv", index=False)
map_vs_sklearn.to_csv(HERE / "map_vs_sklearn.csv")
pd.concat(
    [
        valid_ids.assign(split="valid", y=y_valid, p=p_valid, p_low=np.percentile(p_valid_draws, 5.5, axis=0), p_high=np.percentile(p_valid_draws, 94.5, axis=0)),
        score_ids.assign(split="score", y=np.nan, p=p_score, p_low=lo, p_high=hi, p_in_top=in_top),
    ]
).to_csv(HERE / "predictions.csv", index=False)
(HERE / "metrics.json").write_text(
    json.dumps(m | {"ppc_observed": int(y_train.sum()), "ppc_interval": [float(ppc_low), float(ppc_high)]}, indent=2)
)
