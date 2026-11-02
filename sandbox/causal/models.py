"""Bayesian logistic models shared by the approaches.

Priors:
    covariates    beta ~ N(0, 0.25) on standardized covariates (the prior predictive check from bayesian/fit.py)
    wave          alpha_w ~ N(logit(base rate), 1), one intercept per wave
    treatment     tau ~ N(0, 0.5) on log-odds: weakly informative, an odds ratio between ~0.4 and ~2.7 a priori
    heterogeneity tau_k = mu_tau + sigma_tau * z_k, sigma_tau ~ HalfNormal(0.3), non-centered
"""

import numpy as np
import pymc as pm
from scipy.special import expit, logit

DRAWS = dict(draws=1000, tune=1000, chains=4, nuts_sampler="nutpie", progressbar=False)


def fit_logit(
    y: np.ndarray,
    X: np.ndarray,
    wave_idx: np.ndarray,
    treated: np.ndarray | None = None,
    group_idx: np.ndarray | None = None,
    seed: int = 42,
    target_accept: float = 0.8,
) -> dict[str, np.ndarray]:
    """Fit and return flattened posterior draws: alpha (S x W), beta (S x K), tau (S,) or (S x G)."""
    n_waves = int(wave_idx.max()) + 1
    with pm.Model():
        alpha = pm.Normal("alpha", logit(max(y.mean(), 1e-3)), 1.0, shape=n_waves)
        beta = pm.Normal("beta", 0, 0.25, shape=X.shape[1])
        eta = alpha[wave_idx] + pm.math.dot(X, beta)
        if treated is not None and group_idx is None:
            tau = pm.Normal("tau", 0, 0.5)
            eta = eta + tau * treated
        elif treated is not None:
            mu_tau = pm.Normal("mu_tau", 0, 0.5)
            sigma_tau = pm.HalfNormal("sigma_tau", 0.3)
            z = pm.Normal("z", 0, 1, shape=int(group_idx.max()) + 1)
            tau = pm.Deterministic("tau", mu_tau + sigma_tau * z)
            eta = eta + tau[group_idx] * treated
        pm.Bernoulli("y", logit_p=eta, observed=y)
        idata = pm.sample(random_seed=seed, target_accept=target_accept, **DRAWS)
    n_div = int(idata.sample_stats["diverging"].sum().item())
    post = idata.posterior
    out = {k: post[k].values.reshape(post.sizes["chain"] * post.sizes["draw"], *post[k].shape[2:]) for k in post.data_vars if k != "z"}
    out["divergences"] = np.array(n_div)
    # nutpie stores no r_hat on the dict; compute a cheap split-chain check for the treatment term.
    if "tau" in post:
        chains = post["tau"].values.reshape(post.sizes["chain"], post.sizes["draw"], -1)
        within = chains.var(axis=1, ddof=1).mean(axis=0)
        between = post.sizes["draw"] * chains.mean(axis=1).var(axis=0, ddof=1)
        out["tau_rhat"] = np.sqrt(((post.sizes["draw"] - 1) / post.sizes["draw"] * within + between / post.sizes["draw"]) / within)
    return out


def standardize(draws: dict[str, np.ndarray], X: np.ndarray, wave_idx: np.ndarray, rows: np.ndarray, group_idx: np.ndarray | None = None, thin: int = 1000) -> dict[str, np.ndarray]:
    """g-computation: per posterior draw, average P(Y | do(T=1)) and P(Y | do(T=0)) over the chosen rows."""
    rng = np.random.default_rng(0)
    pick = rng.choice(len(draws["tau"]), min(thin, len(draws["tau"])), replace=False)
    Xr, wr = X[rows], wave_idx[rows]
    gr = group_idx[rows] if group_idx is not None else None
    p1, p0 = [], []
    for j in pick:
        eta0 = draws["alpha"][j][wr] + Xr @ draws["beta"][j]
        tau = draws["tau"][j] if gr is None else draws["tau"][j][gr]
        p0.append(expit(eta0).mean())
        p1.append(expit(eta0 + tau).mean())
    p1, p0 = np.array(p1), np.array(p0)
    return {"p1": p1, "p0": p0, "rd": p1 - p0, "rr": p1 / p0}


def summarize(x: np.ndarray, scale: float = 100) -> dict[str, float]:
    lo, hi = np.percentile(x, [5.5, 94.5])
    return {"mean": float(x.mean() * scale), "lo89": float(lo * scale), "hi89": float(hi * scale), "p_below_0": float((x < 0).mean())}
