"""Shared evaluation so both attempts are scored on exactly the same terms."""

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

CAPACITY = 2_500  # patients Meridian can contact per quarter (2-3k from past waves)


def calibration_line(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Intercept and slope of y ~ logit(p). Perfect calibration is (0, 1)."""
    fit = LogisticRegression(C=np.inf).fit(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1), y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    order = np.argsort(-p)
    top_k = order[: min(CAPACITY, len(p))]
    top_decile = order[: len(p) // 10]
    intercept, slope = calibration_line(y, p)
    return {
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "observed_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        f"top_{CAPACITY}_capture": float(y[top_k].sum() / y.sum()),
        f"top_{CAPACITY}_rate": float(y[top_k].mean()),
        "top_decile_capture": float(y[top_decile].sum() / y.sum()),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p})
    df["decile"] = pd.qcut(df.p.rank(method="first"), bins, labels=False) + 1
    return df.groupby("decile").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = rng.uniform(0.001, 0.2, 20_000)
    y = rng.binomial(1, p)
    m = metrics(y, p)
    assert abs(m["calibration_slope"] - 1) < 0.15 and abs(m["calibration_intercept"]) < 0.3, m
    assert m["auc"] > 0.6 and m["top_decile_capture"] > 0.1, m
    m_bad = metrics(y, p * 2)
    assert m_bad["mean_predicted"] > m["mean_predicted"] and m_bad["calibration_intercept"] < m["calibration_intercept"]
    print("evaluate self-check passed")
