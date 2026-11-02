"""12-month surgery risk: L2 logistic regression on standardized features."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from assignment.features import FEATURES


def train(panel: pd.DataFrame) -> dict:
    """Fit on labeled snapshots; standardize with the training means/SDs."""
    X = panel[FEATURES].to_numpy()
    mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
    sd[sd == 0] = 1
    # Plain arrays, not pandas objects: the pickle then loads in any environment (no pyarrow dependency).
    return {
        "model": LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, panel.y.to_numpy()),
        "mu": mu,
        "sd": sd,
    }


def predict(artifact: dict, df: pd.DataFrame) -> np.ndarray:
    """P(surgery while covered within 12 months)."""
    return artifact["model"].predict_proba((df[FEATURES].to_numpy() - artifact["mu"]) / artifact["sd"])[:, 1]
