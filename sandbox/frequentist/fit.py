# %% [markdown]
# # Attempt 1: frequentist logistic regression
#
# An L2-penalised logistic regression on standardized features. `C = 1` puts an implied N(0, 1) prior on
# each coefficient: sklearn minimises C·Σlogloss + ½‖β‖², which is the negative log-posterior under
# β ~ N(0, C). There's no tuning, as the brief asks. The model is fit on 2024-07..2025-04 and evaluated
# out-of-time on 2025-07.

# %%
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from evaluate import CAPACITY, calibration_table, metrics  # noqa: E402
from panel import FEATURES, build_panel, design  # noqa: E402

panel = build_panel()
splits = design(panel)
X_train, y_train, _ = splits["train"]
X_valid, y_valid, valid_ids = splits["valid"]
X_score, _, score_ids = splits["score"]

# %%
start = time.perf_counter()
model = LogisticRegression(C=1.0, max_iter=2000).fit(X_train, y_train)
fit_seconds = time.perf_counter() - start

p_valid = model.predict_proba(X_valid)[:, 1]
p_score = model.predict_proba(X_score)[:, 1]
m = metrics(y_valid, p_valid) | {"fit_seconds": fit_seconds}
print(json.dumps(m, indent=2))
print(calibration_table(y_valid, p_valid).round(4))

# %% [markdown]
# ## Coefficients and uncertainty
#
# sklearn gives no standard errors. A patient-clustered bootstrap gives them: patients repeat across
# index dates, so rows aren't independent. 200 refits take a few seconds.

# %%
coef = pd.DataFrame({"feature": FEATURES, "coef": model.coef_[0]})
train_ids = splits["train"][2].patient_token.to_numpy()
patients = np.unique(train_ids)
rows_by_patient = pd.Series(np.arange(len(train_ids))).groupby(train_ids).apply(np.array)
rng = np.random.default_rng(42)
boot, boot_intercept = [], []
for _ in range(200):
    idx = np.concatenate(rows_by_patient.loc[rng.choice(patients, len(patients))].to_numpy())
    fit = LogisticRegression(C=1.0, max_iter=2000).fit(X_train[idx], y_train[idx])
    boot.append(fit.coef_[0])
    boot_intercept.append(fit.intercept_[0])
boot, boot_intercept = np.array(boot), np.array(boot_intercept)
# Saved so per-patient confidence intervals can be computed later: the frequentist model does have them.
np.savez(HERE / "bootstrap.npz", coef=boot, intercept=boot_intercept)
coef["se_bootstrap"] = boot.std(axis=0)
coef["ci_low"], coef["ci_high"] = np.percentile(boot, [5.5, 94.5], axis=0)  # 89% to match ArviZ defaults
coef["odds_ratio_per_sd"] = np.exp(coef.coef)
coef = coef.sort_values("coef", key=abs, ascending=False)
print(coef.round(3).to_string(index=False))

# %%
coef.to_csv(HERE / "coefficients.csv", index=False)
pd.concat(
    [
        valid_ids.assign(split="valid", y=y_valid, p=p_valid),
        score_ids.assign(split="score", y=np.nan, p=p_score),
    ]
).to_csv(HERE / "predictions.csv", index=False)
(HERE / "metrics.json").write_text(json.dumps(m | {"intercept": float(model.intercept_[0])}, indent=2))
print(f"top {CAPACITY} of the 2026-07 population: mean predicted risk {np.sort(p_score)[-CAPACITY:].mean():.2%}")
