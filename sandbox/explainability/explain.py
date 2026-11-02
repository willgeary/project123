"""Explain the fitted Bayesian surgery-risk model: posterior feature effects and two-way sensitivity.

Reads the saved posterior (sandbox/bayesian/trace.zarr) and the cached panel; does not refit anything.
Every effect is an association conditional on the other 35 features, not what an intervention would do.

    uv run --group bayes python sandbox/explainability/explain.py
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent
sys.path.insert(0, str(SANDBOX))
from panel import FEATURES, build_panel  # noqa: E402

BREAK_EVEN = 0.017  # risk at which outreach pays for itself (sandbox/README.md)
N_DRAWS = 1_000
INTERVAL = (5.5, 94.5)  # 89% equal-tailed

# ---------------------------------------------------------------- data + posterior

panel = build_panel()
train = panel[panel.split == "train"]
# Must match panel.design(): the posterior lives on this standardized scale.
MU, SD = train[FEATURES].mean(), train[FEATURES].std().replace(0, 1)
POPULATIONS = {"score": panel[panel.split == "score"].reset_index(drop=True), "valid": panel[panel.split == "valid"].reset_index(drop=True)}

post = az.from_zarr(str(SANDBOX / "bayesian" / "trace.zarr")).posterior
assert list(post["beta"].coords["feature"].values) == FEATURES
alpha_all = post["alpha"].values.reshape(-1)
beta_all = post["beta"].values.reshape(-1, len(FEATURES))
keep = np.random.default_rng(42).choice(len(alpha_all), N_DRAWS, replace=False)
ALPHA, BETA = alpha_all[keep], beta_all[keep]
B = pd.DataFrame(BETA, columns=FEATURES)


def risk(raw: pd.DataFrame) -> np.ndarray:
    """Predicted 12-month risk, shape (draws, patients)."""
    X = ((raw[FEATURES] - MU) / SD).to_numpy()
    return expit(ALPHA[:, None] + BETA @ X.T)


Setter = Callable[[pd.DataFrame], pd.DataFrame]


def assign(**values) -> Setter:
    return lambda df: df.assign(**values)


def at_age(age: float) -> Setter:
    # under_65 is a separate feature, so it must move with age or the counterfactual is incoherent.
    return lambda df: df.assign(age=float(age), under_65=float(age < 65))


def age_plus(years: float) -> Setter:
    return lambda df: df.assign(age=df.age + years, under_65=(df.age + years < 65).astype(float))


def observed(df: pd.DataFrame) -> pd.DataFrame:
    return df


def compose(*setters: Setter) -> Setter:
    def apply(df: pd.DataFrame) -> pd.DataFrame:
        for s in setters:
            df = s(df)
        return df

    return apply


BMI_OFF = dict(bmi_25_29=0.0, bmi_30_34=0.0, bmi_35_plus=0.0, no_bmi=0.0)
BMI_LEVELS = {
    "BMI <25": assign(**BMI_OFF),
    "BMI 25–29": assign(**BMI_OFF | {"bmi_25_29": 1.0}),
    "BMI 30–34": assign(**BMI_OFF | {"bmi_30_34": 1.0}),
    "BMI 35+": assign(**BMI_OFF | {"bmi_35_plus": 1.0}),
    "No BMI on file": assign(**BMI_OFF | {"no_bmi": 1.0}),
}
TENURE_LEVELS = {
    "<6 months": assign(tenure_6_12m=0.0, tenure_12m_plus=0.0),
    "6–12 months": assign(tenure_6_12m=1.0, tenure_12m_plus=0.0),
    "12+ months": assign(tenure_6_12m=0.0, tenure_12m_plus=1.0),
}
CONSULT_LEVELS = {"No prior consult": assign(prior_consult=0.0), "Prior consult": assign(prior_consult=1.0)}
AGES = list(range(50, 90, 5))


def flag_or(feature: str) -> np.ndarray:
    return np.exp(B[feature].to_numpy() / SD[feature])


# (label, group, on, off, odds-ratio draws, odds-ratio caption)
EFFECTS = [
    ("BMI 25–29", "BMI band (vs <25)", BMI_LEVELS["BMI 25–29"], BMI_LEVELS["BMI <25"], flag_or("bmi_25_29")),
    ("BMI 30–34", "BMI band (vs <25)", BMI_LEVELS["BMI 30–34"], BMI_LEVELS["BMI <25"], flag_or("bmi_30_34")),
    ("BMI 35+", "BMI band (vs <25)", BMI_LEVELS["BMI 35+"], BMI_LEVELS["BMI <25"], flag_or("bmi_35_plus")),
    ("No BMI on file", "BMI band (vs <25)", BMI_LEVELS["No BMI on file"], BMI_LEVELS["BMI <25"], flag_or("no_bmi")),
    ("Prior surgical consult", "Care history", assign(prior_consult=1.0), assign(prior_consult=0.0), flag_or("prior_consult")),
    ("Imaging in last 180d", "Care history", assign(imaging_180d=1.0), assign(imaging_180d=0.0), flag_or("imaging_180d")),
    ("Injection in last 365d", "Care history", assign(injection_365d=1.0), assign(injection_365d=0.0), flag_or("injection_365d")),
    ("First LBP visit 6–12m ago", "Tenure (vs <6m)", TENURE_LEVELS["6–12 months"], TENURE_LEVELS["<6 months"], flag_or("tenure_6_12m")),
    ("First LBP visit 12m+ ago", "Tenure (vs <6m)", TENURE_LEVELS["12+ months"], TENURE_LEVELS["<6 months"], flag_or("tenure_12m_plus")),
    # Odds ratio is the continuous age term only; the under_65 step (beta ~ -0.007) is negligible.
    ("Age +10 years", "Demographics", age_plus(10), observed, np.exp(10 * B["age"].to_numpy() / SD["age"])),
]


def interval(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(x, INTERVAL)
    return float(lo), float(hi)


def summarize(x: np.ndarray) -> dict:
    lo, hi = interval(x)
    return {"mean": float(x.mean()), "lo89": lo, "hi89": hi, "p_positive": float((x > 0).mean())}


# ---------------------------------------------------------------- 1. single-feature effects

def average_predictive_comparison(pop: pd.DataFrame, on: Setter, off: Setter) -> np.ndarray:
    """Per draw: mean risk with the feature on minus mean risk with it off, over the whole population."""
    return risk(on(pop.copy())).mean(axis=1) - risk(off(pop.copy())).mean(axis=1)


apc = {split: {label: average_predictive_comparison(pop, on, off) for label, _, on, off, _ in EFFECTS} for split, pop in POPULATIONS.items()}
effects = pd.DataFrame(
    [
        {"feature": label, "group": group}
        | {f"or_{k}": v for k, v in summarize(or_draws).items() if k != "p_positive"}
        | {f"pp_{split}_{k}": v * (100 if k != "p_positive" else 1) for split in POPULATIONS for k, v in summarize(apc[split][label]).items()}
        for label, group, _, _, or_draws in EFFECTS
    ]
)
effects.to_csv(HERE / "effects.csv", index=False)
print(effects.round(3).to_string(index=False))

# ---------------------------------------------------------------- 2. two-feature sensitivity (2026-07 population)

pop = POPULATIONS["score"]


def grid(rows: dict[str, Setter], cols: dict[str, Setter]) -> tuple[pd.DataFrame, dict]:
    cells, draws = [], {}
    for r, set_r in rows.items():
        for c, set_c in cols.items():
            p = risk(compose(set_r, set_c)(pop.copy()))
            avg = p.mean(axis=1)
            lo, hi = interval(avg)
            draws[r, c] = avg
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "mean_risk": float(avg.mean()),
                    "lo89": lo,
                    "hi89": hi,
                    "width89": hi - lo,
                    "p_mean_above_break_even": float((avg > BREAK_EVEN).mean()),
                    "share_patients_above_break_even": float((p.mean(axis=0) > BREAK_EVEN).mean()),
                }
            )
    return pd.DataFrame(cells), draws


bmi_age, bmi_age_draws = grid(BMI_LEVELS, {str(a): at_age(a) for a in AGES})
consult_tenure, consult_tenure_draws = grid(CONSULT_LEVELS, TENURE_LEVELS)
bmi_age.to_csv(HERE / "sensitivity_bmi_age.csv", index=False)
consult_tenure.to_csv(HERE / "sensitivity_consult_tenure.csv", index=False)

# Effect of one feature at each level of the other, in probability points; on the logit scale these are constant.
bmi_effect_by_age = {
    band: {a: bmi_age_draws[band, str(a)] - bmi_age_draws["BMI <25", str(a)] for a in AGES} for band in list(BMI_LEVELS)[1:]
}
consult_effect_by_tenure = {t: consult_tenure_draws["Prior consult", t] - consult_tenure_draws["No prior consult", t] for t in TENURE_LEVELS}
contrasts = {
    "bmi_30_34_effect_age55_minus_age80_pp": summarize(100 * (bmi_effect_by_age["BMI 30–34"][55] - bmi_effect_by_age["BMI 30–34"][80])),
    "consult_effect_lt6m_minus_12mplus_pp": summarize(100 * (consult_effect_by_tenure["<6 months"] - consult_effect_by_tenure["12+ months"])),
    "bmi_effect_by_age_pp": {band: {a: summarize(100 * d) for a, d in by_age.items()} for band, by_age in bmi_effect_by_age.items()},
    "consult_effect_by_tenure_pp": {t: summarize(100 * d) for t, d in consult_effect_by_tenure.items()},
    "observed_mean_risk_score": float(risk(pop).mean()),
}
(HERE / "interactions.json").write_text(json.dumps(contrasts, indent=2))
print(consult_tenure.round(4).to_string(index=False))
print(json.dumps({k: v for k, v in contrasts.items() if "by" not in k}, indent=2))

# ---------------------------------------------------------------- plots

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
CAPTION = "Associations conditional on the other model features, not causal effects. 2026-07 scoring population (n = 16,010), 1,000 posterior draws."

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 160,
    }
)


def ridgeline(draws: dict[str, np.ndarray], groups: dict[str, str], xlabel: str, fmt: Callable[[float], str], ref: float, log: bool, path: Path, title: str, caption: str):
    labels = list(draws)
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(labels) + 1.9))
    values = np.concatenate(list(draws.values()))
    lo_all, hi_all = values.min(), values.max()
    lo_all, hi_all = min(lo_all, ref), max(hi_all, ref)
    pad = (hi_all - lo_all) * 0.04 if not log else 0
    for i, label in enumerate(labels):
        x = np.log(draws[label]) if log else draws[label]
        y0 = len(labels) - 1 - i
        kde = gaussian_kde(x)
        xs = np.linspace(x.min(), x.max(), 400)
        dens = kde(xs)
        dens = 0.82 * dens / dens.max()
        xs_plot = np.exp(xs) if log else xs
        lo, hi = interval(x)
        inside = (xs >= lo) & (xs <= hi)
        ax.fill_between(xs_plot, y0, y0 + dens, color=SERIES[0], alpha=0.10, lw=0)
        ax.fill_between(xs_plot[inside], y0, y0 + dens[inside], color=SERIES[0], alpha=0.35, lw=0)
        ax.plot(xs_plot, y0 + dens, color=SERIES[0], lw=2, solid_capstyle="round")
        mean = x.mean()
        ax.plot([np.exp(mean) if log else mean] * 2, [y0, y0 + kde(mean)[0] * 0.82 / kde(xs).max()], color=SURFACE, lw=1.5)
        m, l, h = (np.exp(v) if log else v for v in (mean, lo, hi))
        # Label the posterior mean of the quantity itself (for odds ratios, not the geometric mean).
        m = draws[label].mean()
        ax.text(1.01, y0 + 0.3, f"{fmt(m)}  [{fmt(l)}, {fmt(h)}]", transform=ax.get_yaxis_transform(), color=INK, va="center", fontsize=9.5)
    ax.axvline(ref, color=AXIS, lw=1, zorder=0)
    ax.set_yticks(np.arange(len(labels))[::-1] + 0.3, [f"{lab}" for lab in labels])
    ax.tick_params(axis="y", length=0)
    for tick, label in zip(ax.get_yticklabels(), labels):
        tick.set_color(INK)
    if log:
        ax.set_xscale("log")
        ax.set_xlim(lo_all * 0.95, hi_all * 1.05)
        ticks = [t for t in [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5] if lo_all * 0.95 <= t <= hi_all * 1.05]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
    else:
        ax.set_xlim(lo_all - pad, hi_all + pad)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.15, len(labels))
    ax.set_xlabel(xlabel)
    # Group separators: hairline between feature families.
    prev = None
    for i, label in enumerate(labels):
        if prev is not None and groups[label] != prev:
            ax.axhline(len(labels) - i - 0.08, color=GRID, lw=1)
        prev = groups[label]
    ax.text(1.01, 1.0, "Mean  [89% interval]", transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.02, 0.01, caption + " Dark band = 89% interval; white tick = mean.", color=INK_2, fontsize=8.5)
    fig.subplots_adjust(left=0.2, right=0.75, top=0.91, bottom=0.11)
    fig.savefig(path)
    plt.close(fig)


groups = {label: group for label, group, *_ in EFFECTS}
ridgeline(
    {label: 100 * apc["score"][label] for label in groups},
    groups,
    "Change in average 12-month surgery risk (percentage points)",
    lambda v: f"{v:+.2f}",
    0.0,
    False,
    HERE / "effects_pp_kde.png",
    "Change in 12-month surgery probability when each feature is switched on for every patient",
    CAPTION,
)
ridgeline(
    {label: or_draws for label, _, _, _, or_draws in EFFECTS},
    groups,
    "Odds ratio (log scale): feature on vs off; BMI vs <25, tenure vs <6m, age per 10 years",
    lambda v: f"{v:.2f}",
    1.0,
    True,
    HERE / "effects_or_kde.png",
    "Posterior odds ratios",
    "Associations conditional on the other model features, not causal effects. Same for every patient (no interaction terms); 1,000 posterior draws.",
)


def ink_for(fill) -> str:
    r, g, b = to_rgb(fill)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK


def heatmap(ax, cells: pd.DataFrame, rows: list[str], cols: list[str], col_label: str, vmax: float, detail: bool):
    mean = cells.pivot(index="row", columns="col", values="mean_risk").loc[rows, cols].to_numpy()
    lo = cells.pivot(index="row", columns="col", values="lo89").loc[rows, cols].to_numpy()
    hi = cells.pivot(index="row", columns="col", values="hi89").loc[rows, cols].to_numpy()
    share = cells.pivot(index="row", columns="col", values="share_patients_above_break_even").loc[rows, cols].to_numpy()
    im = ax.imshow(100 * mean, cmap=SEQUENTIAL, vmin=0, vmax=100 * vmax, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            color = ink_for(SEQUENTIAL(mean[i, j] / vmax))
            text = f"{100 * mean[i, j]:.1f}%\n{100 * lo[i, j]:.1f}–{100 * hi[i, j]:.1f}"
            if detail:
                text += f"\n{100 * share[i, j]:.0f}% > 1.7%"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8.5 if not detail else 9.5, linespacing=1.3)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(col_label)
    return im


# BMI x age
fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
vmax = bmi_age.mean_risk.max()
im = heatmap(ax_h, bmi_age, list(BMI_LEVELS), [str(a) for a in AGES], "Age (years), set for every patient", vmax, detail=False)
cbar = fig.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02)
cbar.set_label("Average 12-month risk (%)", color=INK_2)
cbar.outline.set_visible(False)
ax_h.set_title("Average risk: posterior mean and 89% interval", loc="left", color=INK, fontsize=11)
for k, band in enumerate(bmi_effect_by_age):
    d = np.array([100 * bmi_effect_by_age[band][a] for a in AGES])
    mean, (lo, hi) = d.mean(axis=1), np.percentile(d, INTERVAL, axis=1)
    if band == "BMI 30–34":  # one band only: four overlapping washes turn to mud
        ax_l.fill_between(AGES, lo, hi, color=SERIES[k], alpha=0.12, lw=0, label="89% interval (BMI 30–34)")
    ax_l.plot(AGES, mean, color=SERIES[k], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=band, solid_capstyle="round")
ax_l.axhline(0, color=AXIS, lw=1)
ax_l.set_xlim(AGES[0] - 1, AGES[-1] + 1)
ax_l.set_xticks(AGES)
ax_l.grid(axis="y", color=GRID, lw=1)
ax_l.set_axisbelow(True)
ax_l.set_xlabel("Age (years)")
ax_l.set_ylabel("Change vs BMI <25 (percentage points)")
ax_l.legend(frameon=False, loc="upper right", fontsize=9)
ax_l.set_title("Same logit shift, bigger probability shift where risk is higher", loc="left", color=INK, fontsize=11)
fig.suptitle("BMI band × age: 12-month surgery risk when both are set for every patient", x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.02, 0.01, CAPTION + " Other features at observed values; band = 89% interval.", color=INK_2, fontsize=8.5)
fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14, wspace=0.25)
fig.savefig(HERE / "sensitivity_bmi_age.png")
plt.close(fig)

# Prior consult x tenure
fig, (ax_h, ax_d) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
vmax = consult_tenure.mean_risk.max()
im = heatmap(ax_h, consult_tenure, list(CONSULT_LEVELS), list(TENURE_LEVELS), "Time since first LBP visit, set for every patient", vmax, detail=True)
cbar = fig.colorbar(im, ax=ax_h, fraction=0.04, pad=0.02)
cbar.set_label("Average 12-month risk (%)", color=INK_2)
cbar.outline.set_visible(False)
ax_h.set_title("Average risk [89% interval], share of patients above 1.7% break-even", loc="left", color=INK, fontsize=10.5)
for j, t in enumerate(TENURE_LEVELS):
    d = 100 * consult_effect_by_tenure[t]
    lo, hi = interval(d)
    ax_d.plot([lo, hi], [j, j], color=SERIES[0], lw=2, solid_capstyle="round")
    ax_d.plot(d.mean(), j, "o", color=SERIES[0], ms=9, mec=SURFACE, mew=2)
    ax_d.text(hi + 0.4, j, f"+{d.mean():.2f} [{lo:.2f}, {hi:.2f}]", va="center", color=INK, fontsize=9)
ax_d.set_yticks(range(len(TENURE_LEVELS)), list(TENURE_LEVELS))
ax_d.invert_yaxis()
ax_d.tick_params(axis="y", length=0)
ax_d.spines["left"].set_visible(False)
ax_d.set_xlim(0, max(100 * consult_effect_by_tenure["<6 months"].max() * 1.35, 1))
ax_d.grid(axis="x", color=GRID, lw=1)
ax_d.set_axisbelow(True)
ax_d.set_xlabel("Prior consult effect (percentage points)")
ax_d.set_title("Prior-consult effect by tenure band", loc="left", color=INK, fontsize=10.5)
fig.suptitle("Prior surgical consult × time since first back-pain visit", x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.02, 0.01, CAPTION + " 99.9% of this population is actually 12m+.", color=INK_2, fontsize=8.5)
fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.17, wspace=0.3)
fig.savefig(HERE / "sensitivity_consult_tenure.png")
plt.close(fig)
print("wrote", sorted(p.name for p in HERE.iterdir() if p.suffix in {".png", ".csv", ".json"}))
