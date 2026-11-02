"""Out-of-time evaluation of the Surgery Risk Model: metrics, charts, tables, report and PDF deck in out/evaluation/.

The model is refit on 2024-07 to 2025-04 and tested on the 2025-07 snapshot, which it never saw. Production
(`main.py train`) also trains on 2025-07, so this measures how the model does on a quarter it hasn't seen.

    uv run python -m assignment.evaluation
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from assignment import assumptions as A
from assignment import features, model, slides

TEST_DATE = features.EXTRACT - features.HORIZON  # 2025-07-01: the latest snapshot with a full 12-month label
N_BOOT = 500
PREDICTED, OBSERVED, BASELINE = slides.ACCENT, slides.GRAY, slides.LIGHT


def flag_top(p: np.ndarray, k: int) -> np.ndarray:
    """1 for the k highest-risk patients: the list Meridian would contact."""
    flag = np.zeros(len(p), dtype=int)
    flag[np.argsort(-p)[:k]] = 1
    return flag


def metrics(y: np.ndarray, p: np.ndarray, k: int) -> dict[str, float]:
    """Discrimination, calibration, and classification metrics for a top-k contact list."""
    flag = flag_top(p, k)
    return {
        "AUC": roc_auc_score(y, p),
        "Mean predicted risk": p.mean(),
        "Observed surgery rate": y.mean(),
        "Precision": precision_score(y, flag),
        "Recall (surgeries caught on the list)": recall_score(y, flag),
        "F1": f1_score(y, flag),
        "Accuracy": (flag == y).mean(),
    }


def bootstrap(y: np.ndarray, p: np.ndarray, k: int) -> pd.DataFrame:
    """Point estimate and 95% interval for each metric, resampling test patients (the model is fixed)."""
    rng = np.random.default_rng(A.RANDOM_SEED)
    k_scaled = lambda n: round(k * n / len(y))  # keep the list the same share of the population
    draws = pd.DataFrame(
        [metrics(y[i], p[i], k_scaled(len(i))) for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))]
    )
    table = pd.DataFrame({"value": metrics(y, p, k), "ci_low": draws.quantile(0.025), "ci_high": draws.quantile(0.975)})
    return table.rename_axis("metric")


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed surgery rate by risk decile, with a 95% interval on the observed rate."""
    df = pd.DataFrame({"y": y, "p": p, "decile": pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False) + 1})
    t = df.groupby("decile").agg(patients=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    t["observed_se"] = np.sqrt(t.observed * (1 - t.observed) / t.patients)
    return t


def gains(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    """Share of surgeries captured as the list grows, highest risk first."""
    captured = np.cumsum(y[np.argsort(-p)]) / y.sum()
    share = np.arange(1, len(y) + 1) / len(y)
    return pd.DataFrame({"share_contacted": share, "share_captured": captured})


def examples(test: pd.DataFrame, k: int) -> pd.DataFrame:
    """Five real test patients, one per situation a stakeholder asks about."""
    t = test.assign(rank=test.p.rank(ascending=False, method="first").astype(int)).reset_index(drop=True)
    surg, none = t[t.y == 1], t[t.y == 0]
    picks = {
        "Top of the list, had surgery": surg.p.idxmax(),
        "High risk, no surgery": none[none["rank"] > 20].p.idxmax(),
        f"On the cut-off line (rank {k:,})": (none.p - t.p[t["rank"] == k].iloc[0]).abs().idxmin(),
        "Typical patient": (none.p - t.p.median()).abs().idxmin(),
        "Missed: low risk, had surgery": surg.p.idxmin(),
    }
    bmi = lambda r: "not recorded" if np.isnan(r.bmi) else f"{r.bmi:.0f}"
    flags = lambda r: ", ".join(
        name
        for col, name in [
            ("prior_consult", "prior consult"),
            ("imaging_180d", "recent imaging"),
            ("injection_365d", "injection"),
        ]
        if r[col]
    )
    rows = []
    for situation, i in picks.items():
        r = t.loc[i]
        profile = f"{r.age:.0f}{'M' if r.male else 'F'}, BMI {bmi(r)}" + (f", {flags(r)}" if flags(r) else "")
        rows.append(
            {
                "situation": situation,
                "profile": profile,
                "predicted_risk": r.p,
                "rank": r["rank"],
                "on_list": r["rank"] <= k,
                "outcome": "Surgery" if r.y else "No surgery",
            }
        )
    return pd.DataFrame(rows)


def fmt(metric: str, v: float) -> str:
    """AUC and F1 as decimals, rates as percentages (risk to two places so predicted vs observed is visible)."""
    return f"{v:.3f}" if metric in ("AUC", "F1") else f"{v:.2%}" if "rate" in metric or "risk" in metric else f"{v:.1%}"


def plot_calibration(ax: plt.Axes, cal: pd.DataFrame) -> None:
    x = cal.index.to_numpy()
    ax.plot(x, cal.predicted * 100, "o-", color=PREDICTED, lw=2, ms=6, label="Predicted (Surgery Risk Model)")
    ax.errorbar(
        x,
        cal.observed * 100,
        yerr=1.96 * cal.observed_se * 100,
        fmt="o-",
        color=OBSERVED,
        lw=2,
        ms=6,
        capsize=3,
        label="Observed (95% interval)",
    )
    ax.set(xlabel="Risk decile (10 = highest predicted risk)", ylabel="12-month surgery rate (%)", xticks=x)
    slides.style(ax)


def plot_gains(ax: plt.Axes, g: pd.DataFrame, k: int) -> None:
    ax.plot(g.share_contacted * 100, g.share_captured * 100, color=PREDICTED, lw=2, label="Surgery Risk Model")
    ax.plot([0, 100], [0, 100], color=BASELINE, lw=2, ls="--", label="Random list")
    share_k, captured_k = k / len(g) * 100, g.share_captured.iloc[k - 1] * 100
    ax.plot([share_k], [captured_k], "o", color=PREDICTED, ms=7)
    ax.annotate(
        f"Top {k:,}: {captured_k:.0f}% of surgeries\nfrom {share_k:.1f}% of patients",
        (share_k, captured_k),
        xytext=(12, -36),
        textcoords="offset points",
        fontsize=slides.SIZE["caption"],
    )
    ax.set(
        xlabel="Patients contacted, highest risk first (%)",
        ylabel="Surgeries captured (%)",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    slides.style(ax)


def save_chart(draw, title: str, path: Path, *args) -> None:
    """Draw one chart to a standalone PNG."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    draw(ax, *args)
    ax.set_title(title, loc="left", fontsize=slides.SIZE["caption"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(table: pd.DataFrame, cm: np.ndarray, ex: pd.DataFrame, n_train: int, test: pd.DataFrame, k: int) -> str:
    """Markdown summary with formatted tables, linking the charts."""
    pct = lambda x: f"{x:.1%}"
    metric_rows = "\n".join(
        f"| {m} | {fmt(m, r.value)} | {fmt(m, r.ci_low)} – {fmt(m, r.ci_high)} |" for m, r in table.iterrows()
    )
    ex_rows = "\n".join(
        f"| {r.situation} | {r.profile} | {pct(r.predicted_risk)} | {r.rank:,} | {'Yes' if r.on_list else 'No'} | {r.outcome} |"
        for r in ex.itertuples()
    )
    (tn, fp), (fn, tp) = cm
    return f"""# Surgery Risk Model: out-of-time evaluation

Trained on {n_train:,} snapshots (2024-07 to 2025-04). Tested on **{len(test):,} patients eligible on 2025-07-01**, a quarter the model never saw; {int(test.y.sum()):,} had surgery while covered within 12 months.

## Metrics

Precision, recall, F1 and accuracy use a fixed list of the top {k:,} patients by predicted risk, about the size of Meridian's past outreach waves (2,189–3,183 patients each) and {k / len(test):.1%} of the test population. How many patients to actually contact is a separate decision, covered in `out/list_depth/how_far_down.pdf`. 95% intervals resample test patients.

| Metric | Value | 95% interval |
|---|---|---|
{metric_rows}

- **AUC:** the chance a patient who had surgery is ranked above one who didn't.
- **Precision:** the surgery rate among listed patients, against {pct(test.y.mean())} for everyone.
- **Recall:** the share of all surgeries on the list.
- **Accuracy is not informative here.** With a {pct(test.y.mean())} surgery rate, contacting nobody scores {pct(1 - test.y.mean())}.

### Confusion matrix, top {k:,}

| | Had surgery | No surgery |
|---|---|---|
| **On the list** | {tp:,} | {fp:,} |
| **Not on the list** | {fn:,} | {tn:,} |

## Predicted vs observed

![Calibration](calibration.png)

## Surgeries captured as the list grows

![Gains](gains.png)

## Five real patients

| Situation | Profile | Predicted risk | Rank | On the list | Outcome |
|---|---|---|---|---|---|
{ex_rows}

A predicted risk is a probability, not a verdict. Most high-risk patients don't have surgery within a year, and some low-risk ones do. What matters is that, across groups, predicted and observed rates match (chart above).

The slide version of this report is `surgery_risk_model_evaluation.pdf`.
"""


def deck(
    path: Path,
    table: pd.DataFrame,
    cal: pd.DataFrame,
    g: pd.DataFrame,
    cm: np.ndarray,
    ex: pd.DataFrame,
    n_train: int,
    test: pd.DataFrame,
    k: int,
) -> None:
    """The evaluation as a PDF slide deck."""
    m = table.value
    recall = m["Recall (surgeries caught on the list)"]
    (tn, fp), (fn, tp) = cm
    base = test.y.mean()
    with PdfPages(path, metadata={"Title": "Vi Labs · Meridian · Surgery Risk Model evaluation"}) as pdf:
        slides.section(pdf, "Surgery Risk Model · Out-of-time evaluation")
        fig = slides.page(pdf, "How well does the\nSurgery Risk Model predict?", dark=True)
        slides.text(
            fig,
            0.6,
            3.85,
            f"Out-of-time test on {len(test):,} patients eligible on 2025-07-01, a quarter the model never saw.",
            role="subtitle",
            color=slides.BEIGE,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "The bottom line")
        slides.card(
            fig,
            0.6,
            3.45,
            3.85,
            2.3,
            f"{m['AUC']:.2f}",
            f"AUC: ranks a surgery patient above a non-surgery patient {m['AUC']:.0%} of the time",
        )
        slides.card(
            fig,
            4.74,
            3.45,
            3.85,
            2.3,
            f"{m['Mean predicted risk']:.2%}",
            f"predicted risk, against {m['Observed surgery rate']:.2%} observed: calibrated on a new quarter",
        )
        slides.card(
            fig,
            8.88,
            3.45,
            3.85,
            2.3,
            f"{recall:.0%}",
            f"of surgeries caught by contacting the top {k:,} ({k / len(test):.1%} of patients)",
            PREDICTED,
        )
        slides.text(fig, 0.6, 2.85, "What this means", role="lead")
        slides.text(
            fig,
            0.6,
            2.35,
            "\n".join(
                [
                    "•  The model's risk estimates can be taken at face value: across risk groups, predicted and observed surgery rates match.",
                    f"•  Contacting the top {k:,} reaches {recall / (k / len(test)):.1f}× as many future surgeries as a random list of the same size.",
                    f"•  Surgery is rare ({base:.1%} a year), so most listed patients won't have surgery: the list concentrates risk, it doesn't guarantee it.",
                ]
            ),
            role="body",
            linespacing=1.8,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "How the test works")
        slides.text(
            fig,
            0.6,
            slides.TOP,
            "\n".join(
                [
                    f"•  Train: {n_train:,} patient snapshots, quarterly from 2024-07 to 2025-04.",
                    f"•  Test: the 2025-07 snapshot. {len(test):,} eligible patients, {int(test.y.sum()):,} of whom had surgery",
                    "   while covered within 12 months.",
                    "•  The model never sees the test quarter, so this is how it performs on the next one",
                    "   Meridian will score.",
                    "•  Eligible: covered, back-pain history, no surgery yet, no surgical consult in the last 90 days.",
                    f"•  Precision and recall use a fixed list of the top {k:,}, about the size of Meridian's past waves.",
                    "   How many to actually contact is a separate decision (the 'How far down the list' deck).",
                    "•  95% intervals come from resampling test patients 500 times.",
                ]
            ),
            role="body",
            linespacing=1.9,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Metrics on the test quarter")
        meaning = {
            "AUC": "Chance a surgery patient ranks above a non-surgery one",
            "Mean predicted risk": "Average of the model's predictions",
            "Observed surgery rate": "What actually happened",
            "Precision": f"Surgery rate among listed patients (vs {base:.1%} overall)",
            "Recall (surgeries caught on the list)": f"Share of all surgeries in the top {k:,}",
            "F1": "Balance of precision and recall",
            "Accuracy": f"Misleading at a {base:.1%} rate: listing nobody scores {1 - base:.1%}",
        }
        rows = [
            [mt, fmt(mt, r.value), f"{fmt(mt, r.ci_low)} – {fmt(mt, r.ci_high)}", meaning[mt]]
            for mt, r in table.iterrows()
        ]
        slides.table(
            fig,
            0.6,
            0.85,
            12.13,
            slides.TOP - 0.85,
            ["Metric", "Value", "95% interval", "What it means"],
            rows,
            [0.27, 0.1, 0.17, 0.46],
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, f"Who the top {k:,} catches, and who it misses")
        slides.table(
            fig,
            0.6,
            3.1,
            6.2,
            slides.TOP - 3.1,
            ["", "Had surgery", "No surgery"],
            [["On the list", f"{tp:,}", f"{fp:,}"], ["Not on the list", f"{fn:,}", f"{tn:,}"]],
            [0.36, 0.32, 0.32],
        )
        slides.text(
            fig,
            7.3,
            slides.TOP,
            "\n".join(
                [
                    f"•  {tp:,} of the {int(test.y.sum()):,} surgeries are on the list:",
                    f"   recall {recall:.0%}.",
                    f"•  {tp:,} of {k:,} listed patients had surgery:",
                    f"   precision {m['Precision']:.1%}, {m['Precision'] / base:.1f}× the base rate.",
                    f"•  {fn:,} surgeries came from patients ranked below {k:,}.",
                    "   Some surgeries happen to patients who look low-risk.",
                ]
            ),
            role="body",
            linespacing=1.9,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Predicted risk matches what happened")
        plot_calibration(slides.box(fig, 0.9, 1.3, 7.9, slides.TOP - 1.3), cal)
        slides.text(
            fig,
            9.3,
            slides.TOP,
            "\n".join(
                [
                    "Each point is a tenth of the",
                    "test patients, sorted by risk.",
                    "",
                    f"Top decile: {cal.predicted.iloc[-1]:.1%} predicted,",
                    f"{cal.observed.iloc[-1]:.1%} observed.",
                    "",
                    "Bottom deciles run slightly",
                    f"low ({cal.predicted.iloc[0]:.1%} vs {cal.observed.iloc[0]:.1%}), which",
                    "doesn't affect who makes",
                    "the list.",
                ]
            ),
            role="body",
            linespacing=1.6,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, f"A {k:,}-patient list catches {recall:.0%} of surgeries")
        plot_gains(slides.box(fig, 0.9, 1.3, 7.9, slides.TOP - 1.3), g, k)
        slides.text(
            fig,
            9.3,
            slides.TOP,
            "\n".join(
                [
                    "Contact patients in risk order",
                    "and count how many of the",
                    "year's surgeries you've reached.",
                    "",
                    f"A random list of {k:,} catches",
                    f"{k / len(test):.1%}. This one catches {recall:.0%}.",
                    "",
                    "The curve flattens: past the",
                    "top third, extra contacts",
                    "find few extra surgeries.",
                ]
            ),
            role="body",
            linespacing=1.6,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Five real patients from the test quarter")
        rows = [
            [
                r.situation,
                textwrap.fill(r.profile, 40),
                f"{r.predicted_risk:.1%}",
                f"{r.rank:,}",
                "Yes" if r.on_list else "No",
                r.outcome,
            ]
            for r in ex.itertuples()
        ]
        slides.table(
            fig,
            0.6,
            1.55,
            12.13,
            slides.TOP - 1.55,
            ["Situation", "Profile", "Predicted risk", "Rank", "On list", "Outcome"],
            rows,
            [0.23, 0.37, 0.12, 0.09, 0.08, 0.11],
        )
        slides.text(
            fig,
            0.6,
            1.3,
            "A predicted risk is a probability, not a verdict: most high-risk patients don't have surgery within a year, and some low-risk ones do.",
            role="body",
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Confident enough to rank\nMeridian's list", dark=True)
        slides.text(
            fig,
            0.6,
            3.9,
            "\n".join(
                [
                    f"•  Calibrated on a quarter it never saw: {m['Mean predicted risk']:.2%} predicted vs {m['Observed surgery rate']:.2%} observed.",
                    f"•  Ranks well: AUC {m['AUC']:.2f}, and the top {k:,} catch {recall:.0%} of surgeries.",
                    "•  Limits: half of next quarter's patients are 24+ months past their first visit, beyond the",
                    "   training range, so their risks may run slightly high. Re-check calibration every quarter.",
                ]
            ),
            role="subtitle",
            color=slides.BEIGE,
            linespacing=1.8,
        )
        slides.emit(pdf, fig)


def test_predictions(d: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training snapshots, and the 2025-07 test snapshot with `p` from a model refit without it."""
    panel = features.training_panel(d)
    train, test = panel[panel.index_date < TEST_DATE], panel[panel.index_date == TEST_DATE].copy()
    test["p"] = model.predict(model.train(train), test)
    return train, test


def run(
    data_dir: Path = Path("data"), out_dir: Path = Path("out/evaluation"), k: int = A.EVALUATION_LIST_SIZE
) -> pd.DataFrame:
    """Refit on the training quarters, score the 2025-07 test snapshot, write charts, tables, report and deck."""
    train, test = test_predictions(features.load(data_dir))
    y, p = test.y.to_numpy(), test.p.to_numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    table, cal, g, ex = bootstrap(y, p, k), calibration(y, p), gains(y, p), examples(test, k)
    cm = confusion_matrix(y, flag_top(p, k))
    table.to_csv(out_dir / "metrics.csv")
    cal.to_csv(out_dir / "calibration.csv")
    ex.to_csv(out_dir / "examples.csv", index=False)
    save_chart(
        plot_calibration,
        "Predicted vs observed surgery rate by risk decile, 2025-07 test set",
        out_dir / "calibration.png",
        cal,
    )
    save_chart(
        plot_gains, "Share of 2025-07 surgeries captured vs share of patients contacted", out_dir / "gains.png", g, k
    )
    (out_dir / "report.md").write_text(report(table, cm, ex, len(train), test, k))
    deck(out_dir / "surgery_risk_model_evaluation.pdf", table, cal, g, cm, ex, len(train), test, k)
    print(table.round(4).to_string())
    return table


if __name__ == "__main__":
    run()
