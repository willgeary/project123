"""The final presentation for Meridian: out/presentation/final.pdf, following out/presentation/outline.md.

Every number is computed: the ranked list from out/ (run `main.py score` first), the out-of-time test from a
refit model, past-outreach results from sandbox/causal, and feature effects from sandbox/explainability.

    uv run python -m assignment.presentation
"""

import json
import pickle
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from assignment import assumptions as A
from assignment import evaluation, features, list_depth, model, slides, targeting

CAUSAL = Path("sandbox/causal/deck/causal_deck_data.json")
EFFECTS = Path("sandbox/explainability/effects.csv")
MODEL_PATH = Path("artifacts/model.pkl")
pct = lambda x, d=1: f"{x:.{d}%}"
# The plan slide and the section dividers both read this, so the agenda always matches the deck.
SECTIONS = [
    ("Who is eligible", "Covered, on a path to surgery, not yet operated on, and reachable."),
    ("How the list is ranked", "A 12-month surgery-risk model, tested on a quarter it never saw."),
    ("Which program", "Weight management or physical therapy, by BMI and obesity diagnosis."),
    ("How many to contact", "Contact while expected savings beat the cost of a contact."),
    ("How to measure", "What past outreach can and can't tell us, and how each wave will measure it."),
    ("Assumptions", "What the answer rests on, and what Meridian should confirm."),
]


def funnel(d: dict, ranked: pd.DataFrame, t: pd.Timestamp) -> list[tuple[str, int]]:
    """Patients remaining after each eligibility rule and exclusion."""
    steps = features.eligibility(d, t)
    eligible = list(steps.values())[-1]
    not_recent = eligible & ~d["patients"].index.isin(targeting.recent_enrollees(d["enrollments"], t))
    return [
        ("All patients in the extract", len(d["patients"])),
        *[(name.replace("the index date", "2026-07-01"), int(mask.sum())) for name, mask in steps.items()],
        (f"Not enrolled in PT/WM in the last {A.RECENT_ENROLLMENT_EXCLUSION_DAYS} days", int(not_recent.sum())),
        ("At least one usable channel", len(ranked)),
    ]


def problem(d: dict, t: pd.Timestamp) -> dict:
    """The opening slide's facts: last year's surgeries while covered, and how past outreach ran."""
    surgery, end = d["first_surgery"], d["patients"].coverage_end
    covered = surgery[(surgery >= t - features.HORIZON) & (surgery < t) & (surgery <= end.reindex(surgery.index))]
    o, occ = d["outreach"], d["occurred"]
    first_lbp = occ[occ.event_category == "LBP Encounter"].groupby("patient_token").event_date.min()
    first_contact = o.groupby("patient_token").timestamp.min()
    per_wave = o.groupby(o.timestamp.dt.to_period("Q")).patient_token.nunique()
    per_wave = per_wave[per_wave > 100]  # waves; drop stray late attempts
    return {
        "surgeries": len(covered),
        "delay_days": (first_contact - first_lbp.reindex(first_contact.index)).dt.days.median(),
        "wave_min": int(per_wave.min()),
        "wave_max": int(per_wave.max()),
    }


# One plain-language line per funnel step, in funnel() order.
# Hypothetical patients scored by the trained model: (title, signals shown on the slide, features; unset = absent).
EXAMPLE_PATIENTS = [
    (
        "Patient A: higher risk",
        [
            "58, female, BMI 36 with an obesity diagnosis",
            "Saw a spine surgeon 6 months ago",
            "4 back-pain visits this year, 2 in the last 3 months",
            "Lumbar MRI; sciatica diagnosis",
            "First back-pain visit 8 months ago",
        ],
        {
            "age": 58, "under_65": 1, "bmi_35_plus": 1, "obesity_dx": 1, "prior_consult": 1,
            "log_lbp_90d": np.log1p(2), "log_lbp_365d": np.log1p(4), "days_since_last_lbp": 21 / 365,
            "mri_365d": 1, "radiculopathy": 1, "tenure_6_12m": 1,
        },
    ),
    (
        "Patient B: lower risk",
        [
            "72, male, BMI 23",
            "Never saw a spine surgeon",
            "1 back-pain visit this year, 8 months ago",
            "No imaging, injections or spine diagnoses",
            "First back-pain visit over a year ago",
        ],
        {
            "age": 72, "male": 1, "log_lbp_365d": np.log1p(1), "days_since_last_lbp": 240 / 365,
            "tenure_12m_plus": 1,
        },
    ),
]  # fmt: skip

# Model signals by code name, for the feature tables: (column, shown name, plain-language description).
FEATURE_TABLES = [
    (
        "Back pain and care use",
        [
            ("prior_consult", "Prior surgical consult", "Saw a spine surgeon 90–365 days ago"),
            ("log_lbp_90d", "Back-pain visits, 90 days", "Low back pain visits in the last 3 months"),
            ("log_lbp_365d", "Back-pain visits, 12 months", "Low back pain visits in the last year"),
            ("days_since_last_lbp", "Time since last back-pain visit", "Years since the latest visit, capped at 2"),
            ("tenure_6_12m", "First back-pain visit 6–12m ago", "Vs. a first visit under 6 months ago"),
            ("tenure_12m_plus", "First back-pain visit 12m+ ago", "Vs. a first visit under 6 months ago"),
            ("imaging_180d", "Imaging, 180 days", "Any spine imaging in the last 6 months"),
            ("mri_365d", "Lumbar MRI, 12 months", "A lumbar spine MRI in the last year"),
            ("injection_365d", "Spinal injection, 12 months", "An epidural or other spine injection"),
            ("pt_referral_365d", "PT referral, 12 months", "Referred to physical therapy"),
            ("log_pt_visits", "PT visits", "Physical therapy sessions billed"),
            ("log_ortho_pain_visits", "Ortho / pain visits", "Visits to orthopedics or pain management"),
            ("log_ed_visits", "ED visits", "Emergency department visits, any cause"),
            ("log_rx_fills", "Prescription fills", "Pharmacy fills, any drug"),
            ("prior_program", "Earlier PT / WM enrollment", "Enrolled 90–365 days ago"),
            ("radiculopathy", "Radiculopathy", "Nerve pain down the leg (sciatica)"),
            ("stenosis", "Spinal stenosis", "Narrowed spinal canal"),
            ("spondylolisthesis", "Spondylolisthesis", "A vertebra slipped out of place"),
            ("disc_degeneration", "Disc degeneration", "Worn lumbar discs"),
        ],
    ),
    (
        "Who the patient is",
        [
            ("age", "Age", "Years, at the list date"),
            ("under_65", "Under 65", "Younger than Medicare age"),
            ("male", "Male", "Vs. female"),
            ("sex_unknown", "Sex unknown", "Sex not recorded"),
            ("bmi_25_29", "BMI 25–29", "Overweight, vs. BMI under 25"),
            ("bmi_30_34", "BMI 30–34", "Obese, vs. BMI under 25"),
            ("bmi_35_plus", "BMI 35+", "Severely obese, vs. BMI under 25"),
            ("no_bmi", "No BMI on file", "No BMI code in the last year"),
            ("obesity_dx", "Obesity diagnosis", "Coded obesity (E66)"),
            ("diabetes", "Diabetes", "Type 2 diabetes"),
            ("hypertension", "Hypertension", "High blood pressure"),
            ("heart_disease", "Heart disease", "Coronary artery disease"),
            ("copd", "COPD", "Chronic lung disease"),
            ("ckd", "Kidney disease", "Chronic kidney disease, stage 3"),
            ("depression_anxiety", "Depression / anxiety", "Either diagnosis"),
            ("smoker", "Smoker", "Nicotine dependence, cigarettes"),
            ("osteoporosis", "Osteoporosis", "Low bone density"),
        ],
    ),
]
assert sorted(f for _, rows in FEATURE_TABLES for f, *_ in rows) == sorted(features.FEATURES)

FUNNEL_NOTES = [
    "Everyone in Meridian's data, current members or not.",
    "Still a Meridian member on the day the list is built.",
    "Has seen a clinician for low back pain at least once.",
    "Hasn't had lumbar spine surgery yet, so there is still one to prevent.",
    "Hasn't seen a surgeon recently, when surgery is likely already planned. Older consults stay in.",
    "Isn't already part-way through PT or weight management. Earlier enrollees stay in.",
    "Has a phone, text or email we may use: no opt-outs, invalid numbers or bounces.",
]


def plot_funnel(ax: plt.Axes, steps: list[tuple[str, int]]) -> None:
    names, counts = zip(*steps)
    y = np.arange(len(steps))[::-1]
    ax.barh(y, counts, color=[slides.LIGHT] * (len(steps) - 1) + [slides.ACCENT], height=0.62)
    for yi, c in zip(y, counts):
        ax.text(c + 300, yi, f"{c:,}", va="center", fontsize=slides.SIZE["small"], color=slides.INK)
    ax.set_yticks(y, names, fontsize=slides.SIZE["small"])
    ax.set_ylim(-0.5, len(steps) - 0.5)  # fixed, so notes beside the chart can line up with each bar
    ax.set_xlim(0, max(counts) * 1.25)
    ax.xaxis.set_visible(False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)


def plot_drivers(ax: plt.Axes, effects: pd.DataFrame) -> None:
    e = effects.sort_values("pp_score_mean")
    ax.barh(
        e.feature, e.pp_score_mean, color=[slides.ACCENT if v > 0 else slides.GRAY for v in e.pp_score_mean], height=0.6
    )
    for yi, v in enumerate(e.pp_score_mean):
        ax.text(
            v + (0.06 if v > 0 else -0.06),
            yi,
            f"{v:+.1f}",
            va="center",
            ha="left" if v > 0 else "right",
            fontsize=slides.SIZE["caption"],
        )
    ax.axvline(0, color=slides.LIGHT, lw=1)
    ax.set_xlabel("Change in 12-month surgery risk (percentage points)")
    ax.set_xlim(-3.4, 3.6)
    slides.style(ax)
    ax.tick_params(axis="y", labelsize=slides.SIZE["caption"] + 1)


def draw_dag(ax: plt.Axes, randomized: bool) -> None:
    """Outreach -> Enrollment -> Surgery, with confounders; randomization cuts the arrows into Outreach."""
    ax.set(xlim=(0, 10), ylim=(0, 6))
    ax.axis("off")
    style = lambda fc, ec, ls="-": dict(boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.5, ls=ls)
    node = lambda x, y, s, **kw: ax.text(
        x,
        y,
        s,
        ha="center",
        va="center",
        fontsize=slides.SIZE["small"],
        bbox=kw.get("bbox"),
        weight=slides.MEDIUM if kw.get("bold") else "normal",
    )
    out = node(1.4, 3, "Outreach", bbox=style("#dce9f8", slides.ACCENT), bold=True)
    enr = node(5, 3, "Enrollment", bbox=style("#dce9f8", slides.ACCENT), bold=True)
    sur = node(8.6, 3, "Surgery", bbox=style("#dce9f8", slides.ACCENT), bold=True)
    unseen = node(5, 5.3, "What Meridian's team saw\n(not in the data)", bbox=style("#fbe3d6", slides.ORANGE, "--"))
    seen = node(5, 0.7, "What we can see\n(37 covariates)", bbox=style(slides.PAPER, slides.GRAY))

    arrow = lambda a, b, color, ls="-", lw=2.0, alpha=1.0: slides.connect(ax, a, b, color, ls, lw, alpha)
    for a, b in [(out, enr), (enr, sur)]:
        arrow(a, b, slides.ACCENT, lw=2.5)
    arrow(unseen, sur, slides.ORANGE, "--")
    arrow(seen, sur, slides.GRAY)
    cut = 0.25 if randomized else 1.0
    arrow(unseen, out, slides.ORANGE, "--", alpha=cut)
    arrow(seen, out, slides.GRAY, alpha=cut)
    if randomized:
        rnd = node(1.4, 5.3, "Randomization\n(coin flip)", bbox=style("#dce9f8", slides.ACCENT), bold=True)
        arrow(rnd, out, slides.ACCENT, lw=2.5)
        for x, y in [(3.1, 4.25), (3.1, 1.8)]:
            ax.text(x, y, "×", fontsize=26, color=slides.ORANGE, ha="center", va="center", weight=slides.MEDIUM)
        ax.text(
            8.6,
            5.3,
            "Backdoor closed:\nonly the coin flip\ndecides who's contacted",
            fontsize=slides.SIZE["caption"],
            color=slides.ACCENT,
            ha="center",
            va="center",
        )
    else:
        ax.text(
            8.6,
            5.3,
            "Backdoor path:\nOutreach ← unseen\ninformation → Surgery",
            fontsize=slides.SIZE["caption"],
            color=slides.ORANGE,
            ha="center",
            va="center",
        )
        ax.text(5.0, 1.85, "adjusted for", fontsize=slides.SIZE["caption"], color=slides.GRAY, ha="center")


def assumption_rows(attempts_per_patient: float, contacts: int) -> list[list[str]]:
    """Every judgment call with its value, how solid it is, and where it comes from."""
    return [
        [
            "Enrollment lift from contact",
            pct(A.ENROLLMENT_LIFT),
            "Measured",
            "Past campaigns, doubly robust (89%: 10.6–12.4%)",
        ],
        [
            "Surgery reduction per enrollment",
            pct(A.SURGERY_REDUCTION_PER_ENROLLMENT, 0),
            "Assumed",
            "Literature; history can't test it (section 5)",
        ],
        ["Surgery cost", f"${A.SURGERY_COST:,}", "Published", "Medicare 90-day episode blend; to confirm"],
        ["Outreach cost", f"${A.CONTACT_COST}", "Vi call", "Amortized per patient, same for every channel ($5–$10)"],
        [
            "Late-stage window",
            f"{A.LATE_STAGE_CONSULT_DAYS} days",
            "Judgment",
            "A consult this recent means already late-stage",
        ],
        ["Recent-enrollment exclusion", f"{A.RECENT_ENROLLMENT_EXCLUSION_DAYS} days", "Agreed", "Probably mid-course"],
        ["Prior-consult patients", "Kept, labeled", "Agreed", "High value for cost; Meridian can filter"],
        ["Weight-management rule", f"BMI ≥ {A.WM_BMI_THRESHOLD} or E66", "Clinical", "Obesity cut-off"],
        [
            "Holdout: tier 1 / learning band",
            f"{pct(A.TIER1_HOLDOUT_SHARE, 0)} / {pct(A.TIER2_HOLDOUT_SHARE, 0)}",
            "Agreed",
            "Meridian accepts a holdout",
        ],
        ["Learning band size", f"{A.LEARNING_BAND_SIZE:,}", "Design", "Patients just below break-even, randomized"],
        [
            "Capacity",
            f"{contacts:,} patients",
            "To confirm",
            f"≈ {contacts * attempts_per_patient:,.0f} contact attempts at past rates",
        ],
    ]


def deck(
    path: Path,
    d: dict,
    ranked: pd.DataFrame,
    targets: pd.DataFrame,
    test: pd.DataFrame,
    n_train: int,
    causal: dict,
    effects: pd.DataFrame,
    artifact: dict,
) -> None:
    t, k = features.EXTRACT, A.EVALUATION_LIST_SIZE
    s = list_depth.summary(ranked)
    e, be, n1, n2, contact = s["e"], s["be"], s["n1"], s["n2"], s["contact"]
    y, p = test.y.to_numpy(), test.p.to_numpy()
    m, cal, g = evaluation.metrics(y, p, k), evaluation.calibration(y, p), evaluation.gains(y, p)
    recall = m["Recall (surgeries caught on the list)"]
    attempts = len(d["outreach"]) / d["outreach"].patient_token.nunique()
    listed = ranked[ranked.tier > 0]
    raw, cal_rd = causal["raw"], causal["landmark"]["calibrated_rd"]

    with PdfPages(path, metadata={"Title": "Vi Labs · Meridian · Outreach list for the next wave"}) as pdf:
        # Cover
        fig = slides.page(pdf, "Preventing back surgery\nwith targeted outreach", dark=True)
        slides.text(
            fig,
            0.6,
            3.85,
            "Who to contact, with which program, and how we'll know it worked.",
            role="subtitle",
            color=slides.BEIGE,
        )
        slides.text(fig, 0.6, 0.6, "Vi Labs  ·  Prepared for Meridian Health", "small", color=slides.BEIGE, va="bottom")
        slides.emit(pdf, fig)

        # 0. Opening: the problem and the opportunity
        slides.section(pdf, "Meridian · Back-surgery outreach")
        prob = problem(d, t)
        fig = slides.page(pdf, "Targeted outreach can avoid expensive back surgeries")
        panels = [
            (
                "The problem",
                f"{prob['surgeries']:,} lumbar spine surgeries in the past year among Meridian's covered patients: "
                f"about ${prob['surgeries'] * A.SURGERY_COST / 1e6:.1f}M at ~${A.SURGERY_COST:,} each. "
                "Under its value-based contract, Meridian pays the full cost.",
            ),
            (
                "What's been tried",
                f"Outreach for PT and weight management reached {prob['wave_min']:,}–{prob['wave_max']:,} patients per wave, "
                f"a median {prob['delay_days']:.0f} days after their first back-pain visit. "
                "It was never measured: there was no holdout.",
            ),
            (
                "The opportunity",
                "Conservative care can prevent some surgeries. Reach the patients most likely to need surgery before "
                "they see a surgeon, offer the right program, and measure what it does.",
            ),
        ]
        for i, (title, body) in enumerate(panels):
            slides.panel(fig, 0.6 + i * 4.14, 2.9, 3.85, slides.TOP - 2.9, title, body, width=33)
        slides.text(
            fig,
            0.6,
            2.45,
            "What Meridian asked for: a ranked list for the next wave, a program for each patient,\n"
            "how far down the list to go, and how we'll know it worked.",
            role="lead",
            linespacing=1.5,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "The expected savings of an outreach must beat its cost")
        slides.text(fig, slides.W / 2, 4.6, "DECISION RULE", "kicker", size=14, ha="center")
        slides.text(
            fig,
            slides.W / 2,
            4.05,
            "P(surgery) × enrollment lift × surgery reduction × surgery cost\n>  outreach cost",
            "title",
            size=26,
            ha="center",
            multialignment="center",
            linespacing=1.6,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Two patients: one worth an outreach, one not")
        value_per_risk = A.ENROLLMENT_LIFT * A.SURGERY_REDUCTION_PER_ENROLLMENT * A.SURGERY_COST
        examples = [
            ("Patient A: recent MRI, opioid prescriptions", 0.05, "Reach out"),
            ("Patient B: one back-pain visit, nothing since", 0.01, "Skip"),
        ]
        for i, (who, p_surg, verdict) in enumerate(examples):
            x, savings = 0.6 + i * 6.165, p_surg * value_per_risk
            slides.panel(fig, x, 1.3, 5.965, slides.TOP - 1.3, who, "")
            lines = [
                ("Chance of surgery in 12 months", pct(p_surg, 0)),
                ("× enrollment lift", f"{A.ENROLLMENT_LIFT * 100:.1f} points"),
                ("× surgery reduction", f"{A.SURGERY_REDUCTION_PER_ENROLLMENT:.0%}"),
                ("× surgery cost", f"${A.SURGERY_COST:,}"),
                ("= expected savings", f"${savings:,.2f}"),
                ("Outreach cost", f"${A.CONTACT_COST:,.2f}"),
            ]
            for j, (label, value) in enumerate(lines):
                y = slides.TOP - 0.95 - j * 0.42
                role = "heading" if label.startswith("=") else "body"
                slides.text(fig, x + 0.25, y, label, role)
                slides.text(fig, x + 5.7, y, value, role, ha="right")
            color = slides.ACCENT if savings > A.CONTACT_COST else slides.MUTED
            sign = ">" if savings > A.CONTACT_COST else "<"
            slides.text(fig, x + 0.25, 2.3, verdict, "title", color=color)
            slides.text(
                fig,
                x + 0.25,
                1.7,
                f"${savings:,.2f} {sign} ${A.CONTACT_COST:,.0f}: savings "
                + ("beat the cost." if sign == ">" else "fall short of the cost."),
                "body",
                color=color,
            )
        slides.text(
            fig,
            0.6,
            0.95,
            f"Hypothetical patients. Each 1% of surgery risk is worth ${value_per_risk / 100:,.2f}, "
            f"so the line falls at {pct(A.CONTACT_COST / value_per_risk, 2)}.",
            "small",
            color=slides.MUTED,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "The plan")
        for i, (title, question) in enumerate(SECTIONS):
            y = slides.TOP - 0.1 - i * 0.8
            slides.text(fig, 0.6, y, f"{i + 1:02d}", "title", color=slides.ACCENT)
            slides.text(fig, 1.6, y + 0.02, title, "lead")
            slides.text(fig, 1.6, y - 0.36, question, "body", color=slides.MUTED)
        slides.emit(pdf, fig)

        # 1. Who is eligible
        slides.divider(pdf, 1, *SECTIONS[0])
        steps = funnel(d, ranked, t)
        fig = slides.page(pdf, f"From {steps[0][1]:,} patients to {steps[-1][1]:,} we can reach")
        y0, h = 0.9, slides.TOP - 0.9
        plot_funnel(slides.box(fig, 3.75, y0, 3.6, h), steps)
        assert len(FUNNEL_NOTES) == len(steps)
        for i, note in enumerate(FUNNEL_NOTES):
            y = y0 + (len(steps) - i - 0.5) / len(steps) * h
            slides.text(fig, 7.75, y, textwrap.fill(note, 64), "small", color=slides.INK, va="center", linespacing=1.35)
        slides.emit(pdf, fig)

        # 2. How the list is ranked
        slides.divider(pdf, 2, *SECTIONS[1])
        fig = slides.page(pdf, "What the model predicts")
        slides.text(
            fig,
            0.6,
            slides.TOP,
            "Risk  =  P( lumbar spine surgery within 12 months, while covered  |  signals before the list date )",
            role="lead",
        )
        model_panels = [
            (
                "The model: logistic regression",
                "Each signal gets a weight learned from past patients; the weighted sum becomes a probability. "
                f"Fit on {n_train + len(test):,} quarterly patient snapshots, 2024-07 to 2025-07.",
            ),
            (
                "The target: surgery within 12 months, while covered",
                "1 if lumbar spine surgery (EHR or paid claim) in the 12 months after the list date, before "
                "leaving Meridian; otherwise 0. Meridian only pays while a patient is covered.",
            ),
            (
                f"The signals: {len(features.FEATURES)}, from the 12 months before the list date",
                "Back-pain care (visits, surgical consults, imaging, injections, PT), spine diagnoses, BMI and "
                "obesity, chronic conditions, age and sex. Each one is defined on the next slide.",
            ),
        ]
        for i, (title, body) in enumerate(model_panels):
            slides.panel(fig, 0.6, slides.TOP - 0.6 - (i + 1) * 1.42, 7.3, 1.3, title, body, width=92, role="small")
        sorted_p = ranked.score.to_numpy()
        slides.table(
            fig,
            8.2,
            2.5,
            4.53,
            2.6,
            ["Rank", "12-month risk"],
            [
                ["1 (highest)", pct(sorted_p[0])],
                [f"{n1:,} (the cut-off)", pct(sorted_p[n1 - 1], 2)],
                [f"{n1 + n2:,} (end of band)", pct(sorted_p[n1 + n2 - 1], 2)],
                ["Average patient", pct(sorted_p.mean(), 2)],
            ],
            [0.6, 0.4],
        )
        slides.text(
            fig,
            8.2,
            2.2,
            f"Rank 1 is {sorted_p[0] / sorted_p.mean():.0f}× the average patient's risk.",
            role="small",
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Two patients through the model")
        rows = [{f: 0.0 for f in features.FEATURES} | x for *_, x in EXAMPLE_PATIENTS]
        risks = model.predict(artifact, pd.DataFrame(rows))
        for i, ((title, signals, _), risk) in enumerate(zip(EXAMPLE_PATIENTS, risks)):
            x, above = 0.6 + i * 6.165, risk > be
            color = slides.ACCENT if above else slides.MUTED
            slides.panel(fig, x, 1.3, 5.965, slides.TOP - 1.3, title, "")
            slides.text(fig, x + 0.25, slides.TOP - 0.7, "\n".join(f"•  {s}" for s in signals), "body", linespacing=1.8)
            slides.text(fig, x + 0.25, 3.0, pct(risk), "display", color=color)
            rank = int((ranked.score > risk).sum()) + 1
            slides.text(
                fig,
                x + 0.25,
                2.1,
                f"12-month surgery risk; would rank #{rank:,} of {len(ranked):,}.\n"
                + (
                    f"Above the {pct(be, 2)} break-even: on the list."
                    if above
                    else f"Below the {pct(be, 2)} break-even: not contacted."
                ),
                "body",
                color=color,
                linespacing=1.5,
            )
        slides.text(
            fig,
            0.6,
            0.95,
            f"Hypothetical patients scored by the trained model. Signals not listed are absent. "
            f"Average risk across the {len(ranked):,} ranked patients: {pct(ranked.score.mean())}.",
            "small",
            color=slides.MUTED,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, f"What it looks at: {len(features.FEATURES)} signals from the past 12 months")
        slides.text(
            fig,
            0.6,
            slides.TOP,
            "Counts are log-scaled; every signal covers the 12 months before the list date unless noted.",
            role="body",
            color=slides.MUTED,
        )
        for i, (heading, rows) in enumerate(FEATURE_TABLES):
            x = 0.6 + i * 6.165
            slides.text(fig, x, slides.TOP - 0.45, f"{heading.upper()}  ·  {len(rows)}", "kicker", color=slides.ACCENT)
            n = len(FEATURE_TABLES[0][1]) + 1
            h = (slides.TOP - 0.75 - 0.85) * (len(rows) + 1) / n
            slides.table(
                fig,
                x,
                slides.TOP - 0.75 - h,
                5.965,
                h,
                ["Signal", "What it means"],
                [r[1:] for r in rows],
                [0.45, 0.55],
            )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Biggest drivers of risk")
        slides.text(
            fig,
            0.6,
            slides.TOP,
            "Change in a patient's 12-month surgery risk when the signal is present.\n"
            "Associations, not causes. Outreach history is deliberately not a signal.",
            role="body",
            color=slides.MUTED,
        )
        plot_drivers(slides.box(fig, 3.2, 1.3, 9.0, slides.TOP - 1.95), effects)
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "How it works: a weighted checklist")
        ax = slides.box(fig, 0.6, 3.2, 12.13, 2.5)
        ax.set(xlim=(0, 10), ylim=(0, 2))
        ax.axis("off")
        stages = [
            (f"{len(features.FEATURES)} signals", "per patient"),
            ("× weights", "learned from past patients"),
            ("= a score", "higher means riskier"),
            ("→ chance of surgery", "in the next 12 months"),
        ]
        nodes = []
        for i, (head, sub) in enumerate(stages):
            x = 1.1 + i * 2.6
            nodes.append(
                ax.text(
                    x,
                    1.15,
                    head,
                    ha="center",
                    va="center",
                    fontsize=slides.SIZE["heading"],
                    weight=slides.MEDIUM,
                    color=slides.INK,
                    bbox=dict(boxstyle="round,pad=0.6", fc=slides.PAPER, ec="none"),
                )
            )
            ax.text(x, 0.35, sub, ha="center", va="center", fontsize=slides.SIZE["small"], color=slides.MUTED)
        for a, b in zip(nodes, nodes[1:]):
            slides.connect(ax, a, b)
        slides.text(
            fig,
            0.6,
            2.85,
            "\n".join(
                [
                    "•  Logistic regression: the score is the log-odds of surgery, so chance = 1 / (1 + exp(−score)).",
                    f"•  Trained on {n_train + len(test):,} patient snapshots, taken quarterly from 2024-07 to 2025-07.",
                ]
            ),
            role="body",
            linespacing=1.8,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, f"Better than chance: {recall / (k / len(test)):.1f}× a random list")
        slides.card(fig, 0.6, 3.45, 3.85, 2.3, f"{m['AUC']:.2f}", "AUC, against 0.50 for random ranking", slides.ACCENT)
        slides.card(
            fig,
            4.74,
            3.45,
            3.85,
            2.3,
            pct(recall, 0),
            f"of surgeries caught in the top {k:,}, against {pct(k / len(test))} at random",
            slides.ACCENT,
        )
        slides.card(
            fig,
            8.88,
            3.45,
            3.85,
            2.3,
            pct(m["Precision"]),
            f"surgery rate on that list, against {pct(m['Observed surgery rate'])} overall",
            slides.ACCENT,
        )
        evaluation.plot_gains(slides.box(fig, 0.9, 1.35, 6.4, 1.9), g, k)
        slides.text(
            fig,
            7.9,
            3.05,
            "\n".join(
                [
                    f"Refit on 2024-07 to 2025-04, tested on the {len(test):,}",
                    "patients of 2025-07: a quarter it never saw.",
                    "",
                    f"The {k:,} is only the evaluation list size,",
                    "about the size of past waves. How many",
                    "to contact is section 4.",
                ]
            ),
            role="small",
            linespacing=1.5,
        )
        slides.emit(pdf, fig)

        fig = slides.page(
            pdf, f"Predicted {pct(m['Mean predicted risk'], 2)}, observed {pct(m['Observed surgery rate'], 2)}"
        )
        evaluation.plot_calibration(slides.box(fig, 0.9, 1.3, 7.9, slides.TOP - 1.3), cal)
        slides.text(
            fig,
            9.3,
            slides.TOP,
            "\n".join(
                [
                    "Each point is a tenth of the",
                    "test patients, sorted by risk.",
                    "",
                    f"Top decile: {pct(cal.predicted.iloc[-1])} predicted,",
                    f"{pct(cal.observed.iloc[-1])} observed.",
                    "",
                    "Why it matters: the cut-off in",
                    "section 4 uses the actual",
                    "probabilities, not just the order.",
                ]
            ),
            role="body",
            linespacing=1.6,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "What the list looks like")
        top = targets.head(10)
        yes = lambda v: "Yes" if v else ""
        rows = [
            [
                f"{r.rank:,}",
                r.patient_token[:8],
                pct(r.score),
                r.program_recommended.replace("_", " "),
                r.allowed_channels.replace("|", ", "),
                yes(r.prior_consult),
            ]
            for r in top.itertuples()
        ]
        slides.table(
            fig,
            0.6,
            1.5,
            12.13,
            slides.TOP - 1.5,
            ["Rank", "Patient", "Risk", "Program", "Channels", "Prior consult"],
            rows,
            [0.08, 0.14, 0.1, 0.24, 0.28, 0.16],
        )
        slides.text(
            fig,
            0.6,
            1.25,
            "The first 10 rows of out/targets.csv. Ranks skip patients randomized to the holdout. Patient tokens are shortened.",
            role="small",
        )
        slides.emit(pdf, fig)

        # 3. Which program
        slides.divider(pdf, 3, *SECTIONS[2])
        fig = slides.page(pdf, "Weight management where obesity is a factor")
        slides.text(
            fig,
            0.6,
            slides.TOP,
            f"Rule: BMI ≥ {A.WM_BMI_THRESHOLD} or an obesity diagnosis → weight management.\nEveryone else, including patients with no BMI on file → physical therapy.",
            role="lead",
            linespacing=1.5,
        )
        by = listed.groupby("tier").program_recommended.value_counts().unstack(fill_value=0)
        slides.table(
            fig,
            0.6,
            2.4,
            6.0,
            2.2,
            ["Tier", "Weight management", "Physical therapy"],
            [
                [f"{tier}", f"{by.loc[tier, A.PROGRAMS['WM']]:,}", f"{by.loc[tier, A.PROGRAMS['PT']]:,}"]
                for tier in (1, 2)
            ],
            [0.2, 0.4, 0.4],
        )
        slides.panel(
            fig,
            7.1,
            2.4,
            5.63,
            2.2,
            "What we can't yet tell you",
            "Two-thirds of patients have no BMI on file and default to PT. History can't pick the program per patient; the pilot could randomize PT vs weight management among patients eligible for both.",
            width=52,
            role="small",
        )
        slides.text(fig, 0.6, 1.9, "BMI 30+ carries about twice the surgery risk of BMI under 25.", role="body")
        slides.emit(pdf, fig)

        # 4. How many to contact
        slides.divider(pdf, 4, *SECTIONS[3])
        fig = slides.page(pdf, "The rule: savings must beat the cost of an outreach")
        slides.text(
            fig,
            0.6,
            slides.TOP,
            "P(surgery) × enrollment lift × surgery reduction × surgery cost  >  outreach cost",
            role="lead",
        )
        slides.table(
            fig,
            0.6,
            1.6,
            12.13,
            slides.TOP - 1.0 - 1.6,
            ["Input", "Value", "Source"],
            list_depth.rule_rows(be),
            [0.3, 0.12, 0.58],
        )
        slides.text(
            fig,
            0.6,
            1.3,
            f"Break-even: contact anyone whose 12-month surgery risk is above {pct(be, 2)}. "
            f"Above it, an outreach saves more than its ${A.CONTACT_COST:,.0f} cost; below it, less.",
            role="body",
        )
        slides.emit(pdf, fig)

        curve = list_depth.value_curve(ranked.score, e)
        fig = slides.page(pdf, f"The list is worth most at {n1:,} patients")
        list_depth.plot_value(slides.box(fig, 0.9, 1.3, 7.9, slides.TOP - 1.3), curve, n1)
        slides.text(
            fig,
            9.3,
            slides.TOP,
            "\n".join(
                [
                    "Tier-1 contacts are expected to",
                    f"prevent about {s['surgeries_avoided']:.1f} surgeries",
                    f"this wave: {slides.money(s['net_value'])} net of costs.",
                    "",
                    "That's thin for one wave.",
                    "The value compounds with",
                    "repeated waves, and depends",
                    "on Meridian's real costs and",
                    "the surgery reduction (section 6).",
                ]
            ),
            role="body",
            linespacing=1.6,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, f"Contact {n1:,}, test {n2:,}, hold out a share")
        slides.table(
            fig,
            0.6,
            3.1,
            12.13,
            slides.TOP - 3.1,
            list_depth.TIER_HEADER,
            list_depth.tier_rows(ranked, s),
            list_depth.TIER_WIDTHS,
        )
        slides.text(
            fig,
            0.6,
            2.7,
            "\n".join(
                [
                    "•  Randomized within tier and program, with a fixed seed: the assignment is in out/ranking.csv (arm).",
                    f"•  The learning band costs about {slides.money(s['band_cost'])} in expected value: the price of checking the cut-off.",
                    f"•  {contact.sum():,} patients to contact ≈ {contact.sum() * attempts:,.0f} contact attempts at past rates ({attempts:.1f} per patient).",
                ]
            ),
            role="body",
            linespacing=1.8,
        )
        slides.emit(pdf, fig)

        # 5. How to measure
        slides.divider(pdf, 5, *SECTIONS[4])
        fig = slides.page(pdf, "Why history can't settle the surgery question")
        draw_dag(slides.box(fig, 0.4, 1.25, 7.0, slides.TOP - 1.05), randomized=False)
        evidence = [
            (
                "The raw comparison misleads",
                f"Contacted patients had surgery {pct(raw['1']['surgery_12m'] / 100)} of the time vs {pct(raw['0']['surgery_12m'] / 100)} for the rest. At face value, outreach causes surgery.",
            ),
            (
                "A negative control flags the bias",
                f"They also had more surgeries in the first 30 days ({pct(raw['1']['nc_surgery_30d'] / 100, 2)} vs {pct(raw['0']['nc_surgery_30d'] / 100, 2)}), before any program could act. They were already in the pipeline.",
            ),
            (
                "Adjusting isn't enough",
                f"After 37 covariates and the negative-control correction: {cal_rd['mean']:+.2f} points [{cal_rd['lo89']:+.2f}, {cal_rd['hi89']:+.2f}]. Indistinguishable from zero, and wider than any realistic effect.",
            ),
        ]
        for i, (title, body) in enumerate(evidence):
            slides.panel(fig, 7.6, slides.TOP - (i + 1) * 1.62 + 0.12, 5.13, 1.5, title, body, width=61, role="small")
        slides.text(
            fig,
            0.6,
            1.0,
            f"Confounder: something that drives both who got contacted and who had surgery. We can estimate the enrollment lift (+{A.ENROLLMENT_LIFT:.1%}); not the surgery effect.",
            role="small",
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "A random holdout closes the backdoor")
        draw_dag(slides.box(fig, 0.4, 1.25, 7.0, slides.TOP - 1.05), randomized=True)
        ax = slides.box(fig, 7.7, 3.75, 5.0, 1.9)
        ax.set(xlim=(0, 10), ylim=(0, 3))
        ax.axis("off")
        ax.plot([0.5, 9.5], [1.6, 1.6], color=slides.LIGHT, lw=3)
        for x, top_label, bottom_label in [
            (0.8, "Wave starts", "contact vs holdout"),
            (4.6, "90 days", f"enrollment (detects {pct(s['mde_all'])})"),
            (8.9, "12 months", "surgery (large effects only)"),
        ]:
            ax.plot([x], [1.6], "o", color=slides.ACCENT, ms=12)
            ax.text(
                x, 2.35, top_label, ha="center", fontsize=slides.SIZE["small"], weight=slides.MEDIUM, color=slides.INK
            )
            ax.text(x, 0.75, bottom_label, ha="center", fontsize=slides.SIZE["caption"], color=slides.MUTED)
        slides.text(
            fig,
            7.7,
            3.45,
            "\n".join(
                [
                    "•  A coin flip decides who's contacted, so",
                    "   nothing else can explain the difference.",
                    f"•  Historical lift: {pct(A.ENROLLMENT_LIFT)}; the pilot detects {pct(s['mde_all'])}.",
                    "•  A 0.3-point surgery drop needs about",
                    "   34,000 patients per arm: the surgery",
                    "   readout can only rule out large effects.",
                    "•  The one ask: Meridian's own campaigns",
                    "   skip holdout patients for 12 months.",
                ]
            ),
            role="body",
            linespacing=1.5,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Measure every wave the same way")
        slides.table(
            fig,
            0.6,
            3.05,
            12.13,
            slides.TOP - 3.05,
            ["Question", "Compare", "Read out", "One wave can detect"],
            [
                [
                    "Does contact raise enrollment?",
                    "Contacted vs held out, all tiers",
                    "90 days",
                    f"{pct(s['mde_all'])} lift (historical: {pct(A.ENROLLMENT_LIFT)})",
                ],
                [
                    "Is the cut-off in the right place?",
                    "Lift in the learning band vs tier 1",
                    "90 days",
                    f"{pct(s['mde_t1'])} lift in tier 1 alone",
                ],
                [
                    "Does it prevent surgery?",
                    "Surgery while covered, contacted vs held out",
                    "12 months",
                    "Large effects only: pool waves",
                ],
                [
                    "Which program works better?",
                    "PT vs weight management, randomized if eligible for both",
                    "90 days, 12 months",
                    "Optional next step",
                ],
            ],
            [0.24, 0.38, 0.14, 0.24],
        )
        measure = [
            (
                "Keep the holdout clean",
                "Meridian's own campaigns skip holdout patients for 12 months. Compare by assignment, not by who answered.",
            ),
            (
                "Randomize every wave",
                "Within tier and program, with a fixed seed, recorded in out/ranking.csv. A new holdout each wave keeps evidence building.",
            ),
            (
                "Feed results back",
                "Replace the assumed enrollment lift and surgery reduction in assumptions.py with measured values, then re-score.",
            ),
        ]
        for i, (title, body) in enumerate(measure):
            slides.panel(fig, 0.6 + i * 4.14, 0.85, 3.85, 1.95, title, body, width=40, role="small")
        slides.emit(pdf, fig)

        # 6. Assumptions
        slides.divider(pdf, 6, *SECTIONS[5])
        fig = slides.page(pdf, "One assumption moves the list most")
        list_depth.plot_sensitivity(
            slides.box(fig, 0.9, 1.3, 7.9, slides.TOP - 1.3), list_depth.list_size_by_reduction(ranked.score), n1
        )
        sens = targeting.sensitivity(ranked.score)
        slides.text(
            fig,
            9.3,
            slides.TOP,
            "\n".join(
                [
                    "Patients above break-even as",
                    "the surgery reduction varies.",
                    "",
                    "It's the number section 5",
                    "showed history can't measure.",
                    "",
                    "Across every assumption range,",
                    f"the list runs from {sens.above_break_even.min():,}",
                    f"to {sens.above_break_even.max():,} patients. The rule",
                    "stays; only the cut-off moves.",
                ]
            ),
            role="body",
            linespacing=1.6,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Every assumption, with its source")
        slides.table(
            fig,
            0.6,
            1.5,
            12.13,
            slides.TOP - 1.5,
            ["Assumption", "Value", "How solid", "Source"],
            assumption_rows(attempts, int(contact.sum())),
            [0.29, 0.17, 0.13, 0.41],
        )
        slides.text(
            fig,
            0.6,
            1.3,
            "All live in assignment/assumptions.py: change one, re-run the scoring step, and the list updates in seconds.\nMeridian to confirm: capacity, outreach cost, contracted surgery cost.",
            role="small",
            linespacing=1.4,
        )
        slides.emit(pdf, fig)

        # Close


def run(
    data_dir: Path = Path("data"), out_dir: Path = Path("out"), path: Path = Path("out/presentation/final.pdf")
) -> None:
    d = features.load(data_dir)
    train, test = evaluation.test_predictions(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    deck(
        path,
        d,
        pd.read_csv(out_dir / "ranking.csv"),
        pd.read_csv(out_dir / "targets.csv"),
        test,
        len(train),
        json.loads(CAUSAL.read_text()),
        pd.read_csv(EFFECTS),
        pickle.loads(MODEL_PATH.read_bytes()),
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    run()
