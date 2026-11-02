"""How far down the list? A PDF deck on how many patients Meridian should contact, in out/list_depth/.

Reads the scored 2026-07 population from out/ranking.csv (run `main.py score` first) and recomputes every number
with the targeting rules, so the deck can't drift from the list.

    uv run python -m assignment.list_depth
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from assignment import assumptions as A
from assignment import slides, targeting

REDUCTIONS = np.linspace(0.05, 0.6, 56)


def value_curve(score: pd.Series, e: dict) -> pd.DataFrame:
    """Expected net value of each contact and of the whole list, contacting in risk order."""
    p = np.sort(score.to_numpy())[::-1][:6_000]  # the plotted range; value only falls beyond it
    per_contact = p * targeting.value_per_risk(e) - e["CONTACT_COST"]
    return pd.DataFrame(
        {"rank": np.arange(1, len(p) + 1), "per_contact": per_contact, "cumulative": per_contact.cumsum()}
    )


def list_size_by_reduction(score: pd.Series) -> pd.DataFrame:
    """Patients above break-even across surgery-reduction values, at each end of the contact-cost range."""
    return pd.DataFrame(
        {
            cost: [
                int(
                    (
                        score
                        > targeting.break_even(
                            targeting.economics(SURGERY_REDUCTION_PER_ENROLLMENT=r, CONTACT_COST=cost)
                        )
                    ).sum()
                )
                for r in REDUCTIONS
            ]
            for cost in A.SENSITIVITY["CONTACT_COST"]
        },
        index=REDUCTIONS,
    )


def plot_value(ax: plt.Axes, curve: pd.DataFrame, n1: int) -> None:
    ax.plot(
        curve["rank"],
        curve.cumulative / 1000,
        color=slides.ACCENT,
        lw=2,
        label="Expected net value of contacting the top N",
    )
    ax.axvline(n1, color=slides.GRAY, lw=1, ls="--")
    ax.annotate(
        f"Peaks at {n1:,}: the break-even rank",
        (n1, curve.cumulative.iloc[n1 - 1] / 1000),
        xytext=(14, 8),
        textcoords="offset points",
        fontsize=slides.SIZE["caption"],
    )
    ax.axhline(0, color=slides.LIGHT, lw=1)
    ax.set(
        xlabel="Patients contacted, highest risk first", ylabel="Expected net value ($ thousands)", xlim=(0, len(curve))
    )
    slides.style(ax)


def plot_sensitivity(ax: plt.Axes, sizes: pd.DataFrame, n1: int) -> None:
    for cost, color in zip(sizes.columns, (slides.GRAY, slides.ACCENT)):
        ax.plot(sizes.index * 100, sizes[cost], color=color, lw=2, label=f"${cost} per contact")
    base = A.SURGERY_REDUCTION_PER_ENROLLMENT * 100
    ax.plot([base], [n1], "o", color=slides.ACCENT, ms=7)
    ax.annotate(
        f"Base case: {n1:,}", (base, n1), xytext=(10, -18), textcoords="offset points", fontsize=slides.SIZE["caption"]
    )
    ax.set(xlabel="Surgery reduction per enrollment (%)", ylabel="Patients above break-even", ylim=(0, None))
    slides.style(ax)


def summary(ranked: pd.DataFrame) -> dict:
    """Headline numbers for the list: tiers, arms, expected value, and the pilot's detectable effect."""
    e = targeting.economics()
    n1, n2 = targeting.tier_sizes(ranked.score, e)
    arms = ranked[ranked.tier > 0].groupby(["tier", "arm"]).size()
    contact, holdout = arms.xs("contact", level="arm"), arms.xs("holdout", level="arm")
    tier1_contacts = ranked[(ranked.tier == 1) & (ranked.arm == "contact")]
    return {
        "e": e, "be": targeting.break_even(e), "n1": n1, "n2": n2, "contact": contact, "holdout": holdout,
        "surgeries_avoided": (tier1_contacts.score * e["ENROLLMENT_LIFT"] * e["SURGERY_REDUCTION_PER_ENROLLMENT"]).sum(),
        "net_value": tier1_contacts.expected_net_value.sum(),
        "band_cost": -ranked.loc[(ranked.tier == 2) & (ranked.arm == "contact"), "expected_net_value"].sum(),
        "mde_all": targeting.detectable_effect(contact.sum(), holdout.sum()),
        "mde_t1": targeting.detectable_effect(contact[1], holdout[1]),
    }  # fmt: skip


def rule_rows(be: float) -> list[list[str]]:
    """The break-even inputs with their values and sources."""
    return [
        [
            "Enrollment lift",
            f"{A.ENROLLMENT_LIFT:.1%}",
            "Estimated from past campaigns (doubly robust, 89% interval 10.6–12.4%)",
        ],
        [
            "Surgery reduction per enrollment",
            f"{A.SURGERY_REDUCTION_PER_ENROLLMENT:.0%}",
            "Literature assumption: the data can't test it",
        ],
        ["Surgery cost", f"${A.SURGERY_COST:,}", "Medicare 90-day episode, fusion/decompression blend"],
        ["Outreach cost", f"${A.CONTACT_COST}", "Vi call: amortized cost per patient, same for every channel ($5–$10)"],
        [
            "Break-even risk",
            f"{be:.2%}",
            f"${A.CONTACT_COST} ÷ ({A.ENROLLMENT_LIFT} × {A.SURGERY_REDUCTION_PER_ENROLLMENT} × ${A.SURGERY_COST:,})",
        ],
    ]


def tier_rows(ranked: pd.DataFrame, s: dict) -> list[list[str]]:
    """One row per tier: who, randomization, arm counts, program mix, prior consults."""
    mix = (
        ranked[ranked.tier > 0]
        .groupby("tier")
        .agg(wm=("program_recommended", lambda p: (p == A.PROGRAMS["WM"]).sum()), consult=("prior_consult", "sum"))
    )
    row = lambda t, who, share: [
        str(t),
        who,
        f"{1 - share:.0%} / {share:.0%}",
        f"{s['contact'][t]:,}",
        f"{s['holdout'][t]:,}",
        f"{mix.wm[t]:,}",
        f"{int(mix.consult[t]):,}",
    ]
    return [
        row(1, f"Above break-even ({s['n1']:,})", A.TIER1_HOLDOUT_SHARE),
        row(2, f"Learning band ({s['n2']:,})", A.TIER2_HOLDOUT_SHARE),
        ["—", f"Below rank {s['n1'] + s['n2']:,}", "Not contacted this wave", "", "", "", ""],
    ]


TIER_HEADER = ["Tier", "Who", "Contact / hold out", "Contacted", "Held out", "Weight mgmt", "Prior consult"]
TIER_WIDTHS = [0.06, 0.24, 0.2, 0.12, 0.12, 0.13, 0.13]


def deck(path: Path, ranked: pd.DataFrame) -> None:
    s = summary(ranked)
    e, be, n1, n2, contact, holdout = s["e"], s["be"], s["n1"], s["n2"], s["contact"], s["holdout"]
    surgeries_avoided, net_value, band_cost, mde_all, mde_t1 = (
        s["surgeries_avoided"],
        s["net_value"],
        s["band_cost"],
        s["mde_all"],
        s["mde_t1"],
    )
    score = ranked.score
    curve, sizes, sens = value_curve(score, e), list_size_by_reduction(score), targeting.sensitivity(score)

    with PdfPages(path, metadata={"Title": "Vi Labs · Meridian · How far down the list"}) as pdf:
        slides.section(pdf, "How far down the list")
        fig = slides.page(pdf, "How many patients\nshould Meridian contact?", dark=True)
        slides.text(
            fig,
            0.6,
            3.85,
            f"{n1:,} whose expected savings beat the cost of contact, plus a {n2:,}-patient learning band.",
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
            f"{n1:,}",
            f"patients above the {be:.2%} break-even risk: contact them",
            slides.ACCENT,
        )
        slides.card(
            fig,
            4.74,
            3.45,
            3.85,
            2.3,
            f"{n2:,}",
            "more just below it, randomized 50/50 to learn whether outreach works there too",
        )
        slides.card(
            fig,
            8.88,
            3.45,
            3.85,
            2.3,
            f"{contact.sum():,}",
            f"contacts in total; {holdout.sum():,} held out to measure the effect",
        )
        slides.text(fig, 0.6, 2.85, "What this means", role="lead")
        slides.text(
            fig,
            0.6,
            2.35,
            "\n".join(
                [
                    f"•  The {contact[1]:,} tier-1 contacts are expected to prevent about {surgeries_avoided:.1f} surgeries this wave: {slides.money(net_value)} net of contact costs.",
                    f"•  The learning band costs about {slides.money(band_cost)} in expected value: the price of checking the cut-off is in the right place.",
                    "•  The number depends most on how much enrollment reduces surgery, which the data can't measure.",
                    "   The pilot is built around that.",
                ]
            ),
            role="body",
            linespacing=1.8,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "The rule: contact while savings beat cost")
        slides.text(
            fig,
            0.6,
            slides.TOP,
            "P(surgery) × enrollment lift × surgery reduction × surgery cost  >  contact cost",
            role="lead",
        )
        rows = rule_rows(be)
        slides.table(
            fig, 0.6, 1.6, 12.13, slides.TOP - 1.0 - 1.6, ["Input", "Value", "Source"], rows, [0.3, 0.12, 0.58]
        )
        slides.text(
            fig,
            0.6,
            1.3,
            "P(surgery) comes from the Surgery Risk Model, calibrated out-of-time (2.16% predicted vs 2.16% observed).",
            role="body",
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, f"The list is worth most at {n1:,} patients")
        plot_value(slides.box(fig, 0.9, 1.3, 7.9, slides.TOP - 1.3), curve, n1)
        slides.text(
            fig,
            9.3,
            slides.TOP,
            "\n".join(
                [
                    "Each contact adds its expected",
                    "savings minus its cost.",
                    "",
                    f"Above {be:.2%} risk, contacts",
                    "pay for themselves; below it,",
                    "each one subtracts value.",
                    "",
                    f"Peak: {slides.money(curve.cumulative.iloc[n1 - 1])} expected",
                    "net value, if every listed",
                    "patient were contacted.",
                ]
            ),
            role="body",
            linespacing=1.6,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "One assumption moves the answer most")
        plot_sensitivity(slides.box(fig, 0.9, 1.3, 7.9, slides.TOP - 1.3), sizes, n1)
        lo, hi = sizes.loc[np.isclose(sizes.index, 0.10)].iloc[0], sizes.loc[np.isclose(sizes.index, 0.50)].iloc[0]
        slides.text(
            fig,
            9.3,
            slides.TOP,
            "\n".join(
                [
                    "Patients above break-even as",
                    "the surgery reduction varies:",
                    "",
                    f"10% reduction: {lo[A.CONTACT_COST]:,}",
                    f"50% reduction: {hi[A.CONTACT_COST]:,}",
                    "",
                    "Halving the contact cost to $5",
                    f"lifts the base case to {sizes.loc[np.isclose(sizes.index, A.SURGERY_REDUCTION_PER_ENROLLMENT), 5].iloc[0]:,}.",
                    "",
                    "The reduction is a literature",
                    "assumption; the data can't test it.",
                ]
            ),
            role="body",
            linespacing=1.6,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Every assumption at its low and high value")
        labels = {
            "base": "Base case",
            "SURGERY_COST": "Surgery cost",
            "CONTACT_COST": "Contact cost",
            "SURGERY_REDUCTION_PER_ENROLLMENT": "Surgery reduction",
            "ENROLLMENT_LIFT": "Enrollment lift",
        }
        fmt_value = lambda name, v: (
            ""
            if name == "base"
            else f"${v:,.0f}"
            if "COST" in name
            else f"{v:.0%}"
            if "REDUCTION" in name
            else f"{v:.1%}"
        )
        rows = [
            [labels[r.assumption], fmt_value(r.assumption, r.value), f"{r.break_even:.2%}", f"{r.above_break_even:,}"]
            for r in sens.itertuples()
            if not (r.assumption != "base" and r.value == getattr(A, r.assumption))
        ]
        slides.table(
            fig,
            0.6,
            0.85,
            8.0,
            slides.TOP - 0.85,
            ["Assumption", "Value", "Break-even risk", "Patients to contact"],
            rows,
            [0.33, 0.17, 0.24, 0.26],
        )
        slides.text(
            fig,
            9.0,
            slides.TOP,
            "\n".join(
                [
                    "Read across a row to see what",
                    "changes if one input is wrong.",
                    "",
                    "The list ranges from",
                    f"{sens.above_break_even.min():,} to {sens.above_break_even.max():,} patients.",
                    "",
                    "The rule stays the same; only",
                    "the cut-off moves. Re-running",
                    "the scoring step with a new",
                    "assumption updates the list.",
                ]
            ),
            role="body",
            linespacing=1.6,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "Tiers: contact, and learn as we go")
        slides.table(fig, 0.6, 3.1, 12.13, slides.TOP - 3.1, TIER_HEADER, tier_rows(ranked, s), TIER_WIDTHS)
        slides.text(
            fig,
            0.6,
            2.7,
            "\n".join(
                [
                    "•  Randomized within tier and program with a fixed seed. Held-out patients get usual care,",
                    "   and Meridian's own campaigns must skip them for 12 months.",
                    f"•  Primary outcome: enrollment within 90 days. Detectable lift: {mde_all:.1%} pooled, {mde_t1:.1%} in tier 1 alone,",
                    f"   against {A.ENROLLMENT_LIFT:.1%} measured historically.",
                    "•  Surgery can't be measured this way: a 0.3-point drop needs about 34,000 patients per arm.",
                ]
            ),
            role="body",
            linespacing=1.7,
        )
        slides.emit(pdf, fig)

        fig = slides.page(pdf, "What Meridian needs to confirm")
        for i, (title, body) in enumerate(
            [
                (
                    "Capacity",
                    f"The list asks for {contact.sum():,} contacts this wave. Past waves reached 2,189–3,183 patients. If Meridian can do fewer, shrink the learning band first, then tier 1 from the bottom.",
                ),
                (
                    "Contact cost",
                    f"${A.CONTACT_COST} per contacted patient is the conservative end of $5–$10. At $5, the list grows to {sens.loc[(sens.assumption == 'CONTACT_COST') & (sens.value == 5), 'above_break_even'].iloc[0]:,}.",
                ),
                (
                    "Surgery cost",
                    f"${A.SURGERY_COST:,} is a published Medicare blend. Meridian's contracted rates would replace it: at ${A.SENSITIVITY['SURGERY_COST'][0]:,} the list is {sens.loc[(sens.assumption == 'SURGERY_COST') & (sens.value == A.SENSITIVITY['SURGERY_COST'][0]), 'above_break_even'].iloc[0]:,}.",
                ),
            ]
        ):
            x = 0.6 + i * 4.14
            slides.panel(fig, x, 2.6, 3.85, slides.TOP - 2.6, title, body)
        slides.emit(pdf, fig)

        fig = slides.page(pdf, f"Contact {n1:,}, learn from {n2:,} more", dark=True)
        slides.text(
            fig,
            0.6,
            3.85,
            "\n".join(
                [
                    f"•  Contact everyone above {be:.2%} risk: {n1:,} patients, {contact[1]:,} after the holdout.",
                    f"•  Randomize the next {n2:,} 50/50 to test whether the cut-off is in the right place.",
                    "•  Re-run the list each wave with updated assumptions; the rule stays, the cut-off moves.",
                ]
            ),
            role="subtitle",
            color=slides.BEIGE,
            linespacing=1.8,
        )
        slides.emit(pdf, fig)


def run(ranking: Path = Path("out/ranking.csv"), out_dir: Path = Path("out/list_depth")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    deck(out_dir / "how_far_down.pdf", pd.read_csv(ranking))
    print(f"wrote {out_dir / 'how_far_down.pdf'}")


if __name__ == "__main__":
    run()
