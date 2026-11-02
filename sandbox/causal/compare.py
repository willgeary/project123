"""Side-by-side estimates from every approach, plus an interval chart. Run after all approach scripts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
load = lambda name: json.loads((HERE / f"{name}.json").read_text())
A, B, C, E = load("a_gcomp"), load("b_subclass"), load("c_aipw"), load("e_landmark")

rows = []
for label, res in [("A. Outcome model (g-computation)", A), ("B. Propensity strata, Beta-Binomial", B), ("C. Doubly robust (AIPW)", C)]:
    for o in ["enroll_180d", "surgery_12m", "nc_surgery_30d", "nc_acute_12m", "nc_acute_pre"]:
        rows.append({"approach": label, "outcome": o, **res[o]["att_rd_pts"]})
rows.append({"approach": "A, first wave per patient only", "outcome": "surgery_12m", **A["surgery_12m"]["first_wave_only_att_rd_pts"]})
rows.append({"approach": "E. Landmark day 30", "outcome": "nc_surgery_30_90", **E["surgery_30_90"]["att_rd_pts"]})
rows.append({"approach": "E. Landmark day 30", "outcome": "surgery_90_365", **E["surgery_90_365"]["att_rd_pts"]})
rows.append({"approach": "E. Landmark, NC-calibrated", "outcome": "surgery_90_365", **E["calibrated_rd_90_365_pts"]})
table = pd.DataFrame(rows)
table.to_csv(HERE / "estimates.csv", index=False)
print(table.pivot_table(index="approach", columns="outcome", values="mean").round(2).to_string())

surgery = table[table.outcome.isin(["surgery_12m", "surgery_90_365"])].reset_index(drop=True)
controls = table[table.outcome.isin(["nc_surgery_30d", "nc_surgery_30_90"])].reset_index(drop=True)
fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), gridspec_kw={"height_ratios": [len(surgery), len(controls)]})
for ax, part, title, color in [(axes[0], surgery, "Effect of contact on surgery (percentage points)", "#2a78d6"),
                                (axes[1], controls, "Negative controls (true effect is 0)", "#52514e")]:
    labels = part.approach + "  ·  " + part.outcome.str.replace("_", " ")
    y = range(len(part))[::-1]
    ax.errorbar(part["mean"], list(y), xerr=[part["mean"] - part["lo89"], part["hi89"] - part["mean"]], fmt="o", color=color, ecolor=color, capsize=3, ms=6, lw=2)
    ax.axvline(0, color="#b4b2a9", lw=1, ls="--")
    ax.set_yticks(list(y), labels, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e3e6ea", lw=0.5)
axes[1].set_xlabel("ATT, posterior mean and 89% interval")
fig.tight_layout()
fig.savefig(HERE / "estimates.png", dpi=150)
print("wrote estimates.csv, estimates.png")
