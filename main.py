"""Entrypoint.

uv run main.py train
uv run main.py score
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd

from assignment import features, model, targeting
from assignment.utils import validate_targets

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
OUT_DIR = Path("out")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
LIST_COLUMNS = [
    "patient_token",
    "program_recommended",
    "score",
    "rank",
    "tier",
    "expected_net_value",
    "allowed_channels",
    "prior_consult",
    "prior_enrollment_365d",
]


def train() -> None:
    """Fit the risk model on every labeled quarterly snapshot (2024-07 to 2025-07)."""
    panel = features.training_panel(features.load(DATA_DIR))
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps(model.train(panel)))
    print(f"trained on {len(panel):,} snapshots, {int(panel.y.sum()):,} surgeries -> {MODEL_PATH}")


def score() -> pd.DataFrame:
    """Score the 2026-07-01 population and write the contact list, full ranking (with arm) and sensitivity table."""
    d, t = features.load(DATA_DIR), features.EXTRACT
    snap = features.snapshot(d, t)
    snap["score"] = model.predict(pickle.loads(MODEL_PATH.read_bytes()), snap)
    snap = targeting.apply_exclusions(snap, d["enrollments"], d["outreach"], t)
    snap["program_recommended"] = targeting.recommend_program(snap)
    ranked = targeting.assign(snap.rename(columns={"prior_program": "prior_enrollment_365d"}))

    targets = ranked.loc[ranked.arm == "contact", LIST_COLUMNS]
    validate_targets(targets)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets.to_csv(OUT_DIR / "targets.csv", index=False)
    ranked[[*LIST_COLUMNS, "arm"]].to_csv(OUT_DIR / "ranking.csv", index=False)
    targeting.sensitivity(ranked.score).to_csv(OUT_DIR / "sensitivity.csv", index=False)

    arms = ranked[ranked.tier > 0].groupby(["tier", "arm"]).size()
    e = targeting.economics()
    print(f"eligible {len(ranked):,}; break-even risk {targeting.break_even(e):.2%}")
    print(arms.to_string())
    print(
        f"pilot detects an enrollment lift of {targeting.detectable_effect(arms.xs('contact', level='arm').sum(), arms.xs('holdout', level='arm').sum()):.1%}"
    )
    return targets


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Fit on data/, write model artifacts to artifacts/.")
    sub.add_parser("score", help="Score patients, write out/targets.csv.")
    if parser.parse_args().command == "train":
        train()
    else:
        score()


if __name__ == "__main__":
    cli()
