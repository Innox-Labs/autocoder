"""Golden-set evaluation harness (design doc Section 11).

Each golden case is a pair of files in the golden directory:
  <name>.txt   the synthetic note (NEVER real PHI)
  <name>.json  {"date_of_service": "...", "encounter_type": "...",
                "expected_codes": ["E11.9", ...], "first_listed": "E11.9"}

Metrics: code-level precision/recall/F1 (exact and 3-char category),
first-listed accuracy, abstention rate.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from autocoder.pipeline.intake import load_document
from autocoder.pipeline.reasoner import ReasonerClient
from autocoder.pipeline.run import code_chart
from autocoder.terminology.store import TerminologyStore


def run_golden(
    golden_dir: str | Path,
    store: TerminologyStore,
    client: ReasonerClient,
) -> dict:
    golden_dir = Path(golden_dir)
    cases = sorted(golden_dir.glob("*.json"))
    if not cases:
        raise FileNotFoundError(f"No golden cases (*.json) in {golden_dir}")

    tp = fp = fn = 0
    cat_tp = cat_fp = cat_fn = 0
    first_listed_hits = 0
    abstained = 0
    per_case = []

    for meta_path in cases:
        meta = json.loads(meta_path.read_text())
        note_path = meta_path.with_suffix(".txt")
        document = load_document(
            note_path,
            date_of_service=date.fromisoformat(meta["date_of_service"]),
            encounter_type=meta.get("encounter_type", "office_visit"),
        )
        chart = code_chart(document, store, client)

        expected = set(meta["expected_codes"])
        produced = {c.code for c in chart.codes}
        tp += len(expected & produced)
        fp += len(produced - expected)
        fn += len(expected - produced)

        expected_cat = {c.split(".")[0] for c in expected}
        produced_cat = {c.split(".")[0] for c in produced}
        cat_tp += len(expected_cat & produced_cat)
        cat_fp += len(produced_cat - expected_cat)
        cat_fn += len(expected_cat - produced_cat)

        produced_first = next(
            (c.code for c in chart.codes if c.first_listed), None
        )
        first_ok = produced_first == meta.get("first_listed")
        if first_ok:
            first_listed_hits += 1
        if chart.abstentions:
            abstained += 1

        per_case.append(
            {
                "case": meta_path.stem,
                "expected": sorted(expected),
                "produced": sorted(produced),
                "missing": sorted(expected - produced),
                "extra": sorted(produced - expected),
                "first_listed_ok": first_ok,
                "routing": chart.routing,
            }
        )

    return {
        "cases": len(cases),
        "exact": _prf(tp, fp, fn),
        "category": _prf(cat_tp, cat_fp, cat_fn),
        "first_listed_accuracy": round(first_listed_hits / len(cases), 3),
        "abstention_rate": round(abstained / len(cases), 3),
        "per_case": per_case,
    }


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }
