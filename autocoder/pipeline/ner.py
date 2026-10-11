"""Stage 3 — Clinical NER via OpenMed registry models (design doc Section 5.3).

Runs a disease/condition model and a pharma model over the note, merges
overlapping spans, and returns plain dict entities:
    {"text", "label", "start", "end", "score", "model"}
"""

from __future__ import annotations

from typing import Any

DEFAULT_MODELS = (
    "disease_detection_superclinical",
    "pharma_detection_superclinical",
)

# NER labels that name a drug rather than a clinical finding.
MEDICATION_LABELS = {"DRUG", "MEDICATION", "CHEM", "CHEMICAL", "TREATMENT"}


def extract_entities(
    text: str,
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    confidence_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    from openmed import analyze_text

    entities: list[dict[str, Any]] = []
    for model_name in models:
        result = analyze_text(
            text,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            # DeBERTa-v2 (the SuperClinical backbone) does not support SDPA
            # attention in current transformers; force the eager path.
            attn_implementation="eager",
        )
        for ent in result.entities:
            start = getattr(ent, "start", None)
            end = getattr(ent, "end", None)
            if start is None or end is None:
                continue
            entities.append(
                {
                    "text": ent.text,
                    "label": str(ent.label).upper(),
                    "start": int(start),
                    "end": int(end),
                    "score": float(getattr(ent, "confidence", 0.0) or 0.0),
                    "model": model_name,
                }
            )
    return merge_overlaps(entities)


def merge_overlaps(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate overlapping spans across models; keep the higher-scoring one.

    Same-label adjacent merging is already handled inside each model run by
    OpenMed's aggregation; here we only resolve cross-model overlap.
    """
    ordered = sorted(entities, key=lambda e: (e["start"], -(e["end"] - e["start"])))
    kept: list[dict[str, Any]] = []
    for ent in ordered:
        overlap = next(
            (k for k in kept if ent["start"] < k["end"] and k["start"] < ent["end"]),
            None,
        )
        if overlap is None:
            kept.append(ent)
        elif ent["score"] > overlap["score"]:
            kept[kept.index(overlap)] = ent
    return kept
