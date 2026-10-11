"""Stage 4 — Assertion + outpatient codability filter (design doc Section 5.4).

OpenMed resolves the context axes (negation, temporality, certainty,
experiencer, severity/laterality); the outpatient rules table below is ours:

  negated                          -> drop
  experiencer != patient           -> drop
  hypothetical                     -> drop
  uncertain CONDITION              -> keep, flagged "uncertain_dx"
                                      (outpatient: never code an uncertain dx;
                                      the reasoner codes documented symptoms)
  historical                       -> keep, flagged "historical" (Z-code path)
  medication entities              -> keep (supports "actively managed" and
                                      use-additional-code rules like Z79.4)
"""

from __future__ import annotations

from typing import Any

from autocoder.models import CodableFact, Section
from autocoder.pipeline.ner import MEDICATION_LABELS
from autocoder.pipeline.sections import section_for_offset

# Sections whose facts are dropped outright for outpatient dx coding.
_FAMILY_SECTIONS = {"family_history"}


def build_facts(
    text: str,
    entities: list[dict[str, Any]],
    sections: list[Section],
) -> tuple[list[CodableFact], list[CodableFact]]:
    """Return (codable_facts, dropped_facts) with full context fields on both."""
    from openmed.clinical import (
        resolve_experiencer,
        resolve_temporality,
        resolve_uncertainty,
        scan_context_cues,
    )
    from openmed.clinical.severity_laterality import extract_severity_laterality

    finding_spans = [
        {"start": e["start"], "end": e["end"], "label": e["label"]}
        for e in entities
        if e["label"] not in MEDICATION_LABELS
    ]
    modifiers = extract_severity_laterality(finding_spans, text)
    modifiers_by_target: dict[tuple[int, int], dict[str, str]] = {}
    for mod in modifiers:
        target = tuple(mod["target_offset"])
        modifiers_by_target.setdefault(target, {})[str(mod["kind"])] = str(
            mod["normalized"]
        )

    # Build span payloads first, then scan ConText cues once for all spans so
    # the resolvers see properly scoped negation/temporality/uncertainty cues
    # ("denies chest pain" must resolve negated).
    payloads = []
    for ent in entities:
        section = section_for_offset(sections, ent["start"])
        payloads.append(
            {
                "text": ent["text"],
                "document_text": text,
                "start": ent["start"],
                "end": ent["end"],
                "section": section.label if section else "body",
            }
        )
    cue_hits = scan_context_cues(text, payloads)

    codable: list[CodableFact] = []
    dropped: list[CodableFact] = []
    for i, (ent, span_payload) in enumerate(zip(entities, payloads), start=1):
        section_label = span_payload["section"]
        hits = cue_hits[span_payload]
        section_experiencer = (
            "family" if section_label in _FAMILY_SECTIONS else None
        )
        fact = CodableFact(
            fact_id=f"fact_{i}",
            entity_text=ent["text"],
            label=ent["label"],
            span=(ent["start"], ent["end"]),
            section=section_label,
            negation=_negation_from_hits(text, hits),
            temporality=str(resolve_temporality(span_payload, hits)),
            certainty=str(resolve_uncertainty(span_payload, hits)),
            experiencer=str(
                resolve_experiencer(
                    text, span_payload, section_experiencer=section_experiencer
                ).experiencer
            ),
            attributes=modifiers_by_target.get((ent["start"], ent["end"]), {}),
            ner_score=ent["score"],
        )
        fact = _apply_outpatient_rules(fact)
        if "dropped" in fact.flags:
            dropped.append(fact)
        else:
            codable.append(fact)
    return codable, dropped


def _negation_from_hits(text: str, hits) -> str:
    """Negation from per-span scoped ConText cues.

    ``scan_context_cues`` already restricted these hits to cues whose scope
    reaches the span, so any negation-category cue means the mention is
    negated. (We bypass ``resolve_negation``'s clause-level parity count,
    which mis-fires on sentences with two independent denials such as
    "Denies chest pain, denies polyuria".)
    """
    for hit in hits:
        if getattr(hit, "category", None) == "negation":
            return "negated"
    return "affirmed"


def _apply_outpatient_rules(fact: CodableFact) -> CodableFact:
    """Apply the outpatient codability table; mutates flags and returns fact."""
    is_medication = fact.label in MEDICATION_LABELS

    if fact.negation == "negated":
        fact.flags.append("dropped")
        fact.flags.append("reason:negated")
    elif fact.experiencer != "patient":
        fact.flags.append("dropped")
        fact.flags.append(f"reason:experiencer_{fact.experiencer}")
    elif fact.temporality == "hypothetical":
        fact.flags.append("dropped")
        fact.flags.append("reason:hypothetical")
    elif fact.certainty == "uncertain" and not is_medication:
        # Outpatient guideline: never code an uncertain diagnosis. Keep the
        # fact visible so the reasoner codes the documented symptoms instead.
        fact.flags.append("uncertain_dx")
    elif fact.temporality == "historical":
        fact.flags.append("historical")

    if is_medication and "dropped" not in fact.flags:
        fact.flags.append("medication")
    return fact
