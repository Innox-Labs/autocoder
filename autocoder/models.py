"""Typed contracts between pipeline stages (prototype design doc, Section 6).

Offsets always index into ``NormalizedDocument.text``.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

ENCOUNTER_TYPES = ("office_visit", "clinic", "urgent_care")


class NormalizedDocument(BaseModel):
    doc_id: str
    text: str
    # For PDFs: OpenMed's char-offset -> page/bbox spans (kept opaque here).
    offset_map: Optional[list[dict[str, Any]]] = None
    date_of_service: date
    encounter_type: str

    @model_validator(mode="after")
    def _check_encounter(self) -> "NormalizedDocument":
        if self.encounter_type not in ENCOUNTER_TYPES:
            raise ValueError(
                f"encounter_type must be one of {ENCOUNTER_TYPES}, "
                f"got {self.encounter_type!r}"
            )
        return self


class Section(BaseModel):
    label: str  # canonical label, "other", or "body"
    raw_header: Optional[str] = None
    start: int
    end: int


class CodableFact(BaseModel):
    fact_id: str
    entity_text: str
    label: str  # CONDITION | MEDICATION | SYMPTOM | ...
    span: tuple[int, int]
    section: str
    negation: str = "affirmed"  # affirmed | negated
    temporality: str = "recent"  # recent | historical | hypothetical
    certainty: str = "certain"  # certain | uncertain
    experiencer: str = "patient"  # patient | family | other
    attributes: dict[str, str] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)  # ["uncertain_dx"], ["historical"]
    ner_score: float = 0.0


class Candidate(BaseModel):
    code: str  # dotted, e.g. "E11.621"
    description: str
    score: float
    source: str  # "lexical" | "semantic"
    billable: bool = True
    excludes1: list[str] = Field(default_factory=list)
    excludes2: list[str] = Field(default_factory=list)
    code_first: list[str] = Field(default_factory=list)
    use_additional: list[str] = Field(default_factory=list)
    parent: Optional[str] = None


class CodeAssignment(BaseModel):
    code: str
    release_version: str
    description: str = ""
    first_listed: bool = False
    rationale: str = ""
    evidence_spans: list[tuple[int, int]] = Field(min_length=1)
    supported_facts: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    rule_flags: list[str] = Field(default_factory=list)


class Abstention(BaseModel):
    reason: str
    fact_ids: list[str] = Field(default_factory=list)


class CodedChart(BaseModel):
    chart_id: str
    document: NormalizedDocument
    sections: list[Section] = Field(default_factory=list)
    facts: list[CodableFact] = Field(default_factory=list)
    candidates: dict[str, list[Candidate]] = Field(default_factory=dict)
    codes: list[CodeAssignment] = Field(default_factory=list)
    abstentions: list[Abstention] = Field(default_factory=list)
    routing: str = "needs_review"  # "auto" | "needs_review"
    routing_reasons: list[str] = Field(default_factory=list)
    pipeline_versions: dict[str, str] = Field(default_factory=dict)
