"""Stage 7a — Deterministic rules engine, checks R1-R8 (design doc Section 9).

Pure functions over the terminology store for the release in force. The
reasoner proposes; these rules dispose. Violations either auto-fix (drop) or
are returned as messages for one self-correction round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass

from autocoder.models import CodableFact, CodeAssignment
from autocoder.terminology.store import TerminologyStore, code_in_expr


@dataclass
class Violation:
    rule: str  # "R1".."R8"
    message: str
    codes: list[str]
    auto_fixed: bool = False


def apply_rules(
    codes: list[CodeAssignment],
    facts: list[CodableFact],
    store: TerminologyStore,
    release_id: int,
) -> tuple[list[CodeAssignment], list[Violation]]:
    """Validate and auto-fix where safe. Returns (kept_codes, violations)."""
    violations: list[Violation] = []
    kept = list(codes)

    kept = _r1_validity(kept, store, release_id, violations)
    _r2_billable(kept, store, release_id, violations)
    _r3_excludes1(kept, store, release_id, violations)
    kept = _r4_redundancy(kept, store, release_id, violations)
    _r5_code_first(kept, store, release_id, violations)
    _r6_specificity(kept, facts, store, release_id, violations)
    _r7_first_listed(kept, facts, violations)
    kept = _r8_symptom_integral(kept, violations)

    for violation in violations:
        for code in kept:
            if code.code in violation.codes:
                code.rule_flags.append(f"{violation.rule}:{violation.message[:80]}")
    return kept, violations


def unresolved(violations: list[Violation]) -> list[Violation]:
    return [v for v in violations if not v.auto_fixed]


# ------------------------------------------------------------------ checks


def _r1_validity(codes, store, release_id, violations) -> list[CodeAssignment]:
    kept = []
    for code in codes:
        concept = store.get_concept(release_id, code.code)
        if concept is None:
            violations.append(
                Violation(
                    "R1",
                    f"{code.code} does not exist in this release.",
                    [code.code],
                )
            )
        else:
            code.description = code.description or concept.description
            kept.append(code)
    return kept


def _r2_billable(codes, store, release_id, violations) -> None:
    for code in codes:
        concept = store.get_concept(release_id, code.code)
        if concept and not concept.billable:
            children = store.children(release_id, code.code)
            options = ", ".join(c.code for c in children[:8] if c)
            violations.append(
                Violation(
                    "R2",
                    f"{code.code} is not billable; choose a specific child "
                    f"(e.g. {options}).",
                    [code.code],
                )
            )


def _r3_excludes1(codes, store, release_id, violations) -> None:
    for i, a in enumerate(codes):
        concept = store.get_concept(release_id, a.code)
        if concept is None:
            continue
        excludes = concept.rules.get("excludes1", [])
        for b in codes[i + 1 :]:
            for expr in excludes:
                for token in _code_tokens(expr):
                    if code_in_expr(b.code, token):
                        violations.append(
                            Violation(
                                "R3",
                                f"Excludes1: {a.code} and {b.code} cannot be "
                                f"coded together ({expr!r}). Keep one.",
                                [a.code, b.code],
                            )
                        )


def _r4_redundancy(codes, store, release_id, violations) -> list[CodeAssignment]:
    assigned = {c.code for c in codes}
    kept = []
    for code in codes:
        ancestors = set(store.ancestors(release_id, code.code))
        redundant_parents = ancestors & assigned
        kept.append(code)
        for parent in redundant_parents:
            violations.append(
                Violation(
                    "R4",
                    f"{parent} is an ancestor of assigned {code.code}; "
                    "dropping the less specific code.",
                    [parent],
                    auto_fixed=True,
                )
            )
    to_drop = {v.codes[0] for v in violations if v.rule == "R4"}
    return [c for c in kept if c.code not in to_drop]


def _r5_code_first(codes, store, release_id, violations) -> None:
    assigned = [c.code for c in codes]
    for code in codes:
        concept = store.get_concept(release_id, code.code)
        if concept is None:
            continue
        for expr in concept.rules.get("code_first", []):
            tokens = _code_tokens(expr)
            if tokens and not any(
                code_in_expr(other, token)
                for token in tokens
                for other in assigned
                if other != code.code
            ):
                violations.append(
                    Violation(
                        "R5",
                        f"{code.code} has a code-first note ({expr!r}) and no "
                        "matching etiology code is assigned.",
                        [code.code],
                    )
                )


def _r6_specificity(codes, facts, store, release_id, violations) -> None:
    facts_by_id = {f.fact_id: f for f in facts}
    for code in codes:
        lateralities = {
            facts_by_id[fid].attributes.get("laterality")
            for fid in code.supported_facts
            if fid in facts_by_id
        } - {None}
        if not lateralities:
            continue
        laterality = next(iter(lateralities)).lower()
        concept = store.get_concept(release_id, code.code)
        if concept is not None and concept.parent_code:
            siblings = [
                (s.code, s.description)
                for s in store.children(release_id, concept.parent_code)
                if s is not None
            ]
        else:
            siblings = store.concepts_matching_prefix(
                release_id, code.code.split(".")[0]
            )
        better = [
            sib_code
            for sib_code, sib_desc in siblings
            if laterality in sib_desc.lower() and sib_code != code.code
        ]
        if better:
            violations.append(
                Violation(
                    "R6",
                    f"Documentation supports laterality '{laterality}' but "
                    f"{code.code} does not encode it; consider {better[0]}.",
                    [code.code],
                )
            )


def _r7_first_listed(codes, facts, violations) -> None:
    if not codes:
        return
    flagged = [c for c in codes if c.first_listed]
    if len(flagged) != 1:
        violations.append(
            Violation(
                "R7",
                f"Exactly one first-listed code required, found {len(flagged)}.",
                [c.code for c in codes],
            )
        )
        return
    facts_by_id = {f.fact_id: f for f in facts}
    sections = {
        facts_by_id[fid].section
        for fid in flagged[0].supported_facts
        if fid in facts_by_id
    }
    if sections and sections <= {"past_medical_history"}:
        violations.append(
            Violation(
                "R7",
                f"First-listed {flagged[0].code} is supported only by past "
                "medical history; the first-listed diagnosis must reflect the "
                "reason for today's visit.",
                [flagged[0].code],
            )
        )


def _r8_symptom_integral(codes, violations) -> list[CodeAssignment]:
    """Drop R-chapter symptom codes whose facts also support a confirmed
    (non-R) coded diagnosis."""
    confirmed_facts: set[str] = set()
    for code in codes:
        if not code.code.startswith("R"):
            confirmed_facts.update(code.supported_facts)
    kept = []
    for code in codes:
        if (
            code.code.startswith("R")
            and code.supported_facts
            and set(code.supported_facts) <= confirmed_facts
        ):
            violations.append(
                Violation(
                    "R8",
                    f"Symptom code {code.code} is integral to a coded "
                    "confirmed diagnosis; dropped.",
                    [code.code],
                    auto_fixed=True,
                )
            )
        else:
            kept.append(code)
    return kept


# ------------------------------------------------------------------ helpers

import re  # noqa: E402

_CODE_TOKEN_RE = re.compile(r"[A-Z][0-9][0-9A-Z]*(?:\.[0-9A-Z-]+)?(?:-[A-Z][0-9][0-9A-Z.]*)?")


def _code_tokens(expr: str) -> list[str]:
    """Pull code/range tokens out of an instructional-note string like
    'diabetes mellitus due to underlying condition (E08.-)'."""
    return _CODE_TOKEN_RE.findall(expr.upper())
