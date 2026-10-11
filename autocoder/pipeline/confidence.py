"""Stage 7b — Confidence scoring and routing (design doc Section 10).

Prototype formula: reasoner self-score, damped by weak retrieval margin, weak
supporting NER scores, and rule flags. Calibration against reviewed charts
replaces this once real data exists.
"""

from __future__ import annotations

from autocoder.models import Abstention, Candidate, CodableFact, CodeAssignment

AUTO_THRESHOLD = 0.85


def score_codes(
    codes: list[CodeAssignment],
    facts: list[CodableFact],
    candidates: dict[str, list[Candidate]],
) -> None:
    facts_by_id = {f.fact_id: f for f in facts}
    for code in codes:
        confidence = max(0.05, min(1.0, code.confidence or 0.5))
        margin = _retrieval_margin(code, candidates)
        confidence *= 0.85 + 0.15 * margin
        ner_scores = [
            facts_by_id[fid].ner_score
            for fid in code.supported_facts
            if fid in facts_by_id
        ]
        if ner_scores and min(ner_scores) < 0.7:
            confidence *= 0.9
        if code.rule_flags:
            confidence *= 0.75 ** len(code.rule_flags)
        code.confidence = round(confidence, 3)


def route(
    codes: list[CodeAssignment],
    abstentions: list[Abstention],
    unresolved_violations: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not codes:
        reasons.append("No codes assigned.")
    for code in codes:
        if code.confidence < AUTO_THRESHOLD:
            reasons.append(
                f"{code.code} confidence {code.confidence} < {AUTO_THRESHOLD}."
            )
    if abstentions:
        reasons.append(f"{len(abstentions)} open question(s) from the reasoner.")
    if unresolved_violations:
        reasons.append(f"{unresolved_violations} unresolved rule violation(s).")
    return ("needs_review" if reasons else "auto"), reasons


def _retrieval_margin(
    code: CodeAssignment, candidates: dict[str, list[Candidate]]
) -> float:
    """1.0 = chosen code clearly outranked alternatives; 0.0 = coin flip."""
    margins = []
    for fid in code.supported_facts:
        cands = candidates.get(fid, [])
        chosen = next((c for c in cands if c.code == code.code), None)
        if chosen is None or len(cands) < 2:
            continue
        runner_up = max((c.score for c in cands if c.code != code.code), default=0.0)
        margins.append(max(0.0, min(1.0, chosen.score - runner_up + 0.5)))
    return sum(margins) / len(margins) if margins else 0.5
