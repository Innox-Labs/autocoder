from datetime import date

import pytest

from autocoder.models import CodableFact, CodeAssignment
from autocoder.pipeline.rules import apply_rules, unresolved
from autocoder.terminology.store import TerminologyStore


@pytest.fixture
def env(tmp_path):
    store = TerminologyStore(tmp_path / "term.sqlite")
    release_id = store.create_release(
        system="icd10cm",
        version="FY2026",
        effective_start=date(2025, 10, 1),
        effective_end=date(2026, 9, 30),
    )
    store.add_concepts(
        release_id,
        [
            ("E11", "Type 2 diabetes mellitus", False, None),
            ("E11.9", "Type 2 DM without complications", True, "E11"),
            ("E11.621", "Type 2 DM with foot ulcer", True, "E11"),
            ("R05", "Cough", False, None),
            ("R05.9", "Cough, unspecified", True, "R05"),
            ("J20.9", "Acute bronchitis, unspecified", True, None),
            ("I10", "Essential hypertension", True, None),
            ("H60.31", "Otitis externa, right ear", True, None),
            ("H60.32", "Otitis externa, left ear", True, "H60.3"),
            ("H60.3", "Other infective otitis externa", False, None),
        ],
    )
    store.add_rules(
        release_id,
        [("I10", "excludes1", ["hypertensive disease complicating pregnancy (O10-O11)"])],
    )
    return store, release_id


def _code(code, facts=("fact_1",), first=False):
    return CodeAssignment(
        code=code,
        release_version="FY2026",
        first_listed=first,
        evidence_spans=[(0, 10)],
        supported_facts=list(facts),
        confidence=0.9,
    )


def _fact(fact_id="fact_1", section="assessment", **attrs):
    return CodableFact(
        fact_id=fact_id,
        entity_text="x",
        label="CONDITION",
        span=(0, 10),
        section=section,
        attributes=attrs,
        ner_score=0.9,
    )


def test_r1_unknown_code_removed(env):
    store, release_id = env
    kept, violations = apply_rules([_code("Z99.999", first=True)], [_fact()], store, release_id)
    assert kept == []
    assert any(v.rule == "R1" for v in violations)


def test_r2_nonbillable_flagged(env):
    store, release_id = env
    kept, violations = apply_rules([_code("E11", first=True)], [_fact()], store, release_id)
    r2 = [v for v in violations if v.rule == "R2"]
    assert r2 and "E11.9" in r2[0].message


def test_r4_parent_dropped_when_child_present(env):
    store, release_id = env
    kept, violations = apply_rules(
        [_code("E11.9", first=True), _code("E11")], [_fact()], store, release_id
    )
    assert {c.code for c in kept} == {"E11.9"}
    assert any(v.rule == "R4" and v.auto_fixed for v in violations)


def test_r6_laterality_specificity(env):
    store, release_id = env
    fact = _fact(laterality="left")
    kept, violations = apply_rules(
        [_code("H60.31", first=True)], [fact], store, release_id
    )
    r6 = [v for v in violations if v.rule == "R6"]
    assert r6 and "H60.32" in r6[0].message


def test_r7_pmh_only_first_listed_flagged(env):
    store, release_id = env
    fact = _fact(section="past_medical_history")
    kept, violations = apply_rules(
        [_code("E11.9", first=True)], [fact], store, release_id
    )
    assert any(v.rule == "R7" for v in violations)


def test_r8_symptom_integral_dropped(env):
    store, release_id = env
    kept, violations = apply_rules(
        [_code("J20.9", facts=("fact_1",), first=True), _code("R05.9", facts=("fact_1",))],
        [_fact()],
        store,
        release_id,
    )
    assert {c.code for c in kept} == {"J20.9"}
    r8 = [v for v in violations if v.rule == "R8"]
    assert r8 and r8[0].auto_fixed


def test_unresolved_excludes_autofixed(env):
    store, release_id = env
    kept, violations = apply_rules(
        [_code("E11.9", first=True), _code("E11")], [_fact()], store, release_id
    )
    assert all(v.rule != "R4" for v in unresolved(violations))
