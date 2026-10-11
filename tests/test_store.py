from datetime import date

import pytest

from autocoder.terminology.store import TerminologyStore, code_in_expr


@pytest.fixture
def store(tmp_path):
    return TerminologyStore(tmp_path / "term.sqlite")


def _seed(store: TerminologyStore, version="FY2026", start="2025-10-01"):
    release_id = store.create_release(
        system="icd10cm",
        version=version,
        effective_start=date.fromisoformat(start),
        effective_end=date.fromisoformat(start).replace(
            year=date.fromisoformat(start).year + 1, month=9, day=30
        ),
    )
    store.add_concepts(
        release_id,
        [
            ("E11", "Type 2 diabetes mellitus", False, None),
            ("E11.9", "Type 2 diabetes mellitus without complications", True, "E11"),
            ("E11.621", "Type 2 diabetes mellitus with foot ulcer", True, "E11"),
            ("I10", "Essential (primary) hypertension", True, None),
        ],
    )
    store.add_aliases(
        release_id,
        [
            ("E11.9", "Type 2 diabetes mellitus without complications", "description"),
            ("E11.9", "type 2 diabetes", "synonym"),
            ("I10", "hypertension", "synonym"),
        ],
    )
    store.add_rules(
        release_id,
        [("E11.621", "use_additional", ["code to identify site of ulcer (L97.4-)"])],
    )
    return release_id


def test_release_for_selects_by_date(store):
    _seed(store)
    store.activate("icd10cm", "FY2026")
    release = store.release_for(date(2026, 3, 1))
    assert release.version == "FY2026"


def test_release_for_requires_active(store):
    _seed(store)
    with pytest.raises(LookupError):
        store.release_for(date(2026, 3, 1))


def test_release_boundary_october(store):
    _seed(store, "FY2026", "2025-10-01")
    _seed(store, "FY2027", "2026-10-01")
    store.activate("icd10cm", "FY2026")
    store.activate("icd10cm", "FY2027")
    assert store.release_for(date(2026, 9, 30)).version == "FY2026"
    assert store.release_for(date(2026, 10, 1)).version == "FY2027"


def test_concept_lookup_with_rules(store):
    release_id = _seed(store)
    concept = store.get_concept(release_id, "E11.621")
    assert concept is not None
    assert concept.billable
    assert concept.rules["use_additional"]


def test_exact_alias(store):
    release_id = _seed(store)
    matches = store.exact_alias(release_id, "Type 2 Diabetes")
    assert ("E11.9", "Type 2 diabetes mellitus without complications") in matches


def test_children_and_ancestors(store):
    release_id = _seed(store)
    child_codes = {c.code for c in store.children(release_id, "E11")}
    assert child_codes == {"E11.9", "E11.621"}
    assert store.ancestors(release_id, "E11.9") == ["E11"]


def test_code_in_expr():
    assert code_in_expr("E11.9", "E11")
    assert code_in_expr("E11.621", "E08-E13")
    assert code_in_expr("L97.423", "L97.4-")
    assert not code_in_expr("I10", "E08-E13")
    assert not code_in_expr("E14.9", "E08-E13")
