from autocoder.models import CodableFact
from autocoder.pipeline.assertion import _apply_outpatient_rules


def _fact(**overrides) -> CodableFact:
    base = dict(
        fact_id="fact_1",
        entity_text="type 2 diabetes",
        label="CONDITION",
        span=(0, 15),
        section="assessment",
        ner_score=0.95,
    )
    base.update(overrides)
    return CodableFact(**base)


def test_negated_is_dropped():
    fact = _apply_outpatient_rules(_fact(negation="negated"))
    assert "dropped" in fact.flags
    assert "reason:negated" in fact.flags


def test_family_experiencer_is_dropped():
    fact = _apply_outpatient_rules(_fact(experiencer="family"))
    assert "dropped" in fact.flags


def test_hypothetical_is_dropped():
    fact = _apply_outpatient_rules(_fact(temporality="hypothetical"))
    assert "dropped" in fact.flags


def test_uncertain_condition_flagged_not_dropped():
    fact = _apply_outpatient_rules(_fact(certainty="uncertain"))
    assert "dropped" not in fact.flags
    assert "uncertain_dx" in fact.flags


def test_historical_flagged():
    fact = _apply_outpatient_rules(_fact(temporality="historical"))
    assert "dropped" not in fact.flags
    assert "historical" in fact.flags


def test_medication_kept_with_flag():
    fact = _apply_outpatient_rules(
        _fact(label="DRUG", entity_text="metformin", certainty="uncertain")
    )
    assert "dropped" not in fact.flags
    assert "medication" in fact.flags
    assert "uncertain_dx" not in fact.flags
