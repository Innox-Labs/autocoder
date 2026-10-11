"""Pipeline orchestrator: NormalizedDocument -> CodedChart.

One self-correction round-trip: if the rules engine finds non-auto-fixed
violations, the reasoner is called once more with the violation messages
appended; anything still failing routes the chart to needs_review.
"""

from __future__ import annotations

import uuid

from autocoder.models import CodedChart, NormalizedDocument
from autocoder.pipeline import assertion, confidence, ner, reasoner, retrieval, rules
from autocoder.pipeline.reasoner import ReasonerClient
from autocoder.pipeline.sections import segment
from autocoder.terminology.embed import load_index
from autocoder.terminology.store import TerminologyStore

import autocoder


def code_chart(
    document: NormalizedDocument,
    store: TerminologyStore,
    client: ReasonerClient,
) -> CodedChart:
    release = store.release_for(document.date_of_service)
    semantic = load_index(store, release.id)

    sections = segment(document.text)
    entities = ner.extract_entities(document.text)
    facts, dropped = assertion.build_facts(document.text, entities, sections)
    candidates = retrieval.retrieve_candidates(facts, store, release.id, semantic)
    guidelines = store.guidelines(release.id)

    codes, abstentions = reasoner.propose_codes(
        client,
        document.text,
        facts,
        candidates,
        guidelines,
        document.encounter_type,
        release.version,
    )
    codes, violations = rules.apply_rules(codes, facts, store, release.id)

    open_violations = rules.unresolved(violations)
    if open_violations:
        messages = [f"[{v.rule}] {v.message}" for v in open_violations]
        codes, retry_abstentions = reasoner.propose_codes(
            client,
            document.text,
            facts,
            candidates,
            guidelines,
            document.encounter_type,
            release.version,
            extra_messages=messages,
        )
        abstentions.extend(retry_abstentions)
        codes, violations = rules.apply_rules(codes, facts, store, release.id)
        open_violations = rules.unresolved(violations)

    confidence.score_codes(codes, facts, candidates)
    routing, reasons = confidence.route(codes, abstentions, len(open_violations))

    return CodedChart(
        chart_id=uuid.uuid4().hex[:12],
        document=document,
        sections=sections,
        facts=facts + dropped,
        candidates=candidates,
        codes=codes,
        abstentions=abstentions,
        routing=routing,
        routing_reasons=reasons,
        pipeline_versions={
            "autocoder": autocoder.__version__,
            "terminology": f"{release.system}/{release.version}",
            "reasoner": type(client).__name__,
        },
    )
