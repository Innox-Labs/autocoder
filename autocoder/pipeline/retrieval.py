"""Stage 5 — Candidate retrieval (design doc Section 5.5).

Lexical (exact + fuzzy alias match, following OpenMed's Icd10cmLinker ranking:
prefer billable leaf codes) unioned with semantic (per-release embedding
index). Each candidate is enriched with its instructional rule metadata so the
reasoner and rules engine can see Excludes1 / code-first constraints.
"""

from __future__ import annotations

from autocoder.models import Candidate, CodableFact
from autocoder.terminology.embed import SemanticIndex
from autocoder.terminology.store import TerminologyStore

TOP_K = 10
FUZZY_CUTOFF = 82.0
_HISTORY_Z_PREFIXES = ("Z85", "Z86", "Z87")


def retrieve_candidates(
    facts: list[CodableFact],
    store: TerminologyStore,
    release_id: int,
    semantic: SemanticIndex | None = None,
) -> dict[str, list[Candidate]]:
    results: dict[str, list[Candidate]] = {}
    for fact in facts:
        if "medication" in fact.flags:
            continue  # medications support rules; they don't get dx candidates
        merged: dict[str, Candidate] = {}
        for candidate in _lexical(fact.entity_text, store, release_id):
            merged[candidate.code] = candidate
        if semantic is not None:
            for code, score in semantic.search(fact.entity_text, k=TOP_K):
                if code in merged:
                    merged[code].score = max(merged[code].score, score)
                else:
                    candidate = _enrich(code, score, "semantic", store, release_id)
                    if candidate is not None:
                        merged[code] = candidate
        if "historical" in fact.flags:
            for candidate in _history_z_codes(fact.entity_text, store, release_id):
                merged.setdefault(candidate.code, candidate)
        ranked = sorted(
            merged.values(),
            key=lambda c: (-c.score, -len(c.code.replace(".", "")), c.code),
        )
        results[fact.fact_id] = ranked[:TOP_K]
    return results


def _lexical(
    query: str, store: TerminologyStore, release_id: int
) -> list[Candidate]:
    by_code: dict[str, Candidate] = {}
    for code, _desc in store.exact_alias(release_id, query):
        enriched = _enrich(code, 1.0, "lexical", store, release_id)
        if enriched is not None:
            by_code[code] = enriched

    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return _expand_billable(query, list(by_code.values()), store, release_id)

    corpus = store.alias_corpus(release_id)
    if corpus:
        matches = process.extract(
            query.lower(),
            [text.lower() for text, _cid in corpus],
            scorer=fuzz.WRatio,
            limit=TOP_K * 3,
            score_cutoff=FUZZY_CUTOFF,
        )
        for matched_text, _wratio, idx in matches:
            concept = store.concept_by_id(corpus[idx][1])
            if concept is None or concept[0] in by_code:
                continue
            # WRatio recalls substring matches (any "...type 2 diabetes
            # mellitus with <rare complication>..." alias scores ~90), so it
            # cannot be the ranking score. Rank by whole-string token
            # similarity instead: "without complications" stays close to the
            # query, verbose complication aliases fall away.
            score = fuzz.token_sort_ratio(query.lower(), matched_text) / 100.0
            enriched = _enrich(concept[0], score, "lexical", store, release_id)
            if enriched is not None:
                by_code[concept[0]] = enriched
    return _expand_billable(query, list(by_code.values()), store, release_id)


def _expand_billable(
    query: str,
    candidates: list[Candidate],
    store: TerminologyStore,
    release_id: int,
    depth: int = 3,
) -> list[Candidate]:
    """Non-billable matches (category headers like E11) are expanded with their
    billable descendants so the reasoner sees the specificity ladder.

    Each child is scored by its description's similarity to the query (scaled
    by the parent match score), so 'type 2 diabetes mellitus' ranks the
    'without complications' child above rare complication codes.
    """
    seen = {c.code for c in candidates}
    result = list(candidates)
    for candidate in candidates:
        if candidate.billable:
            continue
        frontier = [candidate.code]
        for _ in range(depth):
            next_frontier = []
            for code in frontier:
                for child in store.children(release_id, code):
                    if child is None or child.code in seen:
                        continue
                    seen.add(child.code)
                    next_frontier.append(child.code)
                    if not child.billable:
                        continue
                    # Score children by how few EXTRA concepts their
                    # description adds beyond the query: for "type 2 diabetes
                    # mellitus", the "without complications" child (2 extra
                    # tokens) outranks "with diabetic macular edema, resolved
                    # ..." (many extra tokens). Capped below exact-match 1.0.
                    query_tokens = set(_tokens(query))
                    extra = [
                        t
                        for t in _tokens(child.description)
                        if t not in query_tokens
                    ]
                    similarity = max(0.3, 0.95 - 0.05 * len(extra))
                    result.append(
                        Candidate(
                            code=child.code,
                            description=child.description,
                            score=round(candidate.score * similarity, 4),
                            source="lexical",
                            billable=True,
                            excludes1=child.rules.get("excludes1", []),
                            excludes2=child.rules.get("excludes2", []),
                            code_first=child.rules.get("code_first", []),
                            use_additional=child.rules.get("use_additional", []),
                            parent=child.parent_code,
                        )
                    )
            frontier = next_frontier
    return result


def _history_z_codes(
    query: str, store: TerminologyStore, release_id: int
) -> list[Candidate]:
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return []
    candidates = []
    for prefix in _HISTORY_Z_PREFIXES:
        for code, desc in store.concepts_matching_prefix(release_id, prefix):
            score = fuzz.partial_ratio(query.lower(), desc.lower()) / 100.0
            if score >= 0.75:
                enriched = _enrich(code, score, "lexical", store, release_id)
                if enriched is not None:
                    candidates.append(enriched)
    return sorted(candidates, key=lambda c: -c.score)[:3]


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def _enrich(
    code: str,
    score: float,
    source: str,
    store: TerminologyStore,
    release_id: int,
) -> Candidate | None:
    concept = store.get_concept(release_id, code)
    if concept is None:
        return None
    return Candidate(
        code=concept.code,
        description=concept.description,
        score=round(float(score), 4),
        source=source,
        billable=concept.billable,
        excludes1=concept.rules.get("excludes1", []),
        excludes2=concept.rules.get("excludes2", []),
        code_first=concept.rules.get("code_first", []),
        use_additional=concept.rules.get("use_additional", []),
        parent=concept.parent_code,
    )
