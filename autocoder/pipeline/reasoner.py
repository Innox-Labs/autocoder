"""Stage 6 — LLM coding reasoner (design doc Sections 5.6 and 8).

One ``ReasonerClient`` protocol, three implementations:
  - OllamaClient      local model, PHI never leaves the machine (product path)
  - OpenAICompatClient  hosted API; note excerpts pass through OpenMed
                        de-identification first (prototype convenience)
  - MockReasoner      no LLM at all: picks the top candidate per fact.
                      Lets the whole pipeline run end-to-end in tests/CI.

Hard constraints enforced in code, not prompt: output must parse against the
schema (one retry), every chosen code must come from a candidate list, every
code needs evidence spans, exactly one first_listed when codes exist.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

import httpx

from autocoder.models import Abstention, Candidate, CodableFact, CodeAssignment

SYSTEM_PROMPT = (
    "You are an expert outpatient medical coder. Assign ICD-10-CM diagnosis "
    "codes for the encounter, following the outpatient rules provided. You may "
    "ONLY choose codes from the candidate lists given. If documentation is "
    "insufficient for a specific code, abstain via the queries array instead "
    "of guessing. Respond with a single JSON object and nothing else."
)

OUTPUT_SPEC = """Respond with JSON exactly in this shape:
{
  "codes": [
    {
      "code": "<candidate code>",
      "first_listed": true,
      "rationale": "<one or two sentences>",
      "evidence_spans": [[start, end]],
      "supported_facts": ["fact_1"],
      "confidence": 0.0
    }
  ],
  "queries": [
    {"reason": "<why you cannot code confidently>", "fact_ids": ["fact_2"]}
  ]
}
Rules: exactly one code has first_listed=true when any codes exist; every code
must appear in some fact's candidate list; evidence_spans are character offsets
into the note text; do not code uncertain diagnoses (facts flagged
uncertain_dx) - code their documented symptoms instead."""


class ReasonerClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ReasonerError(RuntimeError):
    pass


# ------------------------------------------------------------------ clients


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:32b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class OpenAICompatClient:
    """Hosted OpenAI-compatible API. Prompts are de-identified with OpenMed
    before leaving the machine (design rule: never send raw PHI to a cloud
    model)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        deidentify: bool = True,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.deidentify = deidentify
        self.timeout = timeout
        if not self.api_key:
            raise ReasonerError("OPENAI_API_KEY is not set for the hosted reasoner.")

    def complete(self, system: str, user: str) -> str:
        if self.deidentify:
            from openmed import deidentify

            user = deidentify(user, method="replace").deidentified_text
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class MockReasoner:
    """No-LLM fallback: top candidate per non-flagged fact; first-listed is the
    best-scoring fact from an assessment/chief-complaint section. Exists so the
    pipeline, rules engine, UI, and eval harness run without any model."""

    def complete(self, system: str, user: str) -> str:
        raw = user.split("=== INPUT ===", 1)[1].lstrip()
        payload, _end = json.JSONDecoder().raw_decode(raw)
        by_code: dict[str, dict] = {}
        queries: list[dict] = []
        priority_sections = {"assessment", "assessment_and_plan", "chief_complaint"}
        ranks: dict[str, float] = {}
        for fact in payload["facts"]:
            if "uncertain_dx" in fact["flags"] or "medication" in fact["flags"]:
                continue
            candidates = payload["candidates"].get(fact["fact_id"], [])
            billable = [c for c in candidates if c.get("billable", True)]
            if not billable:
                queries.append(
                    {
                        "reason": f"No candidate codes for '{fact['entity_text']}'.",
                        "fact_ids": [fact["fact_id"]],
                    }
                )
                continue
            if "historical" in fact["flags"]:
                z_history = [c for c in billable if c["code"].startswith("Z8")]
                billable = z_history or billable
            # Without an LLM, prefer the least specific billable code on score
            # ties (unspecified beats a guessed complication).
            top = min(billable, key=lambda c: (-c["score"], len(c["code"])))
            entry = by_code.setdefault(
                top["code"],
                {
                    "code": top["code"],
                    "first_listed": False,
                    "rationale": f"Top retrieval match for '{fact['entity_text']}'.",
                    "evidence_spans": [],
                    "supported_facts": [],
                    "confidence": 0.0,
                },
            )
            entry["evidence_spans"].append(list(fact["span"]))
            entry["supported_facts"].append(fact["fact_id"])
            # Cap below the auto-finalize threshold: the mock is a plumbing
            # aid, not a coder - its charts must always route to review.
            entry["confidence"] = max(entry["confidence"], min(0.8, top["score"]))
            boost = 1.0 if fact["section"] in priority_sections else 0.0
            ranks[top["code"]] = max(
                ranks.get(top["code"], 0.0), top["score"] + boost
            )
        codes = list(by_code.values())
        if codes:
            best = max(codes, key=lambda c: ranks.get(c["code"], 0.0))
            best["first_listed"] = True
        return json.dumps({"codes": codes, "queries": queries})


# ------------------------------------------------------------------ prompt


def build_prompt(
    text: str,
    facts: list[CodableFact],
    candidates: dict[str, list[Candidate]],
    guidelines: list[tuple[str, str]],
    encounter_type: str,
) -> str:
    guideline_text = "\n".join(f"- {body}" for _sec, body in guidelines)
    payload = {
        "encounter_type": encounter_type,
        "facts": [
            {
                "fact_id": f.fact_id,
                "entity_text": f.entity_text,
                "label": f.label,
                "span": list(f.span),
                "section": f.section,
                "flags": f.flags,
                "attributes": f.attributes,
                "context_sentence": _sentence_around(text, f.span),
            }
            for f in facts
        ],
        "candidates": {
            fid: [
                {
                    "code": c.code,
                    "description": c.description,
                    "score": c.score,
                    "billable": c.billable,
                    "excludes1": c.excludes1,
                    "code_first": c.code_first,
                    "use_additional": c.use_additional,
                }
                for c in cands
            ]
            for fid, cands in candidates.items()
        },
    }
    return (
        f"OUTPATIENT CODING RULES:\n{guideline_text}\n\n"
        f"{OUTPUT_SPEC}\n\n=== INPUT ===\n{json.dumps(payload, indent=1)}"
    )


def _sentence_around(text: str, span: tuple[int, int], window: int = 240) -> str:
    start = max(0, span[0] - window)
    end = min(len(text), span[1] + window)
    chunk = text[start:end]
    # Trim to sentence-ish boundaries.
    left = chunk.rfind(". ", 0, span[0] - start)
    right = chunk.find(". ", span[1] - start)
    if left != -1:
        chunk = chunk[left + 2 :]
        right -= left + 2
    if right != -1:
        chunk = chunk[: right + 1]
    return chunk.strip()


# ------------------------------------------------------------------ parsing


def propose_codes(
    client: ReasonerClient,
    text: str,
    facts: list[CodableFact],
    candidates: dict[str, list[Candidate]],
    guidelines: list[tuple[str, str]],
    encounter_type: str,
    release_version: str,
    extra_messages: list[str] | None = None,
) -> tuple[list[CodeAssignment], list[Abstention]]:
    prompt = build_prompt(text, facts, candidates, guidelines, encounter_type)
    if extra_messages:
        prompt += "\n\n=== CORRECTIONS REQUIRED ===\n" + "\n".join(extra_messages)

    raw = client.complete(SYSTEM_PROMPT, prompt)
    try:
        parsed = _parse(raw)
    except (json.JSONDecodeError, KeyError, TypeError):
        raw = client.complete(
            SYSTEM_PROMPT, prompt + "\n\nYour previous output was not valid JSON. "
            "Output ONLY the JSON object."
        )
        parsed = _parse(raw)

    allowed = {c.code for cands in candidates.values() for c in cands}
    codes: list[CodeAssignment] = []
    abstentions = [
        Abstention(reason=q.get("reason", ""), fact_ids=q.get("fact_ids", []))
        for q in parsed.get("queries", [])
    ]
    for item in parsed.get("codes", []):
        code = str(item.get("code", "")).strip().upper()
        spans = [
            (int(s), int(e))
            for s, e in item.get("evidence_spans", [])
            if 0 <= int(s) < int(e) <= len(text)
        ]
        if code not in allowed:
            abstentions.append(
                Abstention(
                    reason=f"Reasoner proposed {code} outside the candidate lists; "
                    "discarded (candidate-constrained decoding).",
                    fact_ids=item.get("supported_facts", []),
                )
            )
            continue
        if not spans:
            spans = _spans_from_facts(item.get("supported_facts", []), facts)
        if not spans:
            continue
        codes.append(
            CodeAssignment(
                code=code,
                release_version=release_version,
                first_listed=bool(item.get("first_listed", False)),
                rationale=str(item.get("rationale", "")),
                evidence_spans=spans,
                supported_facts=list(item.get("supported_facts", [])),
                confidence=float(item.get("confidence", 0.0)),
            )
        )

    _normalize_first_listed(codes)
    return codes, abstentions


def _parse(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        raise json.JSONDecodeError("no JSON object found", raw, 0)
    return json.loads(match.group(0))


def _spans_from_facts(
    fact_ids: list[str], facts: list[CodableFact]
) -> list[tuple[int, int]]:
    by_id = {f.fact_id: f for f in facts}
    return [by_id[fid].span for fid in fact_ids if fid in by_id]


def _normalize_first_listed(codes: list[CodeAssignment]) -> None:
    flagged = [c for c in codes if c.first_listed]
    if not codes:
        return
    if len(flagged) != 1:
        for code in codes:
            code.first_listed = False
        best = max(codes, key=lambda c: c.confidence)
        best.first_listed = True
