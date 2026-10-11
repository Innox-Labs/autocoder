"""Stage 2 — Section segmenter (design doc Section 5.2).

Detects headers with simple regexes, then canonicalizes header names through
OpenMed's alias table (knows "PMH", "HPI", "A/P", ...). The whole document is
covered: text before the first header is labeled ``body``; unknown headers
keep label ``other`` with their raw name preserved.
"""

from __future__ import annotations

import re

from autocoder.models import Section

# A header line is one of:
#  - an ALL-CAPS line (possibly with spaces/&/-), max ~6 words
#  - a short "Title:" line (title case or caps before the colon)
_HEADER_RE = re.compile(
    r"^(?P<header>"
    r"[A-Z][A-Z0-9 &/\-]{2,60}"  # ALL CAPS line
    r"|[A-Z][A-Za-z &/\-]{1,60}:"  # "Assessment and Plan:" style
    r")\s*$",
    re.MULTILINE,
)


# Outpatient headers OpenMed's alias table does not cover (it knows HPI, PMH,
# family/social history, assessment, plan). Keys are casefolded.
_EXTRA_ALIASES = {
    "chief complaint": "chief_complaint",
    "cc": "chief_complaint",
    "reason for visit": "chief_complaint",
    "assessment and plan": "assessment_and_plan",
    "assessment & plan": "assessment_and_plan",
    "a&p": "assessment_and_plan",
    "a/p": "assessment_and_plan",
    "impression and plan": "assessment_and_plan",
    "medications": "medications",
    "current medications": "medications",
    "meds": "medications",
    "physical exam": "physical_exam",
    "physical examination": "physical_exam",
    "exam": "physical_exam",
    "results": "results",
    "labs": "results",
    "laboratory": "results",
    "imaging": "results",
    "allergies": "allergies",
    "review of systems": "review_of_systems",
    "ros": "review_of_systems",
    "discharge diagnoses": "discharge_diagnoses",
    "hospital course": "hospital_course",
}


def _canonical(raw: str) -> str:
    try:
        from openmed.clinical import canonical_section_label
    except ImportError:
        canonical_section_label = None
    if canonical_section_label is not None:
        label = canonical_section_label(raw)
        if label:
            return label
    return _EXTRA_ALIASES.get(raw.casefold().strip(), "other")


def segment(text: str) -> list[Section]:
    """Split note text into labeled sections covering the whole document."""
    matches = [
        m for m in _HEADER_RE.finditer(text) if _looks_like_header(m.group("header"))
    ]
    sections: list[Section] = []

    if not matches or matches[0].start() > 0:
        end = matches[0].start() if matches else len(text)
        sections.append(Section(label="body", raw_header=None, start=0, end=end))

    for i, match in enumerate(matches):
        raw = match.group("header").rstrip(":").strip()
        # Section starts at the header itself so the document is fully covered.
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(
            Section(label=_canonical(raw), raw_header=raw, start=start, end=end)
        )
    return sections


def section_for_offset(sections: list[Section], offset: int) -> Section | None:
    for section in sections:
        if section.start <= offset < section.end:
            return section
    return sections[-1] if sections else None


def _looks_like_header(line: str) -> bool:
    stripped = line.strip()
    # Reject sentence-like lines that happen to match (e.g. drug names in caps
    # lists) by capping word count.
    return 0 < len(stripped.split()) <= 6
