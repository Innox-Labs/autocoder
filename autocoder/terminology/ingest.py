"""Release ingestion: official CMS ICD-10-CM files -> terminology store.

Two CMS artifacts feed one release (design doc Section 7):

1. "Code Descriptions in Tabular Order" zip  -> concepts (code, description,
   billable flag, parent) from the fixed-width ``icd10cm_order_*.txt`` file.
2. "Code Tables, Tabular and Index" zip (optional but recommended)
   -> instructional rules (Excludes1/2, codeFirst, useAdditionalCode,
   inclusionTerm) from the tabular XML, and alias terms from the index XML.

Both files are downloaded by the user from cms.gov (they are public data) and
passed as local paths; a URL with --sha256 also works, reusing the checksummed
download pattern from OpenMed's vocab loader.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

from autocoder.terminology.store import TerminologyStore

# Plain-language digest of ICD-10-CM Official Guidelines Section IV
# (outpatient). Restated in our own words; seeded per release so prompt
# guidance is versioned data, replaceable by ingesting the real guideline text.
OUTPATIENT_GUIDELINE_DIGEST = [
    (
        "IV.first_listed",
        "List first the diagnosis chiefly responsible for today's visit. "
        "Outpatient encounters have a first-listed diagnosis, not a principal "
        "diagnosis.",
    ),
    (
        "IV.uncertain",
        "Never code a diagnosis documented as probable, suspected, likely, "
        "questionable, rule out, or working diagnosis. Code the condition to "
        "the highest degree of certainty, such as the documented signs and "
        "symptoms.",
    ),
    (
        "IV.coexisting",
        "Code all documented conditions that coexist at the time of the visit "
        "and require or affect care. Chronic conditions count when they are "
        "treated, monitored, or medicated. Do not code resolved conditions; "
        "use personal-history Z codes where relevant.",
    ),
    (
        "IV.specificity",
        "Code to the highest level of specificity the documentation supports, "
        "including laterality and complication detail. Use combination codes "
        "when one code captures linked conditions.",
    ),
    (
        "IV.symptoms",
        "Do not separately code signs and symptoms that are integral to a "
        "confirmed, coded diagnosis.",
    ),
]

_ORDER_FILE_RE = re.compile(r"icd10cm[_-]?order[_-]?(\d{4})\.txt$", re.IGNORECASE)
_TABULAR_FILE_RE = re.compile(r"icd10cm[_-]?tabular[_-]?(\d{4})\.xml$", re.IGNORECASE)
_INDEX_FILE_RE = re.compile(r"icd10cm[_-]?index[_-]?(\d{4})\.xml$", re.IGNORECASE)


def fiscal_year_window(version: str) -> tuple[date, date]:
    """FY2026 -> (2025-10-01, 2026-09-30)."""
    year = int(version.upper().removeprefix("FY"))
    return date(year - 1, 10, 1), date(year, 9, 30)


def ingest_icd10cm(
    store: TerminologyStore,
    *,
    version: str,
    descriptions_zip: str | Path,
    tables_zip: str | Path | None = None,
) -> int:
    """Ingest one ICD-10-CM release as status=staged. Returns release_id."""
    desc_bytes = Path(descriptions_zip).read_bytes()
    checksum = hashlib.sha256(desc_bytes).hexdigest()
    start, end = fiscal_year_window(version)
    release_id = store.create_release(
        system="icd10cm",
        version=version,
        effective_start=start,
        effective_end=end,
        source_checksum=f"sha256:{checksum}",
    )

    concepts = _parse_order_file(desc_bytes)
    store.add_concepts(release_id, concepts)
    # Every description is also a lookup alias.
    store.add_aliases(
        release_id, [(code, desc, "description") for code, desc, _, _ in concepts]
    )

    if tables_zip is not None:
        tables_bytes = Path(tables_zip).read_bytes()
        rules, inclusion_aliases = _parse_tabular_xml(tables_bytes)
        store.add_rules(release_id, rules)
        store.add_aliases(release_id, inclusion_aliases)
        index_aliases = _parse_index_xml(tables_bytes)
        store.add_aliases(release_id, index_aliases)

    store.add_guidelines(release_id, OUTPATIENT_GUIDELINE_DIGEST)
    return release_id


# ------------------------------------------------------------- order file


def _parse_order_file(zip_bytes: bytes) -> list[tuple[str, str, bool, str | None]]:
    """Parse icd10cm_order_YYYY.txt -> (code, description, billable, parent).

    Fixed-width columns per CMS spec:
      [0:5] order number, [6:13] code (undotted), [14] header flag
      (0 = non-billable header, 1 = valid/billable), [16:76] short
      description, [77:] long description.
    """
    name = _member(zip_bytes, _ORDER_FILE_RE)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        raw = archive.read(name).decode("utf-8", errors="replace")

    seen_codes: list[str] = []
    rows: list[tuple[str, str, bool, str | None]] = []
    for line in raw.splitlines():
        if len(line) < 16:
            continue
        code_plain = line[6:13].strip()
        if not code_plain:
            continue
        billable = line[14:15].strip() == "1"
        long_desc = line[77:].strip() or line[16:76].strip()
        code = _dotted(code_plain)
        parent = _nearest_parent(code_plain, seen_codes)
        rows.append((code, long_desc, billable, _dotted(parent) if parent else None))
        seen_codes.append(code_plain)
    if not rows:
        raise ValueError("Order file parsed to zero concepts; check the source zip.")
    return rows


def _dotted(code_plain: str) -> str:
    code_plain = code_plain.strip().upper()
    if len(code_plain) <= 3 or "." in code_plain:
        return code_plain
    return f"{code_plain[:3]}.{code_plain[3:]}"


def _nearest_parent(code_plain: str, seen: list[str]) -> str | None:
    """The parent is the longest previously-seen strict prefix of this code."""
    for length in range(len(code_plain) - 1, 2, -1):
        prefix = code_plain[:length]
        if prefix in seen:
            return prefix
    return None


# ------------------------------------------------------------- tabular XML


def _parse_tabular_xml(
    zip_bytes: bytes,
) -> tuple[list[tuple], list[tuple]]:
    """Extract instructional notes from the CMS tabular XML.

    Returns (rules, inclusion_aliases):
      rules:   (code, rule_type, payload_list)
      aliases: (code, text, "inclusion_term")
    """
    try:
        name = _member(zip_bytes, _TABULAR_FILE_RE)
    except FileNotFoundError:
        return [], []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        root = ElementTree.fromstring(archive.read(name))

    note_map = {
        "excludes1": "excludes1",
        "excludes2": "excludes2",
        "codeFirst": "code_first",
        "useAdditionalCode": "use_additional",
        "includes": "includes",
    }
    rules: list[tuple] = []
    aliases: list[tuple] = []

    for diag in root.iter("diag"):
        code_el = diag.find("name")
        if code_el is None or not (code_el.text or "").strip():
            continue
        code = code_el.text.strip()
        for xml_tag, rule_type in note_map.items():
            # Only this diag's direct notes, not descendants'.
            for block in diag.findall(xml_tag):
                notes = [
                    (n.text or "").strip() for n in block.findall("note")
                ]
                notes = [n for n in notes if n]
                if notes:
                    rules.append((code, rule_type, notes))
        for block in diag.findall("inclusionTerm"):
            for note in block.findall("note"):
                text = (note.text or "").strip()
                if text:
                    aliases.append((code, text, "inclusion_term"))
    return rules, aliases


# ------------------------------------------------------------- index XML


def _parse_index_xml(zip_bytes: bytes) -> list[tuple]:
    """Best-effort alias extraction from the CMS alphabetic index XML.

    Walks mainTerm/term trees; each node with a <code> child contributes the
    joined title path as an alias for that code. 7th-character placeholders
    and dash-suffixed non-leaf references are skipped.
    """
    try:
        name = _member(zip_bytes, _INDEX_FILE_RE)
    except FileNotFoundError:
        return []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        root = ElementTree.fromstring(archive.read(name))

    aliases: list[tuple] = []

    def walk(term_el: ElementTree.Element, path: list[str]) -> None:
        title_el = term_el.find("title")
        # Direct text only: skip <nemod> children (parenthesized nonessential
        # modifiers) so "Hypertension (accelerated)(benign)..." indexes as
        # "Hypertension".
        title = (title_el.text or "").strip() if title_el is not None else ""
        title = title.rstrip(",").strip()
        # Main-term titles bundle synonyms: "Hypertension, hypertensive".
        variants = [v.strip() for v in title.split(",") if v.strip()]
        head = variants[0] if variants else ""
        current_path = path + [head] if head else path
        code_el = term_el.find("code")
        if code_el is not None and (code_el.text or "").strip():
            code = code_el.text.strip()
            if "-" not in code:  # skip non-leaf cross references like "L97.4-"
                if not path:
                    for variant in variants:
                        aliases.append((code, variant, "index"))
                else:
                    # Coder index convention: "Hypertension, essential".
                    alias = ", ".join(p for p in current_path if p)
                    if alias:
                        aliases.append((code, alias, "index"))
        for child in term_el.findall("term"):
            walk(child, current_path)

    for main in root.iter("mainTerm"):
        walk(main, [])
    return aliases


# ------------------------------------------------------------- helpers


def _member(zip_bytes: bytes, pattern: re.Pattern) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            if pattern.search(Path(name).name):
                return name
    raise FileNotFoundError(f"No member matching {pattern.pattern} in archive")
