"""Stage 1 — Intake: .txt or digital .pdf -> NormalizedDocument.

Scanned PDFs (no text layer) are rejected: OCR is out of prototype scope.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from autocoder.models import NormalizedDocument


class IntakeError(ValueError):
    pass


def load_document(
    path: str | Path,
    *,
    date_of_service: date,
    encounter_type: str,
) -> NormalizedDocument:
    """Load a clinical note from a .txt or digital .pdf file."""
    path = Path(path)
    if not path.exists():
        raise IntakeError(f"Note file does not exist: {path}")

    suffix = path.suffix.lower()
    offset_map = None
    if suffix in (".txt", ".text", ".md"):
        text = path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        text, offset_map = _extract_pdf(path)
    else:
        raise IntakeError(
            f"Unsupported file type {suffix!r}: prototype accepts .txt and digital .pdf"
        )

    if not text.strip():
        raise IntakeError(
            f"No text extracted from {path.name}. If this is a scanned PDF, "
            "OCR is out of prototype scope."
        )

    doc_id = hashlib.sha256(
        f"{path.name}:{len(text)}:{date_of_service}".encode()
    ).hexdigest()[:12]
    return NormalizedDocument(
        doc_id=doc_id,
        text=text,
        offset_map=offset_map,
        date_of_service=date_of_service,
        encounter_type=encounter_type,
    )


def _extract_pdf(path: Path) -> tuple[str, list[dict] | None]:
    try:
        from openmed.multimodal import extract_pdf
    except ImportError as exc:
        raise IntakeError(
            "PDF intake requires the multimodal extra: "
            'uv pip install -e "../openmed[hf,grounding,multimodal]"'
        ) from exc

    extracted = extract_pdf(path)
    text = extracted.text
    spans = getattr(extracted, "spans", None)
    offset_map = None
    if spans:
        offset_map = []
        for span in spans:
            entry = {}
            for field in ("start", "end", "page", "bbox"):
                value = getattr(span, field, None)
                if value is not None:
                    entry[field] = list(value) if field == "bbox" else value
            offset_map.append(entry)
    return text, offset_map
