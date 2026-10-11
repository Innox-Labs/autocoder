"""Review UI: upload a note, view coded charts with evidence highlighting.

Three pages (design doc Section 4 of the guide):
  GET  /              upload form
  POST /charts        run the pipeline on an uploaded note
  GET  /charts        worklist
  GET  /charts/{id}   chart view: highlighted note + code table + debug dump
"""

from __future__ import annotations

import html
import json
import os
import tempfile
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from autocoder.models import CodedChart
from autocoder.terminology.store import DEFAULT_DB_PATH, TerminologyStore

CHARTS_DIR = Path(os.environ.get("AUTOCODER_CHARTS_DIR", "data/charts"))
DB_PATH = Path(os.environ.get("AUTOCODER_DB", str(DEFAULT_DB_PATH)))
REASONER = os.environ.get("AUTOCODER_REASONER", "mock")
REASONER_MODEL = os.environ.get("AUTOCODER_REASONER_MODEL")

_PALETTE = [
    "#ffe08a", "#b5e3b5", "#bcd9f7", "#f7c8c8", "#e3c8f7",
    "#c8f7ef", "#f7ddb0", "#d9e8a8",
]

app = FastAPI(title="AutoCoder Review")
_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"),
    autoescape=select_autoescape(["html"]),
)


def _render(name: str, **context) -> HTMLResponse:
    return HTMLResponse(_env.get_template(name).render(**context))


def _store() -> TerminologyStore:
    return TerminologyStore(DB_PATH)


def _reasoner():
    from autocoder.pipeline.reasoner import (
        MockReasoner,
        OllamaClient,
        OpenAICompatClient,
    )

    if REASONER == "ollama":
        return OllamaClient(model=REASONER_MODEL or "qwen2.5:32b")
    if REASONER == "openai":
        return OpenAICompatClient(model=REASONER_MODEL or "gpt-4o")
    return MockReasoner()


@app.get("/", response_class=HTMLResponse)
def index():
    return _render("index.html", today=date.today().isoformat())


@app.post("/charts")
async def create_chart(
    file: UploadFile = File(...),
    dos: str = Form(...),
    encounter: str = Form("office_visit"),
):
    from autocoder.pipeline.intake import load_document
    from autocoder.pipeline.run import code_chart

    suffix = Path(file.filename or "note.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        document = load_document(
            tmp_path,
            date_of_service=date.fromisoformat(dos),
            encounter_type=encounter,
        )
        chart = code_chart(document, _store(), _reasoner())
    finally:
        tmp_path.unlink(missing_ok=True)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    (CHARTS_DIR / f"{chart.chart_id}.json").write_text(chart.model_dump_json(indent=2))
    return RedirectResponse(f"/charts/{chart.chart_id}", status_code=303)


@app.get("/charts", response_class=HTMLResponse)
def worklist():
    charts = []
    if CHARTS_DIR.exists():
        for path in sorted(CHARTS_DIR.glob("*.json"), reverse=True):
            data = json.loads(path.read_text())
            charts.append(
                {
                    "chart_id": data["chart_id"],
                    "routing": data["routing"],
                    "n_codes": len(data["codes"]),
                    "dos": data["document"]["date_of_service"],
                    "encounter": data["document"]["encounter_type"],
                }
            )
    return _render("worklist.html", charts=charts)


@app.get("/charts/{chart_id}", response_class=HTMLResponse)
def chart_view(chart_id: str):
    path = CHARTS_DIR / f"{chart_id}.json"
    if not path.exists():
        return HTMLResponse("<h1>Chart not found</h1>", status_code=404)
    chart = CodedChart.model_validate_json(path.read_text())

    colors = {
        code.code: _PALETTE[i % len(_PALETTE)] for i, code in enumerate(chart.codes)
    }
    return _render(
        "chart.html",
        chart=chart,
        colors=colors,
        note_html=_highlight(chart, colors),
        candidates_json=json.dumps(
            chart.model_dump(mode="json")["candidates"], indent=1
        ),
    )


def _highlight(chart: CodedChart, colors: dict[str, str]) -> str:
    """Wrap each code's evidence spans in <mark data-code=...> tags.

    Spans are applied back-to-front so earlier offsets stay valid; overlapping
    spans keep the first (higher-listed code) mark.
    """
    text = chart.document.text
    events: list[tuple[int, int, str]] = [
        (start, end, code.code)
        for code in chart.codes
        for start, end in code.evidence_spans
    ]
    events.sort()

    used: list[tuple[int, int, str]] = []
    for start, end, code in events:
        if not any(s < end and start < e for s, e, _ in used):
            used.append((start, end, code))

    pieces: list[tuple[int, int, str | None]] = []
    prev = 0
    for start, end, code in used:
        pieces.append((prev, start, None))
        pieces.append((start, end, code))
        prev = end
    pieces.append((prev, len(text), None))

    out = []
    for start, end, code in pieces:
        chunk = html.escape(text[start:end])
        if code is None:
            out.append(chunk)
        else:
            out.append(
                f'<mark data-code="{code}" style="background:{colors.get(code, "#ff0")}"'
                f' title="{code}">{chunk}</mark>'
            )
    return "".join(out).replace("\n", "<br>")
