"""AutoCoder CLI.

Commands:
  autocoder terminology ingest    stage an ICD-10-CM release from CMS files
  autocoder terminology activate  flip a staged release to active
  autocoder terminology releases  list releases
  autocoder terminology embed     build the semantic index for a release
  autocoder code                  code one note file
  autocoder eval                  run the golden-set evaluation
  autocoder serve                 start the review UI
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from autocoder.terminology.store import DEFAULT_DB_PATH, TerminologyStore

app = typer.Typer(help="AutoCoder — outpatient ICD-10-CM coding prototype")
terminology_app = typer.Typer(help="Manage versioned terminology releases")
app.add_typer(terminology_app, name="terminology")

CHARTS_DIR = Path("data/charts")


def _store(db: Path) -> TerminologyStore:
    return TerminologyStore(db)


def _reasoner(name: str, model: str | None):
    from autocoder.pipeline.reasoner import (
        MockReasoner,
        OllamaClient,
        OpenAICompatClient,
    )

    if name == "mock":
        return MockReasoner()
    if name == "ollama":
        return OllamaClient(model=model or "qwen2.5:32b")
    if name == "openai":
        return OpenAICompatClient(model=model or "gpt-4o")
    raise typer.BadParameter(f"Unknown reasoner {name!r}: use mock|ollama|openai")


@terminology_app.command("ingest")
def terminology_ingest(
    version: str = typer.Option(..., help="Release version, e.g. FY2026"),
    descriptions_zip: Path = typer.Option(
        ..., help="CMS 'Code Descriptions in Tabular Order' zip"
    ),
    tables_zip: Path = typer.Option(
        None, help="CMS 'Code Tables, Tabular and Index' zip (rules + index aliases)"
    ),
    db: Path = typer.Option(DEFAULT_DB_PATH),
):
    from autocoder.terminology.ingest import ingest_icd10cm

    store = _store(db)
    release_id = ingest_icd10cm(
        store,
        version=version,
        descriptions_zip=descriptions_zip,
        tables_zip=tables_zip,
    )
    count = store.conn.execute(
        "SELECT COUNT(*) FROM concept WHERE release_id=?", (release_id,)
    ).fetchone()[0]
    rules = store.conn.execute(
        "SELECT COUNT(*) FROM rule WHERE release_id=?", (release_id,)
    ).fetchone()[0]
    typer.echo(
        f"Staged icd10cm/{version} (release_id={release_id}): "
        f"{count} concepts, {rules} rules. Next: autocoder terminology activate "
        f"--version {version}"
    )


@terminology_app.command("activate")
def terminology_activate(
    version: str = typer.Option(...),
    db: Path = typer.Option(DEFAULT_DB_PATH),
):
    store = _store(db)
    store.activate("icd10cm", version)
    typer.echo(f"icd10cm/{version} is now active.")


@terminology_app.command("releases")
def terminology_releases(db: Path = typer.Option(DEFAULT_DB_PATH)):
    store = _store(db)
    for release in store.releases():
        typer.echo(
            f"{release.system}/{release.version}  {release.effective_start} .. "
            f"{release.effective_end}  [{release.status}]"
        )


@terminology_app.command("embed")
def terminology_embed(
    version: str = typer.Option(...),
    db: Path = typer.Option(DEFAULT_DB_PATH),
):
    from autocoder.terminology.embed import build_index

    store = _store(db)
    release = next(
        (r for r in store.releases() if r.version == version), None
    )
    if release is None:
        raise typer.BadParameter(f"No release {version}")
    path = build_index(store, release.id)
    typer.echo(f"Semantic index written to {path}")


@app.command("code")
def code_note(
    note: Path = typer.Argument(..., help=".txt or digital .pdf clinical note"),
    dos: str = typer.Option(..., help="Date of service, YYYY-MM-DD"),
    encounter: str = typer.Option("office_visit"),
    reasoner: str = typer.Option("mock", help="mock | ollama | openai"),
    model: str = typer.Option(None, help="Model name for the chosen reasoner"),
    db: Path = typer.Option(DEFAULT_DB_PATH),
    out: Path = typer.Option(CHARTS_DIR, help="Directory for chart JSON output"),
):
    from autocoder.pipeline.intake import load_document
    from autocoder.pipeline.run import code_chart

    document = load_document(
        note, date_of_service=date.fromisoformat(dos), encounter_type=encounter
    )
    chart = code_chart(document, _store(db), _reasoner(reasoner, model))

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{chart.chart_id}.json"
    path.write_text(chart.model_dump_json(indent=2))

    typer.echo(f"\nChart {chart.chart_id}  routing={chart.routing}")
    for code in chart.codes:
        marker = "*" if code.first_listed else " "
        typer.echo(
            f"  {marker} {code.code:<9} conf={code.confidence:<5} {code.description}"
        )
    for abstention in chart.abstentions:
        typer.echo(f"  ? {abstention.reason}")
    for reason in chart.routing_reasons:
        typer.echo(f"  ! {reason}")
    typer.echo(f"\nSaved: {path}  (view in UI: autocoder serve)")


@app.command("eval")
def run_eval(
    golden: Path = typer.Option(Path("autocoder/evalx/golden")),
    reasoner: str = typer.Option("mock"),
    model: str = typer.Option(None),
    db: Path = typer.Option(DEFAULT_DB_PATH),
):
    from autocoder.evalx.harness import run_golden

    report = run_golden(golden, _store(db), _reasoner(reasoner, model))
    typer.echo(json.dumps(report, indent=2))


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
):
    import uvicorn

    uvicorn.run("autocoder.ui.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
