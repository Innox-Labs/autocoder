# AutoCoder

**v0.1.0 · In development**

AutoCoder is an early medical coding assistant that helps turn clinical notes into suggested diagnosis codes.

The goal is to make coding easier to review, with clear links between suggested
codes and the supporting words in a note.

## What the prototype offers

- Submit a clinical note with its date of service.
- View suggested diagnosis codes and supporting text.
- Review uncertain results before deciding what to use.
- Explore results through a simple review interface.

## Current status

AutoCoder is being built and evaluated. Suggestions can be incomplete or
incorrect, and production accuracy has not been established. Qualified medical
coding professionals should review every result before use.

The current focus is diagnosis coding. Procedure coding and broader inpatient
support are planned areas of development.

## Local development

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
autocoder --help
```

Start the local review interface:

```bash
autocoder serve
```

Open `http://127.0.0.1:8000` in your browser. Coding requires reference data to
be configured separately. The first model run may download model files.

## Development checks

```bash
pytest -q
```

