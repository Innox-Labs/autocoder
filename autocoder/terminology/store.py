"""Versioned terminology store (design doc Section 7).

SQLite for the prototype; plain-SQL schema that moves to Postgres unchanged.
No ICD code, description, or rule appears anywhere in source code — this store
is the single source of truth, selected per chart by date of service.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DEFAULT_DB_PATH = Path("data/terminology.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS release (
    id INTEGER PRIMARY KEY,
    system TEXT NOT NULL,
    version TEXT NOT NULL,
    effective_start TEXT NOT NULL,
    effective_end TEXT NOT NULL,
    source_checksum TEXT,
    status TEXT NOT NULL DEFAULT 'staged',
    UNIQUE (system, version)
);
CREATE TABLE IF NOT EXISTS concept (
    id INTEGER PRIMARY KEY,
    release_id INTEGER NOT NULL REFERENCES release(id),
    code TEXT NOT NULL,
    description TEXT NOT NULL,
    billable INTEGER NOT NULL DEFAULT 1,
    parent_code TEXT,
    UNIQUE (release_id, code)
);
CREATE INDEX IF NOT EXISTS idx_concept_code ON concept(release_id, code);
CREATE INDEX IF NOT EXISTS idx_concept_parent ON concept(release_id, parent_code);
CREATE TABLE IF NOT EXISTS alias (
    id INTEGER PRIMARY KEY,
    concept_id INTEGER NOT NULL REFERENCES concept(id),
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'synonym'
);
CREATE INDEX IF NOT EXISTS idx_alias_text ON alias(text);
CREATE TABLE IF NOT EXISTS rule (
    id INTEGER PRIMARY KEY,
    release_id INTEGER NOT NULL REFERENCES release(id),
    code_or_range TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rule_code ON rule(release_id, code_or_range);
CREATE TABLE IF NOT EXISTS guideline (
    id INTEGER PRIMARY KEY,
    release_id INTEGER NOT NULL REFERENCES release(id),
    section TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS emb_index (
    id INTEGER PRIMARY KEY,
    release_id INTEGER NOT NULL REFERENCES release(id),
    model_id TEXT NOT NULL,
    artifact_path TEXT NOT NULL
);
"""


@dataclass
class Release:
    id: int
    system: str
    version: str
    effective_start: date
    effective_end: date
    status: str


@dataclass
class Concept:
    code: str
    description: str
    billable: bool
    parent_code: str | None
    rules: dict[str, list] = field(default_factory=dict)


class TerminologyStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._alias_cache: dict[int, list[tuple[str, int]]] = {}

    # ------------------------------------------------------------- releases

    def create_release(
        self,
        *,
        system: str,
        version: str,
        effective_start: date,
        effective_end: date,
        source_checksum: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT OR REPLACE INTO release "
            "(system, version, effective_start, effective_end, source_checksum, status) "
            "VALUES (?, ?, ?, ?, ?, 'staged')",
            (
                system,
                version,
                effective_start.isoformat(),
                effective_end.isoformat(),
                source_checksum,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def activate(self, system: str, version: str) -> None:
        self.conn.execute(
            "UPDATE release SET status='active' WHERE system=? AND version=?",
            (system, version),
        )
        self.conn.commit()

    def releases(self, system: str = "icd10cm") -> list[Release]:
        rows = self.conn.execute(
            "SELECT * FROM release WHERE system=? ORDER BY effective_start", (system,)
        ).fetchall()
        return [self._release_from_row(r) for r in rows]

    def release_for(
        self, date_of_service: date, system: str = "icd10cm"
    ) -> Release:
        """Select the active release in force on the date of service."""
        row = self.conn.execute(
            "SELECT * FROM release WHERE system=? AND status='active' "
            "AND effective_start<=? AND effective_end>=? "
            "ORDER BY effective_start DESC LIMIT 1",
            (system, date_of_service.isoformat(), date_of_service.isoformat()),
        ).fetchone()
        if row is None:
            raise LookupError(
                f"No active {system} release covers {date_of_service}. "
                "Run `autocoder terminology ingest` and `activate` first."
            )
        return self._release_from_row(row)

    @staticmethod
    def _release_from_row(row: sqlite3.Row) -> Release:
        return Release(
            id=row["id"],
            system=row["system"],
            version=row["version"],
            effective_start=date.fromisoformat(row["effective_start"]),
            effective_end=date.fromisoformat(row["effective_end"]),
            status=row["status"],
        )

    # ------------------------------------------------------------- concepts

    def add_concepts(self, release_id: int, rows: list[tuple]) -> None:
        """rows: (code, description, billable, parent_code)"""
        self.conn.executemany(
            "INSERT OR IGNORE INTO concept "
            "(release_id, code, description, billable, parent_code) "
            "VALUES (?, ?, ?, ?, ?)",
            [(release_id, c, d, int(b), p) for c, d, b, p in rows],
        )
        self.conn.commit()

    def add_aliases(self, release_id: int, rows: list[tuple]) -> None:
        """rows: (code, alias_text, source)"""
        code_ids = {
            r["code"]: r["id"]
            for r in self.conn.execute(
                "SELECT id, code FROM concept WHERE release_id=?", (release_id,)
            )
        }
        payload = [
            (code_ids[code], text.strip(), source)
            for code, text, source in rows
            if code in code_ids and text.strip()
        ]
        self.conn.executemany(
            "INSERT INTO alias (concept_id, text, source) VALUES (?, ?, ?)", payload
        )
        self.conn.commit()

    def add_rules(self, release_id: int, rows: list[tuple]) -> None:
        """rows: (code_or_range, rule_type, payload)  payload is JSON-serializable"""
        self.conn.executemany(
            "INSERT INTO rule (release_id, code_or_range, rule_type, payload_json) "
            "VALUES (?, ?, ?, ?)",
            [(release_id, c, t, json.dumps(p)) for c, t, p in rows],
        )
        self.conn.commit()

    def add_guidelines(self, release_id: int, rows: list[tuple[str, str]]) -> None:
        self.conn.executemany(
            "INSERT INTO guideline (release_id, section, text) VALUES (?, ?, ?)",
            [(release_id, s, t) for s, t in rows],
        )
        self.conn.commit()

    def get_concept(self, release_id: int, code: str) -> Concept | None:
        row = self.conn.execute(
            "SELECT * FROM concept WHERE release_id=? AND code=?", (release_id, code)
        ).fetchone()
        if row is None:
            return None
        concept = Concept(
            code=row["code"],
            description=row["description"],
            billable=bool(row["billable"]),
            parent_code=row["parent_code"],
        )
        for rule in self.conn.execute(
            "SELECT rule_type, payload_json FROM rule "
            "WHERE release_id=? AND code_or_range=?",
            (release_id, code),
        ):
            concept.rules.setdefault(rule["rule_type"], []).extend(
                json.loads(rule["payload_json"])
            )
        return concept

    def children(self, release_id: int, code: str) -> list[Concept]:
        rows = self.conn.execute(
            "SELECT code FROM concept WHERE release_id=? AND parent_code=?",
            (release_id, code),
        ).fetchall()
        return [self.get_concept(release_id, r["code"]) for r in rows]

    def ancestors(self, release_id: int, code: str) -> list[str]:
        chain: list[str] = []
        current = self.get_concept(release_id, code)
        while current is not None and current.parent_code:
            chain.append(current.parent_code)
            current = self.get_concept(release_id, current.parent_code)
        return chain

    def guidelines(self, release_id: int) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT section, text FROM guideline WHERE release_id=?", (release_id,)
        ).fetchall()
        return [(r["section"], r["text"]) for r in rows]

    # ------------------------------------------------------------- lookup

    def exact_alias(self, release_id: int, query: str) -> list[tuple[str, str]]:
        """Return [(code, description)] whose alias exactly matches (case-fold)."""
        rows = self.conn.execute(
            "SELECT c.code, c.description FROM alias a "
            "JOIN concept c ON c.id = a.concept_id "
            "WHERE c.release_id=? AND lower(a.text)=lower(?)",
            (release_id, query.strip()),
        ).fetchall()
        return [(r["code"], r["description"]) for r in rows]

    def alias_corpus(self, release_id: int) -> list[tuple[str, int]]:
        """All (alias_text, concept_id) for the release, cached for fuzzy search."""
        if release_id not in self._alias_cache:
            rows = self.conn.execute(
                "SELECT a.text, a.concept_id FROM alias a "
                "JOIN concept c ON c.id = a.concept_id WHERE c.release_id=?",
                (release_id,),
            ).fetchall()
            self._alias_cache[release_id] = [(r["text"], r["concept_id"]) for r in rows]
        return self._alias_cache[release_id]

    def concept_by_id(self, concept_id: int) -> tuple[str, str] | None:
        row = self.conn.execute(
            "SELECT code, description FROM concept WHERE id=?", (concept_id,)
        ).fetchone()
        return (row["code"], row["description"]) if row else None

    def concepts_matching_prefix(
        self, release_id: int, prefix: str
    ) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT code, description FROM concept "
            "WHERE release_id=? AND code LIKE ? ORDER BY code",
            (release_id, prefix + "%"),
        ).fetchall()
        return [(r["code"], r["description"]) for r in rows]

    def emb_index_path(self, release_id: int) -> tuple[str, str] | None:
        row = self.conn.execute(
            "SELECT model_id, artifact_path FROM emb_index WHERE release_id=? "
            "ORDER BY id DESC LIMIT 1",
            (release_id,),
        ).fetchone()
        return (row["model_id"], row["artifact_path"]) if row else None

    def set_emb_index(self, release_id: int, model_id: str, path: str) -> None:
        self.conn.execute(
            "INSERT INTO emb_index (release_id, model_id, artifact_path) "
            "VALUES (?, ?, ?)",
            (release_id, model_id, path),
        )
        self.conn.commit()


# ----------------------------------------------------------------- helpers

_RANGE_RE = re.compile(r"^([A-Z][0-9][0-9A-Z]{1,5})\s*-\s*([A-Z][0-9][0-9A-Z]{1,5})$")


def code_in_expr(code: str, expr: str) -> bool:
    """Does dotted ``code`` fall under an ICD code-or-range expression?

    Supports single codes/prefixes ("E11", "E11.9", "L97.4-") and ranges
    ("E10-E13"). Comparison is on the undotted form.
    """
    plain = code.replace(".", "").upper()
    expr = expr.replace(".", "").upper().strip()
    match = _RANGE_RE.match(expr)
    if match:
        low, high = match.group(1), match.group(2)
        # Prefix-range semantics: E10-E13 covers E11621.
        return low <= plain[: len(low)] and plain[: len(high)] <= high
    expr = expr.rstrip("-")
    return plain.startswith(expr)
