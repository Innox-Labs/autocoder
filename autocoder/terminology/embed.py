"""Per-release embedding index for semantic candidate retrieval.

Built once at ingestion time (never at query time). Stored as a .npz of
L2-normalized vectors + concept codes; cosine search is a matrix multiply, so
no FAISS dependency is needed at prototype scale (~100k codes).

Requires the optional extra:  uv pip install -e ".[embed]"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from autocoder.terminology.store import TerminologyStore

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def build_index(
    store: TerminologyStore,
    release_id: int,
    *,
    model_id: str = DEFAULT_EMBED_MODEL,
    out_dir: str | Path = "data/emb",
) -> Path:
    from sentence_transformers import SentenceTransformer

    rows = store.conn.execute(
        "SELECT c.code, c.description, group_concat(a.text, ' | ') AS aliases "
        "FROM concept c LEFT JOIN alias a ON a.concept_id = c.id "
        "WHERE c.release_id=? AND c.billable=1 GROUP BY c.id",
        (release_id,),
    ).fetchall()
    codes = [r["code"] for r in rows]
    texts = [
        f"{r['description']}. {r['aliases'] or ''}"[:512] for r in rows
    ]

    model = SentenceTransformer(model_id)
    vectors = model.encode(
        texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"release_{release_id}.npz"
    np.savez_compressed(path, vectors=vectors.astype(np.float32), codes=np.array(codes))
    store.set_emb_index(release_id, model_id, str(path))
    return path


class SemanticIndex:
    """Loads a prebuilt index and answers top-k cosine queries."""

    def __init__(self, artifact_path: str | Path, model_id: str) -> None:
        data = np.load(artifact_path, allow_pickle=False)
        self.vectors: np.ndarray = data["vectors"]
        self.codes: list[str] = [str(c) for c in data["codes"]]
        self.model_id = model_id
        self._encoder = None

    def _encode(self, text: str) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_id)
        return self._encoder.encode([text], normalize_embeddings=True)[0]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        vector = self._encode(query)
        scores = self.vectors @ vector
        top = np.argsort(-scores)[:k]
        return [(self.codes[i], float(scores[i])) for i in top]


def load_index(store: TerminologyStore, release_id: int) -> SemanticIndex | None:
    entry = store.emb_index_path(release_id)
    if entry is None:
        return None
    model_id, path = entry
    if not Path(path).exists():
        return None
    return SemanticIndex(path, model_id)
