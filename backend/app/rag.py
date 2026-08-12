"""
RAG-backed "tool memory" for the agent.

Replaces the old `get_policy()` tool (which just dumped the entire raw
markdown file into every run) with a real retrieval-augmented memory:

  - vendor_policy.md is split into semantic chunks (one per policy clause /
    ## section).
  - Each chunk is embedded with the Gemini embedding model and cached
    in-memory as a small vector store (this project's scale doesn't
    warrant a real vector DB -- swap `_VECTORS` for pgvector/FAISS/etc.
    if the corpus grows).
  - The agent calls `retrieve_policy(query, k)` as a tool. It gets back
    only the k most relevant clauses (with a stable `chunk_id` used for
    citations), not the whole document -- this is what makes it "memory"
    rather than a static file read.

This module deliberately has ONE seam for testability: `_embed_texts`.
Production always calls Gemini through it. Tests monkeypatch it with a
deterministic stub so the retrieval pipeline (chunking -> embed -> cosine
similarity -> top-k) is exercised without real network access, while the
shipped code path never silently falls back to a non-Gemini template.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np

from .config import DATA_DIR, GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL


class EmbeddingUnavailableError(RuntimeError):
    """Raised when Gemini embeddings cannot be produced. No offline fallback."""


@dataclass
class Chunk:
    chunk_id: str
    source: str          # e.g. "vendor_policy.md"
    tier: int            # authority tier, 1 = policy
    text: str


def _split_policy_into_chunks(markdown_text: str) -> list[Chunk]:
    """Split vendor_policy.md into one chunk per ## section."""
    sections = re.split(r"\n(?=## )", markdown_text.strip())
    chunks: list[Chunk] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        header_match = re.match(r"^#{1,2}\s*(.+)", section)
        title = header_match.group(1).strip() if header_match else "Overview"
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        chunks.append(Chunk(
            chunk_id=f"policy#{slug}",
            source="vendor_policy.md",
            tier=1,
            text=section,
        ))
    return chunks


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts with Gemini. Raises EmbeddingUnavailableError on
    any failure -- there is intentionally no deterministic offline fallback
    in the production path; the agent requires Gemini to be reachable."""
    if not GEMINI_API_KEY:
        raise EmbeddingUnavailableError(
            "GEMINI_API_KEY is not set. This agent requires Gemini for both "
            "reasoning/explanation and policy-memory retrieval; set it in "
            "backend/.env (see .env.example)."
        )
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        vectors = []
        for t in texts:
            resp = genai.embed_content(model=GEMINI_EMBEDDING_MODEL, content=t)
            vectors.append(resp["embedding"])
        return np.array(vectors, dtype=float)
    except EmbeddingUnavailableError:
        raise
    except Exception as exc:  # network error, bad key, quota, SDK error, ...
        raise EmbeddingUnavailableError(f"Gemini embedding call failed: {exc}") from exc


class PolicyMemoryStore:
    """In-memory vector store over the policy document (and, optionally,
    other reference text). Built once and cached for the process lifetime."""

    def __init__(self):
        self._lock = threading.Lock()
        self._chunks: list[Chunk] | None = None
        self._vectors: np.ndarray | None = None

    def build(self, force: bool = False):
        with self._lock:
            if self._chunks is not None and not force:
                return
            policy_text = (DATA_DIR / "vendor_policy.md").read_text()
            chunks = _split_policy_into_chunks(policy_text)
            vectors = _embed_texts([c.text for c in chunks])
            self._chunks = chunks
            self._vectors = vectors

    def is_ready(self) -> bool:
        return self._chunks is not None

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        if self._chunks is None:
            self.build()
        query_vec = _embed_texts([query])[0]
        sims = self._cosine_sim(self._vectors, query_vec)
        top_idx = np.argsort(-sims)[:k]
        return [
            {
                "chunk_id": self._chunks[i].chunk_id,
                "source": self._chunks[i].source,
                "tier": self._chunks[i].tier,
                "text": self._chunks[i].text,
                "score": float(sims[i]),
            }
            for i in top_idx
        ]

    @staticmethod
    def _cosine_sim(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        v_norm = vector / (np.linalg.norm(vector) + 1e-9)
        return m_norm @ v_norm


# Process-wide singleton -- built once at FastAPI startup (see main.py) and
# reused across every agent run so requests don't pay the embedding cost
# of re-indexing the (static) policy document every time.
policy_memory = PolicyMemoryStore()