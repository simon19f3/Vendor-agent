"""
Test fixtures.

The production agent (app/llm.py, app/rag.py) has NO offline fallback --
it requires a real Gemini API key and raises a clear error without one.
For the test suite to run deterministically and without network access,
we monkeypatch the two narrow seams those modules expose for exactly this
purpose (`app.llm._call_gemini` and `app.rag._embed_texts`) with small,
deterministic stubs. This is standard dependency injection for tests, not
a production fallback: nothing in app/ ships with a template response.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must be set before app.config is imported anywhere (it's read once at
# import time). A non-empty value is enough for the "is Gemini configured"
# checks to pass; the actual network call is monkeypatched below.
os.environ.setdefault("GEMINI_API_KEY", "test-key-for-pytest")

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.seed import load_vendor_risk, load_vendor_documents
from app.tools import injector
from app import llm as llm_module
from app import rag as rag_module


def _fake_call_gemini(prompt: str) -> str:
    """Deterministic stand-in for app.llm._call_gemini. Extracts the
    Decision/Reasons/Citations already embedded in the prompt by
    generate_explanation() and reformats them -- so tests can assert on
    stable text without hitting the real Gemini API."""
    decision_match = re.search(r"Decision: (\w+)", prompt)
    reasons_match = re.search(r"Reasons: (.+)", prompt)
    citations_match = re.search(r"Citations \(source ids\): (.+)", prompt)
    decision = decision_match.group(1) if decision_match else "UNKNOWN"
    reasons = reasons_match.group(1) if reasons_match else "[]"
    citations = citations_match.group(1) if citations_match else "[]"
    return f"Decision: {decision}. Reason: {reasons}. Evidence cited: {citations}."


_EMBED_DIM = 64


def _fake_embed_texts(texts):
    """Deterministic bag-of-words hashing 'embedding' so cosine similarity
    still meaningfully reflects word overlap between the query and each
    policy chunk, without calling the real Gemini embeddings endpoint."""
    vectors = []
    for text in texts:
        vec = np.zeros(_EMBED_DIM)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            idx = hash(word) % _EMBED_DIM
            vec[idx] += 1.0
        vectors.append(vec)
    return np.array(vectors, dtype=float)


@pytest.fixture(autouse=True)
def stub_gemini(monkeypatch):
    monkeypatch.setattr(llm_module, "_call_gemini", _fake_call_gemini)
    monkeypatch.setattr(rag_module, "_embed_texts", _fake_embed_texts)
    # Force a clean re-index per test so every test's monkeypatched embedder
    # is actually the one used (the store is otherwise a process-wide singleton).
    rag_module.policy_memory._chunks = None
    rag_module.policy_memory._vectors = None
    yield
    rag_module.policy_memory._chunks = None
    rag_module.policy_memory._vectors = None


@pytest.fixture()
def db():
    """Fresh in-memory SQLite DB per test, seeded with the reference data
    (vendor_risk.csv, vendor_documents.json) exactly like production seed.py."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    load_vendor_risk(session)
    load_vendor_documents(session)
    yield session
    session.close()


@pytest.fixture(autouse=True)
def reset_injector():
    """Ensure fault-injection plans never leak between tests."""
    injector.reset()
    yield
    injector.reset()


def base_request(**overrides):
    req = {
        "request_id": "TEST-0000",
        "vendor_name": "SafeCloud",
        "product": "TeamDocs",
        "cost": 8000,
        "intended_use": "Store internal team documents",
        "data_type": "internal",
    }
    req.update(overrides)
    return req