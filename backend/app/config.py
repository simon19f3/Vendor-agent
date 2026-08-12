import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load backend/.env (see backend/.env.example). Explicit path so this works
# regardless of the current working directory the app/tests are launched
# from (uvicorn, pytest, scripts/*, etc.). Real environment variables (e.g.
# set in CI or a container) always take precedence over .env values.
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

DATA_DIR = BASE_DIR / "data"

# Postgres in production, e.g.:
#   postgresql+psycopg2://user:pass@localhost:5432/vendor_agent
# Defaults to a local SQLite file so the project runs out-of-the-box
# for tests/demo without a Postgres server installed.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'vendor_agent.db'}")

# --- Gemini ---------------------------------------------------------------
# Required. Set GEMINI_API_KEY in backend/.env (copy from .env.example).
# The agent has no offline/deterministic fallback for reasoning or policy
# retrieval: if this is unset, llm.py and rag.py raise a clear error rather
# than silently degrading to a canned template.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

EVALUATION_DATE = os.getenv("EVALUATION_DATE", "2026-08-01")
EVIDENCE_MAX_AGE_DAYS = int(os.getenv("EVIDENCE_MAX_AGE_DAYS", "180"))
COST_ESCALATION_THRESHOLD = float(os.getenv("COST_ESCALATION_THRESHOLD", "10000"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "8"))
MAX_RETRIES_PER_TOOL = int(os.getenv("MAX_RETRIES_PER_TOOL", "2"))  # in addition to the first attempt
GEMINI_CALL_MAX_RETRIES = int(os.getenv("GEMINI_CALL_MAX_RETRIES", "2"))  # transient API-call retries