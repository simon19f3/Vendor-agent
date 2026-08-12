"""
Thin wrapper around Gemini used to produce the final natural-language
explanation from the agent's *already-decided* facts (decision, citations,
policy clauses applied). The LLM is deliberately kept OUT of the control
flow / tool-selection loop for this agent: tool orchestration, retries and
the decision itself are computed deterministically in agent.py so the
system stays auditable and testable. Gemini is used only to phrase the
final rationale in clear prose that cites evidence.

IMPORTANT: unlike an earlier version of this module, there is NO offline
/ deterministic template fallback. This agent is designed to run with
Gemini available; if GEMINI_API_KEY is missing or every call attempt
fails, `generate_explanation` raises `LLMUnavailableError` rather than
silently degrading, since a compliance rationale that wasn't actually
reviewed/phrased by the model must never be presented as if it were.

A bounded number of transient-failure retries (GEMINI_CALL_MAX_RETRIES)
is attempted first -- this mirrors the same "retry with a different
approach before giving up" policy applied to the other tools, rather than
failing on the very first network hiccup.
"""
import time

from .config import GEMINI_API_KEY, GEMINI_CALL_MAX_RETRIES, GEMINI_MODEL


class LLMUnavailableError(RuntimeError):
    """Raised when Gemini cannot produce an explanation after all retries."""


def _call_gemini(prompt: str) -> str:
    """Single call attempt. Isolated as its own function so tests can
    monkeypatch it directly instead of mocking the google SDK."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    if not text:
        raise LLMUnavailableError("Gemini returned an empty response.")
    return text


def generate_explanation(decision: str, reasons: list, citations: list) -> str:
    if not GEMINI_API_KEY:
        raise LLMUnavailableError(
            "GEMINI_API_KEY is not set. This agent requires Gemini to phrase "
            "decision rationales; set it in backend/.env (see .env.example) "
            "and restart the app."
        )

    prompt = (
        "You are writing a one-paragraph audit rationale for an automated "
        "vendor-assessment decision. Do not invent facts. Only restate the "
        "reasons and citations given below in clear prose.\n\n"
        f"Decision: {decision}\n"
        f"Reasons: {reasons}\n"
        f"Citations (source ids): {citations}\n"
    )

    last_error: Exception | None = None
    for attempt in range(1 + GEMINI_CALL_MAX_RETRIES):
        try:
            return _call_gemini(prompt)
        except Exception as exc:  # network error, quota, bad key, empty response, ...
            last_error = exc
            if attempt < GEMINI_CALL_MAX_RETRIES:
                time.sleep(min(0.5 * (attempt + 1), 2))  # small backoff between retries
                continue
    raise LLMUnavailableError(
        f"Gemini call failed after {1 + GEMINI_CALL_MAX_RETRIES} attempt(s): {last_error}"
    ) from last_error