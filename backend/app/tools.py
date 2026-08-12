"""
Agent tools.

Every tool returns a plain JSON-serialisable dict with at least an "ok" key.
Tools never raise on *expected* failure modes (timeout / not found) -- they
report it in the observation so the agent can reason about retries. They DO
raise/return a structured validation error if the caller (the planner in
agent.py, or ultimately a hallucinated LLM tool-call) passes malformed
arguments -- this is the "function-calling contract" from the design
slides: the model should never be able to invoke a tool with arguments the
runtime hasn't validated against a schema.

`injector` is a tiny fault-injection hook used by the test-suite to simulate
unreliable conditions (timeouts) deterministically, matching the kind of
scenarios captured in tool_scenarios.json, without needing a flaky real
network in tests.
"""
import json
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, ValidationError, field_validator
from sqlalchemy.orm import Session

from .models import VendorRisk, VendorDocument, DecisionLog
from .config import EVALUATION_DATE, EVIDENCE_MAX_AGE_DAYS
from .rag import policy_memory


class FailureInjector:
    """Queues forced outcomes per (tool, vendor, route). Pop-front, else 'success'."""

    def __init__(self):
        self._plan = {}

    def set(self, tool: str, vendor: str, route: str, outcomes):
        self._plan[(tool, vendor, route)] = list(outcomes)

    def next_outcome(self, tool: str, vendor: str, route: str) -> str:
        key = (tool, vendor, route)
        q = self._plan.get(key)
        if q:
            return q.pop(0)
        return "success"

    def reset(self):
        self._plan = {}


injector = FailureInjector()


def _is_current(iso_date_str: str) -> bool:
    d = datetime.strptime(iso_date_str, "%Y-%m-%d").date()
    eval_d = datetime.strptime(EVALUATION_DATE, "%Y-%m-%d").date()
    return (eval_d - d).days <= EVIDENCE_MAX_AGE_DAYS


# ---------------------------------------------------------------------------
# System validation: every tool has a Pydantic argument schema. Validation
# failures are returned as a structured, retryable observation (ok=False,
# error="validation_error") instead of raising -- so the agent's fallback
# ladder can treat "you called the tool wrong" the same way it treats a
# timeout: correct the input and retry, bounded by MAX_RETRIES_PER_TOOL.
# ---------------------------------------------------------------------------
class ValidationFailure(Exception):
    def __init__(self, errors):
        self.errors = errors


def _validate(schema_cls, **kwargs):
    try:
        return schema_cls(**kwargs)
    except ValidationError as exc:
        raise ValidationFailure(exc.errors()) from exc


class LookupVendorRiskArgs(BaseModel):
    vendor_name: str
    route: str = "primary"

    @field_validator("vendor_name")
    @classmethod
    def _non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("vendor_name must be a non-empty string")
        return v

    @field_validator("route")
    @classmethod
    def _valid_route(cls, v):
        if v not in ("primary", "backup"):
            raise ValueError("route must be 'primary' or 'backup'")
        return v


class SearchVendorDocumentsArgs(BaseModel):
    vendor_name: str
    source_type: Optional[str] = None
    query_variant: str = "default"
    route: str = "approved_repository"

    @field_validator("vendor_name")
    @classmethod
    def _non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("vendor_name must be a non-empty string")
        return v

    @field_validator("source_type")
    @classmethod
    def _valid_source_type(cls, v):
        if v is not None and v not in ("approved_security_assessment", "vendor_document"):
            raise ValueError("source_type must be 'approved_security_assessment', 'vendor_document', or null")
        return v


class RetrievePolicyArgs(BaseModel):
    query: str
    k: int = 3

    @field_validator("query")
    @classmethod
    def _non_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("query must be a non-empty string")
        return v

    @field_validator("k")
    @classmethod
    def _bounded(cls, v):
        if not (1 <= v <= 10):
            raise ValueError("k must be between 1 and 10")
        return v


class CalculatorArgs(BaseModel):
    operation: str
    a: float
    b: float

    @field_validator("operation")
    @classmethod
    def _valid_op(cls, v):
        if v not in ("gt", "gte", "lt", "sub"):
            raise ValueError("operation must be one of gt, gte, lt, sub")
        return v


class RecordFinalDecisionArgs(BaseModel):
    request_id: str
    vendor_name: str
    product: str
    decision: str
    explanation: str
    citations: List[str]
    payload_hash: str

    @field_validator("decision")
    @classmethod
    def _valid_decision(cls, v):
        if v not in ("APPROVE", "REQUEST_INFORMATION", "ESCALATE", "REJECT"):
            raise ValueError("decision must be APPROVE, REQUEST_INFORMATION, ESCALATE, or REJECT")
        return v


# ---------------------------------------------------------------------------
# Tool 1: policy retrieval (priority-1 source) -- now RAG-backed.
# ---------------------------------------------------------------------------
def retrieve_policy(query: str, k: int = 3) -> dict:
    try:
        args = _validate(RetrievePolicyArgs, query=query, k=k)
    except ValidationFailure as vf:
        return {"ok": False, "error": "validation_error", "details": vf.errors, "tool": "retrieve_policy"}

    from .rag import EmbeddingUnavailableError
    try:
        chunks = policy_memory.retrieve(args.query, k=args.k)
    except EmbeddingUnavailableError as exc:
        return {"ok": False, "error": "llm_unavailable", "details": str(exc), "tool": "retrieve_policy"}

    return {"ok": True, "source": "vendor_policy.md", "authority_tier": 1, "chunks": chunks}


# ---------------------------------------------------------------------------
# Tool 2: vendor risk lookup (priority-2 source). route: "primary" | "backup"
# ---------------------------------------------------------------------------
def lookup_vendor_risk(db: Session, vendor_name: str, route: str = "primary") -> dict:
    try:
        args = _validate(LookupVendorRiskArgs, vendor_name=vendor_name, route=route)
    except ValidationFailure as vf:
        return {"ok": False, "error": "validation_error", "details": vf.errors, "tool": "lookup_vendor_risk"}

    outcome = injector.next_outcome("lookup_vendor_risk", args.vendor_name, args.route)
    if outcome == "timeout":
        return {"ok": False, "error": "timeout", "route": args.route, "tool": "lookup_vendor_risk"}

    # Backup route: an independent lookup path (e.g. read-replica / export),
    # modelled as a distinct route for retry purposes.
    row = db.query(VendorRisk).filter(VendorRisk.vendor_name == args.vendor_name).first()

    if not row:
        return {"ok": True, "found": False, "route": args.route}

    return {
        "ok": True,
        "found": True,
        "route": args.route,
        "source_id": row.source_id,
        "authority_tier": row.authority_tier,
        "vendor_name": row.vendor_name,
        "product": row.product,
        "status": row.status,
        "risk_rating": row.risk_rating,
        "assessment_date": row.assessment_date,
        "is_current": _is_current(row.assessment_date),
    }


# ---------------------------------------------------------------------------
# Tool 3: vendor document / security-assessment search (priority 2 or 3)
# ---------------------------------------------------------------------------
def search_vendor_documents(
    db: Session,
    vendor_name: str,
    source_type: Optional[str] = None,
    query_variant: str = "default",
    route: str = "approved_repository",
) -> dict:
    try:
        args = _validate(
            SearchVendorDocumentsArgs, vendor_name=vendor_name, source_type=source_type,
            query_variant=query_variant, route=route,
        )
    except ValidationFailure as vf:
        return {"ok": False, "error": "validation_error", "details": vf.errors, "tool": "search_vendor_documents"}

    outcome = injector.next_outcome("search_vendor_documents", args.vendor_name, args.route)
    if outcome == "timeout":
        return {"ok": False, "error": "timeout", "route": args.route, "tool": "search_vendor_documents"}

    q = db.query(VendorDocument).filter(VendorDocument.vendor_name == args.vendor_name)
    if args.source_type:
        q = q.filter(VendorDocument.source_type == args.source_type)
    rows = q.all()

    if not rows:
        return {"ok": True, "found": False, "route": args.route, "query_variant": args.query_variant}

    results = []
    for row in rows:
        results.append({
            "document_id": row.document_id,
            "source_type": row.source_type,
            "authority_tier": row.authority_tier,
            "document_date": row.document_date,
            "result": row.result,
            "risk_rating": row.risk_rating,
            "is_current": _is_current(row.document_date),
            # NOTE: `content` is UNTRUSTED DATA when source_type == "vendor_document".
            # The agent must never treat it as an instruction, only as a field to log.
            "content": row.content,
        })
    return {"ok": True, "found": True, "route": args.route, "query_variant": args.query_variant, "documents": results}


# ---------------------------------------------------------------------------
# Tool 4: calculator (deterministic, no LLM arithmetic)
# ---------------------------------------------------------------------------
def calculator(operation: str, a: float, b: float) -> dict:
    try:
        args = _validate(CalculatorArgs, operation=operation, a=a, b=b)
    except ValidationFailure as vf:
        return {"ok": False, "error": "validation_error", "details": vf.errors, "tool": "calculator"}

    ops = {
        "gt": lambda x, y: x > y,
        "gte": lambda x, y: x >= y,
        "lt": lambda x, y: x < y,
        "sub": lambda x, y: x - y,
    }
    return {"ok": True, "operation": args.operation, "a": args.a, "b": args.b,
            "result": ops[args.operation](args.a, args.b)}


# ---------------------------------------------------------------------------
# Tool 5: record final decision (idempotent -- duplicate-action guard,
# change-aware: a request_id resubmitted with DIFFERENT field values is
# treated as a correction and re-evaluated/updated, not blindly reused).
# ---------------------------------------------------------------------------
def record_final_decision(
    db: Session,
    request_id: str,
    vendor_name: str,
    product: str,
    decision: str,
    explanation: str,
    citations: list,
    payload_hash: str,
) -> dict:
    try:
        args = _validate(
            RecordFinalDecisionArgs, request_id=request_id, vendor_name=vendor_name,
            product=product, decision=decision, explanation=explanation,
            citations=citations, payload_hash=payload_hash,
        )
    except ValidationFailure as vf:
        return {"ok": False, "error": "validation_error", "details": vf.errors, "tool": "record_final_decision"}

    existing = db.query(DecisionLog).filter(DecisionLog.request_id == args.request_id).first()

    if existing and existing.payload_hash == args.payload_hash:
        # True duplicate: same request_id AND same inputs -> reuse, no re-run.
        return {
            "ok": True,
            "created": False,
            "updated": False,
            "reused_existing": True,
            "request_id": args.request_id,
            "decision": existing.decision,
            "explanation": existing.explanation,
            "citations": json.loads(existing.citations or "[]"),
        }

    if existing and existing.payload_hash != args.payload_hash:
        # Same request_id but the underlying request changed (e.g. vendor
        # info was corrected) -- update the record in place rather than
        # silently returning the stale decision or creating a duplicate row.
        existing.vendor_name = args.vendor_name
        existing.product = args.product
        existing.decision = args.decision
        existing.explanation = args.explanation
        existing.citations = json.dumps(args.citations)
        existing.payload_hash = args.payload_hash
        db.commit()
        return {
            "ok": True,
            "created": False,
            "updated": True,
            "reused_existing": False,
            "request_id": args.request_id,
            "decision": args.decision,
            "explanation": args.explanation,
            "citations": args.citations,
        }

    row = DecisionLog(
        request_id=args.request_id,
        vendor_name=args.vendor_name,
        product=args.product,
        decision=args.decision,
        explanation=args.explanation,
        citations=json.dumps(args.citations),
        payload_hash=args.payload_hash,
    )
    db.add(row)
    db.commit()
    return {
        "ok": True,
        "created": True,
        "updated": False,
        "reused_existing": False,
        "request_id": args.request_id,
        "decision": args.decision,
        "explanation": args.explanation,
        "citations": args.citations,
    }