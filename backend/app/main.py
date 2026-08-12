"""
FastAPI app for the Vendor-Assessment Agent.

Endpoints:
  POST /requests            submit a vendor request (stored, not yet evaluated)
  GET  /requests            list submitted requests
  POST /agent/run/{id}      run the agent for a given request_id, return full trace
  GET  /decisions           list all recorded final decisions
  GET  /decisions/{id}      get the decision + full step trace for one request_id
  POST /seed                (re)load reference data (risk db, documents) - dev helper
"""
import json
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .db import Base, engine, SessionLocal, get_db
from .models import DecisionLog, AgentStepLog
from .schemas import VendorRequestIn, AgentRunResult
from .agent import run_agent
from .seed import seed as seed_reference_data
from .rag import policy_memory

app = FastAPI(title="Vendor-Assessment Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of submitted-but-not-yet-evaluated requests (demo only;
# swap for a DB table if you need it to survive restarts before evaluation).
_PENDING_REQUESTS = {}


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    seed_reference_data()
    try:
        policy_memory.build()
    except Exception as exc:
        # Don't crash the whole app on startup (health checks / static
        # endpoints should still work), but make the failure loud: every
        # agent run will hit the same error until GEMINI_API_KEY is fixed.
        print(f"[startup] WARNING: could not build RAG policy-memory index: {exc}")


@app.post("/requests")
def submit_request(req: VendorRequestIn):
    _PENDING_REQUESTS[req.request_id] = req.model_dump()
    return {"ok": True, "request_id": req.request_id}


@app.get("/requests")
def list_requests():
    return list(_PENDING_REQUESTS.values())


@app.post("/agent/run/{request_id}", response_model=AgentRunResult)
def run(request_id: str, db: Session = Depends(get_db)):
    req = _PENDING_REQUESTS.get(request_id)
    if req is None:
        raise HTTPException(404, f"No submitted request found for request_id={request_id}. "
                                  f"POST /requests first.")
    result = run_agent(db, req)
    return result


@app.post("/agent/run_inline", response_model=AgentRunResult)
def run_inline(req: VendorRequestIn, db: Session = Depends(get_db)):
    """Submit + run in a single call (handy for quick testing / the frontend demo)."""
    result = run_agent(db, req.model_dump())
    return result


@app.get("/decisions")
def list_decisions(db: Session = Depends(get_db)):
    rows = db.query(DecisionLog).order_by(DecisionLog.created_at.desc()).all()
    return [
        {
            "request_id": r.request_id,
            "vendor_name": r.vendor_name,
            "product": r.product,
            "decision": r.decision,
            "explanation": r.explanation,
            "citations": json.loads(r.citations or "[]"),
            "payload_hash": r.payload_hash,
            "created_at": str(r.created_at),
            "updated_at": str(r.updated_at) if r.updated_at else None,
        }
        for r in rows
    ]


@app.get("/decisions/{request_id}")
def get_decision(request_id: str, db: Session = Depends(get_db)):
    decision = db.query(DecisionLog).filter(DecisionLog.request_id == request_id).first()
    if not decision:
        raise HTTPException(404, "No decision recorded for this request_id yet.")
    steps = (
        db.query(AgentStepLog)
        .filter(AgentStepLog.request_id == request_id)
        .order_by(AgentStepLog.step_number.asc())
        .all()
    )
    return {
        "request_id": decision.request_id,
        "vendor_name": decision.vendor_name,
        "product": decision.product,
        "decision": decision.decision,
        "explanation": decision.explanation,
        "citations": json.loads(decision.citations or "[]"),
        "payload_hash": decision.payload_hash,
        "created_at": str(decision.created_at),
        "updated_at": str(decision.updated_at) if decision.updated_at else None,
        "steps": [
            {
                "step_number": s.step_number,
                "thought": s.thought,
                "action": s.action,
                "action_input": json.loads(s.action_input or "{}"),
                "observation": json.loads(s.observation or "{}"),
                "retry_of_step": s.retry_of_step,
            }
            for s in steps
        ],
    }


@app.post("/seed")
def reseed():
    seed_reference_data()
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True}