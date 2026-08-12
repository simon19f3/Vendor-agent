from sqlalchemy import Column, Integer, String, Float, Text, DateTime, UniqueConstraint
from sqlalchemy.sql import func

from .db import Base


class VendorRisk(Base):
    """Internal vendor-risk database (priority-2 source)."""

    __tablename__ = "vendor_risk"

    id = Column(Integer, primary_key=True)
    vendor_id = Column(String, unique=True)
    vendor_name = Column(String, index=True)
    product = Column(String)
    status = Column(String)  # approved | prohibited
    risk_rating = Column(String)  # low | medium | high
    assessment_date = Column(String)  # ISO date
    source_id = Column(String)
    source_type = Column(String)
    authority_tier = Column(Integer)


class VendorDocument(Base):
    """Security assessments (tier 2) and vendor-supplied docs (tier 3, untrusted)."""

    __tablename__ = "vendor_documents"

    id = Column(Integer, primary_key=True)
    document_id = Column(String, unique=True)
    vendor_name = Column(String, index=True)
    product = Column(String)
    source_type = Column(String)  # approved_security_assessment | vendor_document
    authority_tier = Column(Integer)
    document_date = Column(String)
    result = Column(String)  # pass | fail | claimed_pass
    risk_rating = Column(String)
    content = Column(Text)


class DecisionLog(Base):
    """Final, idempotent decision record. One row per request_id (dedup guard)."""

    __tablename__ = "decision_log"
    __table_args__ = (UniqueConstraint("request_id", name="uq_decision_request_id"),)

    id = Column(Integer, primary_key=True)
    request_id = Column(String, index=True, nullable=False)
    vendor_name = Column(String)
    product = Column(String)
    decision = Column(String)  # APPROVE | REQUEST_INFORMATION | ESCALATE | REJECT
    explanation = Column(Text)
    citations = Column(Text)  # JSON-encoded list of source ids
    # Hash of the normalized request payload (vendor_name, product, cost,
    # intended_use, data_type) that produced this decision. Lets the agent
    # tell a true duplicate (same request_id, same inputs -> reuse) apart
    # from a corrected/updated request that happens to reuse an old
    # request_id (different inputs -> must be re-evaluated, not reused).
    payload_hash = Column(String, index=True, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AgentStepLog(Base):
    """Full ReAct trace: one row per agent step (thought/action/observation)."""

    __tablename__ = "agent_step_log"

    id = Column(Integer, primary_key=True)
    request_id = Column(String, index=True, nullable=False)
    run_id = Column(String, index=True, nullable=False)
    step_number = Column(Integer)
    thought = Column(Text)
    action = Column(String)      # tool name, or "final_decision"
    action_input = Column(Text)  # JSON-encoded
    observation = Column(Text)   # JSON-encoded
    retry_of_step = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())