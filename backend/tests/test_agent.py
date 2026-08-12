"""
>= 10 test cases for the Vendor-Assessment Agent, covering:
  - normal approve / reject / escalate paths
  - missing required information
  - tool timeouts (recovered via retry, and exhausted)
  - outdated evidence + fallback search
  - conflicting evidence across two current tier-2 sources
  - prompt-injection resistance (untrusted vendor_document content)
  - duplicate-action prevention (idempotent record_final_decision)
  - max-steps stop condition

Run with:  cd backend && pytest -v
"""
from app.agent import run_agent
from app.tools import injector, VendorDocument
from app.models import VendorDocument as VendorDocumentModel
from .conftest import base_request


# --------------------------------------------------------------------------- #
# 1. Normal approve
# --------------------------------------------------------------------------- #
def test_normal_approve(db):
    req = base_request(request_id="VR-001", vendor_name="SafeCloud", product="TeamDocs",
                        cost=8000, data_type="internal")
    result = run_agent(db, req)
    assert result["decision"] == "APPROVE"
    assert "RISK-001" in result["citations"]
    assert result["stopped_reason"] == "goal_met"
    assert len(result["steps"]) >= 2  # policy + risk lookup at minimum


# --------------------------------------------------------------------------- #
# 2. Missing information (cost is null)
# --------------------------------------------------------------------------- #
def test_missing_cost(db):
    req = base_request(request_id="VR-004", cost=None)
    result = run_agent(db, req)
    assert result["decision"] == "REQUEST_INFORMATION"
    assert result["stopped_reason"] == "missing_information"
    assert "cost" in result["explanation"].lower() or "cost" in str(result["citations"])


# --------------------------------------------------------------------------- #
# 3. Missing information (data_type is null)
# --------------------------------------------------------------------------- #
def test_missing_data_type(db):
    req = base_request(request_id="VR-005", data_type=None)
    result = run_agent(db, req)
    assert result["decision"] == "REQUEST_INFORMATION"
    assert result["stopped_reason"] == "missing_information"


# --------------------------------------------------------------------------- #
# 4. Prohibited vendor -> reject
# --------------------------------------------------------------------------- #
def test_prohibited_vendor_reject(db):
    req = base_request(request_id="VR-003", vendor_name="BlockedSoft", product="SyncNow",
                        cost=5000, data_type="internal")
    result = run_agent(db, req)
    assert result["decision"] == "REJECT"
    assert result["stopped_reason"] == "prohibited_vendor"
    assert "RISK-003" in result["citations"]


# --------------------------------------------------------------------------- #
# 5. Restricted data -> always escalate
# --------------------------------------------------------------------------- #
def test_restricted_data_escalate(db):
    req = base_request(request_id="VR-013", vendor_name="SafeCloud", product="TeamDocs",
                        cost=8000, data_type="restricted")
    result = run_agent(db, req)
    assert result["decision"] == "ESCALATE"
    assert result["stopped_reason"] == "risk_too_high"


# --------------------------------------------------------------------------- #
# 6. Medium risk -> escalate
# --------------------------------------------------------------------------- #
def test_medium_risk_escalate(db):
    req = base_request(request_id="VR-002", vendor_name="DataBridge", product="InsightPro",
                        cost=9000, data_type="internal")
    result = run_agent(db, req)
    assert result["decision"] == "ESCALATE"
    assert "RISK-002" in result["citations"]


# --------------------------------------------------------------------------- #
# 7. Cost above threshold -> escalate
# --------------------------------------------------------------------------- #
def test_cost_over_threshold_escalate(db):
    req = base_request(request_id="VR-012", vendor_name="BudgetSoft", product="NotesPlus",
                        cost=15000, data_type="internal")
    result = run_agent(db, req)
    assert result["decision"] == "ESCALATE"
    assert result["stopped_reason"] == "cost_too_high"


# --------------------------------------------------------------------------- #
# 8. High-risk confidential vendor with a failing security assessment -> reject
# --------------------------------------------------------------------------- #
def test_security_assessment_fail_reject(db):
    req = base_request(request_id="VR-008", vendor_name="SplitWare", product="SecureShare",
                        cost=7000, data_type="confidential")
    result = run_agent(db, req)
    assert result["decision"] == "REJECT"
    assert result["stopped_reason"] == "security_assessment_failed"
    assert "SEC-004" in result["citations"]


# --------------------------------------------------------------------------- #
# 9. Tool timeout on primary route, recovered via backup route -> approve
# --------------------------------------------------------------------------- #
def test_timeout_recovered_via_backup_route(db):
    injector.set("lookup_vendor_risk", "TimeoutLabs", "primary", ["timeout"])
    req = base_request(request_id="VR-006", vendor_name="TimeoutLabs", product="MetricsHub",
                        cost=7500, data_type="internal")
    result = run_agent(db, req)
    assert result["decision"] == "APPROVE"
    # confirm the timeout + retry are both visible in the trace
    actions = [s["action"] for s in result["steps"]]
    observations = [s["observation"] for s in result["steps"]]
    assert actions.count("lookup_vendor_risk") == 2
    assert any(o.get("error") == "timeout" for o in observations)
    assert any(o.get("found") and o.get("route") == "backup" for o in observations)


# --------------------------------------------------------------------------- #
# 10. Tool timeout on every retry route -> retries exhausted -> escalate
# --------------------------------------------------------------------------- #
def test_timeout_exhausted_escalate(db):
    injector.set("lookup_vendor_risk", "FailWare", "primary", ["timeout"])
    injector.set("lookup_vendor_risk", "FailWare", "backup", ["timeout"])
    injector.set("search_vendor_documents", "FailWare", "approved_repository", ["timeout"])
    req = base_request(request_id="VR-007", vendor_name="FailWare", product="OpsMonitor",
                        cost=6000, data_type="confidential")
    result = run_agent(db, req)
    assert result["decision"] == "ESCALATE"
    assert result["stopped_reason"] == "tool_failure_exhausted"
    # exactly 3 attempts were made (1 + MAX_RETRIES_PER_TOOL), not unbounded retries
    risk_attempts = [s for s in result["steps"] if s["action"] in
                      ("lookup_vendor_risk", "search_vendor_documents")]
    assert len(risk_attempts) == 3


# --------------------------------------------------------------------------- #
# 11. Outdated risk evidence, fallback search also outdated -> escalate
# --------------------------------------------------------------------------- #
def test_outdated_evidence_escalate(db):
    req = base_request(request_id="VR-011", vendor_name="OldStack", product="LegacyCRM",
                        cost=4000, data_type="internal")
    result = run_agent(db, req)
    assert result["decision"] == "ESCALATE"
    assert result["stopped_reason"] == "outdated_evidence"
    # confirm the agent actually tried the fallback search before giving up
    assert any(s["action"] == "search_vendor_documents" for s in result["steps"])


# --------------------------------------------------------------------------- #
# 12. Prompt-injection attempt inside a vendor-supplied document is ignored;
#     the agent still approves based only on trusted tier-2 evidence.
# --------------------------------------------------------------------------- #
def test_prompt_injection_ignored(db):
    req = base_request(request_id="VR-009", vendor_name="InjectCorp", product="HelpDesk AI",
                        cost=9000, data_type="confidential")
    result = run_agent(db, req)
    assert result["decision"] == "APPROVE"
    # the malicious tier-3 doc must be flagged/logged, never used as an instruction
    guardrail_steps = [s for s in result["steps"] if s["action"] == "guardrail_check"]
    assert len(guardrail_steps) == 1
    assert guardrail_steps[0]["observation"]["ignored_as_instruction"] is True
    assert "VENDOR-006" not in result["citations"]  # untrusted doc never cited as evidence
    assert "SEC-005" in result["citations"]          # trusted tier-2 assessment cited instead


# --------------------------------------------------------------------------- #
# 13. Duplicate-action prevention: same request_id evaluated twice
# --------------------------------------------------------------------------- #
def test_duplicate_request_returns_existing_decision(db):
    req = base_request(request_id="VR-010", vendor_name="DuplicateCo", product="FlowDesk",
                        cost=6500, data_type="internal")
    first = run_agent(db, req)
    second = run_agent(db, req)
    assert first["decision"] == "APPROVE"
    assert second["decision"] == first["decision"]
    assert second["stopped_reason"] == "duplicate_reused"
    # the second run must not repeat all the tool calls of the first
    assert len(second["steps"]) < len(first["steps"])

    from app.models import DecisionLog
    assert db.query(DecisionLog).filter(DecisionLog.request_id == "VR-010").count() == 1


# --------------------------------------------------------------------------- #
# 14. Conflicting current tier-2 evidence -> escalate
# --------------------------------------------------------------------------- #
def test_conflicting_evidence_escalate(db):
    # Inject a second, conflicting, *current* tier-2 security assessment for
    # SafeCloud so two priority-2 sources disagree on a material fact.
    db.add(VendorDocumentModel(
        document_id="SEC-999",
        vendor_name="SafeCloud",
        product="TeamDocs",
        source_type="approved_security_assessment",
        authority_tier=2,
        document_date="2026-07-30",
        result="fail",
        risk_rating="high",
        content="Conflicting later assessment found critical issues.",
    ))
    db.commit()
    req = base_request(request_id="VR-CONFLICT", vendor_name="SafeCloud", product="TeamDocs",
                        cost=8000, data_type="confidential")
    result = run_agent(db, req)
    assert result["decision"] == "ESCALATE"
    assert result["stopped_reason"] == "conflicting_evidence"


# --------------------------------------------------------------------------- #
# 15. Max-steps stop condition
# --------------------------------------------------------------------------- #
def test_max_steps_hit(db):
    # Force every retryable route to time out so the agent burns through its
    # full step budget without ever reaching a confident decision.
    for route in ("primary", "backup"):
        injector.set("lookup_vendor_risk", "SafeCloud", route, ["timeout"] * 5)
    injector.set("search_vendor_documents", "SafeCloud", "approved_repository", ["timeout"] * 5)
    req = base_request(request_id="VR-MAXSTEPS", vendor_name="SafeCloud", product="TeamDocs",
                        cost=8000, data_type="internal")
    result = run_agent(db, req)
    assert result["decision"] == "ESCALATE"
    assert result["stopped_reason"] in ("tool_failure_exhausted", "max_steps_reached")
    assert len(result["steps"]) <= 8


# --------------------------------------------------------------------------- #
# 16. Change-aware duplicate handling: resubmitting the SAME request_id with
#     DIFFERENT field values must be re-evaluated (not blindly reused), and
#     must update the existing decision row rather than insert a duplicate.
# --------------------------------------------------------------------------- #
def test_same_request_id_changed_payload_is_reevaluated(db):
    first_req = base_request(request_id="VR-014", vendor_name="DataBridge",
                              product="InsightPro", cost=9000, data_type="internal")
    first = run_agent(db, first_req)
    assert first["decision"] == "ESCALATE"  # DataBridge is current medium risk
    assert first["record_action"] == "created"

    # Same request_id, but the vendor/product/data_type were corrected.
    second_req = base_request(request_id="VR-014", vendor_name="SafeCloud",
                               product="TeamDocs", cost=8000, data_type="internal")
    second = run_agent(db, second_req)

    assert second["stopped_reason"] != "duplicate_reused"  # must NOT be a stale reuse
    assert second["decision"] == "APPROVE"                 # re-evaluated against new facts
    assert second["record_action"] == "updated"
    assert second["payload_hash"] != first["payload_hash"]

    # Exactly one row for this request_id -- updated in place, not duplicated.
    from app.models import DecisionLog
    rows = db.query(DecisionLog).filter(DecisionLog.request_id == "VR-014").all()
    assert len(rows) == 1
    assert rows[0].decision == "APPROVE"
    assert rows[0].payload_hash == second["payload_hash"]

    # A genuinely unchanged resubmission (third call, same payload as the
    # second) IS treated as a duplicate and reused.
    third = run_agent(db, second_req)
    assert third["stopped_reason"] == "duplicate_reused"
    assert third["decision"] == "APPROVE"