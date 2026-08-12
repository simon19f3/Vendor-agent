"""
ReAct-style Vendor-Assessment Agent.

Design (see mentor slides):
  - The Brain:   deterministic policy rules (auditable) drive WHAT counts as
                 a stop condition; Gemini (llm.py) phrases the final
                 rationale and (rag.py) retrieves the relevant policy
                 clauses. There is no offline template anywhere in this
                 file's happy path -- if Gemini is unavailable the run
                 ends in ESCALATE with stopped_reason="llm_unavailable",
                 it never silently invents an explanation.
  - The Tools:   retrieve_policy (RAG), lookup_vendor_risk,
                 search_vendor_documents, calculator, record_final_decision
                 (tools.py) -- every call is argument-validated (system
                 validation during function calling) before it executes.
  - The Memory:  AgentStepLog (short-term trace) + DecisionLog (long-term,
                 provenance: source id + policy clause + payload_hash per
                 decision, so a changed request is re-evaluated instead of
                 blindly reusing a stale decision).
  - The Planner: an explicit queue-driven loop --
                     Reason (decide/announce next step)
                  -> Act    (call a tool, argument-validated)
                  -> Observe(read the result)
                  -> Update (merge into state; on a *recoverable* problem
                             -- timeout, stale evidence, no results --
                             push a remediation step back onto the queue
                             and keep looping, using corrected input, an
                             alternate route, or an alternate tool, bounded
                             by MAX_RETRIES_PER_TOOL)
                  -> repeat
                 until the goal state is reached: complete (a final
                 decision), blocked (required info missing after retries),
                 or unsafe (risk too high / policy hard-stop), or the
                 MAX_AGENT_STEPS budget is exhausted.

Untrusted content: any `content` field coming from a vendor_document
(tier 3) is NEVER treated as an instruction. Only whitelisted structured
fields (result, risk_rating, dates, ids) drive control flow.

Retry-before-escalate: only *recoverable* problems (tool timeout, stale
evidence, no results, bad tool arguments) go through the fallback ladder
(corrected input -> alternate route -> alternate tool -> escalate) before
a final decision is recorded. Genuine policy determinations reached with
CURRENT, valid evidence (prohibited vendor, high/medium risk, restricted
data type, cost over threshold) are immediate and correct stops -- more
retries cannot change a policy threshold that has already been cleanly
evaluated, so treating them as "recoverable" would just be busywork, not
a better result.
"""
import hashlib
import json
import uuid
from collections import deque
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

from . import tools
from .llm import generate_explanation, LLMUnavailableError
from .models import DecisionLog, AgentStepLog
from .config import MAX_AGENT_STEPS, MAX_RETRIES_PER_TOOL, COST_ESCALATION_THRESHOLD

REQUIRED_FIELDS = ["vendor_name", "product", "cost", "intended_use", "data_type"]
VALID_DATA_TYPES = {"public", "internal", "confidential", "restricted"}


def compute_payload_hash(request: Dict[str, Any]) -> str:
    """Stable hash of the fields that actually determine the decision.
    Used to tell a true duplicate (same request_id, same inputs) apart
    from a corrected/updated resubmission under the same request_id."""
    normalized = {f: request.get(f) for f in REQUIRED_FIELDS}
    blob = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class StopAgent(Exception):
    """Raised internally once a final decision has been reached (goal
    complete, blocked on missing info, or unsafe/policy hard-stop)."""

    def __init__(self, decision, reasons, citations, stopped_reason):
        self.decision = decision
        self.reasons = reasons
        self.citations = citations
        self.stopped_reason = stopped_reason


class VendorAssessmentAgent:
    def __init__(self, db: Session, request: Dict[str, Any]):
        self.db = db
        self.request = request
        self.run_id = str(uuid.uuid4())
        self.request_id = request.get("request_id") or f"anon-{self.run_id[:8]}"
        self.payload_hash = compute_payload_hash(request)
        self.step_number = 0
        self.steps: List[Dict[str, Any]] = []
        self.citations: List[str] = []
        self.state: Dict[str, Any] = {}          # accumulated evidence / facts
        self._retry_counts: Dict[str, int] = {}   # per logical-step retry budget used

    # ------------------------------------------------------------------ #
    # logging / bookkeeping
    # ------------------------------------------------------------------ #
    def _log_step(self, thought, action, action_input, observation, retry_of=None):
        self.step_number += 1
        entry = {
            "step_number": self.step_number,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation,
            "retry_of_step": retry_of,
        }
        self.steps.append(entry)
        self.db.add(AgentStepLog(
            request_id=self.request_id,
            run_id=self.run_id,
            step_number=entry["step_number"],
            thought=thought,
            action=action,
            action_input=json.dumps(action_input),
            observation=json.dumps(observation),
            retry_of_step=retry_of,
        ))
        self.db.commit()
        if self.step_number >= MAX_AGENT_STEPS:
            raise StopAgent(
                "ESCALATE",
                ["Maximum agent steps reached before a confident decision could be made."],
                self.citations,
                "max_steps_reached",
            )
        return entry

    def _cite(self, source_id: str):
        if source_id and source_id not in self.citations:
            self.citations.append(source_id)

    # ------------------------------------------------------------------ #
    # Generic retry-before-escalate ladder. This is the single mechanism
    # used everywhere a tool call can fail recoverably: it tries up to
    # 1 + MAX_RETRIES_PER_TOOL attempts, each one representing a distinct
    # Reason -> Act -> Observe -> Update iteration of the loop, using a
    # *different* approach each time (different route, different tool,
    # or corrected/broadened query) as required by policy. Only once the
    # ladder is exhausted does the caller get to decide to escalate.
    # ------------------------------------------------------------------ #
    def _attempt_with_retries(self, tool_name, vendor_name, attempts_spec, thought_prefix,
                               step_key: Optional[str] = None):
        """
        attempts_spec: list of dicts like {"route": "primary", "call": callable, "note": "..."}
        Returns the first observation with ok=True, else the last failed observation.
        """
        step_key = step_key or f"{tool_name}:{vendor_name}"
        last_obs = None
        first_step = None
        budget = min(len(attempts_spec), 1 + MAX_RETRIES_PER_TOOL)
        for i, spec in enumerate(attempts_spec[:budget]):
            obs = spec["call"]()
            is_retry = i > 0
            if is_retry:
                self._retry_counts[step_key] = self._retry_counts.get(step_key, 0) + 1
            thought = (
                f"{thought_prefix} (attempt {i + 1}/{budget}, route={spec['route']}"
                f"{', RETRY using a different approach per fallback policy' if is_retry else ''})"
                f": {spec['note']}"
            )
            entry = self._log_step(
                thought=thought,
                action=tool_name,
                action_input={"vendor_name": vendor_name, "route": spec["route"]},
                observation=obs,
                retry_of=first_step,
            )
            if first_step is None:
                first_step = entry["step_number"]
            last_obs = obs
            if obs.get("ok") and obs.get("error") != "validation_error":
                return last_obs
        return last_obs

    # ------------------------------------------------------------------ #
    # main loop: an explicit queue of pending checks. Handlers may push
    # additional steps back onto the front of the queue (e.g. "the risk
    # record is stale -> go gather fresher evidence next") instead of
    # deciding to escalate immediately -- this is what keeps the agent
    # iterating (Reason -> Act -> Observe -> Update -> repeat) toward a
    # *better* result rather than stopping at the first sign of trouble.
    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, Any]:
        self._pending = deque([
            "duplicate_guard",
            "retrieve_policy",
            "validate_required_fields",
            "check_restricted_data",
            "gather_risk_evidence",
            "apply_risk_rules",
            "check_cost",
            "check_evidence_requirements",
            "approve",
        ])
        try:
            while self._pending:
                step_name = self._pending.popleft()
                handler = getattr(self, f"_step_{step_name}")
                handler()
            # Queue drained without an explicit stop -- shouldn't happen
            # ("approve" always raises StopAgent) but fail safe if it does.
            raise StopAgent("ESCALATE", ["Planner exhausted its checklist without reaching a "
                                          "conclusive decision."], self.citations, "planner_incomplete")
        except StopAgent as stop:
            return self._finalize(stop)

    # ------------------------------------------------------------------ #
    # steps
    # ------------------------------------------------------------------ #
    def _step_duplicate_guard(self):
        existing = self.db.query(DecisionLog).filter(
            DecisionLog.request_id == self.request_id
        ).first()
        if not existing:
            return
        if existing.payload_hash == self.payload_hash:
            self._log_step(
                thought="A final decision already exists for this request_id AND the "
                        "request's fields are unchanged; returning the existing decision "
                        "instead of re-running the agent (duplicate-action prevention).",
                action="check_decision_log",
                action_input={"request_id": self.request_id, "payload_hash": self.payload_hash},
                observation={"ok": True, "found": True, "same_payload": True,
                             "decision": existing.decision},
            )
            raise StopAgent(
                existing.decision,
                ["Duplicate request_id with identical inputs: prior decision reused "
                 "without re-evaluation."],
                json.loads(existing.citations or "[]"),
                "duplicate_reused",
            )
        # Same request_id, but the request content changed (e.g. vendor
        # name/cost/data_type corrected) -- do NOT reuse; log why we're
        # continuing, then fall through and re-run the full checklist.
        self._log_step(
            thought=f"A decision already exists for request_id={self.request_id}, but the "
                    f"submitted fields differ from what produced that decision (previous "
                    f"payload_hash={existing.payload_hash}, current={self.payload_hash}). "
                    f"Treating this as a corrected request and re-evaluating from scratch "
                    f"rather than returning the stale decision.",
            action="check_decision_log",
            action_input={"request_id": self.request_id, "payload_hash": self.payload_hash},
            observation={"ok": True, "found": True, "same_payload": False,
                         "previous_decision": existing.decision},
        )

    def _step_retrieve_policy(self):
        attempts = [
            {
                "route": "vector_store",
                "note": "Retrieve the governing policy clauses relevant to intake "
                        "requirements (RAG lookup over vendor_policy.md).",
                "call": lambda: tools.retrieve_policy(
                    "required information cost data type vendor risk evidence approval", k=5),
            },
        ]
        obs = self._attempt_with_retries("retrieve_policy", "policy", attempts,
                                          "Reason: ground this run in the current policy "
                                          "before evaluating anything else",
                                          step_key="retrieve_policy")
        if not obs.get("ok"):
            # No offline fallback -- if Gemini/embeddings are unreachable we
            # cannot ground the decision in policy text, so we stop rather
            # than guess.
            raise StopAgent(
                "ESCALATE",
                [f"Policy memory (RAG over vendor_policy.md) could not be retrieved: "
                 f"{obs.get('details') or obs.get('error')}."],
                self.citations,
                "llm_unavailable",
            )
        for chunk in obs["chunks"]:
            self._cite(chunk["chunk_id"])
        self.state["policy_chunks"] = obs["chunks"]

    def _step_validate_required_fields(self):
        missing = [f for f in REQUIRED_FIELDS if self.request.get(f) in (None, "")]
        if missing:
            raise StopAgent(
                "REQUEST_INFORMATION",
                [f"Missing required field(s): {', '.join(missing)} (policy: 'Required information')"],
                self.citations,
                "missing_information",
            )
        if self.request["data_type"] not in VALID_DATA_TYPES:
            raise StopAgent(
                "REQUEST_INFORMATION",
                [f"data_type '{self.request['data_type']}' is not one of the allowed values."],
                self.citations,
                "missing_information",
            )

    def _step_check_restricted_data(self):
        if self.request["data_type"] == "restricted":
            raise StopAgent(
                "ESCALATE",
                ["data_type is 'restricted', which always requires escalation per policy."],
                self.citations,
                "risk_too_high",
            )

    def _step_gather_risk_evidence(self):
        vendor = self.request["vendor_name"]
        attempts = [
            {
                "route": "primary",
                "note": "Query the internal vendor-risk database (primary route).",
                "call": lambda: tools.lookup_vendor_risk(self.db, vendor, route="primary"),
            },
            {
                "route": "backup",
                "note": "Primary route failed; retry with a DIFFERENT route (backup) per "
                        "the fallback ladder, rather than escalating immediately.",
                "call": lambda: tools.lookup_vendor_risk(self.db, vendor, route="backup"),
            },
            {
                "route": "alternate_tool:search_vendor_documents",
                "note": "Both risk-lookup routes failed; escalate to a DIFFERENT TOOL "
                        "entirely (document search) to try to establish equivalent, "
                        "current risk evidence before giving up.",
                "call": lambda: tools.search_vendor_documents(
                    self.db, vendor, source_type="approved_security_assessment",
                    query_variant="fallback_for_risk", route="approved_repository",
                ),
            },
        ]
        obs = self._attempt_with_retries("lookup_vendor_risk", vendor, attempts,
                                          "Reason: look up the current vendor risk status",
                                          step_key=f"risk_evidence:{vendor}")
        if not obs.get("ok"):
            raise StopAgent(
                "ESCALATE",
                [f"Vendor-risk evidence could not be retrieved for '{vendor}' after "
                 f"exhausting all {1 + MAX_RETRIES_PER_TOOL} retry routes/tools "
                 f"(repeated tool timeouts)."],
                self.citations,
                "tool_failure_exhausted",
            )
        self.state["risk_obs"] = obs

    def _step_apply_risk_rules(self):
        risk_obs = self.state["risk_obs"]
        vendor = self.request["vendor_name"]

        # risk_obs may have come from the alternate-tool fallback (documents payload)
        if "documents" in risk_obs or (risk_obs.get("found") is False and "risk_rating" not in risk_obs
                                        and "documents" not in risk_obs and self._came_from_doc_fallback()):
            docs = risk_obs.get("documents", [])
            current_pass = [d for d in docs if d["is_current"] and d["source_type"] == "approved_security_assessment"
                             and d["result"] == "pass"]
            if not risk_obs.get("found") or not current_pass:
                raise StopAgent(
                    "ESCALATE",
                    [f"No current vendor-risk evidence found for '{vendor}' via any "
                     f"available route (risk database unreachable, no current "
                     f"corroborating assessment on file)."],
                    self.citations,
                    "missing_evidence",
                )
            doc = current_pass[0]
            self._cite(doc["document_id"])
            self.state["risk_rating"] = doc["risk_rating"]
            self.state["risk_is_current"] = True
            self.state["risk_source_id"] = doc["document_id"]
            return

        if not risk_obs.get("found"):
            raise StopAgent(
                "ESCALATE",
                [f"No vendor-risk record on file for '{vendor}'; required current "
                 f"evidence is missing."],
                self.citations,
                "missing_evidence",
            )

        self._cite(risk_obs["source_id"])

        if risk_obs["status"] == "prohibited":
            raise StopAgent(
                "REJECT",
                [f"Vendor/product is marked 'prohibited' in the vendor-risk database "
                 f"({risk_obs['source_id']})."],
                self.citations,
                "prohibited_vendor",
            )

        if risk_obs["risk_rating"] == "high":
            raise StopAgent(
                "REJECT",
                [f"Current vendor risk rating is 'high' ({risk_obs['source_id']})."],
                self.citations,
                "risk_too_high",
            )

        if risk_obs["risk_rating"] == "medium":
            raise StopAgent(
                "ESCALATE",
                [f"Current vendor risk rating is 'medium' ({risk_obs['source_id']}); "
                 f"policy requires escalation."],
                self.citations,
                "risk_too_high",
            )

        # risk_rating == "low" from here on
        if not risk_obs["is_current"]:
            # Recoverable: don't escalate yet -- try a bounded alternate-tool
            # search for fresher evidence first (fallback ladder step 2/3).
            attempts = [
                {
                    "route": "approved_repository",
                    "note": f"Risk record {risk_obs['source_id']} is older than the "
                            f"180-day currency window; search for a more recent "
                            f"approved assessment before escalating.",
                    "call": lambda: tools.search_vendor_documents(
                        self.db, vendor, source_type="approved_security_assessment",
                        query_variant="check_for_newer_evidence", route="approved_repository"),
                },
                {
                    "route": "secondary_repository",
                    "note": "Still stale; retry via a different route (secondary repository).",
                    "call": lambda: tools.search_vendor_documents(
                        self.db, vendor, source_type="approved_security_assessment",
                        query_variant="check_for_newer_evidence", route="secondary_repository"),
                },
            ]
            obs = self._attempt_with_retries(
                "search_vendor_documents", vendor, attempts,
                "Reason: existing risk evidence is outdated, look for something fresher",
                step_key=f"risk_freshness:{vendor}",
            )
            current_docs = [
                d for d in obs.get("documents", [])
                if d["is_current"] and d["result"] == "pass"
            ] if obs.get("ok") and obs.get("found") else []
            if not current_docs:
                raise StopAgent(
                    "ESCALATE",
                    [f"Vendor-risk evidence for '{vendor}' is outdated "
                     f"({risk_obs['source_id']}, dated {risk_obs['assessment_date']}) and no "
                     f"current corroborating evidence could be found after retries."],
                    self.citations,
                    "outdated_evidence",
                )
            self._cite(current_docs[0]["document_id"])

        self.state["risk_rating"] = "low"
        self.state["risk_is_current"] = True
        self.state["risk_source_id"] = risk_obs["source_id"]

    def _step_check_cost(self):
        cost = self.request["cost"]
        obs = tools.calculator("gt", cost, COST_ESCALATION_THRESHOLD)
        if obs.get("error") == "validation_error":
            # Recoverable input problem: try a corrected/coerced value once
            # before giving up and asking the requester for clean input.
            try:
                coerced = float(str(cost).replace(",", "").replace("$", ""))
            except (TypeError, ValueError):
                coerced = None
            self._log_step(
                thought=f"Reason: cost value '{cost}' failed argument validation for the "
                        f"calculator tool; retrying once with a corrected/coerced numeric "
                        f"value before requesting information.",
                action="calculator",
                action_input={"operation": "gt", "a": cost, "b": COST_ESCALATION_THRESHOLD},
                observation=obs,
            )
            if coerced is None:
                raise StopAgent(
                    "REQUEST_INFORMATION",
                    [f"cost '{cost}' is not a valid number and could not be corrected."],
                    self.citations,
                    "missing_information",
                )
            obs = tools.calculator("gt", coerced, COST_ESCALATION_THRESHOLD)
            cost = coerced

        self._log_step(
            thought=f"Reason: check whether cost ({cost}) exceeds the escalation "
                    f"threshold ({COST_ESCALATION_THRESHOLD}).",
            action="calculator",
            action_input={"operation": "gt", "a": cost, "b": COST_ESCALATION_THRESHOLD},
            observation=obs,
        )
        if obs["result"]:
            raise StopAgent(
                "ESCALATE",
                [f"Cost ({cost}) exceeds the USD {COST_ESCALATION_THRESHOLD} escalation threshold."],
                self.citations,
                "cost_too_high",
            )

    def _step_check_evidence_requirements(self):
        data_type = self.request["data_type"]
        if data_type in ("public", "internal"):
            return  # current low-risk record already established, sufficient.

        if data_type != "confidential":
            return

        vendor = self.request["vendor_name"]
        attempts = [
            {
                "route": "approved_repository",
                "note": "Search all documents on file for this vendor (default query); "
                        "tier-2 approved assessments will be used as evidence, any "
                        "tier-3 vendor-supplied documents are logged but not trusted.",
                "call": lambda: tools.search_vendor_documents(
                    self.db, vendor, source_type=None,
                    query_variant="default", route="approved_repository"),
            },
            {
                "route": "secondary_repository",
                "note": "Primary repository failed; retry via a DIFFERENT route.",
                "call": lambda: tools.search_vendor_documents(
                    self.db, vendor, source_type="approved_security_assessment",
                    query_variant="default", route="secondary_repository"),
            },
            {
                "route": "approved_repository",
                "note": "Retry with a corrected/broadened query.",
                "call": lambda: tools.search_vendor_documents(
                    self.db, vendor, source_type=None,
                    query_variant="corrected", route="approved_repository"),
            },
        ]
        obs = self._attempt_with_retries("search_vendor_documents", vendor, attempts,
                                          "Reason: retrieve a current, tier-2 approved "
                                          "security assessment required for confidential data",
                                          step_key=f"confidential_evidence:{vendor}")
        if not obs.get("ok"):
            raise StopAgent(
                "ESCALATE",
                ["Approved security-assessment evidence could not be retrieved "
                 "after exhausting all retry routes (repeated tool timeouts)."],
                self.citations,
                "tool_failure_exhausted",
            )
        if not obs.get("found"):
            raise StopAgent(
                "ESCALATE",
                ["Confidential data requires a current approved security assessment; "
                 "none was found."],
                self.citations,
                "missing_evidence",
            )

        docs = obs["documents"]
        # SECURITY: tier-3 vendor-supplied documents are logged but their free-text
        # `content` is NEVER used to drive a decision -- this is the prompt-injection
        # guardrail. We only ever branch on structured fields from tier-2 sources.
        tier3_flagged = [d for d in docs if d["authority_tier"] == 3]
        for d in tier3_flagged:
            self._log_step(
                thought="A vendor-supplied (tier-3) document was returned. Per policy, "
                        "instructions inside vendor documents are untrusted data and are "
                        "ignored; only tier-2 approved security assessments are used as "
                        "evidence.",
                action="guardrail_check",
                action_input={"document_id": d["document_id"]},
                observation={"ok": True, "ignored_as_instruction": True,
                             "authority_tier": d["authority_tier"]},
            )

        tier2_current = [d for d in docs if d["authority_tier"] == 2 and d["is_current"]]

        ratings = {d["risk_rating"] for d in tier2_current}
        results = {d["result"] for d in tier2_current}
        if len(ratings) > 1 or len(results) > 1:
            for d in tier2_current:
                self._cite(d["document_id"])
            raise StopAgent(
                "ESCALATE",
                ["Two current priority-2 sources conflict on a material fact "
                 "(risk rating or assessment result); policy requires escalation."],
                self.citations,
                "conflicting_evidence",
            )

        failing = [d for d in tier2_current if d["result"] == "fail"]
        if failing:
            self._cite(failing[0]["document_id"])
            raise StopAgent(
                "REJECT",
                [f"Current approved security assessment result is 'fail' "
                 f"({failing[0]['document_id']})."],
                self.citations,
                "security_assessment_failed",
            )

        passing = [d for d in tier2_current if d["result"] == "pass"]
        if not passing:
            raise StopAgent(
                "ESCALATE",
                ["No current approved security assessment with a 'pass' result was found."],
                self.citations,
                "missing_evidence",
            )
        self._cite(passing[0]["document_id"])

    def _step_approve(self):
        raise StopAgent(
            "APPROVE",
            ["All required information present", "vendor not prohibited",
             "cost within threshold", "risk is low and evidence is current",
             "data-type evidence requirements satisfied", "no unresolved conflicts"],
            self.citations,
            "goal_met",
        )

    def _came_from_doc_fallback(self) -> bool:
        return self.state.get("risk_obs", {}).get("route", "") == "approved_repository"

    # ------------------------------------------------------------------ #
    def _finalize(self, stop: StopAgent) -> Dict[str, Any]:
        try:
            explanation = generate_explanation(stop.decision, stop.reasons, stop.citations)
        except LLMUnavailableError as exc:
            # No offline template: the run itself still concludes (the
            # policy decision was already made deterministically), but we
            # surface the LLM failure plainly instead of inventing prose.
            explanation = None
            llm_error = str(exc)
        else:
            llm_error = None

        if explanation is None:
            # Persist a decision even if Gemini couldn't phrase it, so the
            # compliance outcome is never lost -- but the failure is explicit.
            explanation = (
                f"[Gemini unavailable, rationale not generated: {llm_error}] "
                f"Decision basis: {'; '.join(stop.reasons)}"
            )

        obs = tools.record_final_decision(
            self.db,
            request_id=self.request_id,
            vendor_name=self.request.get("vendor_name") or "unknown",
            product=self.request.get("product") or "unknown",
            decision=stop.decision,
            explanation=explanation,
            citations=stop.citations,
            payload_hash=self.payload_hash,
        )
        # Only log the record-decision step if we haven't already hit max steps
        # (the duplicate-guard path already logged its own terminal step).
        if stop.stopped_reason != "duplicate_reused":
            try:
                self._log_step(
                    thought=f"Goal reached ({stop.stopped_reason}); recording final decision.",
                    action="record_final_decision",
                    action_input={"request_id": self.request_id, "decision": stop.decision,
                                   "payload_hash": self.payload_hash},
                    observation=obs,
                )
            except StopAgent:
                pass  # max-steps triggered exactly on the final log; decision still recorded.

        used_existing = obs.get("reused_existing")
        final_decision = obs["decision"] if used_existing else stop.decision
        final_explanation = obs["explanation"] if used_existing else explanation
        final_citations = obs["citations"] if used_existing else stop.citations

        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "decision": final_decision,
            "explanation": final_explanation,
            "citations": final_citations,
            "steps": self.steps,
            "stopped_reason": stop.stopped_reason,
            "payload_hash": self.payload_hash,
            "record_action": "updated" if obs.get("updated") else (
                "reused" if used_existing else "created"),
        }


def run_agent(db: Session, request: Dict[str, Any]) -> Dict[str, Any]:
    return VendorAssessmentAgent(db, request).run()