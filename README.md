# Vendor-Assessment Agent

An autonomous, ReAct-style agent that evaluates software-vendor requests against a written policy and returns one of four decisions: **APPROVE**, **REQUEST_INFORMATION**, **ESCALATE**, or **REJECT**.

**Stack:** FastAPI (backend + agent) · PostgreSQL / SQLite · Gemini (explanation phrasing + policy RAG) · Next.js (frontend)

---

## 1. Agent Workflow

The agent follows a **Reason → Act → Observe → Update → Repeat** loop driven by an explicit planner queue (not free-form LLM tool selection).

```
1. duplicate_guard
2. retrieve_policy          (RAG over vendor_policy.md)
3. validate_required_fields
4. check_restricted_data
5. gather_risk_evidence
6. apply_risk_rules
7. check_cost
8. check_evidence_requirements   (extra rules for confidential data)
9. approve                   (only reached if nothing blocked earlier)
```

Each step:

1. **Reason** — records a thought explaining why the next action is needed  
2. **Act** — calls a tool with validated arguments  
3. **Observe** — reads the structured tool result  
4. **Update** — merges evidence into `self.state` and citations; on recoverable failure, pushes a remediation step back onto the queue  

The loop stops as soon as any terminal condition is met (see §5).

**Important design choice:** Tool orchestration and the decision itself are computed **deterministically** in `agent.py`. Gemini is used only to:

- retrieve relevant policy clauses (RAG embeddings), and  
- phrase the final one-paragraph rationale  

It never chooses tools or invents evidence. If Gemini is unavailable, the run escalates with `stopped_reason="llm_unavailable"` rather than silently inventing an explanation.

---

## 2. Tools

| Tool | Purpose | Authority tier | Notes |
|------|---------|----------------|-------|
| `retrieve_policy` | RAG retrieval over `vendor_policy.md` (chunked by `##` sections) | 1 (highest) | Always first. Returns only the k most relevant clauses + stable `chunk_id` for citations. |
| `lookup_vendor_risk` | Internal vendor-risk database | 2 | Routes: `primary` → `backup`. Returns status, risk_rating, assessment_date, currency flag. |
| `search_vendor_documents` | Security assessments + vendor-supplied docs | 2 (approved) / 3 (vendor-supplied, untrusted) | Multiple routes & query variants for the fallback ladder. |
| `calculator` | Deterministic numeric comparison | n/a | Used for cost-threshold checks. |
| `record_final_decision` | Idempotent write to `decision_log` | n/a | Deduplicates by `(request_id, payload_hash)`. |

Every tool:

- Has a **Pydantic argument schema** (runtime validation before execution)
- Returns a plain JSON-serialisable dict with at least an `"ok"` key
- Never raises on expected failure modes (timeout, not-found); it reports them in the observation so the planner can retry or escalate

---

## 3. State

| Layer | What is stored | Where |
|-------|----------------|-------|
| **Short-term (per run)** | Accumulated evidence, risk rating, policy chunks, citations, full ReAct trace | `self.state`, `self.citations`, `self.steps` → `AgentStepLog` table |
| **Long-term** | Final decision + explanation + citations + payload hash | `DecisionLog` table (unique on `request_id`) |
| **Policy memory** | Embedded chunks of `vendor_policy.md` | In-process vector store (`PolicyMemoryStore` in `rag.py`) |

**Provenance:** Every cited fact carries a stable source id (`chunk_id`, `source_id`, or `document_id`) and authority tier so an auditor can reconstruct *why* a decision was made.

**Change-aware duplicates:** A `payload_hash` of the normalised request fields is stored with every decision.  
- Same `request_id` + same hash → reuse prior decision (no re-evaluation).  
- Same `request_id` + different hash → treat as a correction and re-evaluate, updating the existing row in place.

---

## 4. Stopping Rules

The agent raises an internal `StopAgent` and finalises as soon as any of these is true:

| Condition | Decision | `stopped_reason` |
|-----------|----------|------------------|
| All checks passed | APPROVE | `goal_met` |
| Required field missing or invalid | REQUEST_INFORMATION | `missing_information` |
| `data_type == "restricted"` | ESCALATE | `risk_too_high` |
| Vendor marked prohibited | REJECT | `prohibited_vendor` |
| Risk rating high | REJECT | `risk_too_high` |
| Risk rating medium | ESCALATE | `risk_too_high` |
| Cost > $10 000 | ESCALATE | `cost_too_high` |
| Confidential data, security assessment fails | REJECT | `security_assessment_failed` |
| Conflicting current tier-2 sources | ESCALATE | `conflicting_evidence` |
| Evidence outdated and no fresher source found | ESCALATE | `outdated_evidence` |
| Required evidence missing after retries | ESCALATE | `missing_evidence` |
| Tool failures exhausted (3 attempts) | ESCALATE | `tool_failure_exhausted` |
| Policy / embedding unavailable | ESCALATE | `llm_unavailable` |
| `MAX_AGENT_STEPS` (default 8) reached | ESCALATE | `max_steps_reached` |
| Identical request already decided | reuse prior decision | `duplicate_reused` |

---

## 5. Retries and Fallback Ladder

Retries are **bounded** and **must change approach**:

- Budget: **1 initial attempt + 2 retries** (`MAX_RETRIES_PER_TOOL = 2`)
- Only *recoverable* problems enter the ladder (timeout, stale evidence, no results, validation error)
- Genuine policy determinations (prohibited vendor, high/medium risk, restricted data, cost over threshold) are **immediate stops** — more retries cannot change a cleanly evaluated threshold

**Fallback ladder** (matches the design slides):

1. Retry with corrected input / different route  
2. Try an alternate tool  
3. Request missing information (when appropriate)  
4. Escalate  

Each retry is logged with `retry_of_step` so the full trace remains auditable.

---

## 6. Guardrails

- **Prompt-injection resistance:** Documents returned by `search_vendor_documents` may include tier-3 (vendor-supplied) content. The agent only ever branches on structured fields (`result`, `risk_rating`, `document_date`, `authority_tier`). Free-text `content` is never parsed for instructions. Every tier-3 document is explicitly logged as a `guardrail_check` step.
- **Argument validation:** Every tool call is validated against a Pydantic schema before execution. Malformed arguments return a structured `validation_error` observation (retryable).
- **No silent degradation:** If Gemini is unavailable for policy retrieval or explanation, the agent escalates rather than inventing a rationale.

---

## 7. Setup

### Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Create `backend/.env`:

```env
GEMINI_API_KEY=your-key-here
# Optional overrides:
# GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
# DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/vendor_agent
```

**Windows PowerShell** (if not using `.env`):

```powershell
$env:GEMINI_API_KEY = "your-key-here"
```

Start the server:

```bash
uvicorn app.main:app --reload --port 8000
```


### Frontend

```bash
cd frontend
npm install
$env:NEXT_PUBLIC_API_BASE = "http://localhost:8000"   # PowerShell
npm run dev
```

Open http://localhost:3000.

### Tests

```bash
cd backend
pytest -v
```

16 test cases cover normal approve/reject/escalate paths, missing fields, timeouts (recovered and exhausted), outdated evidence, conflicting sources, prompt injection, duplicates, change-aware resubmission, and max-steps.

### Execution logs

```bash
cd backend
python -m scripts.generate_execution_logs
```

Produces `logs/execution_logs.md` with full step traces for every scenario.

---

## 8. Decision Logic Summary (implements `vendor_policy.md`)

1. Fetch relevant policy clauses (RAG).  
2. Validate required fields → else `REQUEST_INFORMATION`.  
3. Restricted data → `ESCALATE` immediately.  
4. Look up vendor risk (primary → backup → document fallback, bounded retries):  
   - prohibited → `REJECT`  
   - high → `REJECT`  
   - medium → `ESCALATE`  
   - low but outdated, no current corroboration → `ESCALATE`  
   - low and current → continue  
5. Cost > $10 000 → `ESCALATE`.  
6. Confidential data additionally requires a current tier-2 approved security assessment with result `pass`; fail → `REJECT`; conflict → `ESCALATE`; none found → `ESCALATE`.  
7. Nothing blocking → `APPROVE`.  
8. Duplicate `request_id` with identical payload → return existing decision without re-running tools.

---
## 9. Test suite (`backend/tests/test_agent.py`) — 15 cases

| # | Test | Covers |
|---|---|---|
| 1 | `test_normal_approve` | Normal, fully-approvable request |
| 2 | `test_missing_cost` | Missing required field → REQUEST_INFORMATION |
| 3 | `test_missing_data_type` | Missing required field → REQUEST_INFORMATION |
| 4 | `test_prohibited_vendor_reject` | Prohibited vendor → REJECT |
| 5 | `test_restricted_data_escalate` | `restricted` data type always escalates |
| 6 | `test_medium_risk_escalate` | Medium risk rating → ESCALATE |
| 7 | `test_cost_over_threshold_escalate` | Cost > $10,000 → ESCALATE |
| 8 | `test_security_assessment_fail_reject` | Failing tier-2 security assessment → REJECT |
| 9 | `test_timeout_recovered_via_backup_route` | Tool failure, **recovered** via retry/backup route |
| 10 | `test_timeout_exhausted_escalate` | Tool failure, **retries exhausted** (bounded at 3 attempts) → ESCALATE |
| 11 | `test_outdated_evidence_escalate` | Evidence >180 days old, fallback search also stale → ESCALATE |
| 12 | `test_prompt_injection_ignored` | Malicious instructions inside a tier-3 vendor document are logged and ignored; decision uses trusted tier-2 evidence only |
| 13 | `test_duplicate_request_returns_existing_decision` | Same `request_id` run twice → second run reuses the first decision, no re-invoked tools |
| 14 | `test_conflicting_evidence_escalate` | Two current tier-2 sources disagree on a material fact → ESCALATE |
| 15 | `test_max_steps_hit` | Every route exhausted → agent stops at the step budget rather than looping forever |

Each test asserts on the final `decision`, `stopped_reason`, evidence
`citations`, and (where relevant) the shape of the tool-call trace itself
(e.g. exactly 3 attempts were made, not an unbounded retry loop).

## 10. Project Layout

```
vendor-agent/
├── backend/
│   ├── app/
│   │   ├── agent.py      # ReAct loop, planner queue, stop conditions
│   │   ├── tools.py      # 5 tools + argument schemas + fault injector
│   │   ├── llm.py        # Gemini explanation phrasing (no offline template)
│   │   ├── rag.py        # Policy memory (chunk → embed → retrieve)
│   │   ├── models.py     # SQLAlchemy models
│   │   ├── schemas.py    # Pydantic I/O
│   │   ├── db.py / config.py / seed.py / main.py
│   ├── data/             # vendor_policy.md, vendor_risk.csv, vendor_documents.json
│   ├── tests/            # 16 test cases
│   └── scripts/
├── frontend/             # Next.js UI
└── logs/                 # Generated execution traces
```