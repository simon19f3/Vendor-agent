# Architecture Diagram

## High-level flow

```mermaid
flowchart TD
    subgraph Input
        REQ[Vendor Request<br/>request_id, vendor, product,<br/>cost, intended_use, data_type]
    end

    subgraph Planner["Planner (agent.py)"]
        Q[Queue of steps]
        R[Reason: choose next step]
        A[Act: call tool]
        O[Observe: read result]
        U[Update state + plan]
        STOP{Stop condition?}
    end

    subgraph Validation
        PV[Pydantic arg schemas]
        TV[Tool result checks]
    end

    subgraph Tools["Tools (tools.py)"]
        T1[retrieve_policy<br/>RAG / tier 1]
        T2[lookup_vendor_risk<br/>primary / backup]
        T3[search_vendor_documents<br/>tier 2 / 3]
        T4[calculator]
        T5[record_final_decision]
    end

    subgraph Memory
        ST[Short-term<br/>AgentStepLog + self.state]
        LT[Long-term<br/>DecisionLog + payload_hash]
        RAG[PolicyMemoryStore<br/>embedded policy chunks]
    end

    subgraph Decision
        FIN[Final decision<br/>APPROVE / REQUEST_INFO / ESCALATE / REJECT]
        EXP[Gemini explanation phrasing]
        CITE[Citations + stopped_reason]
    end

    REQ --> Q
    Q --> R
    R --> A
    A --> PV
    PV -->|valid| Tools
    PV -->|invalid| O
    Tools --> O
    O --> TV
    TV --> U
    U --> ST
    U --> STOP
    STOP -->|no| R
    STOP -->|yes| FIN
    FIN --> EXP
    FIN --> LT
    FIN --> CITE

    T1 -.-> RAG
    T2 -.-> ST
    T3 -.-> ST
    T5 -.-> LT
```

## Planner / fallback detail

```mermaid
flowchart LR
    subgraph Recoverable Failure
        F1[Timeout / stale / empty / bad args]
    end

    subgraph Fallback Ladder
        L1[1. Corrected input<br/>or different route]
        L2[2. Alternate tool]
        L3[3. Request missing info]
        L4[4. Escalate]
    end

    F1 --> L1
    L1 -->|still failing| L2
    L2 -->|still failing| L3
    L3 -->|still failing| L4

    Budget["Budget: 1 + MAX_RETRIES_PER_TOOL = 3 attempts"]
    Budget -.-> L1
    Budget -.-> L2
```

## Memory & provenance

```mermaid
flowchart TB
    subgraph ShortTerm
        S1[self.state<br/>risk_rating, policy_chunks, …]
        S2[self.citations]
        S3[self.steps → AgentStepLog]
    end

    subgraph LongTerm
        D1[DecisionLog<br/>decision, explanation,<br/>citations, payload_hash]
    end

    subgraph PolicyMemory
        P1[vendor_policy.md]
        P2[Chunk by ## sections]
        P3[Gemini embeddings]
        P4[Cosine top-k retrieve]
    end

    P1 --> P2 --> P3 --> P4
    P4 --> S1
    S1 --> D1
    S2 --> D1
```

## Decision authority tiers

| Tier | Source | Trust |
|------|--------|-------|
| 1 | Policy clauses (`retrieve_policy`) | Highest — always consulted first |
| 2 | Internal risk DB + approved security assessments | Trusted evidence |
| 3 | Vendor-supplied documents | Untrusted — logged via `guardrail_check`, never used as instructions |

Control flow only inspects structured fields (`result`, `risk_rating`, `document_date`, `authority_tier`). Free-text content from tier 3 is never parsed for commands.