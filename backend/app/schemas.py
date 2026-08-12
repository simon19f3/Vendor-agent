from typing import Optional, List, Any, Dict
from pydantic import BaseModel


class VendorRequestIn(BaseModel):
    request_id: str
    vendor_name: Optional[str] = None
    product: Optional[str] = None
    cost: Optional[float] = None
    intended_use: Optional[str] = None
    data_type: Optional[str] = None


class StepTrace(BaseModel):
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: Dict[str, Any]
    retry_of_step: Optional[int] = None


class AgentRunResult(BaseModel):
    request_id: str
    run_id: str
    decision: str
    explanation: str
    citations: List[str]
    steps: List[StepTrace]
    stopped_reason: str
    payload_hash: str
    record_action: str  # "created" | "updated" | "reused"