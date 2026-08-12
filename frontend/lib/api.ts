export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export type VendorRequest = {
  request_id: string;
  vendor_name?: string | null;
  product?: string | null;
  cost?: number | null;
  intended_use?: string | null;
  data_type?: string | null;
};

export type StepTrace = {
  step_number: number;
  thought: string;
  action: string;
  action_input: Record<string, any>;
  observation: Record<string, any>;
  retry_of_step?: number | null;
};

export type AgentRunResult = {
  request_id: string;
  run_id: string;
  decision: string;
  explanation: string;
  citations: string[];
  steps: StepTrace[];
  stopped_reason: string;
};

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function runInline(req: VendorRequest): Promise<AgentRunResult> {
  const res = await fetch(`${API_BASE}/agent/run_inline`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return json(res);
}

export async function listDecisions() {
  const res = await fetch(`${API_BASE}/decisions`, { cache: "no-store" });
  return json<any[]>(res);
}

export async function getDecision(requestId: string) {
  const res = await fetch(`${API_BASE}/decisions/${requestId}`, { cache: "no-store" });
  return json<any>(res);
}
