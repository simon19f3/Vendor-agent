"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { runInline, listDecisions, AgentRunResult, VendorRequest } from "../lib/api";

const DATA_TYPES = ["public", "internal", "confidential", "restricted"];

const EXAMPLES: VendorRequest[] = [
  { request_id: "VR-001", vendor_name: "SafeCloud", product: "TeamDocs", cost: 8000, intended_use: "Store internal team documents", data_type: "internal" },
  { request_id: "VR-003", vendor_name: "BlockedSoft", product: "SyncNow", cost: 5000, intended_use: "Synchronize company files", data_type: "internal" },
  { request_id: "VR-009", vendor_name: "InjectCorp", product: "HelpDesk AI", cost: 9000, intended_use: "Process confidential customer-support records", data_type: "confidential" },
  { request_id: "VR-013", vendor_name: "SafeCloud", product: "TeamDocs", cost: 8000, intended_use: "Store payment credentials", data_type: "restricted" },
];

export default function Home() {
  const [form, setForm] = useState<VendorRequest>(EXAMPLES[0]);
  const [result, setResult] = useState<AgentRunResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<any[]>([]);

  async function refreshHistory() {
    try {
      setHistory(await listDecisions());
    } catch {
      /* backend not running yet -- ignore */
    }
  }

  useEffect(() => {
    refreshHistory();
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await runInline(form);
      setResult(r);
      refreshHistory();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="card">
        <h3>Submit a vendor request</h3>
        <p style={{ fontSize: 13, color: "#666" }}>
          Try an example: {EXAMPLES.map((ex) => (
            <button
              key={ex.request_id}
              type="button"
              style={{ background: "#eee", color: "#111", marginRight: 6, marginTop: 6, padding: "4px 10px" }}
              onClick={() => setForm(ex)}
            >
              {ex.request_id}
            </button>
          ))}
        </p>
        <form onSubmit={submit}>
          <label>Request ID</label>
          <input value={form.request_id} onChange={(e) => setForm({ ...form, request_id: e.target.value })} required />

          <label>Vendor name</label>
          <input value={form.vendor_name ?? ""} onChange={(e) => setForm({ ...form, vendor_name: e.target.value })} />

          <label>Product</label>
          <input value={form.product ?? ""} onChange={(e) => setForm({ ...form, product: e.target.value })} />

          <label>Cost (USD)</label>
          <input type="number" value={form.cost ?? ""} onChange={(e) => setForm({ ...form, cost: e.target.value === "" ? null : Number(e.target.value) })} />

          <label>Intended use</label>
          <input value={form.intended_use ?? ""} onChange={(e) => setForm({ ...form, intended_use: e.target.value })} />

          <label>Data type</label>
          <select value={form.data_type ?? ""} onChange={(e) => setForm({ ...form, data_type: e.target.value || null })}>
            <option value="">-- select --</option>
            {DATA_TYPES.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>

          <button type="submit" disabled={loading}>{loading ? "Running agent…" : "Run agent"}</button>
        </form>
      </div>

      {error && <div className="card" style={{ borderColor: "#e88", color: "#a00" }}>{error}</div>}

      {result && (
        <div className="card">
          <h3>
            Decision:{" "}
            <span className={`badge ${result.decision}`}>{result.decision}</span>
          </h3>
          <p><strong>Stop reason:</strong> {result.stopped_reason}</p>
          <p>{result.explanation}</p>
          <p><strong>Citations:</strong> {result.citations.join(", ") || "none"}</p>

          <h4>Agent trace ({result.steps.length} steps)</h4>
          {result.steps.map((s) => (
            <div className="step" key={s.step_number}>
              <div className="action">
                Step {s.step_number}: {s.action}
                {s.retry_of_step ? ` (retry of step ${s.retry_of_step})` : ""}
              </div>
              <div>{s.thought}</div>
              <pre>{JSON.stringify({ input: s.action_input, observation: s.observation }, null, 2)}</pre>
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <h3>Past decisions</h3>
        {history.length === 0 && <p style={{ color: "#888" }}>No decisions recorded yet.</p>}
        {history.map((d) => (
          <div key={d.request_id} className="step">
            <Link href={`/requests/${d.request_id}`}>
              <strong>{d.request_id}</strong> — {d.vendor_name} / {d.product}
            </Link>{" "}
            <span className={`badge ${d.decision}`}>{d.decision}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
