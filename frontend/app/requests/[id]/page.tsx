"use client";

import { useEffect, useState } from "react";
import { getDecision } from "../../../lib/api";

export default function RequestDetail({ params }: { params: { id: string } }) {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDecision(params.id).then(setData).catch((e) => setError(e.message));
  }, [params.id]);

  if (error) return <div className="card">{error}</div>;
  if (!data) return <div className="card">Loading…</div>;

  return (
    <div className="card">
      <h3>
        {data.request_id} — {data.vendor_name} / {data.product}{" "}
        <span className={`badge ${data.decision}`}>{data.decision}</span>
      </h3>
      <p>{data.explanation}</p>
      <p><strong>Citations:</strong> {data.citations.join(", ") || "none"}</p>
      <h4>Trace</h4>
      {data.steps.map((s: any) => (
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
  );
}
