// The minimal System page (backend health check).
import { WorkspaceLayout } from "../components/Frame.jsx";
import { useHealth } from "../lib/useHealth.js";

export function SystemPage() {
  const { loading, data, error } = useHealth();
  return (
    <WorkspaceLayout>
      <div className="mb-7">
        <div className="label mb-2.5">System</div>
        <h1 className="font-serif text-[44px] leading-[1.05] font-light text-ink">Model status</h1>
        <p className="mt-2.5 text-[15px] text-muted">Live result of the backend health check (GET /api/health).</p>
      </div>
      {loading && <p className="label">Checking…</p>}
      {error && <p className="text-[14px] text-danger">{error}</p>}
      {data && (
        <div className="rounded-panel border border-line bg-panel">
          <div className="flex gap-10 border-b border-line px-6 py-4">
            <div><div className="label">Status</div><div className="num mt-1 text-ink">{data.status}</div></div>
            <div><div className="label">Models loaded</div><div className="num mt-1 text-ink">{data.models_loaded} / {data.models_expected}</div></div>
            {data.onnxruntime && <div><div className="label">ONNX Runtime</div><div className="num mt-1 text-ink">{data.onnxruntime}</div></div>}
          </div>
          {Object.entries(data.models || {}).map(([name, m]) => (
            <div key={name} className="flex items-center gap-4 border-b border-line px-6 py-3 last:border-0">
              <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${m.loaded ? "bg-ok" : "bg-faint"}`} />
              <span className="flex-1 text-[14px] text-ink-2">{m.description}</span>
              <span className="num text-[12px] text-muted">{m.file}</span>
              <span className="num w-20 text-right text-[12px] text-muted">{m.size_mb ? `${m.size_mb} MB` : ""}</span>
              <span className="num w-56 text-right text-[12px] text-muted">{m.loaded ? "loaded" : m.error}</span>
            </div>
          ))}
        </div>
      )}
    </WorkspaceLayout>
  );
}
