// Workspace building blocks shared by all restoration workspaces.
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api.js";
import { CORRUPTIONS, LEVELS, MAX_UPLOAD_MB, paramsText, validateFile } from "../lib/corruptions.js";
import { ChevronIcon, CloseIcon, DownloadIcon, SamplesIcon, UploadIcon } from "./Icons.jsx";

export function Panel({ title, className = "", children }) {
  return (
    <section className={`flex flex-col ${className}`}>
      {title && <h2 className="mb-3 text-[15px] font-medium text-ink">{title}</h2>}
      <div className="flex-1 rounded-panel border border-line bg-panel p-5">{children}</div>
    </section>
  );
}

export function Button({ variant = "secondary", className = "", ...props }) {
  const look = variant === "primary"
    ? "bg-accent text-on-accent hover:bg-accent-strong disabled:bg-accent/40 disabled:text-on-accent/60"
    : "border border-line bg-inset text-ink hover:bg-white/[0.06] disabled:text-faint disabled:hover:bg-inset";
  return (
    <button
      className={`inline-flex h-11 items-center justify-center gap-2 rounded-control px-4 text-[14px] font-medium transition-colors disabled:cursor-not-allowed ${look} ${className}`}
      {...props}
    />
  );
}

export function Toggle({ checked, onChange, label, disabled }) {
  return (
    <label className={`flex items-center justify-between rounded-control border border-line bg-inset px-4 py-3 text-[14px] ${disabled ? "text-faint" : "text-ink-2"}`}>
      {label}
      <button
        type="button" role="switch" aria-checked={checked} disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative h-5 w-9 rounded-full transition-colors ${checked ? "bg-accent" : "bg-line"} disabled:opacity-50`}
      >
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-ink transition-all ${checked ? "left-[18px]" : "left-0.5"}`} />
      </button>
    </label>
  );
}

/** Sample chooser: a floating glass layer listing the bundled clean images from /api/samples. */
function SamplePicker({ kind, onPick, onClose }) {
  const [state, setState] = useState({ loading: true, items: [], error: null });
  useEffect(() => {
    api.samples().then((s) => setState({ loading: false, items: s[kind] || [], error: null }))
      .catch((e) => setState({ loading: false, items: [], error: e.message }));
  }, [kind]);
  return (
    <div className="glass absolute inset-x-0 top-9 z-20 rounded-panel border border-line p-4 shadow-2xl">
      <div className="mb-3 flex items-center justify-between">
        <span className="label">Choose a sample</span>
        <button onClick={onClose} className="text-muted hover:text-ink" aria-label="Close"><CloseIcon /></button>
      </div>
      {state.loading && <p className="label">Loading…</p>}
      {state.error && <p className="text-[13px] text-danger">{state.error}</p>}
      <div className="grid max-h-64 grid-cols-6 gap-2 overflow-y-auto">
        {state.items.map((s) => (
          <button key={s.name} onClick={() => onPick(s)} title={s.name.split("/").pop()}
            className="aspect-square overflow-hidden rounded-control border border-line hover:border-accent">
            <img src={s.url} alt={s.name} className="h-full w-full object-cover" />
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * Source panel: drop zone / browse, "Choose a sample", Clear, and an optional toggle.
 * source = { file?, sample?, name, preview } or null.
 * toggle = { label, checked, onChange } or null (e.g. "Image is already corrupted").
 * tabs   = optional element shown above the drop zone (Face-to-Sketch: Upload | Webcam).
 * children = optional element that REPLACES the drop zone (Face-to-Sketch: the webcam view).
 */
export function UploadPanel({ source, onSource, toggle, tabs, children, disabled, sampleKind = "pets", onError }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const [picking, setPicking] = useState(false);

  const takeFile = (file) => {
    if (!file) return;
    const err = validateFile(file);
    if (err) return onError(err);
    onSource({ file, name: file.name, preview: URL.createObjectURL(file) });
  };

  return (
    <Panel title="Source" className="relative">
      <div className="flex h-full flex-col gap-3">
        {tabs}
        {children ?? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); if (!disabled) takeFile(e.dataTransfer.files[0]); }}
          className={`relative flex min-h-[176px] flex-1 flex-col items-center justify-center rounded-control border border-dashed transition-colors ${
            drag ? "border-accent bg-accent/[0.04]" : "border-line"} ${disabled ? "opacity-60" : ""}`}
        >
          {source && (
            <button onClick={() => onSource(null)} disabled={disabled}
              className="label absolute right-3 top-3 rounded-[6px] border border-line bg-inset px-2 py-0.5 hover:!text-ink">Clear</button>
          )}
          {source ? (
            <div className="flex flex-col items-center gap-2">
              <img src={source.preview} alt="" className="h-20 w-20 rounded-control border border-line object-cover" />
              <span className="num max-w-[240px] truncate text-[12px] text-ink-2">{source.name}</span>
            </div>
          ) : (
            <>
              <button onClick={() => inputRef.current?.click()} disabled={disabled}
                className="mb-3 flex h-10 w-10 items-center justify-center rounded-control bg-inset text-ink-2 hover:text-ink" aria-label="Browse">
                <UploadIcon />
              </button>
              <button onClick={() => inputRef.current?.click()} disabled={disabled} className="text-[14px] font-medium text-ink hover:underline">
                Drop an image or browse
              </button>
              <span className="num mt-1.5 text-[12px] text-muted">JPG or PNG, up to {MAX_UPLOAD_MB} MB</span>
              <button onClick={() => setPicking(true)} disabled={disabled}
                className="mt-4 inline-flex items-center gap-2 rounded-[6px] border border-line bg-inset px-3 py-1.5 text-[13px] text-ink-2 hover:text-ink">
                <SamplesIcon width={14} height={14} /> Choose a sample
              </button>
            </>
          )}
          <input ref={inputRef} type="file" accept="image/jpeg,image/png" className="hidden"
            onChange={(e) => { takeFile(e.target.files[0]); e.target.value = ""; }} />
        </div>
        )}
        {toggle && <Toggle label={toggle.label} checked={toggle.checked} onChange={toggle.onChange} disabled={disabled} />}
      </div>
      {picking && (
        <SamplePicker kind={sampleKind} onClose={() => setPicking(false)}
          onPick={(s) => { setPicking(false); onSource({ sample: s.name, name: s.name.split("/").pop(), preview: s.url }); }} />
      )}
    </Panel>
  );
}

/** Corruption panel: type, intensity, exact parameters, and the action buttons passed as children. */
export function CorruptionPanel({ corruption, level, onCorruption, onLevel, locked, disabled, children }) {
  const off = locked || disabled;
  return (
    <Panel title="Corruption">
      <div className="flex h-full flex-col gap-4">
        <div className={`flex flex-col gap-4 ${off ? "opacity-60" : ""}`}>
        <div>
          <div className="label mb-2">Type</div>
          <div className="relative">
            <select value={corruption} disabled={off} onChange={(e) => onCorruption(e.target.value)}
              className="h-11 w-full appearance-none rounded-control border border-line bg-inset px-4 text-[14px] text-ink disabled:cursor-not-allowed">
              {CORRUPTIONS.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
            </select>
            <ChevronIcon className="pointer-events-none absolute right-3 top-3.5 text-muted" />
          </div>
        </div>
        <div>
          <div className="label mb-2">Intensity</div>
          <div className="grid grid-cols-3 gap-1 rounded-control border border-line bg-inset p-1">
            {LEVELS.map((l) => (
              <button key={l.id} disabled={off || corruption === "none"} onClick={() => onLevel(l.id)}
                className={`h-9 rounded-[6px] text-[13px] transition-colors disabled:cursor-not-allowed ${
                  level === l.id ? "bg-white/[0.08] text-ink" : "text-muted hover:text-ink-2"}`}>
                {l.label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center justify-between rounded-control border border-line bg-inset px-4 py-2.5">
          <span className="label">Parameters</span>
          <span className="num text-[12px] text-ink-2">{locked ? "—" : paramsText(corruption, level)}</span>
        </div>
        </div>
        <div className="mt-auto grid grid-cols-2 gap-3">{children}</div>
      </div>
    </Panel>
  );
}

/** One labelled image area with corner ticks; shows a placeholder message when empty. */
export function ImagePanel({ label, src, empty, icon, busy, children, footer }) {
  return (
    <section className="flex min-w-0 flex-col">
      <div className="label mb-2 flex h-5 items-center justify-between">{label}{children}</div>
      <div className="relative h-[330px] overflow-hidden rounded-panel border border-line bg-inset">
        {["left-3 top-3 border-l border-t", "right-3 top-3 border-r border-t", "left-3 bottom-3 border-l border-b", "right-3 bottom-3 border-r border-b"].map((c) => (
          <span key={c} className={`absolute z-10 h-3 w-3 border-faint ${c}`} aria-hidden />
        ))}
        {src ? (
          <img src={src} alt={label} className="h-full w-full object-contain" />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-muted">
            {icon}
            <span className={`label ${busy ? "animate-pulse" : ""}`}>{empty}</span>
          </div>
        )}
      </div>
      {footer}
    </section>
  );
}

/** LOW -> HIGH legend for error maps (same ramp as the backend's error_map colours). */
export function ErrorLegend() {
  return (
    <div className="mt-2 flex items-center gap-2">
      <span className="label">Low</span>
      <span className="h-1.5 flex-1 rounded-full" style={{ background: "linear-gradient(90deg, #0e0f11, #7a96ba, #ecf0f5)" }} />
      <span className="label">High</span>
    </div>
  );
}

/** Indeterminate progress + the REAL elapsed time (we never show a fake percentage). */
export function ProgressStatus({ label, startedAt }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 100); return () => clearInterval(id); }, []);
  return (
    <div className="mt-6">
      <div className="h-[2px] overflow-hidden rounded-full bg-line">
        <div className="progress-indeterminate h-full w-2/5 bg-accent" />
      </div>
      <div className="mt-2 flex justify-between">
        <span className="label">{label}</span>
        <span className="label num">{((now - startedAt) / 1000).toFixed(1)} s</span>
      </div>
    </div>
  );
}

/** Bottom strip with the run's real settings + Download result. items = [[label, value], ...]. */
export function ResultStrip({ items, downloadUrl, downloadName = "restored.png", downloadLabel = "Download result" }) {
  return (
    <div className="mt-6 flex flex-wrap items-center gap-x-10 gap-y-3 rounded-panel border border-line bg-panel px-6 py-4">
      {items.map(([k, v]) => (
        <div key={k}>
          <div className="label">{k}</div>
          <div className="num mt-1 text-[14px] text-ink">{v}</div>
        </div>
      ))}
      {downloadUrl && (
        <a href={downloadUrl} download={downloadName} className="ml-auto">
          <Button variant="secondary" tabIndex={-1}><DownloadIcon /> {downloadLabel}</Button>
        </a>
      )}
    </div>
  );
}

export function ErrorBanner({ message, onClose }) {
  if (!message) return null;
  return (
    <div role="alert" className="mb-6 flex items-center justify-between rounded-panel border border-danger/40 bg-danger/[0.08] px-5 py-3 text-[14px] text-danger">
      {message}
      <button onClick={onClose} className="text-danger/80 hover:text-danger" aria-label="Dismiss"><CloseIcon /></button>
    </div>
  );
}

/** Horizontal bars for the 4 classifier probabilities (or mixture weights). values = {name: number}. */
export function RoutingBars({ values, highlight }) {
  const rows = Object.entries(values).sort((a, b) => b[1] - a[1]);
  return (
    <div className="flex flex-col gap-4">
      {rows.map(([name, v]) => (
        <div key={name}>
          <div className="mb-1.5 flex justify-between">
            <span className={`text-[13px] ${name === highlight ? "text-ink" : "text-muted"}`}>{name}</span>
            <span className="num text-[13px] text-ink-2">{(v * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-inset">
            <div className={`h-full rounded-full ${name === highlight ? "bg-accent" : "bg-accent/40"}`} style={{ width: `${Math.max(v * 100, 0.5)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Gate -> 4 branches diagram. Each curve's thickness and brightness are proportional to that branch's
 * weight from the backend response; the dominant branch is highlighted. weights = [[name, w], ...] in a fixed order.
 */
export function RoutingDiagram({ weights, dominant }) {
  const W = 1000, H = 230, gateX = 230, nodeX = 560, nodeW = 200;
  const ys = weights.map((_, i) => 30 + i * ((H - 60) / (weights.length - 1)));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-[230px] w-full" role="img" aria-label="Gate routing weights">
      {weights.map(([name, w], i) => (
        <path key={name} d={`M ${gateX + 90} ${H / 2} C ${gateX + 230} ${H / 2}, ${nodeX - 140} ${ys[i]}, ${nodeX} ${ys[i]}`}
          fill="none" stroke="var(--color-accent)" strokeWidth={1 + 9 * w} strokeOpacity={0.18 + 0.82 * w} strokeLinecap="round" />
      ))}
      <rect x={gateX - 90} y={H / 2 - 22} width={180} height={44} rx={8} fill="var(--color-inset)" stroke="var(--color-line)" />
      <text x={gateX} y={H / 2 + 5} textAnchor="middle" fill="var(--color-ink)" fontSize="15" fontFamily="var(--font-sans)">Gate</text>
      {weights.map(([name, w], i) => {
        const top = name === dominant;
        return (
          <g key={name}>
            <rect x={nodeX} y={ys[i] - 17} width={nodeW} height={34} rx={8}
              fill={top ? "rgba(159,184,207,0.16)" : "var(--color-inset)"} stroke={top ? "var(--color-accent)" : "var(--color-line)"} />
            <text x={nodeX + 14} y={ys[i] + 5} fill={top ? "var(--color-ink)" : "var(--color-muted)"} fontSize="14" fontFamily="var(--font-sans)">{name}</text>
            <text x={nodeX + nodeW + 16} y={ys[i] + 5} fill="var(--color-ink-2)" fontSize="13" fontFamily="var(--font-mono)">{(w * 100).toFixed(1)}%</text>
          </g>
        );
      })}
    </svg>
  );
}
