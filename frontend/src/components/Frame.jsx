// The app frame: TopBar, Sidebar, PageHeader and the WorkspaceLayout that combines them.
// The design rules (wordmark, status badge, the four tab names, page header) are implemented ONLY here,
// so every page is identical by construction.
import { Link, NavLink } from "react-router-dom";
import { MLFLOW_URL, MOCK } from "../lib/api.js";
import { useHealth } from "../lib/useHealth.js";

export const WORKSPACES = [
  { path: "/universal", name: "Universal Restoration" },
  { path: "/hard-routed", name: "Hard-Routed Restoration" },
  { path: "/soft-moe", name: "Soft Mixture-of-Experts Restoration" },
  { path: "/face-to-sketch", name: "Face-to-Sketch Generator" },
];

export function Wordmark() {
  return (
    <Link to="/" className="flex shrink-0 items-center gap-3 text-[18px] text-ink whitespace-nowrap hover:text-white">
      <span className="h-2 w-2 rounded-full bg-dot" aria-hidden />
      Image Processing Engine
    </Link>
  );
}

/** Top-right status. Driven by /api/health; in mock mode it is replaced by the MOCK DATA badge. */
export function StatusBadge() {
  const { loading, data, error } = useHealth();
  if (MOCK) {
    return <span className="label rounded-[6px] border border-accent px-2.5 py-1 !text-accent">Mock data</span>;
  }
  let dot = "bg-faint";
  let text = "Checking models";
  if (error) text = "Backend offline";
  else if (!loading && data) {
    const all = data.models_loaded === data.models_expected;
    dot = all ? "bg-ok" : "bg-faint";
    text = all ? "Models loaded" : `${data.models_loaded}/${data.models_expected} models loaded`;
  }
  return (
    <Link to="/system" className="label flex items-center gap-2 whitespace-nowrap !text-ink-2 hover:!text-ink" title="Open System status">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} aria-hidden />
      {text}
    </Link>
  );
}

export function TopBar({ links = false }) {
  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-line bg-bar px-[70px] max-lg:px-6">
      <Wordmark />
      <div className="flex items-center gap-8">
        {links && (
          <nav className="flex items-center gap-8 text-[15px] text-ink max-md:hidden">
            <Link to="/universal" className="hover:text-white">Workspaces</Link>
            <Link to="/system" className="hover:text-white">System</Link>
            <a href={MLFLOW_URL} target="_blank" rel="noreferrer" className="hover:text-white">Experiments</a>
            <span className="h-4 w-px bg-line" aria-hidden />
          </nav>
        )}
        <StatusBadge />
      </div>
    </header>
  );
}

export function Sidebar() {
  const tab = ({ isActive }) =>
    `relative block whitespace-nowrap rounded-control px-4 py-2.5 text-[14px] transition-colors ${
      isActive ? "bg-white/[0.06] text-ink" : "text-muted hover:text-ink-2 hover:bg-white/[0.03]"
    }`;
  const quiet = "block px-4 py-1.5 text-[13px] text-muted hover:text-ink-2";
  return (
    <aside className="glass flex w-[296px] shrink-0 flex-col border-r border-line px-3 py-6">
      <nav className="flex flex-col gap-1">
        {WORKSPACES.map((w) => (
          <NavLink key={w.path} to={w.path} className={tab}>
            {({ isActive }) => (
              <>
                {isActive && <span className="absolute left-0 top-2 bottom-2 w-[2px] rounded-full bg-accent" aria-hidden />}
                {w.name}
              </>
            )}
          </NavLink>
        ))}
      </nav>
      <div className="mt-auto border-t border-line pt-4">
        <Link to="/" className={quiet}>Home</Link>
        <Link to="/system" className={quiet}>System</Link>
        <a href={MLFLOW_URL} target="_blank" rel="noreferrer" className={quiet}>Experiments</a>
      </div>
    </aside>
  );
}

export function PageHeader({ number, title, description }) {
  return (
    <div className="mb-7">
      <div className="label mb-2.5">Workspace {String(number).padStart(2, "0")}</div>
      <h1 className="font-serif text-[44px] leading-[1.05] font-light text-ink">{title}</h1>
      <p className="mt-2.5 text-[15px] text-muted">{description}</p>
    </div>
  );
}

/** Shared page width and margins for every workspace page. */
export function WorkspaceLayout({ children }) {
  return (
    <div className="flex h-full flex-col">
      <TopBar />
      <div className="flex min-h-0 flex-1">
        <Sidebar />
        <main className="min-w-0 flex-1 overflow-y-auto">
          <div className="max-w-[1240px] px-12 py-9">{children}</div>
        </main>
      </div>
    </div>
  );
}
