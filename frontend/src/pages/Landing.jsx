// Landing page (full width, no sidebar). Decorative only: the card images are the Stitch design
// illustrations (not model outputs), and card 3's bars are a visual motif with no numbers.
// Layout: exactly one screen tall (h-screen, no scroll) on landscape laptop screens; the cards
// share the height left between the top bar and the footer strip, so they scale with the window.
import { Link } from "react-router-dom";
import imgHard from "../assets/landing/design_hard.jpg";
import imgSketch from "../assets/landing/design_sketch.jpg";
import imgSoft from "../assets/landing/design_soft.jpg";
import imgUniversal from "../assets/landing/design_universal.jpg";
import { TopBar } from "../components/Frame.jsx";
import { ArrowRightIcon } from "../components/Icons.jsx";

function Card({ n, title, to, img, text, className = "", imgPosition = "center", children }) {
  return (
    <Link to={to}
      className={`group flex min-h-0 flex-col rounded-panel border border-line bg-panel p-[1.6vh] transition-colors hover:border-white/20 ${className}`}>
      <div className="mb-[1.2vh] flex items-baseline gap-3">
        <span className="num text-[12px] text-muted">{n}</span>
        <span className="text-[clamp(14px,1.9vh,18px)] text-ink">{title}</span>
      </div>
      <img src={img} alt="" style={{ objectPosition: imgPosition }}
        className="min-h-0 w-full flex-1 rounded-control border border-line object-cover" />
      <p className="mt-[1.2vh] text-[clamp(12px,1.55vh,14px)] leading-snug text-ink-2">{text}</p>
      {children}
    </Link>
  );
}

// Purely decorative bars (no values shown): they only suggest "weights spread over 4 branches".
const MOTIF = [["Clean", 62], ["Salt", 18], ["Blur", 14], ["Occlusion", 6]];

const STEPS = [
  ["01", "Corrupt", "Noise, blur or occlusion, applied at runtime."],
  ["02", "Classify", "A CNN identifies which corruption is present."],
  ["03", "Restore", "Specialist or mixture-of-experts autoencoders repair it."],
  ["04", "Compare", "Input, output and error map, side by side."],
];

export default function Landing() {
  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <TopBar links />
      <main className="bg-grid flex min-h-0 flex-1 items-stretch gap-[5vw] px-[70px] py-[3.5vh] max-lg:px-6">
        {/* hero */}
        <div className="flex min-w-0 flex-1 flex-col justify-center">
          <h1 className="font-serif text-[clamp(52px,8.6vh,96px)] leading-[1.02] font-light text-ink">Restore.<br />Route.<br />Sketch.</h1>
          <p className="mt-[4vh] max-w-[560px] text-[clamp(14px,1.8vh,17px)] text-muted">Four models for repairing damaged photographs and drawing faces.</p>
          <Link to="/universal"
            className="mt-[4vh] inline-flex h-[clamp(44px,6vh,56px)] w-fit items-center gap-5 rounded-control bg-accent px-7 text-[clamp(15px,1.9vh,17px)] text-on-accent transition-colors hover:bg-accent-strong">
            Enter workspace <ArrowRightIcon width={18} height={18} />
          </Link>
        </div>

        {/* 2 x 2 cards, column 2 staggered down a little (as in the design) */}
        <div className="grid min-h-0 w-[min(860px,56vw)] shrink-0 grid-cols-2 gap-[1.2vw]">
          <div className="flex min-h-0 flex-col gap-[2vh]">
            <Card n="01" title="Universal Restoration" to="/universal" img={imgUniversal} className="flex-1"
              text="Blind degradation removal across mixed artifacts." />
            <Card n="03" title="Soft Mixture-of-Experts" to="/soft-moe" img={imgSoft} className="flex-[1.15]" imgPosition="center 22%"
              text="Continuous blending of specialized sub-networks.">
              <div className="mt-[1.2vh] grid grid-cols-4 gap-x-3 rounded-control bg-inset px-3 py-[1vh]" aria-hidden>
                {MOTIF.map(([k, w]) => (
                  <div key={k}>
                    <span className="label block truncate text-[10px]">{k}</span>
                    <div className="mt-1 h-1 rounded-full bg-line"><div className="h-full rounded-full bg-accent" style={{ width: `${w}%` }} /></div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
          <div className="flex min-h-0 flex-col gap-[2vh] pt-[3vh]">
            <Card n="02" title="Hard-Routed Restoration" to="/hard-routed" img={imgHard} className="flex-1"
              text="Deterministic routing based on a degradation classifier." />
            <Card n="04" title="Face-to-Sketch Generator" to="/face-to-sketch" img={imgSketch} className="flex-1"
              text="Structural contour and anatomical graphite synthesis." />
          </div>
        </div>
      </main>

      <footer className="shrink-0 border-t border-line bg-bar">
        <div className="grid grid-cols-4 gap-10 px-[70px] py-[2.2vh] max-lg:px-6">
          {STEPS.map(([n, title, text]) => (
            <div key={n}>
              <div className="flex items-baseline gap-3"><span className="num text-[12px] text-muted">{n}</span><span className="text-[15px] text-ink">{title}</span></div>
              <p className="mt-1 text-[13px] leading-snug text-muted">{text}</p>
            </div>
          ))}
        </div>
      </footer>
    </div>
  );
}
