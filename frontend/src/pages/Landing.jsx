// Landing page (full width, no sidebar). Card images are REAL outputs of our models
// (made by scripts/make_landing_assets.py); card 3's bars are the real gate weights for that image.
import { Link } from "react-router-dom";
import card1 from "../assets/landing/card1_universal.png";
import card2 from "../assets/landing/card2_hard.png";
import card3 from "../assets/landing/card3_soft.png";
import card3Weights from "../assets/landing/card3_soft.json";
import card4 from "../assets/landing/card4_sketch.png";
import { TopBar } from "../components/Frame.jsx";
import { ArrowRightIcon } from "../components/Icons.jsx";

function Card({ n, title, to, img, alt, text, children }) {
  return (
    <Link to={to} className="group block rounded-panel border border-line bg-panel p-4 transition-colors hover:border-white/20">
      <div className="mb-3 flex items-baseline gap-3">
        <span className="num text-[13px] text-muted">{n}</span>
        <span className="text-[18px] text-ink">{title}</span>
      </div>
      <img src={img} alt={alt} className="aspect-[4/3] w-full rounded-control border border-line object-cover" />
      <p className="mt-3 text-[14px] leading-snug text-ink-2">{text}</p>
      {children}
    </Link>
  );
}

const STEPS = [
  ["01", "Corrupt", "Noise, blur or occlusion, applied at runtime."],
  ["02", "Classify", "A CNN identifies which corruption is present."],
  ["03", "Restore", "Specialist or mixture-of-experts autoencoders repair it."],
  ["04", "Compare", "Input, output and error map, side by side."],
];

export default function Landing() {
  return (
    <div className="flex min-h-full flex-col">
      <TopBar links />
      <main className="bg-grid flex-1">
        <div className="grid grid-cols-[1fr_auto] items-center gap-16 px-[70px] py-10 max-lg:grid-cols-1 max-lg:px-6">
          <div>
            <h1 className="font-serif text-[76px] leading-[1.02] font-light text-ink">Restore.<br />Route.<br />Sketch.</h1>
            <p className="mt-10 max-w-[560px] text-[16px] text-muted">Four models for repairing damaged photographs and drawing faces.</p>
            <Link to="/universal"
              className="mt-10 inline-flex h-14 items-center gap-5 rounded-control bg-accent px-7 text-[17px] text-on-accent transition-colors hover:bg-accent-strong">
              Enter workspace <ArrowRightIcon width={18} height={18} />
            </Link>
          </div>

          {/* staggered 2-column card grid (column 2 starts lower, as in the design) */}
          <div className="grid w-[786px] grid-cols-2 gap-5 max-lg:w-full">
            <div className="flex flex-col gap-5">
              <Card n="01" title="Universal Restoration" to="/universal" img={card1}
                alt="Salt-and-pepper input and the universal autoencoder's output"
                text="One autoencoder removes noise, blur and occlusion without being told which." />
              <Card n="03" title="Soft Mixture-of-Experts" to="/soft-moe" img={card3}
                alt="Occluded input and the soft mixture-of-experts output" text="Continuous blending of specialized sub-networks.">
                <div className="mt-3 flex flex-col gap-2 rounded-control bg-inset px-3 py-2.5">
                  {Object.entries(card3Weights.weights).map(([k, v]) => (
                    <div key={k}>
                      <div className="flex justify-between"><span className="label">{k}</span><span className="num text-[13px] text-ink-2">{v.toFixed(2)}</span></div>
                      <div className="mt-1 h-1 rounded-full bg-line"><div className="h-full rounded-full bg-accent" style={{ width: `${v * 100}%` }} /></div>
                    </div>
                  ))}
                  <span className="mt-0.5 text-[12px] text-faint">{card3Weights.caption}</span>
                </div>
              </Card>
            </div>
            <div className="mt-7 flex flex-col gap-5">
              <Card n="02" title="Hard-Routed Restoration" to="/hard-routed" img={card2}
                alt="Blurred input and the hard-routed blur expert's output" text="A classifier routes each image to one specialist autoencoder." />
              <Card n="04" title="Face-to-Sketch Generator" to="/face-to-sketch" img={card4}
                alt="Face photograph for the sketch generator" text="A style-conditioned GAN draws a face as a pencil sketch." />
            </div>
          </div>
        </div>
      </main>
      <footer className="border-t border-line bg-bar">
        <div className="grid grid-cols-4 gap-10 px-[70px] py-7 max-lg:grid-cols-2 max-lg:px-6">
          {STEPS.map(([n, title, text]) => (
            <div key={n}>
              <div className="flex items-baseline gap-3"><span className="num text-[12px] text-muted">{n}</span><span className="text-[15px] text-ink">{title}</span></div>
              <p className="mt-2 text-[14px] text-muted">{text}</p>
            </div>
          ))}
        </div>
      </footer>
    </div>
  );
}
