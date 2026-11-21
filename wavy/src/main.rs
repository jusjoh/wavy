//! wavy — equatorial Kelvin and Rossby waves (and El Niño / La Niña) as a
//! terminal screensaver.
//!
//! Unlike the Python prototype (precompute → play back), the shallow-water
//! model runs live: it is cheap enough (a few RK4 steps on a small C-grid per
//! frame) that the animation never loops.

mod render;
mod rng;
mod solver;
mod term;
mod wbc;

use std::collections::VecDeque;
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use render::{AutoScale, Marker, Palette, Renderer};
use rng::Rng;
use solver::{Coupling, Geometry, Params, Sim, YEAR};
use wbc::{Slip, Wbc, WbcParams, DAY};

const USAGE: &str = "\
wavy — equatorial Kelvin and Rossby waves in your terminal

USAGE: wavy [OPTIONS]

Modes:
  --mode waves    free equatorial waves, random height bumps (default)
  --mode enso     coupled ocean–atmosphere: El Niño / La Niña cycles
  --mode wbc      cross-equatorial western boundary current (after
                  Goldsworth, Marshall & Johnson 2021, JPO 51)

Physics (linear shallow water on an equatorial beta-plane, Matsuno units:
length ≈ 350 km, time ≈ 1.45 days, 252 units ≈ 1 year):
  --geometry basin|channel  walls at x = 0, Lx or periodic in x   [basin]
  --lx X          domain length in deformation radii              [40]
  --ly Y          half-width in y (solid walls at ±Ly)            [8]
  --nx N          grid cells in x                       [waves 200, enso 100]
  --ny N          grid cells in y                       [waves 80,  enso 40]
  --dt T          RK4 time step                         [waves 0.04, enso 0.08]
  --damping R     Rayleigh damping rate                 [waves 0.008, enso 0.004]
  --filter F      biharmonic grid-noise filter in x     [waves 0, enso 0.001]
  --seed N        random seed                                     [from clock]

Waves mode:
  --kick-every T  drop a random bump every T time units, 0 = never [40]

ENSO mode:
  --mu M          Bjerknes coupling (≈0.43 = self-sustained)      [0.42]
  --eps E         SST relaxation rate                             [0.03]
  --burst-amp A   random wind-burst strength, 0 = none            [0.05]
  --burst-rate R  wind bursts per time unit                       [0.025]
  --wind-x F      centre of the coupled wind patch (fraction of Lx) [0.3]
  --wind-sy Y     meridional half-width of the wind patch         [1.0]
  --init-h H      initial thermocline bump                        [0.5]
  --view sst|h    start showing SST or thermocline depth          [sst]

WBC mode (nonlinear 1.5-layer shallow water, SI units; north is up, the
western coast on the left, drawn in true proportions unless --stretch):
  --slip no|free  lateral boundary condition on the western wall  [no]
  --visc NU       Laplacian viscosity in m²/s                     [300]
  --view pv|zeta  potential or relative vorticity                 [pv]
  --stretch       fill the whole terminal width (distorts the shapes)

Display:
  --fps F         frames per second                               [30]
  --frame-dt T    model time per frame at speed x1      [waves 0.2, enso 0.8,
                                                         wbc 3 (hours)]
  --gain G        amplitude multiplier for the colours            [1]
  --no-hud        hide the status line
  --no-equator    start without the dotted equator line
  --snapshot T    simulate up to time T (append 'y' for years; in wbc mode
                  T is in days), print one frame, exit
  -h, --help      show this help
  -V, --version   show the version

Keys: q quit | space pause | + / - speed | e equator | h status line
      k kick (waves: new bump, enso: westerly wind burst) | r restart
      v toggle view (enso: SST / thermocline, wbc: PV / vorticity)
      b switch no-slip / free-slip and restart (wbc)
      s highlight f·q < 0, the symmetric-instability criterion (wbc)
      a true proportions / fill the screen (wbc)
      ? show / hide the keys
";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Waves,
    Enso,
    Wbc,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Sst,
    Thermocline,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WbcView {
    Pv,
    Vorticity,
}

/// Nominal conversion of the model Niño-3 index to °C (model rms ≈ 0.17 ↔
/// observed ≈ 0.9 °C).
const NINO_DEGC: f64 = 5.0;
/// Fixed colour scales in ENSO mode — amplitude is the signal, so no auto-scaling.
const ENSO_SST_SCALE: f64 = 0.5;
const ENSO_H_SCALE: f64 = 1.0;
/// Minimum wall-clock time a wind-burst label stays in the status line.
const BURST_LABEL_HOLD: Duration = Duration::from_secs(2);

#[derive(Default)]
struct Raw {
    mode: Option<Mode>,
    geometry: Option<Geometry>,
    lx: Option<f64>,
    ly: Option<f64>,
    nx: Option<usize>,
    ny: Option<usize>,
    dt: Option<f64>,
    damping: Option<f64>,
    filter: Option<f64>,
    kick_every: Option<f64>,
    seed: Option<u64>,
    mu: Option<f64>,
    eps: Option<f64>,
    burst_amp: Option<f64>,
    burst_rate: Option<f64>,
    wind_x: Option<f64>,
    wind_sy: Option<f64>,
    init_h: Option<f64>,
    view: Option<View>,
    wbc_view: Option<WbcView>,
    slip: Option<Slip>,
    visc: Option<f64>,
    fps: Option<f64>,
    frame_dt: Option<f64>,
    gain: Option<f64>,
    no_hud: bool,
    no_equator: bool,
    stretch: bool,
    snapshot: Option<f64>,
}

struct Opts {
    mode: Mode,
    params: Params,
    kick_every: f64,
    init_h: f64,
    seed: Option<u64>,
    view: View,
    wbc_view: WbcView,
    slip: Slip,
    visc: f64,
    fps: f64,
    frame_dt: f64,
    gain: f64,
    hud: bool,
    equator: bool,
    stretch: bool,
    snapshot: Option<f64>,
}

fn parse_args() -> Result<Opts, String> {
    let mut r = Raw::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        // accept both "--opt value" and "--opt=value"; option names are case-insensitive
        // so the Python spellings (--Lx, --Ny) work too
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n.to_ascii_lowercase(), Some(v.to_string())),
            _ => (arg.to_ascii_lowercase(), None),
        };
        let mut val = |what: &str| -> Result<String, String> {
            inline
                .clone()
                .or_else(|| args.next())
                .ok_or_else(|| format!("{name} needs a {what}"))
        };
        fn num<T: std::str::FromStr>(name: &str, s: String) -> Result<Option<T>, String> {
            s.parse()
                .map(Some)
                .map_err(|_| format!("invalid value for {name}: {s:?}"))
        }
        match name.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "-v" | "--version" => {
                println!("wavy {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "--mode" => {
                r.mode = Some(match val("mode")?.as_str() {
                    "waves" => Mode::Waves,
                    "enso" => Mode::Enso,
                    "wbc" => Mode::Wbc,
                    m => return Err(format!("unknown mode {m:?} (waves|enso|wbc)")),
                })
            }
            "--geometry" => {
                r.geometry = Some(match val("geometry")?.as_str() {
                    "basin" => Geometry::Basin,
                    "channel" => Geometry::Channel,
                    g => return Err(format!("unknown geometry {g:?} (basin|channel)")),
                })
            }
            "--view" => match val("view")?.as_str() {
                "sst" => r.view = Some(View::Sst),
                "h" | "thermocline" => r.view = Some(View::Thermocline),
                "pv" => r.wbc_view = Some(WbcView::Pv),
                "zeta" | "vorticity" => r.wbc_view = Some(WbcView::Vorticity),
                v => return Err(format!("unknown view {v:?} (sst|h|pv|zeta)")),
            },
            "--slip" => {
                r.slip = Some(match val("slip")?.as_str() {
                    "no" | "no-slip" | "noslip" => Slip::NoSlip,
                    "free" | "free-slip" | "freeslip" => Slip::FreeSlip,
                    v => return Err(format!("unknown slip condition {v:?} (no|free)")),
                })
            }
            "--visc" => r.visc = num(&name, val("number")?)?,
            "--lx" => r.lx = num(&name, val("number")?)?,
            "--ly" => r.ly = num(&name, val("number")?)?,
            "--nx" => r.nx = num(&name, val("integer")?)?,
            "--ny" => r.ny = num(&name, val("integer")?)?,
            "--dt" => r.dt = num(&name, val("number")?)?,
            "--damping" => r.damping = num(&name, val("number")?)?,
            "--filter" => r.filter = num(&name, val("number")?)?,
            "--kick-every" => r.kick_every = num(&name, val("number")?)?,
            "--seed" => r.seed = num(&name, val("integer")?)?,
            "--mu" => r.mu = num(&name, val("number")?)?,
            "--eps" => r.eps = num(&name, val("number")?)?,
            "--burst-amp" => r.burst_amp = num(&name, val("number")?)?,
            "--burst-rate" => r.burst_rate = num(&name, val("number")?)?,
            "--wind-x" => r.wind_x = num(&name, val("number")?)?,
            "--wind-sy" => r.wind_sy = num(&name, val("number")?)?,
            "--init-h" => r.init_h = num(&name, val("number")?)?,
            "--fps" => r.fps = num(&name, val("number")?)?,
            "--frame-dt" => r.frame_dt = num(&name, val("number")?)?,
            "--gain" => r.gain = num(&name, val("number")?)?,
            "--no-hud" => r.no_hud = true,
            "--no-equator" => r.no_equator = true,
            "--stretch" => r.stretch = true,
            "--snapshot" => {
                let v = val("time")?;
                r.snapshot = Some(match v.strip_suffix('y') {
                    Some(years) => num::<f64>(&name, years.to_string())?.unwrap() * YEAR,
                    None => num::<f64>(&name, v.trim_end_matches('d').to_string())?.unwrap(),
                });
            }
            _ => return Err(format!("unknown option {arg:?} (see --help)")),
        }
    }

    let mode = r.mode.unwrap_or(Mode::Waves);
    let enso = mode == Mode::Enso;
    let wbc_mode = mode == Mode::Wbc;
    let pick =
        |v: Option<f64>, waves: f64, enso_v: f64| v.unwrap_or(if enso { enso_v } else { waves });
    let geometry = r.geometry.unwrap_or(Geometry::Basin);
    if enso && geometry != Geometry::Basin {
        return Err("--mode enso needs the basin geometry".into());
    }
    let coupling = enso.then(|| {
        let d = Coupling::default();
        Coupling {
            mu: r.mu.unwrap_or(d.mu),
            eps: r.eps.unwrap_or(d.eps),
            burst_amp: r.burst_amp.unwrap_or(d.burst_amp),
            burst_rate: r.burst_rate.unwrap_or(d.burst_rate),
            wind_x: r.wind_x.unwrap_or(d.wind_x),
            wind_sy: r.wind_sy.unwrap_or(d.wind_sy),
            ..d
        }
    });
    let o = Opts {
        mode,
        params: Params {
            geometry,
            lx: r.lx.unwrap_or(40.0),
            ly: r.ly.unwrap_or(8.0),
            nx: r.nx.unwrap_or(if enso { 100 } else { 200 }),
            ny: r.ny.unwrap_or(if enso { 40 } else { 80 }),
            dt: pick(r.dt, 0.04, 0.08),
            damping: pick(r.damping, 0.008, 0.004),
            filter: pick(r.filter, 0.0, 0.001),
            coupling,
        },
        kick_every: r.kick_every.unwrap_or(40.0),
        init_h: r.init_h.unwrap_or(0.5),
        seed: r.seed,
        view: r.view.unwrap_or(View::Sst),
        wbc_view: r.wbc_view.unwrap_or(WbcView::Pv),
        slip: r.slip.unwrap_or(Slip::NoSlip),
        visc: r.visc.unwrap_or(WbcParams::default().nu),
        fps: r.fps.unwrap_or(30.0),
        frame_dt: if wbc_mode {
            r.frame_dt.unwrap_or(3.0)
        } else {
            pick(r.frame_dt, 0.2, 0.8)
        },
        gain: r.gain.unwrap_or(1.0),
        hud: !r.no_hud,
        equator: !r.no_equator,
        stretch: r.stretch,
        snapshot: r.snapshot,
    };

    if o.visc.is_nan() || o.visc < 0.0 {
        return Err("--visc must be non-negative".into());
    }
    let p = &o.params;
    if !(0.0..0.125).contains(&p.filter) {
        return Err("--filter must be in [0, 1/8)".into());
    }
    if p.nx < 2 || p.ny < 2 {
        return Err("--nx and --ny must be at least 2".into());
    }
    if !(p.lx > 0.0 && p.ly > 0.0 && p.dt > 0.0 && o.fps > 0.0 && o.frame_dt > 0.0) {
        return Err("--lx, --ly, --dt, --fps and --frame-dt must be positive".into());
    }
    // RK4 is stable for c·dt/dx up to ~2.8/sqrt(2) in 2-D; keep a margin
    let dmin = (p.lx / p.nx as f64).min(2.0 * p.ly / p.ny as f64);
    if p.dt > 1.5 * dmin {
        return Err(format!(
            "--dt {} is too large for this grid (keep it below ~{:.3})",
            p.dt,
            1.5 * dmin
        ));
    }
    Ok(o)
}

fn main() -> ExitCode {
    let o = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("wavy: {e}");
            return ExitCode::from(2);
        }
    };
    match run(o) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("wavy: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Niño-3 in °C and its ENSO phase label (±0.5 °C thresholds, like the ONI).
fn enso_phase(nino3: f64) -> (f64, &'static str) {
    let c = nino3 * NINO_DEGC;
    let phase = if c >= 0.5 {
        "El Niño"
    } else if c <= -0.5 {
        "La Niña"
    } else {
        "neutral"
    };
    (c, phase)
}

/// What gets drawn and how it is coloured.
struct Display {
    mode: Mode,
    view: View,
    waves_pal: Palette,
    sst_pal: Palette,
    auto: AutoScale,
}

impl Display {
    fn field<'a>(&self, sim: &'a Sim) -> &'a [f64] {
        match (self.mode, self.view) {
            (Mode::Enso, View::Sst) => &sim.model.sst,
            _ => &sim.model.s.h,
        }
    }

    fn palette(&self) -> &Palette {
        match (self.mode, self.view) {
            (Mode::Enso, View::Sst) => &self.sst_pal,
            _ => &self.waves_pal,
        }
    }

    /// Multiplier that maps the field to [-1, 1] colours.
    fn gain(&self, user_gain: f64) -> f64 {
        user_gain
            / match (self.mode, self.view) {
                (Mode::Waves | Mode::Wbc, _) => self.auto.scale,
                (Mode::Enso, View::Sst) => ENSO_SST_SCALE,
                (Mode::Enso, View::Thermocline) => ENSO_H_SCALE,
            }
    }

    fn update_scale(&mut self, sim: &Sim, dt: f64) {
        if self.mode == Mode::Waves {
            self.auto.update(&sim.model.s.h, dt);
        }
    }
}

/// Monthly Niño-3 history for the status-line sparkline.
struct History {
    values: VecDeque<f64>,
    next_t: f64,
}

impl History {
    const CAP: usize = 240; // 20 years of months
    const EVERY: f64 = YEAR / 12.0;

    fn new() -> Self {
        History {
            values: VecDeque::new(),
            next_t: 0.0,
        }
    }

    fn record(&mut self, t: f64, nino3_c: f64) {
        while t >= self.next_t {
            if self.values.len() == Self::CAP {
                self.values.pop_front();
            }
            self.values.push_back(nino3_c);
            self.next_t += Self::EVERY;
        }
    }

    /// Coloured sparkline of the last `width` months (one bar per month).
    fn sparkline(&self, width: usize, pal: &Palette) -> String {
        const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let n = self.values.len().min(width);
        let mut s = String::new();
        for &v in self.values.iter().skip(self.values.len() - n) {
            let lev = ((v.abs() / 2.5) * 8.0).ceil().clamp(1.0, 8.0) as usize - 1;
            // colour by sign/strength, grey-ish when neutral
            s.push_str(pal.fg_for((v / 2.0).clamp(-1.0, 1.0)));
            s.push(BARS[lev]);
        }
        s.push_str("\x1b[0m");
        s
    }
}

/// Key bindings shown by `?` (key, description).
fn key_help(mode: Mode) -> Vec<(&'static str, &'static str)> {
    let mut keys = vec![
        ("q", "quit"),
        ("space", "pause / resume"),
        ("+ / -", "speed (x1 to x10)"),
    ];
    match mode {
        Mode::Enso => {
            keys.push(("v", "switch SST / thermocline view"));
            keys.push(("k", "westerly wind burst (often starts an El Niño)"));
        }
        Mode::Waves => keys.push(("k", "drop in a random bump")),
        Mode::Wbc => {
            keys.push(("b", "switch no-slip / free-slip (restarts)"));
            keys.push(("s", "highlight f·q < 0 (symmetric instability)"));
            keys.push(("v", "switch PV / relative vorticity"));
            keys.push(("k", "add a small random perturbation"));
            keys.push(("a", "true proportions / fill the screen"));
        }
    }
    keys.extend([
        ("r", "restart"),
        ("e", "dotted equator line"),
        ("h", "status line"),
        ("?", "show / hide this help"),
    ]);
    keys
}

/// Draw the key-help box centred over the top `rows` rows of the screen.
fn draw_help(buf: &mut String, mode: Mode, cols: usize, rows: usize) {
    const KEY_W: usize = 7;
    const BOX: &str = "\x1b[0m\x1b[48;2;24;26;36m\x1b[38;2;120;130;150m"; // border
    const TEXT: &str = "\x1b[38;2;210;214;224m";
    const KEY: &str = "\x1b[1;38;2;255;170;60m";
    const RESET_KEY: &str = "\x1b[22m";

    let items = key_help(mode);
    let desc_max = items
        .iter()
        .map(|(_, d)| d.chars().count())
        .max()
        .unwrap_or(0);
    // "│  " + key + desc + "  │"
    let desc_w = desc_max.min(cols.saturating_sub(4 + KEY_W + 2));
    let inner = 2 + KEY_W + desc_w + 2;
    let width = inner + 2;
    let height = items.len() + 4; // borders + a blank line top and bottom
    if width > cols || height > rows || desc_w < 8 {
        return; // terminal too small — the status line still has the hint
    }
    let x0 = (cols - width) / 2 + 1;
    let y0 = (rows - height) / 2 + 1;
    let mut line = |y: usize, content: &str| {
        buf.push_str(&format!("\x1b[{};{}H{BOX}{content}", y0 + y, x0));
    };

    let title = " keys ";
    let dashes = inner - 1 - title.len();
    line(0, &format!("╭─{TEXT}{title}{BOX}{}╮", "─".repeat(dashes)));
    line(1, &format!("│{}│", " ".repeat(inner)));
    for (i, (k, d)) in items.iter().enumerate() {
        let d: String = d.chars().take(desc_w).collect();
        line(
            2 + i,
            &format!("│  {KEY}{k:<KEY_W$}{RESET_KEY}{TEXT}{d:<desc_w$}  {BOX}│"),
        );
    }
    line(height - 2, &format!("│{}│", " ".repeat(inner)));
    line(height - 1, &format!("╰{}╯", "─".repeat(inner)));
    buf.push_str("\x1b[0m");
}

fn run(o: Opts) -> io::Result<()> {
    let seed = o.seed.unwrap_or_else(Rng::seed_from_time);
    if o.mode == Mode::Wbc {
        return run_wbc(o, seed);
    }
    let geometry = o.params.geometry;
    let dt = o.params.dt;
    let steps_per_frame = ((o.frame_dt / dt).round() as u32).max(1);
    let mut sim = Sim::new(o.params, o.kick_every, seed, o.init_h);
    let (nx, ny) = (sim.model.nx(), sim.model.ny());

    let mut disp = Display {
        mode: o.mode,
        view: o.view,
        waves_pal: Palette::waves(),
        sst_pal: Palette::sst(),
        auto: AutoScale::new(),
    };
    let mut renderer = Renderer::default();
    let mut history = History::new();
    let mut buf = String::new();

    if let Some(t_end) = o.snapshot {
        let n = (t_end / dt).round() as u64;
        let (mut tmax, mut hmax) = (0.0f64, 0.0f64);
        for k in 0..n {
            sim.step();
            if k % steps_per_frame as u64 == 0 {
                disp.update_scale(&sim, dt * steps_per_frame as f64);
                tmax = tmax.max(sim.model.sst.iter().fold(0.0, |m, x| m.max(x.abs())));
                hmax = hmax.max(sim.model.s.h.iter().fold(0.0, |m, x| m.max(x.abs())));
            }
        }
        disp.update_scale(&sim, dt * steps_per_frame as f64);
        let (cols, rows) = if term::stdout_is_tty() {
            term::size()
        } else {
            (100, 30)
        };
        renderer.render(
            &mut buf,
            disp.field(&sim),
            nx,
            ny,
            cols,
            rows,
            disp.gain(o.gain),
            Marker::None,
            disp.palette(),
        );
        buf.push_str("\x1b[0m\n");
        io::stdout().lock().write_all(buf.as_bytes())?;
        let m = &sim.model;
        eprint!(
            "t = {:.2} ({:.2} yr)  energy = {:.6}  seed = {seed}",
            m.t,
            m.t / YEAR,
            m.energy()
        );
        if m.coupled() {
            let (c, phase) = enso_phase(m.nino3());
            eprintln!(
                "  nino3 = {:+.5} ({c:+.2} °C, {phase})  max|T| = {tmax:.3}  max|h| = {hmax:.3}",
                m.nino3()
            );
        } else {
            eprintln!("  colour scale = {:.4}", disp.auto.scale);
        }
        return Ok(());
    }

    term::install_signal_handlers();
    let screen = term::Screen::enter()?;
    let interactive = term::stdin_is_tty();
    let frame_time = Duration::from_secs_f64(1.0 / o.fps);
    let (mut speed, mut paused, mut hud, mut equator) = (1u32, false, o.hud, o.equator);
    let mut help = false;
    let mut keybuf = [0u8; 32];
    let mut last_size = (0, 0);
    let mut wind_label: (&str, Instant) = ("", Instant::now());
    let mut out = io::stdout().lock();

    disp.update_scale(&sim, 0.0);
    'outer: while !term::quit_requested() {
        let t0 = Instant::now();

        if !paused {
            for _ in 0..speed * steps_per_frame {
                sim.step();
            }
            disp.update_scale(&sim, dt * (speed * steps_per_frame) as f64);
        }
        if o.mode == Mode::Enso {
            history.record(sim.model.t, sim.model.nino3() * NINO_DEGC);
        }

        let (cols, rows) = term::size();
        buf.clear();
        if (cols, rows) != last_size {
            buf.push_str("\x1b[0m\x1b[2J");
            last_size = (cols, rows);
        }
        let body_rows = if hud { rows.saturating_sub(1) } else { rows };
        renderer.render(
            &mut buf,
            disp.field(&sim),
            nx,
            ny,
            cols,
            body_rows,
            disp.gain(o.gain),
            if equator {
                Marker::Horizontal(0.5)
            } else {
                Marker::None
            },
            disp.palette(),
        );
        if help {
            draw_help(&mut buf, o.mode, cols, body_rows);
        }
        if hud && rows > 0 {
            let pause = if paused { " (paused)" } else { "" };
            let (msg, spark) = match o.mode {
                Mode::Waves | Mode::Wbc => (
                    format!(
                        " t = {:7.1}   speed x{speed}{pause}   {}   Kelvin → east   \
                         ← Rossby west   [? keys]",
                        sim.model.t,
                        geometry.name(),
                    ),
                    None,
                ),
                Mode::Enso => {
                    let (c, phase) = enso_phase(sim.model.nino3());
                    let view = match disp.view {
                        View::Sst => "SST",
                        View::Thermocline => "thermocline",
                    };
                    // Bursts last only ~0.5 s of wall time at x1 and overlap at higher
                    // speeds, so the label may change at most once per BURST_LABEL_HOLD
                    // (a shown label stays at least that long; one that is still blowing
                    // when the hold ends is re-armed).
                    let now = Instant::now();
                    if now >= wind_label.1 {
                        let cur = match sim.burst_active() {
                            Some(s) if s > 0.0 => "westerly burst",
                            Some(_) => "easterly burst",
                            None => "",
                        };
                        let hold = if cur.is_empty() {
                            Duration::ZERO
                        } else {
                            BURST_LABEL_HOLD
                        };
                        wind_label = (cur, now + hold);
                    }
                    let wind = wind_label.0;
                    // every field has a fixed width so the line (and the sparkline
                    // after it) never shifts sideways
                    (
                        format!(
                            " year {:5.1}  x{speed:<2}{pause:<9} Niño-3 {c:+.1}°C {phase:<7}  \
                             {view:<11}  {wind:<14}  [? keys] ",
                            sim.model.t / YEAR,
                        ),
                        Some(()),
                    )
                }
            };
            let len = msg.chars().count();
            let msg: String = msg.chars().take(cols).collect();
            buf.push_str(&format!("\x1b[{rows};1H\x1b[0m\x1b[2K{msg}"));
            if spark.is_some() && cols > len + 8 {
                buf.push_str(&history.sparkline(cols - len - 1, &disp.sst_pal));
            }
        }
        out.write_all(buf.as_bytes())?;
        out.flush()?;

        // wait out the rest of the frame, handling keys as they arrive
        // Input is polled at least once per frame, even when the frame already
        // took longer than its budget (high speeds) — otherwise keys, including
        // q, would never be read.
        loop {
            if term::quit_requested() {
                break;
            }
            let left = frame_time.saturating_sub(t0.elapsed());
            if !interactive {
                std::thread::sleep(left);
                break;
            }
            for &k in term::poll_keys(left.as_millis() as i32, &mut keybuf) {
                match k {
                    b'q' | b'Q' | 3 | 4 => break 'outer, // q, Ctrl-C, Ctrl-D
                    b' ' => paused = !paused,
                    b'+' | b'=' => speed = (speed + 1).min(10),
                    b'-' | b'_' => speed = speed.saturating_sub(1).max(1),
                    b'e' => equator = !equator,
                    b'h' => {
                        hud = !hud;
                        last_size = (0, 0); // force a full clear
                    }
                    b'v' if o.mode == Mode::Enso => {
                        disp.view = match disp.view {
                            View::Sst => View::Thermocline,
                            View::Thermocline => View::Sst,
                        }
                    }
                    b'k' => sim.kick(),
                    b'?' => help = !help,
                    27 => help = false, // Esc
                    b'r' => {
                        sim.restart();
                        disp.auto.reset();
                        disp.update_scale(&sim, 0.0);
                        history = History::new();
                    }
                    _ => {}
                }
            }
            if t0.elapsed() >= frame_time {
                break;
            }
        }
    }
    drop(out);
    drop(screen);
    Ok(())
}

/// Fill `out` with the WBC display field: columns = longitude (western coast
/// on the left), rows = latitude (row 0 = south, drawn at the bottom). Only the
/// rows outside the sponges are shown, as in the paper's figures.
/// Values are scaled by f(1000 km): PV as q·H/f_ref, vorticity as ζ/f_ref.
/// With `crit`, points where f·q < 0 become NaN (drawn highlighted).
/// Returns (width, height) of the field.
fn wbc_field(m: &mut Wbc, view: WbcView, crit: bool, out: &mut Vec<f64>) -> (usize, usize) {
    let (nx, ny) = (m.nx(), m.ny());
    let (fref, h0) = (m.f_ref(), m.h0());
    let fs: Vec<f64> = (0..=ny).map(|j| m.f_corner(j)).collect();
    let (lo, hi) = m.visible_rows();
    let (zeta, q) = m.vorticity();
    let w = nx + 1;
    out.clear();
    for (j, &f) in fs.iter().enumerate().take(hi + 1).skip(lo) {
        for i in 0..w {
            let k = j * w + i;
            let (qk, zk) = (q[k] as f64, zeta[k] as f64);
            out.push(if crit && f * qk < 0.0 {
                f64::NAN
            } else {
                match view {
                    WbcView::Pv => qk * h0 / fref,
                    WbcView::Vorticity => zk / fref,
                }
            });
        }
    }
    (w, hi - lo + 1)
}

/// Where the WBC strip goes on screen: (first column, width). With true
/// proportions the strip is as wide as the domain's aspect ratio allows
/// (pixels are square: one cell = 1 wide × 2 tall), centred; otherwise it
/// fills the width.
fn wbc_layout(m: &Wbc, cols: usize, rows: usize, stretch: bool) -> (usize, usize) {
    if stretch || cols == 0 {
        return (0, cols);
    }
    let px = 2.0 * rows as f64 * m.lx() / m.visible_height();
    let w = (px.round() as usize).clamp(1, cols);
    ((cols - w) / 2, w)
}

/// Blank the parts of the first `rows` lines outside columns col0..col0+w
/// (they may hold the help box or a previous layout).
fn clear_margins(buf: &mut String, cols: usize, rows: usize, col0: usize, w: usize) {
    let right = cols.saturating_sub(col0 + w);
    for r in 1..=rows {
        if col0 > 0 {
            buf.push_str(&format!("\x1b[{r};1H\x1b[0m{}", " ".repeat(col0)));
        }
        if right > 0 {
            buf.push_str(&format!(
                "\x1b[{r};{}H\x1b[0m{}",
                col0 + w + 1,
                " ".repeat(right)
            ));
        }
    }
}

/// Share of corners north of 250 km with anomalous PV (f·q < 0).
fn anomalous_share(m: &mut Wbc) -> f64 {
    let (nx, ny) = (m.nx(), m.ny());
    let lat0 = m.equator_frac() * ny as f64;
    let j0 = (lat0 + 250e3 / ((m.p_y_extent()) / ny as f64)).ceil() as usize;
    let fs: Vec<f64> = (0..=ny).map(|j| m.f_corner(j)).collect();
    let (_, q) = m.vorticity();
    let (mut n, mut bad) = (0usize, 0usize);
    for j in j0.min(ny)..=ny {
        for i in 0..=nx {
            n += 1;
            if fs[j] * (q[j * (nx + 1) + i] as f64) < 0.0 {
                bad += 1;
            }
        }
    }
    bad as f64 / n.max(1) as f64
}

fn run_wbc(o: Opts, seed: u64) -> io::Result<()> {
    let params = WbcParams {
        slip: o.slip,
        nu: o.visc,
        ..WbcParams::default()
    };
    let mut m = Wbc::new(params, seed);
    let steps_per_frame = ((o.frame_dt * 3600.0 / m.dt()).round() as u32).max(1);
    let pv_pal = Palette::pv();
    let zeta_pal = Palette::sst();
    let mut view = o.wbc_view;
    let mut crit = false;
    let mut renderer = Renderer::default();
    let mut field = Vec::new();
    let mut buf = String::new();
    // equator position within the visible rows
    let marker = |on: bool, m: &Wbc| {
        let (lo, hi) = m.visible_rows();
        let jeq = m.equator_frac() * m.ny() as f64;
        if on {
            Marker::Horizontal((jeq - lo as f64 + 0.5) / (hi - lo + 1) as f64)
        } else {
            Marker::None
        }
    };
    let palette = |v: WbcView| match v {
        WbcView::Pv => &pv_pal,
        WbcView::Vorticity => &zeta_pal,
    };

    if let Some(days) = o.snapshot {
        let n = (days * DAY / m.dt()).round() as u64;
        for _ in 0..n {
            m.step();
        }
        let (cols, rows) = if term::stdout_is_tty() {
            term::size()
        } else {
            (100, 30)
        };
        let (w, h) = wbc_field(&mut m, view, false, &mut field);
        let (col0, sw) = wbc_layout(&m, cols, rows, o.stretch);
        renderer.render_region(
            &mut buf,
            &field,
            w,
            h,
            col0,
            sw,
            rows,
            o.gain,
            marker(true, &m),
            palette(view),
        );
        buf.push_str("\x1b[0m\n");
        io::stdout().lock().write_all(buf.as_bytes())?;
        eprintln!(
            "day {:.1}  {}  min h = {:.1} m  max |u| = {:.2} m/s  anomalous PV north of 250 km = {:.1} %  seed = {seed}",
            m.t / DAY,
            m.slip().name(),
            m.min_h(),
            m.max_speed(),
            100.0 * anomalous_share(&mut m),
        );
        return Ok(());
    }

    term::install_signal_handlers();
    let screen = term::Screen::enter()?;
    let interactive = term::stdin_is_tty();
    let frame_time = Duration::from_secs_f64(1.0 / o.fps);
    let (mut speed, mut paused, mut hud, mut equator) = (1u32, false, o.hud, o.equator);
    let mut stretch = o.stretch;
    let mut help = false;
    let mut keybuf = [0u8; 32];
    let mut last_size = (0, 0);
    let mut out = io::stdout().lock();

    'outer: while !term::quit_requested() {
        let t0 = Instant::now();
        if !paused {
            for _ in 0..speed * steps_per_frame {
                m.step();
            }
        }

        let (cols, rows) = term::size();
        buf.clear();
        if (cols, rows) != last_size {
            buf.push_str("\x1b[0m\x1b[2J");
            last_size = (cols, rows);
        }
        let body_rows = if hud { rows.saturating_sub(1) } else { rows };
        let (w, h) = wbc_field(&mut m, view, crit, &mut field);
        let (col0, sw) = wbc_layout(&m, cols, body_rows, stretch);
        clear_margins(&mut buf, cols, body_rows, col0, sw);
        renderer.render_region(
            &mut buf,
            &field,
            w,
            h,
            col0,
            sw,
            body_rows,
            o.gain,
            marker(equator, &m),
            palette(view),
        );
        if help {
            draw_help(&mut buf, o.mode, cols, body_rows);
        }
        if hud && rows > 0 {
            let pause = if paused { " (paused)" } else { "" };
            let v = match view {
                WbcView::Pv => "PV",
                WbcView::Vorticity => "vorticity",
            };
            let c = if crit { "f·q<0 marked" } else { "" };
            let msg = format!(
                " day {:6.1}  x{speed:<2}{pause:<9} {:<9}  {v:<9}  {c:<12}  \
                 equator ┄  north ↑  coast ←   [? keys]",
                m.t / DAY,
                m.slip().name(),
            );
            let msg: String = msg.chars().take(cols).collect();
            buf.push_str(&format!("\x1b[{rows};1H\x1b[0m\x1b[2K{msg}"));
        }
        out.write_all(buf.as_bytes())?;
        out.flush()?;

        // Input is polled at least once per frame, even when the frame already
        // took longer than its budget (high speeds) — otherwise keys, including
        // q, would never be read.
        loop {
            if term::quit_requested() {
                break;
            }
            let left = frame_time.saturating_sub(t0.elapsed());
            if !interactive {
                std::thread::sleep(left);
                break;
            }
            for &k in term::poll_keys(left.as_millis() as i32, &mut keybuf) {
                match k {
                    b'q' | b'Q' | 3 | 4 => break 'outer,
                    b' ' => paused = !paused,
                    b'+' | b'=' => speed = (speed + 1).min(10),
                    b'-' | b'_' => speed = speed.saturating_sub(1).max(1),
                    b'e' => equator = !equator,
                    b'h' => {
                        hud = !hud;
                        last_size = (0, 0);
                    }
                    b'b' => {
                        m.set_slip(m.slip().other());
                        m.restart();
                    }
                    b's' => crit = !crit,
                    b'v' => {
                        view = match view {
                            WbcView::Pv => WbcView::Vorticity,
                            WbcView::Vorticity => WbcView::Pv,
                        }
                    }
                    b'k' => m.perturb(0.5),
                    b'a' => stretch = !stretch,
                    b'r' => m.restart(),
                    b'?' => help = !help,
                    27 => help = false,
                    _ => {}
                }
            }
            if t0.elapsed() >= frame_time {
                break;
            }
        }
    }
    drop(out);
    drop(screen);
    Ok(())
}
