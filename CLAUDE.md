# wavy — equatorial ocean dynamics as a terminal screensaver

Inspired by `pipes.sh` (Arch package, github.com/pipeseroni/pipes.sh): a full-screen
terminal animation, but showing equatorial ocean physics. Four modes:

| mode | what | model |
|---|---|---|
| `waves` (default) | free equatorial Kelvin + Rossby waves, random height bumps | linear shallow water, Matsuno units |
| `enso` | El Niño / La Niña cycles | linear ocean + SST + Bjerknes wind feedback (stripped-down Cane–Zebiak) |
| `wbc` | cross-equatorial western boundary current, no-slip vs free-slip | nonlinear 1.5-layer shallow water, SI units |
| `kh` | Kelvin–Helmholtz billows in a stratified shear layer | 2-D Boussinesq x–z slice, vorticity–streamfunction |

Two implementations:
- **`wavy/` — Rust, the shipped executable.** Simulates *live* while rendering, never loops.
  All four modes.
- **`src/wavy/` — Python, research/prototyping.** Precompute-then-play for the waves
  mode, plus the ENSO prototype used for parameter tuning (with plots). No WBC prototype.

## State of the repo (read first after moving)
- Renamed from eqwaves / terminal-wave to **`wavy`** (2026-11-21). Repo
  `/home/schuejo/git/wavy`, branch `main`, remote `git@github.com:jusjoh/wavy.git`. Rust crate
  dir `wavy/` (binary `wavy`); Python package `wavy` in `src/wavy/` with scripts `wavy-play` /
  `wavy-simulate` (not `wavy`, so they don't clash with the Rust binary on PATH).
- `./terminal-wave/` (if still present) is a leftover clone of the old `jusjoh/terminal-wave`
  repo, README only — not part of the project, git-ignored, delete it.
- Things that should be committed: `CLAUDE.md`, `README.md` (empty — still to write),
  `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `src/`, `wavy/` (incl.
  `Cargo.lock`, excl. `target/`), `tools/`.
- Generated / don't commit: `wavy/target/`, `.venv/`, `__pycache__/` (all in .gitignore).
  `waves.npz` (7 MB, output of `wavy-simulate`) is regenerable — better left out.
- The source paper for the WBC mode (Goldsworth et al. 2021, PDF) was attached in chat, not
  stored in the repo; reference below.

## Layout
```
wavy/                 Rust crate (binary `wavy`), only dependency: libc
  Cargo.toml             release profile: lto, codegen-units = 1
  src/main.rs            CLI, mode dispatch, frame loops, HUD, help box, WBC display layout
  src/solver.rs          linear equatorial shallow water (waves + enso modes), wind bursts
  src/wbc.rs             nonlinear 1.5-layer WBC model (wbc mode)
  src/kh.rs              2-D Boussinesq KH model, own FFT + Poisson solver (kh mode)
  src/render.rs          truecolor half-block renderer, palettes, auto colour scale
  src/term.rs            termios/alt-screen/poll/ioctl/signals via libc
  src/rng.rs             SplitMix64 + Box–Muller (no rand crate)
src/wavy/       Python package (uv project, pyproject.toml)
  simulate.py            precompute waves -> waves.npz  (CHANNEL ONLY, see below)
  play.py                truecolor player for waves.npz
  enso.py                ENSO prototype: basin solver + SST + coupling, diagnostics, plots
  enso_scan.py           parallel parameter scans over enso.py
tools/screen2png.py      ANSI terminal capture -> PNG (visual testing, see "Testing")
tools/co2.py             CO2 estimate from Claude Code's local token logs (stdlib only; rates are guesses)
```

## Toolchains and commands
- **Rust**: installed with rustup by the user; binaries in `~/.cargo/bin`. Non-interactive
  shells (agents) may not have it on PATH → `export PATH=$HOME/.cargo/bin:$PATH`.
- **Python**: uv project, Python 3.13 (`.python-version`), dependency `numpy` only;
  matplotlib is pulled ad hoc with `uv run --with matplotlib ...`.

```
cd wavy
cargo build --release                       # -> wavy/target/release/wavy
cargo test --release                        # 10 tests, ~12 s
cargo clippy --release --all-targets        # kept at zero warnings
cargo fmt
cargo install --path wavy                # (from repo root) puts wavy in ~/.cargo/bin

./target/release/wavy                    # waves
./target/release/wavy --mode enso
./target/release/wavy --mode wbc [--slip no|free]
./target/release/wavy --mode kh [--ri 0.08]
./target/release/wavy --help             # all options

uv run wavy-simulate && uv run wavy-play waves.npz       # Python waves
uv run --with matplotlib python -m wavy.enso --plot enso.png # 40 model yr, ~3 min
uv run --with matplotlib python -m wavy.enso_scan OUTDIR '{"mu":[0.38,0.42],"years":[40]}'
```

## wavy CLI and keys
Options are case-insensitive (`--Lx` works) and accept `--opt value` or `--opt=value`.
Mode-dependent defaults are resolved after parsing (`Raw` → `Opts` in main.rs).

- Common: `--mode waves|enso|wbc|kh`, `--fps 30`, `--frame-dt` (model time per frame at x1:
  waves 0.2, enso 0.8 [Matsuno units], wbc 3 [hours], kh 0.1), `--gain`, `--no-hud`, `--no-equator`,
  `--seed N` (default from clock), `--snapshot T` (simulate to T, print one frame to stdout,
  stats to stderr, exit; `30y` = years; in wbc T is days, `35` or `35d`).
- Linear model: `--geometry basin|channel`, `--lx 40`, `--ly 8`, `--nx/--ny` (waves 200x80,
  enso 100x40), `--dt` (0.04 / 0.08), `--damping` (0.008 / 0.004), `--filter` (0 / 0.001).
- Waves: `--kick-every 40`.
- ENSO: `--mu 0.42`, `--eps 0.03`, `--burst-amp 0.05`, `--burst-rate 0.025`, `--wind-x 0.3`,
  `--wind-sy 1.0`, `--init-h 0.5`, `--view sst|h`.
- WBC: `--slip no|free` (default no), `--visc 100` (m²/s), `--view pv|zeta`, `--true-proportions`
  (alias `--no-stretch`; `--stretch`, now the default, is still accepted).
- KH: `--ri 0.08`, `--re 1500`, `--cycle 150` (0 = never rebuild), `--view temp|zeta`.

Keys (all modes): `q`/Ctrl-C quit, `space` pause, `+`/`-` speed x1..x10, `e` equator line,
`h` status line, `r` restart, `?` key-help box (`?`/Esc closes). Mode keys:
waves `k` random bump · enso `v` SST/thermocline, `k` strong westerly burst ·
wbc `b` switch no-slip/free-slip **and restart**, `s` highlight f·q<0, `v` PV/vorticity,
`k` small random perturbation, `a` fill width / true proportions ·
kh `v` temperature/vorticity, `[`/`]` Richardson number ∓0.02, `k` random perturbation,
`m` rebuild the shear layer now (no equator line in kh).

## Rust architecture notes
- **Rendering** (`render.rs`): upper half block `▀` → 2 square-ish pixels per cell (fg = top,
  bg = bottom); truecolor escapes emitted only when the colour changes; whole frame built in
  one `String` and written once. Field is row-major, row 0 drawn at the bottom; **bilinear**
  resampling to the terminal size each frame (handles resize). Palettes: `waves` (blue–black–
  orange), `sst` (blue–black–red), `pv` (grey scale: white = negative/SH PV, mid grey 0,
  black = positive/NH PV; was teal–pink like the paper until the user asked for this),
  48 levels per sign, gamma 0.6, plus a yellow HIGHLIGHT entry used for NaN pixels and a
  MARKER entry (brightest positive colour; orange for `pv`). `Marker::Horizontal(frac)`
  draws the dotted equator. `render_region` draws into a column range (used for WBC's
  centred strip). `AutoScale` (waves mode only): EMA (tau 60 units) of the per-frame 99.7th
  percentile of |h|, floor 0.05.
- **Terminal** (`term.rs`): cbreak (ICANON/ECHO/ISIG off) + alt screen + hidden cursor,
  restored by `Drop` (also on panic). Ctrl-C arrives as byte 3 (ISIG off); SIGINT/TERM/HUP
  handlers set a flag. Size via TIOCGWINSZ, fallback $COLUMNS/$LINES, then 100x30 (snapshot
  to a pipe uses 100x30).
- **Frame loops** (three: `run` for waves/enso, `run_wbc`, `run_kh`): step the model `speed ×
  steps_per_frame` times, render, then wait out the frame while handling keys. Input is
  **always polled at least once per frame** — before that fix, frames slower than the
  budget (wbc at x10) never read keys, not even `q`.
- **HUD**: every field fixed-width (nothing shifts sideways). ENSO: model year, Niño-3 in
  nominal °C (model index × 5), phase (±0.5 °C like the ONI), wind-burst label (changes at
  most every 2 s wall-clock; bursts last ~0.5 s at x1 and overlap when fast), monthly
  Niño-3 sparkline (last up to 240 months). The help box is drawn after the field, centred.
- **Performance**: waves ~0.5 ms/step (200x80), enso ~0.17 ms/step (100x40), wbc ~0.6 ms/step
  (80x200, f32). Waves/enso hold 30 fps at x10; wbc drops to ~18 fps at x10 (55 ms/frame).
  kh ~2 ms/step (256x128, f32), one step per frame at x1, holds 30 fps at x10.
  All solvers use row-slice loops with edge cases outside the inner loop (bounds checks
  elided, vectorised) — rewrites from naive indexing gave 1.5–1.75x.

## Mode: waves — linear equatorial shallow water (solver.rs)
Matsuno non-dimensional units (length = equatorial deformation radius ≈ 350 km, time ≈ 1.45 d,
Kelvin speed = 1; 252 units ≈ 1 year):

    u_t - y v = -h_x - r u + tau
    v_t + y u = -h_y - r v
    h_t + u_x + v_y = -r h

- Arakawa C-grid, RK4, solid walls at y = ±Ly (v = 0). Layout: h (Ny,Nx); u x-faces, face i =
  **west** face of cell i (basin (Ny,Nx+1) with u = 0 on both walls; channel (Ny,Nx)
  periodic); v (Ny+1,Nx) with wall rows 0/Ny. (The Python channel solver used the east face —
  physically identical.)
- Initial unit Gaussian bump at x = 0.3 Lx; random bumps every 40 units (basin: middle 70 %);
  damping 0.008 keeps amplitude bounded.
- Verified: channel, no kicks/damping, t = 40 → energy 1.574059, **identical to the Python
  solver**; both drift +0.21 % (the 4-point Coriolis averaging on the C-grid isn't exactly
  energy-neutral — not a bug; the test tolerance is 0.5 %). Kelvin wave moves east at speed 1
  (test), n=1 Rossby lobes west at ~1/3. Basin: Kelvin hits the east wall at t ≈ 32 →
  poleward coastal Kelvin waves + westward Rossby "beak"; Rossby waves reflect at the west wall
  into a new Kelvin wave (delayed-oscillator picture).
- Known limitation: coastal Kelvin waves near y = ±8 have width ~1/8 vs dy = 0.2 → thin,
  under-resolved lines (`--ny 160` or `--ly 5` help).

## Mode: enso — coupled ENSO (solver.rs `Coupling`, prototype enso.py)
Stripped-down Cane–Zebiak on the basin model (h = thermocline depth anomaly, deep = warm):

    T_t = eps (M(x,y) F(h) - T)                 SST anomaly
    tau = mu · Nino3(T) · G(x,y) + bursts       zonal wind stress added to the u equation

- M = upwelling mask: exp(-y²/(2·1.5²)) × sigmoid((x/Lx - 0.45)/0.06) (eastern cold tongue).
  F = asymmetric tanh: a_warm 1.0 for h > 0, a_cold 0.6 for h < 0 (El Niño > La Niña).
  Niño-3 box x/Lx ∈ (0.56, 0.94), |y| < 1.6. G = Gaussian patch at wind_x 0.3 Lx, half-widths
  0.12 Lx and wind_sy 1.0. Bursts: rate 1/40 per unit, amp ±0.05, 8–20 units long, at
  0.15–0.5 Lx, Gaussian 2.5 (x) × 2.0 (y). Key `k`: westerly burst 4× amp for 30 units.
- Defaults (tuned in Python): mu 0.42, eps 0.03, r 0.004, grid 100x40, dt 0.08, initial bump
  0.5 at 0.5 Lx; frame-dt 0.8 → ~1 model year per 10 s at x1.
- **Tuning results** (enso.py, 20–40 yr runs):
  - eps 0.01 (SST ~5 months): never oscillates — mu ≤ 0.5 decays, mu ≥ 0.6 locks into a
    **stationary permanent El Niño** (tilt feedback beats recharge/discharge). A wide wind
    patch (wind_sy 2–3) or patch at 0.4–0.45 Lx also goes stationary.
  - eps 0.03, wind_x 0.3, wind_sy 1.0: oscillatory mode, **period ~3 yr**; damped at mu 0.40,
    grows at 0.45, limit cycle at 0.50 (Niño-3 rms ~0.33). West heat content leads Niño-3 by
    ~¼ cycle (recharge oscillator).
  - Defaults mu 0.42 (just subcritical) + bursts → irregular ENSO, Niño-3 rms ~0.17, strong
    events every few years, quiet multi-year stretches. Real period 3–7 yr → model is short.
  - `diagnostics()` period = 2·duration/sign changes: biased low when noise is on.
- **Rust vs Python differences** (deliberate):
  - SST is operator-split: wind uses T from the step start, then T relaxes exactly
    (1 - e^(-eps dt)) towards M·F(h_new) once per step — 2x faster. Niño-3 after 5 yr
    (mu 0.45, no noise, init 0.1): Rust 0.04838 vs Python 0.04837.
  - `a·tanh(h/a)` computed via one exp (libm tanh ~3x slower).
  - **x-only biharmonic filter** (`--filter`, enso default 0.001): without it, grid-scale
    stripes (short Rossby waves from the west wall) build up over decades. Must be a
    symmetric operator L² with wall-aware BCs (Neumann for h/v, Dirichlet for u at basin
    walls, periodic in the channel). History: a version that skipped the 2 edge points was
    non-symmetric and **blew up** with coupling; adding a y-filter or 0.003+ visibly shifted
    the ENSO cycle (the Kelvin wave is only ~2.5 cells wide in y).
- Fixed colour scales in enso (SST 0.5, h 1.0) — amplitude is the signal; h saturates in the
  far west at strong events (|h| reaches ~2), accepted to keep eastern structure visible.
- Test `enso_oscillates_above_threshold`: mu 0.5, no noise → period 2–5 yr, rms > 0.05.

## Mode: wbc — cross-equatorial western boundary current (wbc.rs)
After **Goldsworth, Marshall & Johnson (2021): "Symmetric Instability in Cross-Equatorial
Western Boundary Currents", J. Phys. Oceanogr. 51, 2049–2067, doi:10.1175/JPO-D-20-0273.1**
(also Edwards & Pedlosky 1998, JPO 28). The user asked to switch between the paper's
no-slip and free-slip lateral boundary conditions.

    u_t - q V = -B_x + ν (D_x - ζ_y) - r u
    v_t + q U = -B_y + ν (D_y + ζ_x) - r (v - V_jet)
    h_t + U_x + V_y = -r (h - h_ref)
    q = (f + ζ)/h,  f = βy,  B = g'h + |u|²/2,  (U, V) = h (u, v),  D = u_x + v_y

- Nonlinear 1.5-layer reduced gravity, **SI units**: β 2.3e-11, g' 0.02, H 300 m (c 2.45 m/s),
  C-grid, enstrophy-conserving (Sadourny) q·flux terms (q averaged, fluxes 4-point averaged),
  Bernoulli gradient, Laplacian viscosity in vector-invariant form, RK4, dt 1200 s.
  Layout: h (Ny,Nx); u (Ny,Nx+1) zero on both walls; v (Ny+1,Nx) with prescribed open
  boundary rows; ζ, q at corners (Ny+1,Nx+1); h at corners = average of adjacent cells.
- Domain x 0..500 km (nx 80, dx 6.25 km; west wall at x = 0 is the coast, east wall always
  free-slip), y -400..1600 km (ny 200, dy 10 km). **Bickley jet** V0 0.9 m/s, x_mid 40 km,
  δ 30 km (paper values). Transport h0·V_jet prescribed at the S and N open boundaries (equal
  in and out); sponges (100 km S, 300 km N, quadratic ramp to 2e-5 1/s) relax u→0, v→V_jet,
  h→h_ref (geostrophic, g'h_x = fV, flat far east). Initial state: jet everywhere + h_ref +
  seeded noise 0.05 m. `restart` reseeds identically, so `b` compares both BCs from the same
  state.
- **The only difference between the BCs**: ζ on the western wall = 2v/Δx (no-slip, ghost v = -v;
  the wall injects positive vorticity through the viscous stress) or 0 (free-slip).
- Fields are **f32** (`type R`): identical to f64 to all printed digits, ~1.7x faster. Grid
  setup in f64, converted; hot-loop constants precomputed in `Grid`. Public API returns f64.
- Tuning: ν 300 too damped (free-slip never sheds), ν 50 grid speckles by day 50, **ν 100
  default**; dx 10 km over 800 km was too coarse/too much empty ocean → 500 km at 6.25 km.
- **Result matches the paper qualitatively**: no-slip retroflects and sheds anticyclonic eddies
  with negative-PV (white) cores and a dark rim of wall-injected positive PV (paper Figs. 7/10);
  free-slip carries negative PV far north along the coast in elongated bulges without
  retroflection, faster flow (Figs. 12/14). Anomalous PV (f·q < 0) north of 250 km:
  free-slip ~8.5 % vs no-slip ~4–6 %; at day 150 (seed 2): 8.6 % vs 4.2 %. Eddies form around
  days 30–35 (paper: ~26). Stable ≥ 150 days for both BCs (min h ~175 m, max |u| < 3 m/s).
- **Not representable**: the symmetric instability itself (stacked x–z overturning cells,
  ~100 m vertical scale, need vertical structure). Instead `s` highlights where f·q < 0 —
  the instability criterion (paper: fQ < 0) — in yellow (NaN → highlight colour).
- Display: upright like the paper — north up, coast on the left, sponges hidden (visible rows
  -300..1300 km). **Stretched to the full terminal width by default** (user's choice, eddies
  become flat ellipses). `a` / `--true-proportions` switches to a centred undistorted strip
  ~0.6 × the height in columns (pixels are square: 1 cell = 1 wide × 2 tall); its margins
  are blanked every frame (otherwise the help box leaves residue). PV shown as
  q·H/f(1000 km), vorticity as ζ/f(1000 km). Frame-dt 3 h → ~4 model days per second at x1.
- Tests: `stays_bounded` (60 days both BCs: min h > 50 m, |u| < 5 m/s),
  `anomalous_pv_crosses_equator` (free-slip, 30 days, > 10 anomalous corners north of 200 km).

## Mode: kh — Kelvin–Helmholtz instability (kh.rs)
Not equatorial: a vertical x–z slice. Units: half shear-layer thickness, half velocity jump.

    ω_t + u·∇ω = -J θ_x + ∇²ω/Re      θ_t + u·∇θ = ∇²θ/Re      ∇²ψ = ω, u = ψ_z, w = -ψ_x

- Background U = θ = tanh z (warm on top), so min gradient Ri = J at z = 0; unstable for
  J < 1/4. Periodic in x, free-slip lids. Domain 56 x 28 (four fastest-growing billows),
  256x128, dt 0.1, J 0.08, Re 1500, Pr 1. Fills the screen (no true-proportions option).
- Cell-centred; face velocities from corner ψ (exactly non-divergent); 5th-order upwind fluxes
  (implicit LES), **θ limited** (universal limiter — without it |θ| overshot to 1.4); SSP-RK3;
  Poisson = own radix-2 FFT in x (nx power of two, nz even) + Thomas in z. 512x256 works
  (dt 0.05) but costs ~10 ms/step before the FFT speed-up.
- FFT twiddles are stored contiguously per stage so the butterflies vectorise (-40 %).
- A free layer mixes once and stays stable, so every `cycle` 150 units ω and θ relax to the
  background (rate 0.2 for 20 units, a model device, HUD "rebuilding"), then new long-wave noise
  (0.02) seeds the next set. Billows at t ≈ 30, pairing ≈ 55, turbulence ≈ 80–130.
- HUD "Ri" = J of the background; "mean-flow Ri" = min of the x-mean profiles (negative while
  overturning). w_rms < 0.05 in |z| < 2 = "laminar".
- 2-D only: real billows break down through 3-D secondary instabilities, so the late turbulence
  here is 2-D (pairing, inverse cascade).
- Tests: Poisson residual; growth rate 0.1897 ± 0.006 at k = 0.4446, J = 0 (Michalke 1964);
  Miles–Howard (J = 0.1 grows, J = 0.3 does not); bounded over two cycles (|θ| < 1.05).

## Python side (src/wavy/)
- `simulate.py` is the **channel-only** version (periodic in x, kick-every 30, damping 0.008,
  200x80, dt 0.04, t 0..400, ~17 s). The basin geometry exists only in Rust (and enso.py has
  its own basin solver). Output: h normalised by the 99.7th percentile, int8, with `scale`,
  `dt_frame`, `Lx`, `Ly` in a compressed .npz (the existing waves.npz has no geometry key).
- `play.py`: the original player (nearest-neighbour resample, pure-Python per-cell loop —
  slow on big terminals). Keys q/space/+/-/e/h; `--snapshot N`.
- `enso.py`: `Config` dataclass (every field is also a CLI flag), `Model.run()` returns time
  series (nino3, h_west, h_east, burst) + equatorial Hovmöllers; `diagnostics()`; `plot()`
  (Hovmöller h, Hovmöller SST, indices). ~4.6 s per model year on 100x40. It has neither the
  SST splitting nor the filter of the Rust port.
- `enso_scan.py OUTDIR JSON_GRID`: cartesian product of parameters, multiprocessing pool,
  saves run_*.npz/png and prints diagnostics.
- pyproject scripts: `wavy-play` → play.main, `wavy-simulate` → simulate.main.

## Testing and verification recipes
- Unit tests: `cargo test --release` (solver: energy, closed basin mass, Kelvin speed, ENSO
  oscillation; wbc: bounded, anomalous PV crosses the equator; kh: see the mode section).
- Physics cross-checks against Python were done via `--snapshot` stderr numbers (energy,
  Niño-3) — reproduce with the values quoted above.
- **Visual checks without a terminal**: render a snapshot or a live run in a pseudo-terminal
  of known size with `script`, convert with `tools/screen2png.py`, look at the PNG:

      script -qfec "stty cols 160 rows 50; ./wavy/target/release/wavy --mode wbc \
          --seed 1 --snapshot 35" /tmp/s.raw
      uv run --with matplotlib python tools/screen2png.py /tmp/s.png 160 50 /tmp/s.raw

- **Live-loop tests**: pipe scripted keys into the pty, always with a hard timeout:

      ( sleep 2; printf '?'; sleep 1; printf 'b'; sleep 2; printf 'q' ) | \
          timeout 30 script -qfec "stty cols 160 rows 50; ./wavy/target/release/wavy \
          --mode wbc --seed 1" /tmp/live.log
      # then parse the log: HUD lines via regex on '\x1b[50;1H\x1b[0m\x1b[2K', or replay it
      # through `pyte` (uv run --with pyte) to get the final screen text; check that
      # '\x1b[?1049l' (alt screen left) is present.

- Pitfalls hit: `pkill -f wavy` from a shell whose command line contains "wavy" kills
  the shell itself — use `pgrep -a wavy` first. Without `timeout`, a test that can't quit
  the program hangs the session (that's how the frame-loop polling bug was found).

## Ideas / next steps
- WBC: show no-slip and free-slip **side by side** in the empty space around the strip (same
  initial state, like the paper's comparison figures; ~2x compute, fits at x1) — offered,
  not yet decided.
- ENSO: seasonal cycle in the coupling (events peak in boreal winter); longer period (3–7 yr)
  without falling into the stationary mode.
- Waves: Gill forcing (steady heat source) instead of random kicks; better-resolved coastal
  Kelvin waves (finer grid or N/S sponges).
- Packaging: Arch PKGBUILD building the Rust binary (`cargo build --release --locked`,
  install `/usr/bin/wavy`); no numpy / .npz needed any more.
- Windows: works via WSL. Native needs only `term.rs` ported (termios/ioctl/poll/signals);
  rendering is plain VT/truecolor, fine in Windows Terminal. Recommended route: `crossterm`
  (vs. hand-written Win32 console code). User said not yet (2026-11-22).
- README.md is empty — write user-facing docs (modes, keys, screenshots via screen2png).
- Python simulate.py could gain the basin geometry to match Rust; play.py render loop is slow.
