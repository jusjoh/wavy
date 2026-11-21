//! Linear shallow-water equations on an equatorial beta-plane (Matsuno units:
//! length = equatorial deformation radius, Kelvin speed = 1):
//!
//! ```text
//!     u_t - y v = -h_x - r u + tau
//!     v_t + y u = -h_y - r v
//!     h_t + u_x + v_y = -r h
//! ```
//!
//! Arakawa C-grid, RK4, solid walls at y = ±Ly (v = 0).
//!
//! Optional ENSO coupling (stripped-down Cane–Zebiak, port of
//! `src/wavy/enso.py`): h is the thermocline depth anomaly, an SST
//! anomaly T relaxes to a subsurface temperature set by h in the eastern
//! upwelling region, and the wind stress is a patch proportional to the Niño-3
//! SST index plus random wind bursts:
//!
//! ```text
//!     T_t = eps (M(x, y) F(h) - T)
//!     tau = mu · Nino3(T) · G(x, y) + bursts(x, y, t)
//! ```
//!
//! T is slow (1/eps ≈ 33 time units vs dt = 0.08), so it is split off from the
//! RK4 step: the ocean step uses the wind from T at the start of the step, then
//! T relaxes exactly towards M·F(h_new). Agrees with the fully coupled RK4 of
//! the Python prototype to ~1e-4 in Niño-3 after 5 years.
//!
//! Layout (row-major, j = y index, i = x index):
//! - `h`, `sst`: (Ny, Nx) cell centres
//! - `u`: x-faces; face `i` sits at x = i·dx, i.e. on the *west* side of cell `i`.
//!   Basin: (Ny, Nx+1) with u = 0 on faces 0 and Nx (walls).
//!   Channel: (Ny, Nx), periodic — the east face of cell Nx-1 is face 0.
//! - `v`: (Ny+1, Nx) y-faces; row `j` sits at y = -Ly + j·dy, rows 0 and Ny are walls.

use crate::rng::Rng;

/// Model time units per year (1 unit ≈ 1.45 days for c ≈ 2.8 m/s).
pub const YEAR: f64 = 252.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Geometry {
    /// Walls at x = 0 and x = Lx (Pacific-like basin).
    Basin,
    /// Periodic in x.
    Channel,
}

impl Geometry {
    pub fn name(self) -> &'static str {
        match self {
            Geometry::Basin => "basin",
            Geometry::Channel => "channel",
        }
    }
}

/// ENSO coupling parameters (defaults = tuned values from the Python prototype).
#[derive(Clone, Debug)]
pub struct Coupling {
    /// Bjerknes coupling: wind stress per unit Niño-3 SST (threshold ≈ 0.43)
    pub mu: f64,
    /// SST relaxation rate
    pub eps: f64,
    /// saturation of F(h) for a deep / shallow thermocline (El Niño > La Niña)
    pub a_warm: f64,
    pub a_cold: f64,
    /// coupled wind patch: centre and half-width as fraction of Lx, y half-width
    pub wind_x: f64,
    pub wind_sx: f64,
    pub wind_sy: f64,
    /// random wind bursts per time unit, stress amplitude, duration range
    pub burst_rate: f64,
    pub burst_amp: f64,
    pub burst_dur: (f64, f64),
}

impl Default for Coupling {
    fn default() -> Self {
        Coupling {
            mu: 0.42,
            eps: 0.03,
            a_warm: 1.0,
            a_cold: 0.6,
            wind_x: 0.3,
            wind_sx: 0.12,
            wind_sy: 1.0,
            burst_rate: 1.0 / 40.0,
            burst_amp: 0.05,
            burst_dur: (8.0, 20.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Params {
    pub geometry: Geometry,
    pub lx: f64,
    pub ly: f64,
    pub nx: usize,
    pub ny: usize,
    pub dt: f64,
    pub damping: f64,
    /// Biharmonic filter strength per step along x (0 = off): f -= filter·δx⁴ f,
    /// with wall-aware symmetric operators. Removes grid-scale noise (short
    /// Rossby waves from the western wall build up over decades in ENSO mode);
    /// a 2Δx wave loses 16·filter per step, ENSO-scale structure essentially
    /// nothing. Stable for filter < 1/8.
    pub filter: f64,
    pub coupling: Option<Coupling>,
}

/// Precomputed coupling fields.
struct CouplingGrid {
    mu: f64,
    eps: f64,
    a_warm: f64,
    a_cold: f64,
    /// upwelling mask on h points
    m: Vec<f64>,
    /// coupled wind pattern on u points
    g: Vec<f64>,
    /// h-point indices of the Niño-3 box
    nino3: Vec<usize>,
}

impl CouplingGrid {
    /// Subsurface temperature from thermocline depth (asymmetric tanh).
    #[inline]
    fn f(&self, h: f64) -> f64 {
        // a·tanh(h/a) via a single exp (libm tanh is ~3x slower and this is
        // evaluated for every cell in every RK stage): tanh|x| = (1-e)/(1+e), e = exp(-2|x|)
        let a = if h > 0.0 { self.a_warm } else { self.a_cold };
        let e = (-2.0 * h.abs() / a).exp();
        (a * (1.0 - e) / (1.0 + e)).copysign(h)
    }

    fn nino3(&self, t: &[f64]) -> f64 {
        self.nino3.iter().map(|&k| t[k]).sum::<f64>() / self.nino3.len().max(1) as f64
    }
}

/// Immutable grid description shared by the tendency computation.
struct Grid {
    basin: bool,
    nx: usize,
    ny: usize,
    /// number of u-faces per row
    nux: usize,
    dx: f64,
    dy: f64,
    r: f64,
    yc: Vec<f64>,
    yv: Vec<f64>,
    cpl: Option<CouplingGrid>,
}

#[derive(Clone)]
pub struct Fields {
    pub h: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

impl Fields {
    fn zeros(g: &Grid) -> Self {
        Fields {
            h: vec![0.0; g.ny * g.nx],
            u: vec![0.0; g.ny * g.nux],
            v: vec![0.0; (g.ny + 1) * g.nx],
        }
    }

    fn fill_zero(&mut self) {
        self.h.fill(0.0);
        self.u.fill(0.0);
        self.v.fill(0.0);
    }

    /// self = x + a * y
    fn set_axpy(&mut self, x: &Fields, a: f64, y: &Fields) {
        for (d, (xs, ys)) in [
            (&mut self.h, (&x.h, &y.h)),
            (&mut self.u, (&x.u, &y.u)),
            (&mut self.v, (&x.v, &y.v)),
        ] {
            for ((d, &xv), &yv) in d.iter_mut().zip(xs.iter()).zip(ys.iter()) {
                *d = xv + a * yv;
            }
        }
    }
}

/// Tendencies of state `s`, written to `out`. The zonal wind stress is
/// `wind · G + tau_ext` (coupled patch with fixed amplitude `wind` = mu·Niño-3,
/// plus wind bursts); both are only used with coupling.
///
/// Written as row-slice loops over the interior plus explicit edge cells, so the
/// compiler can drop bounds checks and vectorise; the only difference between
/// basin and channel is what lies east of the last cell / west of face 0.
fn tendency(g: &Grid, s: &Fields, wind: f64, tau_ext: &[f64], out: &mut Fields) {
    let (nx, ny, nux, basin) = (g.nx, g.ny, g.nux, g.basin);
    let (idx, idy, r) = (1.0 / g.dx, 1.0 / g.dy, g.r);
    let east_of_last = if basin { nx } else { 0 }; // u-face east of cell nx-1
    let hrow = |j: usize| &s.h[j * nx..(j + 1) * nx];
    let urow = |j: usize| &s.u[j * nux..(j + 1) * nux];
    let vrow = |j: usize| &s.v[j * nx..(j + 1) * nx];

    // continuity: h_t = -(u_x + v_y)
    for j in 0..ny {
        let (hr, ur, vs, vn) = (hrow(j), urow(j), vrow(j), vrow(j + 1));
        let o = &mut out.h[j * nx..(j + 1) * nx];
        for i in 0..nx - 1 {
            o[i] = -((ur[i + 1] - ur[i]) * idx + (vn[i] - vs[i]) * idy) - r * hr[i];
        }
        let i = nx - 1;
        o[i] = -((ur[east_of_last] - ur[i]) * idx + (vn[i] - vs[i]) * idy) - r * hr[i];
    }

    // x-momentum on x-faces: u_t = y v - h_x  (face f lies between cells f-1 and f)
    for j in 0..ny {
        let y = g.yc[j];
        let (hr, ur, vs, vn) = (hrow(j), urow(j), vrow(j), vrow(j + 1));
        let o = &mut out.u[j * nux..(j + 1) * nux];
        for f in 1..nx {
            let v_at_u = 0.25 * (vs[f - 1] + vn[f - 1] + vs[f] + vn[f]);
            o[f] = y * v_at_u - (hr[f] - hr[f - 1]) * idx - r * ur[f];
        }
        if basin {
            o[0] = 0.0; // west wall
            o[nx] = 0.0; // east wall
        } else {
            let w = nx - 1; // periodic: face 0 lies between cell nx-1 and cell 0
            let v_at_u = 0.25 * (vs[w] + vn[w] + vs[0] + vn[0]);
            o[0] = y * v_at_u - (hr[0] - hr[w]) * idx - r * ur[0];
        }
    }

    // y-momentum on y-faces: v_t = -y u - h_y  (rows 0 and ny are walls, v = 0)
    out.v[..nx].fill(0.0);
    out.v[ny * nx..].fill(0.0);
    for j in 1..ny {
        let y = g.yv[j];
        let (hs, hn, us, un, vr) = (hrow(j - 1), hrow(j), urow(j - 1), urow(j), vrow(j));
        let o = &mut out.v[j * nx..(j + 1) * nx];
        for i in 0..nx - 1 {
            let u_at_v = 0.25 * (us[i] + us[i + 1] + un[i] + un[i + 1]);
            o[i] = -y * u_at_v - (hn[i] - hs[i]) * idy - r * vr[i];
        }
        let (i, e) = (nx - 1, east_of_last);
        let u_at_v = 0.25 * (us[i] + us[e] + un[i] + un[e]);
        o[i] = -y * u_at_v - (hn[i] - hs[i]) * idy - r * vr[i];
    }

    // wind stress (coupled patch + bursts), interior faces only — the basin
    // walls must keep u = 0, and coupling is basin-only
    if let Some(c) = &g.cpl {
        for j in 0..ny {
            let row = j * nux;
            let o = &mut out.u[row + 1..row + nx];
            let gp = &c.g[row + 1..row + nx];
            for (o, &gv) in o.iter_mut().zip(gp) {
                *o += wind * gv;
            }
        }
    }
    if !tau_ext.is_empty() {
        for j in 0..ny {
            let row = j * nux;
            for (o, &tv) in out.u[row + 1..row + nx]
                .iter_mut()
                .zip(&tau_ext[row + 1..row + nx])
            {
                *o += tv;
            }
        }
    }
}

/// Boundary condition for one direction of the biharmonic filter.
#[derive(Clone, Copy)]
enum Bc {
    /// mirror (zero gradient): conserves the sum — h, and u/v along walls
    Neumann,
    /// first and last points are walls held at zero (u in x for the basin, v in y)
    Dirichlet,
    Periodic,
}

/// Second difference of `f` into `out` with boundary condition `bc`.
fn lap1d(f: &[f64], out: &mut [f64], bc: Bc) {
    let n = f.len();
    for i in 1..n - 1 {
        out[i] = f[i - 1] - 2.0 * f[i] + f[i + 1];
    }
    match bc {
        Bc::Neumann => {
            out[0] = f[1] - f[0];
            out[n - 1] = f[n - 2] - f[n - 1];
        }
        Bc::Dirichlet => {
            out[0] = 0.0;
            out[n - 1] = 0.0;
        }
        Bc::Periodic => {
            out[0] = f[n - 1] - 2.0 * f[0] + f[1];
            out[n - 1] = f[n - 2] - 2.0 * f[n - 1] + f[0];
        }
    }
}

/// f -= a · L(L(f)) along one line. L is the symmetric second-difference
/// matrix for `bc`, so L² is symmetric positive semi-definite: the filter only
/// ever removes variance (eigenvalues of L² ≤ 16, stable for a < 1/8) and,
/// with Neumann/periodic ends, conserves the sum exactly.
fn biharm1d(f: &mut [f64], a: f64, bc: Bc, l1: &mut Vec<f64>, l2: &mut Vec<f64>) {
    let n = f.len();
    if n < 3 {
        return;
    }
    l1.resize(n, 0.0);
    l2.resize(n, 0.0);
    lap1d(f, l1, bc);
    lap1d(l1, l2, bc);
    for (v, &d) in f.iter_mut().zip(l2.iter()) {
        *v -= a * d;
    }
}

/// Biharmonic filter along x on a (rows × cols) row-major field.
///
/// x only on purpose: the grid noise (short Rossby waves from the western
/// wall) is in x, while a y-filter would also damp the equatorial Kelvin wave,
/// which is only ~2.5 cells wide in y, and noticeably shifts the ENSO cycle.
fn hyperfilter_x(
    f: &mut [f64],
    rows: usize,
    cols: usize,
    a: f64,
    bc: Bc,
    l1: &mut Vec<f64>,
    l2: &mut Vec<f64>,
) {
    for r in 0..rows {
        biharm1d(&mut f[r * cols..(r + 1) * cols], a, bc, l1, l2);
    }
}

pub struct Model {
    pub params: Params,
    g: Grid,
    pub s: Fields,
    /// SST anomaly on h points (empty without coupling)
    pub sst: Vec<f64>,
    k: [Fields; 4],
    tmp: Fields,
    /// external zonal wind stress on u points (sum of active bursts; empty if uncoupled)
    tau_ext: Vec<f64>,
    /// scratch lines for the filter
    l1: Vec<f64>,
    l2: Vec<f64>,
    /// model time
    pub t: f64,
}

impl Model {
    pub fn new(params: Params) -> Self {
        let (nx, ny) = (params.nx, params.ny);
        let basin = params.geometry == Geometry::Basin;
        let dx = params.lx / nx as f64;
        let dy = 2.0 * params.ly / ny as f64;
        let nux = if basin { nx + 1 } else { nx };
        let yc: Vec<f64> = (0..ny)
            .map(|j| -params.ly + (j as f64 + 0.5) * dy)
            .collect();
        let yv: Vec<f64> = (0..=ny).map(|j| -params.ly + j as f64 * dy).collect();

        let cpl = params.coupling.as_ref().map(|c| {
            let lx = params.lx;
            let mut m = vec![0.0; ny * nx];
            let mut nino3 = Vec::new();
            for j in 0..ny {
                for i in 0..nx {
                    let xf = (i as f64 + 0.5) * dx / lx;
                    let y = yc[j];
                    // equatorial cold tongue in the east
                    m[j * nx + i] =
                        (-y * y / (2.0 * 1.5 * 1.5)).exp() / (1.0 + (-(xf - 0.45) / 0.06).exp());
                    // Niño-3 box: 150W–90W, 5S–5N
                    if xf > 0.56 && xf < 0.94 && y.abs() < 1.6 {
                        nino3.push(j * nx + i);
                    }
                }
            }
            let mut g = vec![0.0; ny * nux];
            for j in 0..ny {
                for f in 0..nux {
                    let xf = f as f64 * dx / lx;
                    let ax = (xf - c.wind_x) / c.wind_sx;
                    let ay = yc[j] / c.wind_sy;
                    g[j * nux + f] = (-0.5 * ax * ax - 0.5 * ay * ay).exp();
                }
            }
            CouplingGrid {
                mu: c.mu,
                eps: c.eps,
                a_warm: c.a_warm,
                a_cold: c.a_cold,
                m,
                g,
                nino3,
            }
        });
        let coupled = cpl.is_some();

        let g = Grid {
            basin,
            nx,
            ny,
            nux,
            dx,
            dy,
            r: params.damping,
            yc,
            yv,
            cpl,
        };
        let s = Fields::zeros(&g);
        let k = [s.clone(), s.clone(), s.clone(), s.clone()];
        let tmp = s.clone();
        let tau_ext = vec![0.0; if coupled { ny * nux } else { 0 }];
        let sst = vec![0.0; if coupled { ny * nx } else { 0 }];
        let (l1, l2) = (Vec::with_capacity(nx + 1), Vec::with_capacity(nx + 1));
        Model {
            params,
            g,
            s,
            sst,
            k,
            tmp,
            tau_ext,
            l1,
            l2,
            t: 0.0,
        }
    }

    pub fn nx(&self) -> usize {
        self.g.nx
    }

    pub fn ny(&self) -> usize {
        self.g.ny
    }

    pub fn coupled(&self) -> bool {
        self.g.cpl.is_some()
    }

    /// Niño-3 SST index (0 without coupling).
    pub fn nino3(&self) -> f64 {
        self.g.cpl.as_ref().map_or(0.0, |c| c.nino3(&self.sst))
    }

    pub fn reset(&mut self) {
        self.s.fill_zero();
        self.sst.fill(0.0);
        self.tau_ext.fill(0.0);
        self.t = 0.0;
    }

    /// One step of length dt: RK4 for the ocean (wind held fixed over the
    /// step), then the SST relaxation.
    pub fn step(&mut self) {
        let dt = self.params.dt;
        let wind = self
            .g
            .cpl
            .as_ref()
            .map_or(0.0, |c| c.mu * c.nino3(&self.sst));
        let Model {
            g,
            s,
            k,
            tmp,
            tau_ext,
            sst,
            ..
        } = self;
        let [k1, k2, k3, k4] = k;
        tendency(g, s, wind, tau_ext, k1);
        tmp.set_axpy(s, 0.5 * dt, k1);
        tendency(g, tmp, wind, tau_ext, k2);
        tmp.set_axpy(s, 0.5 * dt, k2);
        tendency(g, tmp, wind, tau_ext, k3);
        tmp.set_axpy(s, dt, k3);
        tendency(g, tmp, wind, tau_ext, k4);
        let c = dt / 6.0;
        for (field, (a, b, cc, d)) in [
            (&mut s.h, (&k1.h, &k2.h, &k3.h, &k4.h)),
            (&mut s.u, (&k1.u, &k2.u, &k3.u, &k4.u)),
            (&mut s.v, (&k1.v, &k2.v, &k3.v, &k4.v)),
        ] {
            for i in 0..field.len() {
                field[i] += c * (a[i] + 2.0 * b[i] + 2.0 * cc[i] + d[i]);
            }
        }
        if self.params.filter > 0.0 {
            let (nx, ny, nux, a) = (g.nx, g.ny, g.nux, self.params.filter);
            let (l1, l2) = (&mut self.l1, &mut self.l2);
            let (bc_c, bc_u) = if g.basin {
                (Bc::Neumann, Bc::Dirichlet) // u = 0 on the east/west walls
            } else {
                (Bc::Periodic, Bc::Periodic)
            };
            hyperfilter_x(&mut s.h, ny, nx, a, bc_c, l1, l2);
            hyperfilter_x(&mut s.u, ny, nux, a, bc_u, l1, l2);
            hyperfilter_x(&mut s.v, ny + 1, nx, a, bc_c, l1, l2);
        }
        // SST: exact relaxation over dt towards M·F(h) with the new h
        if let Some(cp) = &g.cpl {
            let w = 1.0 - (-cp.eps * dt).exp();
            for ((t, &m), &h) in sst.iter_mut().zip(&cp.m).zip(&s.h) {
                *t += w * (m * cp.f(h) - *t);
            }
        }
        self.t += dt;
    }

    /// Add a Gaussian height bump (periodic distance in x for the channel).
    pub fn add_bump(&mut self, x0: f64, y0: f64, sigma: f64, amp: f64) {
        let (nx, lx, dx) = (self.g.nx, self.params.lx, self.g.dx);
        let periodic = !self.g.basin;
        let inv = 1.0 / (2.0 * sigma * sigma);
        for j in 0..self.g.ny {
            let dy2 = (self.g.yc[j] - y0).powi(2);
            for i in 0..nx {
                let mut d = (i as f64 + 0.5) * dx - x0;
                if periodic {
                    d = (d + lx / 2.0).rem_euclid(lx) - lx / 2.0;
                }
                self.s.h[j * nx + i] += amp * (-(d * d + dy2) * inv).exp();
            }
        }
    }

    /// The initial bump: unit Gaussian on the equator at x = 0.3 Lx.
    pub fn initial_bump(&mut self) {
        self.add_bump(0.3 * self.params.lx, 0.0, 1.0, 1.0);
    }

    /// A random bump near the equator; in the basin only within the middle 70 %.
    pub fn kick(&mut self, rng: &mut Rng) {
        let lx = self.params.lx;
        let x0 = match self.params.geometry {
            Geometry::Basin => rng.uniform(0.15 * lx, 0.85 * lx),
            Geometry::Channel => rng.uniform(0.0, lx),
        };
        let y0 = 0.7 * rng.normal();
        let sigma = rng.uniform(0.7, 1.5);
        let amp = rng.sign() * rng.uniform(0.6, 1.2);
        self.add_bump(x0, y0, sigma, amp);
    }

    /// Add `amp · exp(-((x-x0)/2.5)²/2 - (y/2)²/2)` to the external wind stress.
    fn add_burst_field(&mut self, x0: f64, amp: f64) {
        if self.tau_ext.is_empty() {
            return;
        }
        let (nux, dx) = (self.g.nux, self.g.dx);
        for j in 0..self.g.ny {
            let ay = self.g.yc[j] / 2.0;
            for f in 0..nux {
                let ax = (f as f64 * dx - x0) / 2.5;
                self.tau_ext[j * nux + f] += amp * (-0.5 * ax * ax - 0.5 * ay * ay).exp();
            }
        }
    }

    /// Total energy ½∫(h² + u² + v²) — conserved without damping (for testing).
    pub fn energy(&self) -> f64 {
        let sq = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
        0.5 * (sq(&self.s.h) + sq(&self.s.u) + sq(&self.s.v)) * self.g.dx * self.g.dy
    }
}

/// An active wind burst.
struct Burst {
    x0: f64,
    amp: f64,
    t_end: f64,
}

/// Drives the model in time: periodic h-kicks (wave mode) or random wind
/// bursts (ENSO mode).
pub struct Sim {
    pub model: Model,
    pub rng: Rng,
    /// steps between kicks (0 = never)
    kick_n: u64,
    n: u64,
    init_h: f64,
    bursts: Vec<Burst>,
}

impl Sim {
    /// `init_h`: amplitude of the initial thermocline bump in ENSO mode.
    pub fn new(params: Params, kick_every: f64, seed: u64, init_h: f64) -> Self {
        let kick_n = if kick_every > 0.0 && params.coupling.is_none() {
            (kick_every / params.dt).round().max(1.0) as u64
        } else {
            0
        };
        let mut sim = Sim {
            model: Model::new(params),
            rng: Rng::new(seed),
            kick_n,
            n: 0,
            init_h,
            bursts: Vec::new(),
        };
        sim.initial_state();
        sim
    }

    fn initial_state(&mut self) {
        if self.model.coupled() {
            // same as the Python prototype: Gaussian on the equator at 0.5 Lx
            self.model
                .add_bump(0.5 * self.model.params.lx, 0.0, 1.0, self.init_h);
        } else {
            self.model.initial_bump();
        }
    }

    pub fn step(&mut self) {
        if self.kick_n > 0 && self.n > 0 && self.n.is_multiple_of(self.kick_n) {
            self.model.kick(&mut self.rng);
        }
        if let Some(c) = self.model.params.coupling.clone() {
            let dt = self.model.params.dt;
            if c.burst_amp > 0.0 && self.rng.f64() < c.burst_rate * dt {
                let x0 = self.rng.uniform(0.15, 0.5) * self.model.params.lx;
                let dur = self.rng.uniform(c.burst_dur.0, c.burst_dur.1);
                let amp = self.rng.sign() * c.burst_amp;
                self.start_burst(x0, amp, dur);
            }
            self.expire_bursts();
        }
        self.model.step();
        self.n += 1;
    }

    fn start_burst(&mut self, x0: f64, amp: f64, dur: f64) {
        self.model.add_burst_field(x0, amp);
        self.bursts.push(Burst {
            x0,
            amp,
            t_end: self.model.t + dur,
        });
    }

    fn expire_bursts(&mut self) {
        let t = self.model.t;
        let expired: Vec<(f64, f64)> = self
            .bursts
            .iter()
            .filter(|b| b.t_end <= t)
            .map(|b| (b.x0, b.amp))
            .collect();
        if !expired.is_empty() {
            self.bursts.retain(|b| b.t_end > t);
            // rebuild from scratch to avoid accumulating round-off
            self.model.tau_ext.fill(0.0);
            let active: Vec<(f64, f64)> = self.bursts.iter().map(|b| (b.x0, b.amp)).collect();
            for (x0, amp) in active {
                self.model.add_burst_field(x0, amp);
            }
        }
    }

    /// Wave mode: drop in a random bump. ENSO mode: a strong westerly wind
    /// burst in the western basin (tends to trigger an El Niño).
    pub fn kick(&mut self) {
        match self.model.params.coupling.clone() {
            None => self.model.kick(&mut self.rng),
            Some(c) => {
                let x0 = self.rng.uniform(0.15, 0.35) * self.model.params.lx;
                self.start_burst(x0, 4.0 * c.burst_amp.max(0.05), 30.0);
            }
        }
    }

    /// Is a wind burst currently blowing? (sign of the strongest one)
    pub fn burst_active(&self) -> Option<f64> {
        self.bursts
            .iter()
            .max_by(|a, b| a.amp.abs().total_cmp(&b.amp.abs()))
            .map(|b| b.amp.signum())
    }

    pub fn restart(&mut self) {
        self.model.reset();
        self.bursts.clear();
        self.initial_state();
        self.n = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(geometry: Geometry, damping: f64) -> Params {
        Params {
            geometry,
            lx: 40.0,
            ly: 8.0,
            nx: 200,
            ny: 80,
            dt: 0.04,
            damping,
            filter: 0.0,
            coupling: None,
        }
    }

    #[test]
    fn energy_conserved_without_damping() {
        for geom in [Geometry::Basin, Geometry::Channel] {
            let mut m = Model::new(params(geom, 0.0));
            m.initial_bump();
            let e0 = m.energy();
            for _ in 0..1000 {
                m.step();
            }
            // The 4-point Coriolis averaging on the C-grid is not exactly
            // energy-neutral (the Python solver drifts by the same +0.21 %).
            let rel = (m.energy() - e0).abs() / e0;
            assert!(rel < 5e-3, "{geom:?}: relative energy drift {rel}");
        }
    }

    #[test]
    fn basin_walls_stay_closed() {
        let mut m = Model::new(params(Geometry::Basin, 0.0));
        m.initial_bump();
        let mass0: f64 = m.s.h.iter().sum();
        for _ in 0..1000 {
            m.step();
        }
        let mass: f64 = m.s.h.iter().sum();
        assert!(
            (mass - mass0).abs() < 1e-9 * mass0.abs().max(1.0),
            "{mass0} -> {mass}"
        );
    }

    /// The equatorial Kelvin wave travels east at speed 1: after t = 10 the
    /// equatorial h-maximum east of the start should sit ~10 units further east.
    #[test]
    fn kelvin_wave_speed() {
        let mut m = Model::new(params(Geometry::Channel, 0.0));
        m.initial_bump();
        for _ in 0..250 {
            m.step();
        } // t = 10
        let (nx, j_eq) = (m.nx(), m.ny() / 2);
        let dx = 40.0 / nx as f64;
        let row = &m.s.h[j_eq * nx..(j_eq + 1) * nx];
        // search east of the start point (x > 12 + 5) to skip the Rossby part
        let i0 = ((12.0 + 5.0) / dx) as usize;
        let (imax, _) = row[i0..]
            .iter()
            .enumerate()
            .fold((0, f64::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        let x = (i0 + imax) as f64 * dx + 0.5 * dx;
        assert!(
            (x - 22.0).abs() < 0.6,
            "Kelvin peak at x = {x}, expected ~22"
        );
    }

    fn enso_params(mu: f64) -> Params {
        Params {
            geometry: Geometry::Basin,
            lx: 40.0,
            ly: 8.0,
            nx: 100,
            ny: 40,
            dt: 0.08,
            damping: 0.004,
            filter: 0.0,
            coupling: Some(Coupling {
                mu,
                burst_amp: 0.0,
                ..Coupling::default()
            }),
        }
    }

    /// Above the coupling threshold the ENSO mode oscillates (sign changes of
    /// Niño-3 with a period of a few years) instead of locking into a permanent
    /// El Niño or decaying.
    #[test]
    fn enso_oscillates_above_threshold() {
        let mut sim = Sim::new(enso_params(0.5), 0.0, 1, 0.1);
        let steps_per_month = (YEAR / 12.0 / 0.08).round() as usize;
        let mut series = Vec::new();
        for _ in 0..(12 * 15) {
            for _ in 0..steps_per_month {
                sim.step();
            }
            series.push(sim.model.nino3());
        }
        let late = &series[12 * 5..]; // skip 5 years spin-up
        let crossings = late
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        let period = 2.0 * 10.0 / crossings.max(1) as f64;
        let rms = (late.iter().map(|x| x * x).sum::<f64>() / late.len() as f64).sqrt();
        assert!((2.0..5.0).contains(&period), "period {period} yr");
        assert!(rms > 0.05, "rms {rms}");
    }
}
