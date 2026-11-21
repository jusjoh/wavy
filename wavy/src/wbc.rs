//! Cross-equatorial western boundary current (after Goldsworth, Marshall &
//! Johnson 2021, JPO 51, "Symmetric Instability in Cross-Equatorial Western
//! Boundary Currents", and Edwards & Pedlosky 1998).
//!
//! Nonlinear 1.5-layer (reduced-gravity) shallow water on an equatorial
//! β-plane, in SI units, vector-invariant form:
//!
//! ```text
//!     u_t - q V = -B_x + ν (D_x - ζ_y) - r (u - 0)
//!     v_t + q U = -B_y + ν (D_y + ζ_x) - r (v - V_jet)
//!     h_t + U_x + V_y = -r (h - h_ref)
//!     q = (f + ζ)/h,  f = βy,  B = g'h + |u|²/2,  (U, V) = h (u, v),  D = u_x + v_y
//! ```
//!
//! A Bickley jet `V_jet(x) = V0 sech²((x - x_mid)/δ)` flows north along the
//! western wall: its transport is prescribed at the southern and northern
//! open boundaries and sponge layers relax towards it there (as in the
//! paper's 3-D model). Water entering from the south carries negative PV
//! (f < 0) into the Northern Hemisphere, where f·q < 0 — the criterion for
//! symmetric/inertial instability. The jet sheds anticyclonic eddies as it
//! crosses the equator (barotropic instability).
//!
//! The western wall is either **no-slip** (ζ = 2 v/Δx on the wall: the wall
//! injects positive vorticity through the viscous stress) or **free-slip**
//! (ζ = 0 on the wall). The east wall is always free-slip.
//!
//! What a single-layer model cannot show is the symmetric instability itself:
//! its stacked overturning cells live in the x–z plane. Instead the regions
//! where f·q < 0 can be highlighted.
//!
//! Arakawa C-grid, enstrophy-conserving (Sadourny) Coriolis/advection, RK4.
//! Layout (row-major, j = y index, i = x index):
//! - `h`: (Ny, Nx) centres
//! - `u`: (Ny, Nx+1) x-faces, face i at x = i·Δx, u = 0 on both walls
//! - `v`: (Ny+1, Nx) y-faces, row j at y = y_s + j·Δy, rows 0 and Ny prescribed
//! - `ζ`, `q`: (Ny+1, Nx+1) corners

use crate::rng::Rng;

pub const DAY: f64 = 86400.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Slip {
    NoSlip,
    FreeSlip,
}

impl Slip {
    pub fn name(self) -> &'static str {
        match self {
            Slip::NoSlip => "no-slip",
            Slip::FreeSlip => "free-slip",
        }
    }

    pub fn other(self) -> Slip {
        match self {
            Slip::NoSlip => Slip::FreeSlip,
            Slip::FreeSlip => Slip::NoSlip,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WbcParams {
    /// zonal extent (m), western wall at x = 0
    pub lx: f64,
    /// southern / northern boundary (m, equator at 0)
    pub y_south: f64,
    pub y_north: f64,
    pub nx: usize,
    pub ny: usize,
    /// time step (s)
    pub dt: f64,
    /// df/dy (1/(m s))
    pub beta: f64,
    /// reduced gravity (m/s²) and mean layer depth (m)
    pub g_red: f64,
    pub h0: f64,
    /// Bickley jet: peak speed (m/s), distance of the peak from the wall, width (m)
    pub v0: f64,
    pub x_mid: f64,
    pub delta: f64,
    /// Laplacian viscosity (m²/s)
    pub nu: f64,
    pub slip: Slip,
    /// sponge widths at the southern / northern boundary (m) and max rate (1/s)
    pub sponge_south: f64,
    pub sponge_north: f64,
    pub sponge_rate: f64,
    /// amplitude of the random initial h perturbation (m) that seeds instabilities
    pub noise: f64,
}

impl Default for WbcParams {
    fn default() -> Self {
        WbcParams {
            lx: 500e3,
            y_south: -400e3,
            y_north: 1600e3,
            nx: 80,
            ny: 200,
            dt: 1200.0,
            beta: 2.3e-11,
            g_red: 0.02,
            h0: 300.0,
            v0: 0.9,
            x_mid: 40e3,
            delta: 30e3,
            nu: 100.0,
            slip: Slip::NoSlip,
            sponge_south: 100e3,
            sponge_north: 300e3,
            sponge_rate: 2e-5,
            noise: 0.05,
        }
    }
}

/// Storage and arithmetic type of the fields: single precision is plenty
/// (h ≈ 300 m, per-step changes ≈ 1 cm) and doubles the SIMD width.
type R = f32;

/// f64 → R for whole vectors.
fn to_r(v: &[f64]) -> Vec<R> {
    v.iter().map(|&x| x as R).collect()
}

#[derive(Clone)]
struct Fields {
    h: Vec<R>,
    u: Vec<R>,
    v: Vec<R>,
}

impl Fields {
    fn zeros(nx: usize, ny: usize) -> Self {
        Fields {
            h: vec![0.0; ny * nx],
            u: vec![0.0; ny * (nx + 1)],
            v: vec![0.0; (ny + 1) * nx],
        }
    }

    /// self = x + a * y
    fn set_axpy(&mut self, x: &Fields, a: R, y: &Fields) {
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

/// Diagnostic work arrays (recomputed every tendency evaluation).
struct Work {
    /// mass fluxes h·u, h·v
    uf: Vec<R>,
    vf: Vec<R>,
    /// relative vorticity and PV at corners
    zeta: Vec<R>,
    q: Vec<R>,
    /// Bernoulli function and divergence at centres
    b: Vec<R>,
    d: Vec<R>,
    /// h averaged onto u columns (for h at corners)
    hx: Vec<R>,
}

/// Immutable grid and forcing data (hot-loop constants pre-converted to R).
struct Grid {
    p: WbcParams,
    dy: f64,
    /// y of v rows / corners, and f there
    yv: Vec<f64>,
    fv: Vec<R>,
    /// jet profile on v points, reference h, sponge rates on h/u rows and v rows
    vjet: Vec<R>,
    href: Vec<R>,
    rc: Vec<R>,
    rv: Vec<R>,
    /// 1/Δx, 1/Δy, ν, H, g'
    idx: R,
    idy: R,
    nu: R,
    h0: R,
    g_red: R,
}

pub struct Wbc {
    g: Grid,
    s: Fields,
    k: [Fields; 4],
    tmp: Fields,
    w: Work,
    /// model time (s)
    pub t: f64,
    seed: u64,
    rng: Rng,
}

impl Wbc {
    pub fn new(p: WbcParams, seed: u64) -> Self {
        let (nx, ny) = (p.nx, p.ny);
        let dx = p.lx / nx as f64;
        let dy = (p.y_north - p.y_south) / ny as f64;
        let yc: Vec<f64> = (0..ny).map(|j| p.y_south + (j as f64 + 0.5) * dy).collect();
        let yv: Vec<f64> = (0..=ny).map(|j| p.y_south + j as f64 * dy).collect();
        let jet = |x: f64| {
            let s = 1.0 / ((x - p.x_mid) / p.delta).cosh();
            p.v0 * s * s
        };
        let vjet: Vec<f64> = (0..nx).map(|i| jet((i as f64 + 0.5) * dx)).collect();
        // geostrophic reference: g' h_x = f V, flat (h0) far east of the jet
        let mut href = vec![0.0; ny * nx];
        let t_east = |x: f64| {
            p.v0 * p.delta
                * (((p.lx - p.x_mid) / p.delta).tanh() - ((x - p.x_mid) / p.delta).tanh())
        };
        for j in 0..ny {
            let f = p.beta * yc[j];
            for i in 0..nx {
                href[j * nx + i] = p.h0 - f / p.g_red * t_east((i as f64 + 0.5) * dx);
            }
        }
        let sponge = |y: f64| {
            let s = (p.y_south + p.sponge_south - y) / p.sponge_south;
            let n = (y - (p.y_north - p.sponge_north)) / p.sponge_north;
            p.sponge_rate * s.clamp(0.0, 1.0).powi(2).max(n.clamp(0.0, 1.0).powi(2))
        };
        let rc: Vec<f64> = yc.iter().map(|&y| sponge(y)).collect();
        let rv: Vec<f64> = yv.iter().map(|&y| sponge(y)).collect();
        let fv: Vec<f64> = yv.iter().map(|&y| p.beta * y).collect();

        let s = Fields::zeros(nx, ny);
        let k = [s.clone(), s.clone(), s.clone(), s.clone()];
        let tmp = s.clone();
        let w = Work {
            uf: vec![0.0; ny * (nx + 1)],
            vf: vec![0.0; (ny + 1) * nx],
            zeta: vec![0.0; (ny + 1) * (nx + 1)],
            q: vec![0.0; (ny + 1) * (nx + 1)],
            b: vec![0.0; ny * nx],
            d: vec![0.0; ny * nx],
            hx: vec![0.0; ny * (nx + 1)],
        };
        let g = Grid {
            idx: (1.0 / dx) as R,
            idy: (1.0 / dy) as R,
            nu: p.nu as R,
            h0: p.h0 as R,
            g_red: p.g_red as R,
            p,
            dy,
            yv,
            fv: to_r(&fv),
            vjet: to_r(&vjet),
            href: to_r(&href),
            rc: to_r(&rc),
            rv: to_r(&rv),
        };
        let mut m = Wbc {
            g,
            s,
            k,
            tmp,
            w,
            t: 0.0,
            seed,
            rng: Rng::new(seed),
        };
        m.restart();
        m
    }

    /// Back to the initial state: the jet everywhere along the wall, h in
    /// geostrophic balance, plus a little noise to seed the instabilities.
    pub fn restart(&mut self) {
        let (nx, ny) = (self.g.p.nx, self.g.p.ny);
        self.s.u.fill(0.0);
        for j in 0..=ny {
            self.s.v[j * nx..(j + 1) * nx].copy_from_slice(&self.g.vjet);
        }
        self.s.h.copy_from_slice(&self.g.href);
        // same seed every restart: both boundary conditions start identically
        self.rng = Rng::new(self.seed);
        for h in self.s.h.iter_mut() {
            *h += (self.g.p.noise * (2.0 * self.rng.f64() - 1.0)) as R;
        }
        self.t = 0.0;
    }

    pub fn set_slip(&mut self, slip: Slip) {
        self.g.p.slip = slip;
    }

    /// Random small h perturbation (key `k`).
    pub fn perturb(&mut self, amp: f64) {
        for h in self.s.h.iter_mut() {
            *h += (amp * (2.0 * self.rng.f64() - 1.0)) as R;
        }
    }

    pub fn step(&mut self) {
        let dt = self.g.p.dt as R;
        let Wbc {
            g, s, k, tmp, w, ..
        } = self;
        let [k1, k2, k3, k4] = k;
        tendency(g, w, s, k1);
        tmp.set_axpy(s, 0.5 * dt, k1);
        tendency(g, w, tmp, k2);
        tmp.set_axpy(s, 0.5 * dt, k2);
        tendency(g, w, tmp, k3);
        tmp.set_axpy(s, dt, k3);
        tendency(g, w, tmp, k4);
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
        self.t += self.g.p.dt;
    }
}

/// Compute ζ, q, B, D and the mass fluxes of state `s` into `w`.
///
/// Row-slice loops with the wall/boundary cases outside the inner loops, so
/// the compiler can drop bounds checks and vectorise.
fn diagnose(g: &Grid, w: &mut Work, s: &Fields) {
    let (nx, ny) = (g.p.nx, g.p.ny);
    let nux = nx + 1;
    let (idx, idy) = (g.idx, g.idy);
    let hrow = |j: usize| &s.h[j * nx..(j + 1) * nx];
    let urow = |j: usize| &s.u[j * nux..(j + 1) * nux];
    let vrow = |j: usize| &s.v[j * nx..(j + 1) * nx];

    // mass fluxes; walls carry none, open boundaries carry h0 · V_jet
    for j in 0..ny {
        let (hr, ur) = (hrow(j), urow(j));
        let o = &mut w.uf[j * nux..(j + 1) * nux];
        o[0] = 0.0;
        o[nx] = 0.0;
        for i in 1..nx {
            o[i] = 0.5 * (hr[i - 1] + hr[i]) * ur[i];
        }
    }
    for (o, &vj) in w.vf[..nx].iter_mut().zip(&g.vjet) {
        *o = g.h0 * vj;
    }
    for (o, &vj) in w.vf[ny * nx..].iter_mut().zip(&g.vjet) {
        *o = g.h0 * vj;
    }
    for j in 1..ny {
        let (hs, hn, vr) = (hrow(j - 1), hrow(j), vrow(j));
        let o = &mut w.vf[j * nx..(j + 1) * nx];
        for i in 0..nx {
            o[i] = 0.5 * (hs[i] + hn[i]) * vr[i];
        }
    }

    // relative vorticity at corners
    let no_slip = g.p.slip == Slip::NoSlip;
    for j in 0..=ny {
        let vr = vrow(j);
        let o = &mut w.zeta[j * nux..(j + 1) * nux];
        // western wall: ghost v = -v (no-slip, ζ = 2v/Δx) or +v (free-slip, ζ = 0)
        o[0] = if no_slip { 2.0 * vr[0] * idx } else { 0.0 };
        o[nx] = 0.0; // eastern wall, free-slip
        if j == 0 || j == ny {
            // open boundaries: zero normal gradient of u
            for i in 1..nx {
                o[i] = (vr[i] - vr[i - 1]) * idx;
            }
        } else {
            let (us, un) = (urow(j - 1), urow(j));
            for i in 1..nx {
                o[i] = (vr[i] - vr[i - 1]) * idx - (un[i] - us[i]) * idy;
            }
        }
    }

    // h averaged in x onto u columns (edge columns take the edge cell)
    for j in 0..ny {
        let hr = hrow(j);
        let o = &mut w.hx[j * nux..(j + 1) * nux];
        o[0] = hr[0];
        o[nx] = hr[nx - 1];
        for i in 1..nx {
            o[i] = 0.5 * (hr[i - 1] + hr[i]);
        }
    }
    // PV at corners: h averaged over the adjacent rows (clamped at the boundaries)
    for j in 0..=ny {
        let f = g.fv[j];
        let (j0, j1) = (j.saturating_sub(1), j.min(ny - 1));
        let (a, b) = (
            &w.hx[j0 * nux..(j0 + 1) * nux],
            &w.hx[j1 * nux..(j1 + 1) * nux],
        );
        let z = &w.zeta[j * nux..(j + 1) * nux];
        let o = &mut w.q[j * nux..(j + 1) * nux];
        for i in 0..nux {
            let hc = (0.5 * (a[i] + b[i])).max(1.0);
            o[i] = (f + z[i]) / hc;
        }
    }

    // Bernoulli function and divergence at centres
    for j in 0..ny {
        let (hr, ur, vs, vn) = (hrow(j), urow(j), vrow(j), vrow(j + 1));
        let ob = &mut w.b[j * nx..(j + 1) * nx];
        for i in 0..nx {
            let (uw, ue) = (ur[i], ur[i + 1]);
            ob[i] = g.g_red * hr[i] + 0.25 * (uw * uw + ue * ue + vs[i] * vs[i] + vn[i] * vn[i]);
        }
        let od = &mut w.d[j * nx..(j + 1) * nx];
        for i in 0..nx {
            od[i] = (ur[i + 1] - ur[i]) * idx + (vn[i] - vs[i]) * idy;
        }
    }
}

fn tendency(g: &Grid, w: &mut Work, s: &Fields, out: &mut Fields) {
    diagnose(g, w, s);
    let w = &*w;
    let (nx, ny) = (g.p.nx, g.p.ny);
    let nux = nx + 1;
    let (idx, idy, nu) = (g.idx, g.idy, g.nu);
    let crng = |j: usize| j * nx..(j + 1) * nx;
    let urng = |j: usize| j * nux..(j + 1) * nux;

    // continuity
    for j in 0..ny {
        let r = g.rc[j];
        let (uf, vs, vn) = (&w.uf[urng(j)], &w.vf[crng(j)], &w.vf[crng(j + 1)]);
        let (h, hr) = (&s.h[crng(j)], &g.href[crng(j)]);
        let o = &mut out.h[j * nx..(j + 1) * nx];
        for i in 0..nx {
            o[i] = -(uf[i + 1] - uf[i]) * idx - (vn[i] - vs[i]) * idy - r * (h[i] - hr[i]);
        }
    }

    // x-momentum on interior x-faces
    for j in 0..ny {
        let r = g.rc[j];
        let (qs, qn) = (&w.q[urng(j)], &w.q[urng(j + 1)]);
        let (zs, zn) = (&w.zeta[urng(j)], &w.zeta[urng(j + 1)]);
        let (vs, vn) = (&w.vf[crng(j)], &w.vf[crng(j + 1)]);
        let (b, d) = (&w.b[crng(j)], &w.d[crng(j)]);
        let u = &s.u[urng(j)];
        let o = &mut out.u[urng(j)];
        o[0] = 0.0;
        o[nx] = 0.0;
        for i in 1..nx {
            let qbar = 0.5 * (qs[i] + qn[i]);
            let vbar = 0.25 * (vs[i - 1] + vs[i] + vn[i - 1] + vn[i]);
            let grad_b = (b[i] - b[i - 1]) * idx;
            let visc = (d[i] - d[i - 1]) * idx - (zn[i] - zs[i]) * idy;
            o[i] = qbar * vbar - grad_b + nu * visc - r * u[i];
        }
    }

    // y-momentum on interior y-faces (open boundary rows are prescribed)
    out.v[..nx].fill(0.0);
    out.v[ny * nx..].fill(0.0);
    for j in 1..ny {
        let r = g.rv[j];
        let (q, z) = (&w.q[urng(j)], &w.zeta[urng(j)]);
        let (us, un) = (&w.uf[urng(j - 1)], &w.uf[urng(j)]);
        let (bs, bn) = (&w.b[crng(j - 1)], &w.b[crng(j)]);
        let (ds, dn) = (&w.d[crng(j - 1)], &w.d[crng(j)]);
        let v = &s.v[crng(j)];
        let o = &mut out.v[j * nx..(j + 1) * nx];
        for i in 0..nx {
            let qbar = 0.5 * (q[i] + q[i + 1]);
            let ubar = 0.25 * (us[i] + us[i + 1] + un[i] + un[i + 1]);
            let grad_b = (bn[i] - bs[i]) * idy;
            let visc = (dn[i] - ds[i]) * idy + (z[i + 1] - z[i]) * idx;
            o[i] = -qbar * ubar - grad_b + nu * visc - r * (v[i] - g.vjet[i]);
        }
    }
}

impl Wbc {
    pub fn nx(&self) -> usize {
        self.g.p.nx
    }

    pub fn ny(&self) -> usize {
        self.g.p.ny
    }

    /// Fraction of the meridional extent (from the south) where the equator lies.
    pub fn equator_frac(&self) -> f64 {
        -self.g.p.y_south / (self.g.p.y_north - self.g.p.y_south)
    }

    /// Relative vorticity and PV of the current state at corners
    /// ((Ny+1) × (Nx+1), row-major).
    pub fn vorticity(&mut self) -> (&[R], &[R]) {
        diagnose(&self.g, &mut self.w, &self.s);
        (&self.w.zeta, &self.w.q)
    }

    /// f at corner row j.
    pub fn f_corner(&self, j: usize) -> f64 {
        self.g.p.beta * self.g.yv[j]
    }

    /// Planetary vorticity scale for colouring: f at 1000 km.
    pub fn f_ref(&self) -> f64 {
        self.g.p.beta * 1000e3
    }

    pub fn min_h(&self) -> f64 {
        self.s.h.iter().fold(f64::INFINITY, |m, &x| m.min(x as f64))
    }

    /// Corner rows outside the sponges (the part worth showing, as in the
    /// paper's figures): (first, last) inclusive.
    pub fn visible_rows(&self) -> (usize, usize) {
        let p = &self.g.p;
        let dy = self.g.dy;
        let lo = (p.sponge_south / dy).round() as usize;
        let hi = p.ny - (p.sponge_north / dy).round() as usize;
        (lo.min(p.ny), hi.max(lo).min(p.ny))
    }

    /// Zonal extent (m).
    pub fn lx(&self) -> f64 {
        self.g.p.lx
    }

    /// Meridional extent of the visible (non-sponge) rows (m).
    pub fn visible_height(&self) -> f64 {
        let (lo, hi) = self.visible_rows();
        (hi - lo) as f64 * self.g.dy
    }

    /// Meridional extent of the domain (m).
    pub fn p_y_extent(&self) -> f64 {
        self.g.p.y_north - self.g.p.y_south
    }

    pub fn h0(&self) -> f64 {
        self.g.p.h0
    }

    pub fn slip(&self) -> Slip {
        self.g.p.slip
    }

    pub fn dt(&self) -> f64 {
        self.g.p.dt
    }

    pub fn max_speed(&self) -> f64 {
        let m = |v: &[R]| v.iter().fold(0.0f64, |a, &x| a.max((x as f64).abs()));
        m(&self.s.u).max(m(&self.s.v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_days(slip: Slip, days: f64) -> Wbc {
        let mut m = Wbc::new(
            WbcParams {
                slip,
                ..WbcParams::default()
            },
            1,
        );
        let n = (days * DAY / m.dt()).round() as usize;
        for _ in 0..n {
            m.step();
        }
        m
    }

    /// Both boundary conditions stay bounded for two months and keep the
    /// layer thickness positive.
    #[test]
    fn stays_bounded() {
        for slip in [Slip::NoSlip, Slip::FreeSlip] {
            let m = run_days(slip, 60.0);
            assert!(m.min_h() > 50.0, "{slip:?}: min h {}", m.min_h());
            assert!(m.max_speed() < 5.0, "{slip:?}: max speed {}", m.max_speed());
        }
    }

    /// Southern-Hemisphere water (negative PV) is carried across the equator:
    /// after a few weeks there is anomalous PV (f·q < 0) north of the equator.
    #[test]
    fn anomalous_pv_crosses_equator() {
        let mut m = run_days(Slip::FreeSlip, 30.0);
        let (nx, ny) = (m.nx(), m.ny());
        let fs: Vec<f64> = (0..=ny).map(|j| m.f_corner(j)).collect();
        let yv = m.g.yv.clone();
        let (_, q) = m.vorticity();
        let anomalous = (0..=ny)
            .filter(|&j| yv[j] > 200e3)
            .flat_map(|j| (0..=nx).map(move |i| (j, i)))
            .filter(|&(j, i)| fs[j] * (q[j * (nx + 1) + i] as f64) < 0.0)
            .count();
        assert!(
            anomalous > 10,
            "only {anomalous} anomalous-PV corners north of 200 km"
        );
    }
}
