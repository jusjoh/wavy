//! Kelvin–Helmholtz instability of a stratified shear layer.
//!
//! Vertical x–z slice, 2-D Boussinesq, vorticity–streamfunction form, in units
//! of half the shear-layer thickness and half the velocity difference:
//!
//! ```text
//!     ω_t + u ω_x + w ω_z = -J θ_x + ∇²ω / Re
//!     θ_t + u θ_x + w θ_z =           ∇²θ / Re
//!     ∇²ψ = ω,  u = ψ_z,  w = -ψ_x
//! ```
//!
//! Background U = tanh z, θ = tanh z (warm, light water on top). The gradient
//! Richardson number N²/U_z² then has its minimum J at z = 0, and the layer is
//! unstable only for J < 1/4 (Miles–Howard; neutral curve J = k(1 - k)).
//!
//! A free layer rolls up once, pairs, mixes and ends up stable. To keep going,
//! the background is rebuilt every `cycle` time units: for `REBUILD` units ω
//! and θ relax to the background profiles, then new noise is added.
//!
//! Periodic in x, rigid free-slip lids, all fields at cell centres. Advection:
//! flux form with 5th-order upwind-biased values and face velocities from ψ at
//! the corners (exactly non-divergent). SSP-RK3. Poisson: FFT in x, tridiagonal
//! solve in z. Arrays are padded with `G` ghost cells on each side.

use crate::rng::Rng;
use std::f64::consts::TAU;

type R = f32;
const G: usize = 3;
/// Duration and relaxation rate of the background rebuild between cycles.
const REBUILD: f64 = 20.0;
const REBUILD_RATE: f64 = 0.2;

#[derive(Clone, Debug)]
pub struct KhParams {
    pub lx: f64,
    pub lz: f64,
    /// must be a power of two
    pub nx: usize,
    /// must be even
    pub nz: usize,
    pub dt: f64,
    /// Richardson number J
    pub ri: f64,
    /// Reynolds number (Prandtl number 1)
    pub re: f64,
    /// time between rebuilds of the background, 0 = never
    pub cycle: f64,
    /// amplitude of the vorticity noise that seeds the billows
    pub noise: f64,
}

impl Default for KhParams {
    fn default() -> Self {
        KhParams {
            lx: 56.0, // four of the fastest-growing billows
            lz: 28.0,
            nx: 256,
            nz: 128,
            dt: 0.1,
            ri: 0.08,
            re: 1500.0,
            cycle: 600.0,
            noise: 0.02,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Laminar,
    Billows,
    Rebuilding,
}

impl Phase {
    pub fn name(self) -> &'static str {
        match self {
            Phase::Laminar => "laminar",
            Phase::Billows => "billows",
            Phase::Rebuilding => "rebuilding",
        }
    }
}

/// In-place radix-2 complex FFT (forward sign -; unnormalised).
struct Fft {
    n: usize,
    rev: Vec<usize>,
    /// twiddles, the stage with half-size h at h..2h (contiguous: vectorises)
    cos: Vec<R>,
    sin: Vec<R>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 2);
        let bits = n.trailing_zeros();
        let mut w = vec![0.0; n];
        let mut half = 1;
        while half < n {
            for k in 0..half {
                w[half + k] = TAU * k as f64 / (2 * half) as f64;
            }
            half *= 2;
        }
        Fft {
            n,
            rev: (0..n)
                .map(|i| i.reverse_bits() >> (usize::BITS - bits))
                .collect(),
            cos: w.iter().map(|w: &f64| w.cos() as R).collect(),
            sin: w.iter().map(|w: &f64| -w.sin() as R).collect(),
        }
    }

    fn run(&self, re: &mut [R], im: &mut [R]) {
        let n = self.n;
        for (i, &j) in self.rev.iter().enumerate() {
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut half = 1;
        while half < n {
            let (wr, wi) = (&self.cos[half..2 * half], &self.sin[half..2 * half]);
            for (cr, ci) in re
                .chunks_exact_mut(2 * half)
                .zip(im.chunks_exact_mut(2 * half))
            {
                let (ar, br) = cr.split_at_mut(half);
                let (ai, bi) = ci.split_at_mut(half);
                for k in 0..half {
                    let tr = wr[k] * br[k] - wi[k] * bi[k];
                    let ti = wr[k] * bi[k] + wi[k] * br[k];
                    br[k] = ar[k] - tr;
                    bi[k] = ai[k] - ti;
                    ar[k] += tr;
                    ai[k] += ti;
                }
            }
            half *= 2;
        }
    }
}

/// ∇²ψ = ω with ψ = 0 on the lids (half a cell outside the first/last row).
struct Poisson {
    nx: usize,
    nz: usize,
    fft: Fft,
    /// Thomas coefficients per (row, wavenumber): 1/pivot and c'
    piv: Vec<R>,
    cp: Vec<R>,
    idz2: R,
    re: Vec<R>,
    im: Vec<R>,
    row_re: Vec<R>,
    row_im: Vec<R>,
}

impl Poisson {
    fn new(nx: usize, nz: usize, dx: f64, dz: f64) -> Self {
        let nk = nx / 2 + 1;
        let a = 1.0 / (dz * dz);
        let (mut piv, mut cp) = (vec![0.0; nz * nk], vec![0.0; nz * nk]);
        for k in 0..nk {
            let lam = (2.0 - 2.0 * (TAU * k as f64 / nx as f64).cos()) / (dx * dx);
            let mut c_prev = 0.0;
            for j in 0..nz {
                let wall = if j == 0 || j == nz - 1 { a } else { 0.0 };
                let sub = if j == 0 { 0.0 } else { a };
                let p = 1.0 / (-2.0 * a - lam - wall - sub * c_prev);
                c_prev = a * p;
                piv[j * nk + k] = p as R;
                cp[j * nk + k] = c_prev as R;
            }
        }
        Poisson {
            nx,
            nz,
            fft: Fft::new(nx),
            piv,
            cp,
            idz2: a as R,
            re: vec![0.0; nz * nk],
            im: vec![0.0; nz * nk],
            row_re: vec![0.0; nx],
            row_im: vec![0.0; nx],
        }
    }

    /// `om`, `psi`: padded arrays; only the interior of `psi` is written.
    fn solve(&mut self, om: &[R], psi: &mut [R]) {
        let (nx, nz) = (self.nx, self.nz);
        let (nk, st) = (nx / 2 + 1, nx + 2 * G);
        let at = |j: usize| (j + G) * st + G;
        // forward: two real rows per complex transform
        for j in (0..nz).step_by(2) {
            self.row_re.copy_from_slice(&om[at(j)..at(j) + nx]);
            self.row_im.copy_from_slice(&om[at(j + 1)..at(j + 1) + nx]);
            self.fft.run(&mut self.row_re, &mut self.row_im);
            for k in 0..nk {
                let m = (nx - k) % nx;
                let (zr, zi) = (self.row_re[k], self.row_im[k]);
                let (yr, yi) = (self.row_re[m], self.row_im[m]);
                self.re[j * nk + k] = 0.5 * (zr + yr);
                self.im[j * nk + k] = 0.5 * (zi - yi);
                self.re[(j + 1) * nk + k] = 0.5 * (zi + yi);
                self.im[(j + 1) * nk + k] = 0.5 * (yr - zr);
            }
        }
        let a = self.idz2;
        for f in [&mut self.re, &mut self.im] {
            for j in 0..nz {
                let (lo, hi) = f.split_at_mut(j * nk);
                let row = &mut hi[..nk];
                let piv = &self.piv[j * nk..(j + 1) * nk];
                if j == 0 {
                    for (r, p) in row.iter_mut().zip(piv) {
                        *r *= p;
                    }
                } else {
                    let prev = &lo[(j - 1) * nk..];
                    for ((r, p), q) in row.iter_mut().zip(piv).zip(prev) {
                        *r = (*r - a * q) * p;
                    }
                }
            }
            for j in (0..nz - 1).rev() {
                let (lo, hi) = f.split_at_mut((j + 1) * nk);
                let cp = &self.cp[j * nk..(j + 1) * nk];
                for ((r, c), q) in lo[j * nk..].iter_mut().zip(cp).zip(&hi[..nk]) {
                    *r -= c * q;
                }
            }
        }
        // inverse: the forward transform of the conjugate, conjugated
        let norm = 1.0 / nx as R;
        for j in (0..nz).step_by(2) {
            for k in 0..nx {
                // Z(k) = A(k) + i B(k), with A, B Hermitian
                let (m, s) = if k < nk { (k, 1.0) } else { (nx - k, -1.0) };
                let (ar, ai) = (self.re[j * nk + m], s * self.im[j * nk + m]);
                let (br, bi) = (self.re[(j + 1) * nk + m], s * self.im[(j + 1) * nk + m]);
                self.row_re[k] = ar - bi;
                self.row_im[k] = -(ai + br);
            }
            self.fft.run(&mut self.row_re, &mut self.row_im);
            for (p, r) in psi[at(j)..at(j) + nx].iter_mut().zip(&self.row_re) {
                *p = r * norm;
            }
            for (p, r) in psi[at(j + 1)..at(j + 1) + nx].iter_mut().zip(&self.row_im) {
                *p = -r * norm;
            }
        }
    }
}

/// Periodic ghost columns; ghost rows mirrored with `sign` (+1 even, -1 odd).
fn fill_ghosts(f: &mut [R], nx: usize, nz: usize, sign: R) {
    let st = nx + 2 * G;
    for j in G..G + nz {
        let r = &mut f[j * st..(j + 1) * st];
        for g in 0..G {
            r[g] = r[nx + g];
            r[G + nx + g] = r[G + g];
        }
    }
    for g in 0..G {
        for (dst, src) in [(G - 1 - g, G + g), (G + nz + g, G + nz - 1 - g)] {
            let (d, s) = (dst * st, src * st);
            for i in 0..st {
                f[d + i] = sign * f[s + i];
            }
        }
    }
}

/// Advective flux through a row of faces, 5th-order upwind-biased, minus the
/// diffusive flux. `q(o)`: the value `o` cells along the face normal, faces lying
/// between 2 and 3. With `LIMIT` the face value stays between its neighbours
/// (universal limiter), so no new extrema appear at sharp fronts.
#[inline(always)]
fn face_fluxes<const LIMIT: bool>(flux: &mut [R], vel: &[R], q: [&[R]; 6], diff: R) {
    const C: R = 1.0 / 60.0;
    let n = flux.len();
    let [a, b, c, d, e, f] = q;
    let (a, b, c) = (&a[..n], &b[..n], &c[..n]);
    let (d, e, f) = (&d[..n], &e[..n], &f[..n]);
    let vel = &vel[..n];
    for i in 0..n {
        let grad = d[i] - c[i];
        let mid = 37.0 * (c[i] + d[i]) - 8.0 * (b[i] + e[i]) + (a[i] + f[i]);
        let dif = 10.0 * grad - 5.0 * (e[i] - b[i]) + (f[i] - a[i]);
        let (mut left, mut right) = (C * (mid - dif), C * (mid + dif));
        if LIMIT {
            left = c[i] + limit(left - c[i], grad, c[i] - b[i]);
            right = d[i] - limit(d[i] - right, grad, e[i] - d[i]);
        }
        let (vp, vm) = (0.5 * (vel[i] + vel[i].abs()), 0.5 * (vel[i] - vel[i].abs()));
        flux[i] = vp * left + vm * right - diff * grad;
    }
}

/// Clip `delta` to lie between 0 and minmod(down, 2 up).
#[inline(always)]
fn limit(delta: R, down: R, up: R) -> R {
    let up = 2.0 * up;
    let hi = fmin(fmax(down, 0.0), fmax(up, 0.0));
    let lo = fmax(fmin(down, 0.0), fmin(up, 0.0));
    fmax(fmin(delta, hi), lo)
}

/// min / max without NaN handling (single SIMD instructions).
#[inline(always)]
fn fmin(a: R, b: R) -> R {
    if a < b {
        a
    } else {
        b
    }
}

#[inline(always)]
fn fmax(a: R, b: R) -> R {
    if a > b {
        a
    } else {
        b
    }
}

struct Work {
    psi: Vec<R>,
    /// corner ψ, (nz+1) × (nx+1): row j / column i = lower-left corner of cell (j, i)
    corner: Vec<R>,
    /// face velocities: u on west faces, then w on lower faces
    vel: Vec<R>,
    fx: Vec<R>,
    fz0: Vec<R>,
    fz1: Vec<R>,
    d_om: Vec<R>,
    d_th: Vec<R>,
}

/// d = -∇·(flux) of one field (no flux through the lids).
#[allow(clippy::too_many_arguments)]
fn divergence<const LIMIT: bool>(
    (nx, nz, idx, idz, nu): (usize, usize, R, R, R),
    f: &[R],
    vu: &[R],
    vw: &[R],
    fx: &mut [R],
    fz0: &mut Vec<R>,
    fz1: &mut Vec<R>,
    d: &mut [R],
) {
    let st = nx + 2 * G;
    fz0.fill(0.0);
    for j in 0..nz {
        let row = &f[(j + G) * st..][..st];
        let q = std::array::from_fn(|o| &row[o..][..nx + 1]);
        face_fluxes::<LIMIT>(fx, &vu[j * (nx + 1)..][..nx + 1], q, nu * idx);
        if j + 1 < nz {
            let q = std::array::from_fn(|o| &f[(j + 1 + o) * st + G..][..nx]);
            face_fluxes::<LIMIT>(fz1, &vw[(j + 1) * nx..][..nx], q, nu * idz);
        } else {
            fz1.fill(0.0);
        }
        let d = &mut d[(j + G) * st + G..][..nx];
        let (fw, fe) = (&fx[..nx], &fx[1..][..nx]);
        let (lo, hi) = (&fz0[..nx], &fz1[..nx]);
        for (i, d) in d.iter_mut().enumerate() {
            *d = (fw[i] - fe[i]) * idx + (lo[i] - hi[i]) * idz;
        }
        std::mem::swap(fz0, fz1);
    }
}

pub struct Kh {
    p: KhParams,
    dx: f64,
    dz: f64,
    om: Vec<R>,
    th: Vec<R>,
    om0: Vec<R>,
    th0: Vec<R>,
    poisson: Poisson,
    w: Work,
    /// background profiles per row
    om_bg: Vec<R>,
    th_bg: Vec<R>,
    pub t: f64,
    /// time since the current cycle began
    t_cycle: f64,
    rebuild_left: f64,
    rng: Rng,
}

impl Kh {
    pub fn new(p: KhParams, seed: u64) -> Self {
        let (nx, nz) = (p.nx, p.nz);
        assert!(nz.is_multiple_of(2));
        let (dx, dz) = (p.lx / nx as f64, p.lz / nz as f64);
        let z = |j: usize| (j as f64 + 0.5) * dz - 0.5 * p.lz;
        let size = (nx + 2 * G) * (nz + 2 * G);
        let zeros = vec![0.0; size];
        let mut m = Kh {
            dx,
            dz,
            om: zeros.clone(),
            th: zeros.clone(),
            om0: zeros.clone(),
            th0: zeros.clone(),
            poisson: Poisson::new(nx, nz, dx, dz),
            w: Work {
                psi: zeros.clone(),
                corner: vec![0.0; (nx + 1) * (nz + 1)],
                vel: vec![0.0; nz * (nx + 1) + nz * nx],
                fx: vec![0.0; nx + 1],
                fz0: vec![0.0; nx],
                fz1: vec![0.0; nx],
                d_om: zeros.clone(),
                d_th: zeros,
            },
            om_bg: (0..nz).map(|j| z(j).cosh().powi(-2) as R).collect(),
            th_bg: (0..nz).map(|j| z(j).tanh() as R).collect(),
            p,
            t: 0.0,
            t_cycle: 0.0,
            rebuild_left: 0.0,
            rng: Rng::new(seed),
        };
        m.restart();
        m
    }

    /// Background shear layer plus fresh noise.
    pub fn restart(&mut self) {
        let (nx, st) = (self.p.nx, self.stride());
        for j in 0..self.p.nz {
            let a = (j + G) * st + G;
            self.om[a..a + nx].fill(self.om_bg[j]);
            self.th[a..a + nx].fill(self.th_bg[j]);
        }
        self.t = 0.0;
        self.t_cycle = 0.0;
        self.rebuild_left = 0.0;
        self.perturb(self.p.noise);
    }

    /// Long-wave random vorticity in the shear layer (also key `k`). Including
    /// wavelengths longer than the fastest-growing one lets the billows pair.
    pub fn perturb(&mut self, amp: f64) {
        let modes = (self.p.lx / 7.0).round().max(1.0) as usize;
        let waves: Vec<(f64, f64)> = (1..=modes)
            .map(|_| (amp * self.rng.normal(), TAU * self.rng.f64()))
            .collect();
        let (nx, st) = (self.p.nx, self.stride());
        for i in 0..nx {
            let x = TAU * (i as f64 + 0.5) / nx as f64;
            let s: f64 = waves
                .iter()
                .enumerate()
                .map(|(k, (a, ph))| a * ((k + 1) as f64 * x + ph).cos())
                .sum();
            for j in 0..self.p.nz {
                self.om[(j + G) * st + G + i] += s as R * self.om_bg[j];
            }
        }
    }

    pub fn step(&mut self) {
        let dt = self.p.dt as R;
        self.om0.copy_from_slice(&self.om);
        self.th0.copy_from_slice(&self.th);
        // SSP-RK3: s = a s0 + b (s + dt L(s))
        for (a, b) in [(0.0, 1.0), (0.75, 0.25), (1.0 / 3.0, 2.0 / 3.0)] {
            self.tendency();
            for (f, f0, d) in [
                (&mut self.om, &self.om0, &self.w.d_om),
                (&mut self.th, &self.th0, &self.w.d_th),
            ] {
                for ((f, f0), d) in f.iter_mut().zip(f0).zip(d) {
                    *f = a * f0 + b * (*f + dt * d);
                }
            }
        }
        self.t += self.p.dt;
        self.t_cycle += self.p.dt;
        if self.rebuild_left > 0.0 {
            self.relax((self.p.dt * REBUILD_RATE) as R);
            self.rebuild_left -= self.p.dt;
            if self.rebuild_left <= 0.0 {
                self.t_cycle = 0.0;
                self.perturb(self.p.noise);
            }
        } else if self.p.cycle > 0.0 && self.t_cycle >= self.p.cycle {
            self.rebuild();
        }
    }

    /// Start rebuilding the background shear layer.
    pub fn rebuild(&mut self) {
        self.rebuild_left = REBUILD;
    }

    fn relax(&mut self, c: R) {
        let (nx, st) = (self.p.nx, self.stride());
        for j in 0..self.p.nz {
            let a = (j + G) * st + G;
            for (f, bg) in [(&mut self.om, self.om_bg[j]), (&mut self.th, self.th_bg[j])] {
                for v in &mut f[a..a + nx] {
                    *v += c * (bg - *v);
                }
            }
        }
    }

    fn stride(&self) -> usize {
        self.p.nx + 2 * G
    }

    /// Solve for ψ and put it on the corners.
    fn invert(&mut self) {
        let (nx, nz, st) = (self.p.nx, self.p.nz, self.stride());
        self.poisson.solve(&self.om, &mut self.w.psi);
        fill_ghosts(&mut self.w.psi, nx, nz, -1.0);
        for (j, row) in self.w.corner.chunks_exact_mut(nx + 1).enumerate() {
            let a = &self.w.psi[(j + G - 1) * st + G - 1..][..nx + 2];
            let b = &self.w.psi[(j + G) * st + G - 1..][..nx + 2];
            for (i, c) in row.iter_mut().enumerate() {
                *c = 0.25 * (a[i] + a[i + 1] + b[i] + b[i + 1]);
            }
        }
    }

    fn tendency(&mut self) {
        let (nx, nz, st) = (self.p.nx, self.p.nz, self.stride());
        let (idx, idz) = ((1.0 / self.dx) as R, (1.0 / self.dz) as R);
        let nu = (1.0 / self.p.re) as R;
        fill_ghosts(&mut self.om, nx, nz, 1.0);
        fill_ghosts(&mut self.th, nx, nz, 1.0);
        self.invert();
        let Work {
            corner,
            vel,
            fx,
            fz0,
            fz1,
            d_om,
            d_th,
            ..
        } = &mut self.w;
        // face velocities: u = ψ_z on west faces, w = -ψ_x on lower faces
        let (vu, vw) = vel.split_at_mut(nz * (nx + 1));
        for j in 0..nz {
            let c0 = &corner[j * (nx + 1)..][..nx + 1];
            let c1 = &corner[(j + 1) * (nx + 1)..][..nx + 1];
            for ((u, a), b) in vu[j * (nx + 1)..][..nx + 1].iter_mut().zip(c0).zip(c1) {
                *u = (b - a) * idz;
            }
            for ((w, a), b) in vw[j * nx..][..nx].iter_mut().zip(c0).zip(&c0[1..]) {
                *w = (a - b) * idx;
            }
        }
        let grid = (nx, nz, idx, idz, nu);
        divergence::<false>(grid, &self.om, vu, vw, fx, fz0, fz1, d_om);
        divergence::<true>(grid, &self.th, vu, vw, fx, fz0, fz1, d_th);
        // baroclinic torque -J θ_x (4th order)
        let c = (self.p.ri / (12.0 * self.dx)) as R;
        for j in 0..nz {
            let row = &self.th[(j + G) * st..][..st];
            let s: [&[R]; 4] = [
                &row[G - 2..][..nx],
                &row[G - 1..][..nx],
                &row[G + 1..][..nx],
                &row[G + 2..][..nx],
            ];
            for (i, d) in d_om[(j + G) * st + G..][..nx].iter_mut().enumerate() {
                *d -= c * (8.0 * (s[2][i] - s[1][i]) - (s[3][i] - s[0][i]));
            }
        }
    }

    /// Interior of θ or ω as f64 (row 0 = bottom).
    pub fn field(&self, vorticity: bool, out: &mut Vec<f64>) {
        let f = if vorticity { &self.om } else { &self.th };
        let (nx, st) = (self.p.nx, self.stride());
        out.clear();
        for j in 0..self.p.nz {
            let a = (j + G) * st + G;
            out.extend(f[a..a + nx].iter().map(|&v| v as f64));
        }
    }

    /// Root-mean-square vertical velocity in the shear layer (|z| < 2).
    pub fn w_rms(&mut self) -> f64 {
        self.invert();
        let (nx, nz) = (self.p.nx, self.p.nz);
        let (mut sum, mut n) = (0.0, 0usize);
        for j in 0..=nz {
            if (j as f64 * self.dz - 0.5 * self.p.lz).abs() >= 2.0 {
                continue;
            }
            let c = &self.w.corner[j * (nx + 1)..][..nx + 1];
            for i in 0..nx {
                let w = (c[i] - c[i + 1]) as f64 / self.dx;
                sum += w * w;
                n += 1;
            }
        }
        (sum / n.max(1) as f64).sqrt()
    }

    /// Minimum gradient Richardson number of the x-mean profiles where the
    /// mean shear is at least a third of its maximum.
    pub fn min_ri(&self) -> f64 {
        let (nx, nz, st) = (self.p.nx, self.p.nz, self.stride());
        let mean = |f: &[R], j: usize| {
            let a = (j + G) * st + G;
            f[a..a + nx].iter().map(|&v| v as f64).sum::<f64>() / nx as f64
        };
        let shear: Vec<f64> = (0..nz).map(|j| mean(&self.om, j)).collect();
        let th: Vec<f64> = (0..nz).map(|j| mean(&self.th, j)).collect();
        let smax = shear.iter().fold(0.0f64, |a, s| a.max(s.abs()));
        (1..nz - 1)
            .filter(|&j| shear[j].abs() > smax / 3.0)
            .map(|j| self.p.ri * (th[j + 1] - th[j - 1]) / (2.0 * self.dz) / shear[j].powi(2))
            .fold(f64::INFINITY, f64::min)
    }

    pub fn phase(&mut self) -> Phase {
        if self.rebuild_left > 0.0 {
            Phase::Rebuilding
        } else if self.w_rms() < 0.05 {
            Phase::Laminar
        } else {
            Phase::Billows
        }
    }

    pub fn ri(&self) -> f64 {
        self.p.ri
    }

    pub fn set_ri(&mut self, ri: f64) {
        self.p.ri = ri;
    }

    pub fn dt(&self) -> f64 {
        self.p.dt
    }

    pub fn nx(&self) -> usize {
        self.p.nx
    }

    pub fn nz(&self) -> usize {
        self.p.nz
    }

    /// Largest |ω| and |θ| (stability check).
    pub fn max_abs(&self) -> (f64, f64) {
        let m = |f: &[R]| f.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
        (m(&self.om), m(&self.th))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(k: f64, ri: f64) -> KhParams {
        KhParams {
            lx: TAU / k,
            nx: 64,
            nz: 256,
            dt: 0.05,
            ri,
            re: 1e6,
            cycle: 0.0,
            noise: 0.0,
            ..KhParams::default()
        }
    }

    /// Growth rate of w_rms for a small single-wave perturbation.
    fn growth(k: f64, ri: f64) -> f64 {
        let mut m = Kh::new(params(k, ri), 1);
        let (nx, st) = (m.p.nx, m.stride());
        for j in 0..m.p.nz {
            for i in 0..nx {
                let x = TAU * i as f64 / nx as f64;
                m.om[(j + G) * st + G + i] += 1e-3 * x.cos() as R * m.om_bg[j];
            }
        }
        let advance = |m: &mut Kh, t: f64| {
            while m.t < t {
                m.step();
            }
            m.w_rms()
        };
        let a = advance(&mut m, 12.0);
        let b = advance(&mut m, 20.0);
        (b / a).ln() / 8.0
    }

    #[test]
    fn poisson_inverts_laplacian() {
        let (nx, nz, dx, dz) = (64, 32, 0.2, 0.3);
        let st = nx + 2 * G;
        let mut rng = Rng::new(3);
        let mut psi = vec![0.0 as R; st * (nz + 2 * G)];
        for j in 0..nz {
            for i in 0..nx {
                psi[(j + G) * st + G + i] = rng.normal() as R;
            }
        }
        fill_ghosts(&mut psi, nx, nz, -1.0);
        let mut om = vec![0.0 as R; psi.len()];
        for j in G..G + nz {
            for i in G..G + nx {
                let k = j * st + i;
                om[k] = (psi[k + 1] - 2.0 * psi[k] + psi[k - 1]) / (dx * dx) as R
                    + (psi[k + st] - 2.0 * psi[k] + psi[k - st]) / (dz * dz) as R;
            }
        }
        let mut got = vec![0.0 as R; psi.len()];
        Poisson::new(nx, nz, dx, dz).solve(&om, &mut got);
        for j in G..G + nz {
            for i in G..G + nx {
                let k = j * st + i;
                assert!((got[k] - psi[k]).abs() < 1e-3, "{} {}", got[k], psi[k]);
            }
        }
    }

    /// Michalke (1964): σ = 0.1897 at k = 0.4446 for U = tanh z, unstratified.
    #[test]
    fn growth_rate_matches_theory() {
        let s = growth(0.4446, 0.0);
        assert!((s - 0.1897).abs() < 0.006, "growth rate {s}");
    }

    /// Miles–Howard: no growth for J > 1/4; growth well below it.
    #[test]
    fn stable_above_quarter() {
        let (weak, strong) = (growth(0.45, 0.1), growth(0.45, 0.3));
        assert!(weak > 0.05, "J = 0.1: {weak}");
        assert!(strong < 0.01, "J = 0.3: {strong}");
    }

    #[test]
    fn stays_bounded_and_cycles() {
        let mut m = Kh::new(
            KhParams {
                nx: 128,
                nz: 64,
                dt: 0.1,
                ..KhParams::default()
            },
            1,
        );
        let mut peak = 0.0f64;
        let mut rebuilt = false;
        for n in 0..4000 {
            m.step();
            if n % 20 == 0 {
                peak = peak.max(m.w_rms());
                rebuilt |= m.phase() == Phase::Rebuilding;
                let (om, th) = m.max_abs();
                assert!(om < 5.0 && th < 1.05, "t {} ω {om} θ {th}", m.t);
            }
        }
        assert!(peak > 0.1 && rebuilt, "peak w_rms {peak}");
    }
}
