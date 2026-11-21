//! Truecolor ANSI renderer: upper half-block '▀' → two vertical pixels per cell,
//! bilinearly resampled to the terminal size, colour codes only on change.

use std::fmt::Write;

/// Colour quantisation levels per sign.
pub const LEVELS: i32 = 48;
/// Palette index used for masked (NaN) pixels.
const HIGHLIGHT: u8 = (2 * LEVELS + 1) as u8;
/// Palette index used for the dotted reference line.
const MARKER: u8 = (2 * LEVELS + 2) as u8;

/// Optional dotted reference line (the equator), drawn in the brightest colour.
#[derive(Clone, Copy)]
pub enum Marker {
    None,
    /// horizontal line at this fraction of the height, from the bottom
    Horizontal(f64),
}

pub struct Palette {
    fg: Vec<String>,
    bg: Vec<String>,
}

impl Palette {
    /// Thermocline / wave height: blue (trough) – near-black (0) – orange (crest).
    pub fn waves() -> Self {
        Self::diverging([90.0, 170.0, 255.0], [255.0, 170.0, 60.0])
    }

    /// Sea surface temperature anomaly: blue (cold) – near-black – red (warm).
    pub fn sst() -> Self {
        Self::diverging([70.0, 140.0, 255.0], [255.0, 70.0, 50.0])
    }

    /// Potential vorticity: grey scale, white for negative (southern-hemisphere)
    /// PV through mid grey to black for positive (northern-hemisphere) PV, so
    /// negative PV carried across the equator shows as white on black.
    pub fn pv() -> Self {
        let n = LEVELS as f64;
        let mut rgb: Vec<[u8; 3]> = (-LEVELS..=LEVELS)
            .map(|k| {
                let v = k as f64 / n;
                // same gamma as the diverging maps: small anomalies stay visible
                let b = 0.5 - 0.5 * v.signum() * v.abs().powf(0.6);
                let g = (6.0 + 244.0 * b) as u8;
                [g, g, g]
            })
            .collect();
        rgb.push([255, 225, 60]); // HIGHLIGHT
        rgb.push([255, 120, 40]); // MARKER: orange, visible on grey
        Self::from_rgb(&rgb)
    }

    /// Diverging map for dark terminals from `neg_end` through near-black to
    /// `pos_end`, plus a bright yellow highlight colour for masked pixels.
    fn diverging(neg_end: [f64; 3], pos_end: [f64; 3]) -> Self {
        let base = [14.0, 16.0, 24.0];
        let n = LEVELS as usize;
        let mut rgb = Vec::with_capacity(2 * n + 1);
        let lerp = |end: &[f64; 3], k: usize| -> [u8; 3] {
            let g = (k as f64 / n as f64).powf(0.6); // gamma: small amplitudes stay visible
            let c = |a: usize| (base[a] + (end[a] - base[a]) * g) as u8;
            [c(0), c(1), c(2)]
        };
        for k in (0..=n).rev() {
            rgb.push(lerp(&neg_end, k));
        }
        for k in 1..=n {
            rgb.push(lerp(&pos_end, k));
        }
        rgb.push([255, 225, 60]); // HIGHLIGHT
        rgb.push(rgb[2 * n]); // MARKER: brightest positive colour
        Self::from_rgb(&rgb)
    }

    fn from_rgb(rgb: &[[u8; 3]]) -> Self {
        Palette {
            fg: rgb
                .iter()
                .map(|[r, g, b]| format!("\x1b[38;2;{r};{g};{b}m"))
                .collect(),
            bg: rgb
                .iter()
                .map(|[r, g, b]| format!("\x1b[48;2;{r};{g};{b}m"))
                .collect(),
        }
    }

    /// Foreground escape for a value in [-1, 1].
    pub fn fg_for(&self, v: f64) -> &str {
        let lev = (v * LEVELS as f64)
            .round()
            .clamp(-LEVELS as f64, LEVELS as f64) as i32;
        &self.fg[(lev + LEVELS) as usize]
    }
}

/// Bilinear sampling weights: for each destination pixel the two source
/// indices and the weight of the second one (cell-centre aligned, clamped).
fn bilinear_axis(n_src: usize, n_dst: usize, out: &mut Vec<(usize, usize, f64)>) {
    out.clear();
    let last = (n_src - 1) as f64;
    out.extend((0..n_dst).map(|d| {
        let s = ((d as f64 + 0.5) * n_src as f64 / n_dst as f64 - 0.5).clamp(0.0, last);
        let i0 = s.floor() as usize;
        let i1 = (i0 + 1).min(n_src - 1);
        (i0, i1, s - i0 as f64)
    }));
}

/// Reusable buffers so a frame allocates nothing in steady state.
#[derive(Default)]
pub struct Renderer {
    q: Vec<u8>,
    wy: Vec<(usize, usize, f64)>,
    wx: Vec<(usize, usize, f64)>,
}

impl Renderer {
    /// Render `field` (row-major `ny × nx`, row 0 at the bottom) into `out`,
    /// filling `cols × rows` terminal cells starting at the top-left. `gain`
    /// maps field values to [-1, 1]; NaN values are drawn in the highlight colour.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        out: &mut String,
        field: &[f64],
        nx: usize,
        ny: usize,
        cols: usize,
        rows: usize,
        gain: f64,
        marker: Marker,
        pal: &Palette,
    ) {
        self.render_region(out, field, nx, ny, 0, cols, rows, gain, marker, pal);
    }

    /// Like [`Renderer::render`], but into the `cols` terminal columns starting
    /// at column `col0` (0-based), leaving the rest of each line untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn render_region(
        &mut self,
        out: &mut String,
        field: &[f64],
        nx: usize,
        ny: usize,
        col0: usize,
        cols: usize,
        rows: usize,
        gain: f64,
        marker: Marker,
        pal: &Palette,
    ) {
        if cols == 0 || rows == 0 {
            return;
        }
        let py = 2 * rows;
        bilinear_axis(ny, py, &mut self.wy);
        self.wy.reverse(); // north at the top
        bilinear_axis(nx, cols, &mut self.wx);

        self.q.resize(py * cols, 0);
        let scale = gain * LEVELS as f64;
        for (p, &(j0, j1, fy)) in self.wy.iter().enumerate() {
            let r0 = &field[j0 * nx..(j0 + 1) * nx];
            let r1 = &field[j1 * nx..(j1 + 1) * nx];
            let dst = &mut self.q[p * cols..(p + 1) * cols];
            for (d, &(i0, i1, fx)) in dst.iter_mut().zip(&self.wx) {
                let a = r0[i0] + fx * (r0[i1] - r0[i0]);
                let b = r1[i0] + fx * (r1[i1] - r1[i0]);
                let v = a + fy * (b - a);
                *d = if v.is_nan() {
                    HIGHLIGHT
                } else {
                    let lev = (v * scale).round().clamp(-LEVELS as f64, LEVELS as f64);
                    (lev as i32 + LEVELS) as u8
                };
            }
        }
        match marker {
            Marker::None => {}
            Marker::Horizontal(f) => {
                // pixel rows run top to bottom
                let p = ((1.0 - f) * py as f64).floor().clamp(0.0, (py - 1) as f64) as usize;
                for c in self.q[p * cols..(p + 1) * cols].iter_mut().step_by(3) {
                    *c = MARKER;
                }
            }
        }

        for r in 0..rows {
            let _ = write!(out, "\x1b[{};{}H", r + 1, col0 + 1);
            let top = &self.q[2 * r * cols..(2 * r + 1) * cols];
            let bot = &self.q[(2 * r + 1) * cols..(2 * r + 2) * cols];
            let (mut lt, mut lb) = (u8::MAX, u8::MAX);
            for (&t, &b) in top.iter().zip(bot) {
                if t != lt {
                    out.push_str(&pal.fg[t as usize]);
                    lt = t;
                }
                if b != lb {
                    out.push_str(&pal.bg[b as usize]);
                    lb = b;
                }
                out.push('▀');
            }
        }
    }
}

/// Tracks the colour scale: an exponential moving average of the per-frame
/// 99.7th percentile of |h| (the Python version used the same percentile over
/// the whole precomputed run). Slow, so colours don't pulse.
pub struct AutoScale {
    pub scale: f64,
    buf: Vec<f64>,
    initialised: bool,
}

impl AutoScale {
    const FLOOR: f64 = 0.05;
    const TAU: f64 = 60.0; // model time units

    pub fn new() -> Self {
        AutoScale {
            scale: 1.0,
            buf: Vec::new(),
            initialised: false,
        }
    }

    pub fn update(&mut self, field: &[f64], dt: f64) {
        self.buf.clear();
        self.buf.extend(field.iter().map(|x| x.abs()));
        if self.buf.is_empty() {
            return;
        }
        let k = ((self.buf.len() - 1) as f64 * 0.997).round() as usize;
        let (_, p, _) = self.buf.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
        let p = p.max(Self::FLOOR);
        if !self.initialised {
            self.scale = p;
            self.initialised = true;
        } else {
            self.scale += (p - self.scale) * (dt / Self::TAU).min(1.0);
        }
    }

    pub fn reset(&mut self) {
        self.initialised = false;
    }
}
