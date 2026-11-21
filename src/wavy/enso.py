#!/usr/bin/env python3
"""
Minimal coupled ENSO model (stripped-down Cane-Zebiak) on the equatorial beta-plane.

Ocean: linear shallow water in a basin (walls at x = 0, Lx and y = +-Ly), Matsuno
units (length = deformation radius ~350 km, time ~1.45 days, Kelvin speed 1).
h is the thermocline-depth anomaly (deep = warm).

    u_t - y v = -h_x - r u + tau
    v_t + y u = -h_y - r v
    h_t + u_x + v_y = -r h

SST anomaly T (same grid as h) relaxes to a subsurface temperature set by the
thermocline where there is upwelling (the eastern cold tongue):

    T_t = eps * (M(x, y) * F(h) - T),    F(h) = a+ tanh(h/a+) (h>0), a- tanh(h/a-) (h<0)

Atmosphere (Bjerknes feedback): a zonal wind-stress patch in the central basin,
proportional to the Nino-3 SST index, plus random westerly/easterly wind bursts:

    tau = mu * T_nino3(t) * G(x, y) + bursts(x, y, t)

Warm east -> westerlies -> downwelling Kelvin wave -> warmer east  (positive feedback)
Off-equatorial upwelling Rossby waves reflect at the west wall into upwelling
Kelvin waves                                                       (delayed negative feedback)
"""
import argparse
from dataclasses import dataclass, asdict

import numpy as np

YEAR = 252.0  # model time units per year (1 unit ~ 1.45 days)


@dataclass
class Config:
    Lx: float = 40.0
    Ly: float = 8.0
    Nx: int = 100
    Ny: int = 40
    dt: float = 0.08
    r: float = 0.004          # ocean Rayleigh damping (~1/(9 months))
    eps: float = 0.03         # SST relaxation rate (~1.5 months); 0.01 gives no oscillation
    mu: float = 0.42          # Bjerknes coupling (threshold ~0.43: below damped, above limit cycle)
    a_warm: float = 1.0       # saturation of F(h) for deep thermocline
    a_cold: float = 0.6       # ... for shallow thermocline (El Nino > La Nina)
    wind_x: float = 0.3       # centre of coupled wind patch (fraction of Lx)
    wind_sx: float = 0.12     # its zonal half-width (fraction of Lx)
    wind_sy: float = 1.0      # its meridional half-width (wider -> stuck permanent El Nino)
    burst_rate: float = 1 / 40  # wind bursts per time unit (~6 per year)
    burst_amp: float = 0.05   # wind-burst stress amplitude (makes the cycle irregular)
    burst_dur: tuple = (8.0, 20.0)  # duration range (time units, ~2-4 weeks)
    years: float = 40.0
    seed: int = 1
    init_h: float = 0.1       # small initial thermocline bump (to start the oscillation)


class Model:
    def __init__(self, c: Config):
        self.c = c
        self.dx, self.dy = c.Lx / c.Nx, 2 * c.Ly / c.Ny
        self.x = (np.arange(c.Nx) + 0.5) * self.dx              # h / T / v columns
        self.xu = np.arange(c.Nx + 1) * self.dx                 # u columns
        self.yc = -c.Ly + (np.arange(c.Ny) + 0.5) * self.dy     # h / T / u rows
        self.yv = -c.Ly + np.arange(c.Ny + 1) * self.dy         # v rows
        X, Y = np.meshgrid(self.x, self.yc)
        Xu, Yu = np.meshgrid(self.xu, self.yc)
        xf = X / c.Lx
        # upwelling mask: equatorial cold tongue in the east
        self.M = np.exp(-Y**2 / (2 * 1.5**2)) / (1 + np.exp(-(xf - 0.45) / 0.06))
        # Nino-3 box: 150W-90W, 5S-5N  ->  x/Lx in [0.56, 0.94], |y| < 1.6
        self.nino3 = (xf > 0.56) & (xf < 0.94) & (np.abs(Y) < 1.6)
        # coupled wind patch on u points
        self.G = np.exp(-((Xu / c.Lx - c.wind_x) / c.wind_sx) ** 2 / 2
                        - (Yu / c.wind_sy) ** 2 / 2)
        self.Xu, self.Yu = Xu, Yu
        self.rng = np.random.default_rng(c.seed)
        self.bursts = []  # (t_start, t_end, tau field)

    def F(self, h):
        c = self.c
        return np.where(h > 0, c.a_warm * np.tanh(h / c.a_warm), c.a_cold * np.tanh(h / c.a_cold))

    def nino3_index(self, T):
        return T[self.nino3].mean()

    def burst_tau(self, t):
        tau = 0.0
        for t0, t1, f in self.bursts:
            if t0 <= t < t1:
                tau = tau + f
        return tau

    def maybe_new_burst(self, t, dt):
        c = self.c
        if c.burst_amp > 0 and self.rng.random() < c.burst_rate * dt:
            x0 = self.rng.uniform(0.15, 0.5) * c.Lx
            dur = self.rng.uniform(*c.burst_dur)
            sign = self.rng.choice([-1.0, 1.0])
            f = sign * c.burst_amp * np.exp(-((self.Xu - x0) / 2.5) ** 2 / 2 - (self.Yu / 2.0) ** 2 / 2)
            self.bursts.append((t, t + dur, f))
        self.bursts = [b for b in self.bursts if b[1] > t]

    def tendency(self, s, tau_ext):
        h, u, v, T = s
        c, dx, dy = self.c, self.dx, self.dy
        dh = -((u[:, 1:] - u[:, :-1]) / dx + (v[1:] - v[:-1]) / dy) - c.r * h

        tau = c.mu * self.nino3_index(T) * self.G + tau_ext
        vc = 0.5 * (v[1:] + v[:-1])
        du = np.zeros_like(u)
        du[:, 1:-1] = (self.yc[:, None] * 0.5 * (vc[:, 1:] + vc[:, :-1])
                       - (h[:, 1:] - h[:, :-1]) / dx - c.r * u[:, 1:-1]
                       + (tau[:, 1:-1] if np.ndim(tau) else tau))

        uc = 0.5 * (u[:, 1:] + u[:, :-1])
        dv = np.zeros_like(v)
        dv[1:-1] = (-self.yv[1:-1, None] * 0.5 * (uc[1:] + uc[:-1])
                    - (h[1:] - h[:-1]) / dy - c.r * v[1:-1])

        dT = c.eps * (self.M * self.F(h) - T)
        return (dh, du, dv, dT)

    def run(self, record_every=2.0, hov_every=10.0):
        """Integrate; return time series (Nino-3, west h, ...) and equatorial Hovmoellers."""
        c = self.c
        Nx, Ny = c.Nx, c.Ny
        h = c.init_h * np.exp(-((self.x[None, :] - 0.5 * c.Lx) ** 2 + self.yc[:, None] ** 2) / 2)
        s = [h, np.zeros((Ny, Nx + 1)), np.zeros((Ny + 1, Nx)), np.zeros((Ny, Nx))]
        n = int(round(c.years * YEAR / c.dt))
        rec_n = max(1, int(round(record_every / c.dt)))
        hov_n = max(1, int(round(hov_every / c.dt)))
        eq = np.abs(self.yc) < 1.0
        west = (self.x < 0.4 * c.Lx)
        ts = {k: [] for k in ("t", "nino3", "h_west", "h_east", "tau_burst")}
        hov_h, hov_T = [], []
        dt = c.dt
        add = lambda s, k, a: [x + a * y for x, y in zip(s, k)]
        for i in range(n + 1):
            t = i * dt
            if i % rec_n == 0:
                ts["t"].append(t)
                ts["nino3"].append(self.nino3_index(s[3]))
                ts["h_west"].append(s[0][np.ix_(eq, west)].mean())
                ts["h_east"].append(s[0][np.ix_(eq, ~west)].mean())
                tb = self.burst_tau(t)
                ts["tau_burst"].append(float(np.max(np.abs(tb))) if np.ndim(tb) else 0.0)
            if i % hov_n == 0:
                hov_h.append(s[0][eq].mean(axis=0))
                hov_T.append(s[3][eq].mean(axis=0))
            if not np.all(np.isfinite(s[0])) or np.abs(s[0]).max() > 1e3:
                break
            self.maybe_new_burst(t, dt)
            tau_ext = self.burst_tau(t)
            k1 = self.tendency(s, tau_ext)
            k2 = self.tendency(add(s, k1, dt / 2), tau_ext)
            k3 = self.tendency(add(s, k2, dt / 2), tau_ext)
            k4 = self.tendency(add(s, k3, dt), tau_ext)
            s = [x + dt / 6 * (a + 2 * b + 2 * cc + d) for x, a, b, cc, d in zip(s, k1, k2, k3, k4)]
        out = {k: np.array(v) for k, v in ts.items()}
        out["hov_h"], out["hov_T"] = np.array(hov_h), np.array(hov_T)
        out["x"] = self.x
        out["hov_dt"] = hov_n * dt
        self.state = s
        return out


def diagnostics(out, spinup_years=5.0):
    """Nino-3 statistics after spin-up (NOT mean-removed, so a stuck permanent
    El Nino shows up as a large |mean| with few sign changes).

    rms     : root-mean-square of the index
    mean    : its mean (should be ~0 for a real oscillation)
    period  : 2 * duration / number of sign changes (years)
    growth  : log rms(last third) / rms(first third), per year
    """
    t, n3 = out["t"], out["nino3"]
    keep = t > spinup_years * YEAR
    t, x = t[keep], n3[keep]
    if len(x) < 16:
        return dict(rms=np.nan, mean=np.nan, period=np.nan, growth=np.nan)
    crossings = np.count_nonzero(np.diff(np.sign(x)) != 0)
    dur = (t[-1] - t[0]) / YEAR
    period = 2 * dur / crossings if crossings else np.inf
    k = len(x) // 3
    rms = lambda v: np.sqrt((v ** 2).mean())
    growth = np.log((rms(x[-k:]) + 1e-30) / (rms(x[:k]) + 1e-30)) / (dur * 2 / 3)
    return dict(rms=float(rms(x)), mean=float(x.mean()), period=float(period),
                growth=float(growth))


def plot(out, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 7), gridspec_kw=dict(width_ratios=[1, 1, 1.4]))
    yrs = np.arange(len(out["hov_T"])) * out["hov_dt"] / YEAR
    for a, key, lab in ((ax[0], "hov_h", "thermocline h (eq.)"), (ax[1], "hov_T", "SST T (eq.)")):
        d = out[key]
        v = np.abs(d).max() or 1
        a.pcolormesh(out["x"], yrs, d, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        a.set_xlabel("x (west -> east)"); a.set_ylabel("years"); a.set_title(lab)
    t = out["t"] / YEAR
    ax[2].plot(out["nino3"], t, "k", lw=1, label="Nino-3 SST")
    ax[2].plot(out["h_west"], t, "C2", lw=0.8, label="h west (recharge)")
    ax[2].axvline(0, color="gray", lw=0.5)
    ax[2].set_ylim(t[0], t[-1]); ax[2].legend(loc="upper right"); ax[2].set_title("indices")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in asdict(Config()).items():
        if isinstance(v, tuple):
            continue
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--plot", help="save Hovmoeller + index plot to this PNG")
    a = p.parse_args()
    c = Config(**{k: getattr(a, k) for k in asdict(Config()) if hasattr(a, k)})
    out = Model(c).run()
    d = diagnostics(out)
    print(f"mu={c.mu:.3f} r={c.r} eps={c.eps}  rms={d['rms']:.3f} mean={d['mean']:+.3f}  "
          f"period={d['period']:.2f} yr  growth={d['growth']:+.3f}/yr")
    if a.plot:
        plot(out, a.plot, title=f"mu={c.mu}, r={c.r}, eps={c.eps}, bursts={c.burst_amp}")


if __name__ == "__main__":
    main()
