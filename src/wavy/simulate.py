#!/usr/bin/env python3
"""
Precompute equatorial wave dynamics for the terminal animation.

Linear shallow-water equations on an equatorial beta-plane, in Matsuno's
non-dimensional units (length = equatorial deformation radius, time = 1/sqrt(beta c)):

    u_t - y v = -h_x
    v_t + y u = -h_y
    h_t + u_x + v_y = 0

Periodic in x, solid walls at y = +-Ly. Arakawa C-grid, RK4 time stepping.

A Gaussian height bump on/near the equator splits into
  * an eastward Kelvin wave (speed 1, symmetric, no v)
  * westward Rossby waves (n=1 long-wave speed 1/3, two off-equator lobes)
  * fast inertia-gravity waves that radiate away.
New bumps are "kicked" in at intervals so the animation keeps going forever
(like pipes.sh); weak Rayleigh damping keeps the amplitudes bounded.

Output: an .npz with frames of h (float16), shape (T, Ny, Nx), plus metadata.
"""
import argparse
import numpy as np


def build_grid(Lx, Ly, Nx, Ny):
    dx, dy = Lx / Nx, 2 * Ly / Ny
    x = (np.arange(Nx) + 0.5) * dx                 # h / v x-positions
    yc = -Ly + (np.arange(Ny) + 0.5) * dy          # h / u y-positions
    yv = -Ly + np.arange(Ny + 1) * dy              # v y-positions (incl. walls)
    return dx, dy, x, yc, yv


def make_tendency(dx, dy, yc, yv, r):
    yu = yc[:, None]
    yvi = yv[1:-1, None]

    def f(h, u, v):
        # h: (Ny,Nx) centres; u: (Ny,Nx) at x_{i+1/2}; v: (Ny+1,Nx) at y_{j-1/2}
        dh = -((u - np.roll(u, 1, axis=1)) / dx + (v[1:] - v[:-1]) / dy)

        vc = 0.5 * (v[1:] + v[:-1])                       # v at centres
        v_at_u = 0.5 * (vc + np.roll(vc, -1, axis=1))
        du = yu * v_at_u - (np.roll(h, -1, axis=1) - h) / dx

        uc = 0.5 * (u + np.roll(u, 1, axis=1))            # u at centres
        u_at_v = 0.5 * (uc[1:] + uc[:-1])
        dv = np.zeros_like(v)
        dv[1:-1] = -yvi * u_at_v - (h[1:] - h[:-1]) / dy

        if r:
            dh -= r * h
            du -= r * u
            dv -= r * v
        return dh, du, dv

    return f


def rk4(f, h, u, v, dt):
    k1 = f(h, u, v)
    k2 = f(h + 0.5 * dt * k1[0], u + 0.5 * dt * k1[1], v + 0.5 * dt * k1[2])
    k3 = f(h + 0.5 * dt * k2[0], u + 0.5 * dt * k2[1], v + 0.5 * dt * k2[2])
    k4 = f(h + dt * k3[0], u + dt * k3[1], v + dt * k3[2])
    h = h + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    u = u + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    v = v + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return h, u, v


def bump(x, yc, Lx, x0, y0, s, amp):
    # periodic distance in x
    dxp = (x[None, :] - x0 + Lx / 2) % Lx - Lx / 2
    return amp * np.exp(-(dxp ** 2 + (yc[:, None] - y0) ** 2) / (2 * s ** 2))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--out", default="waves.npz")
    p.add_argument("--Lx", type=float, default=40.0, help="domain length (deformation radii)")
    p.add_argument("--Ly", type=float, default=8.0, help="half-width in y")
    p.add_argument("--Nx", type=int, default=200)
    p.add_argument("--Ny", type=int, default=80)
    p.add_argument("--dt", type=float, default=0.04)
    p.add_argument("--t-end", type=float, default=400.0)
    p.add_argument("--save-every", type=float, default=0.2, help="time between frames")
    p.add_argument("--kick-every", type=float, default=30.0,
                   help="add a new bump every N time units (0 = only the initial one)")
    p.add_argument("--damping", type=float, default=0.008, help="Rayleigh damping rate")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    dx, dy, x, yc, yv = build_grid(a.Lx, a.Ly, a.Nx, a.Ny)
    f = make_tendency(dx, dy, yc, yv, a.damping)

    h = bump(x, yc, a.Lx, a.Lx * 0.3, 0.0, 1.0, 1.0)
    u = np.zeros((a.Ny, a.Nx))
    v = np.zeros((a.Ny + 1, a.Nx))

    nsteps = int(round(a.t_end / a.dt))
    save_n = max(1, int(round(a.save_every / a.dt)))
    kick_n = int(round(a.kick_every / a.dt)) if a.kick_every > 0 else 0

    frames = []
    for n in range(nsteps + 1):
        if n % save_n == 0:
            frames.append(h.astype(np.float16))
        if kick_n and n > 0 and n % kick_n == 0:
            h = h + bump(x, yc, a.Lx,
                         x0=rng.uniform(0, a.Lx),
                         y0=rng.normal(0, 0.7),
                         s=rng.uniform(0.7, 1.5),
                         amp=rng.choice([-1, 1]) * rng.uniform(0.6, 1.2))
        h, u, v = rk4(f, h, u, v, a.dt)
        if n % 1000 == 0:
            print(f"\rstep {n}/{nsteps}  t={n * a.dt:7.1f}  max|h|={np.abs(h).max():.3f}",
                  end="", flush=True)
    print()

    frames = np.stack(frames).astype(np.float32)
    # normalise and quantise to int8 (player only uses ~100 colour levels anyway)
    scale = float(np.percentile(np.abs(frames), 99.7)) or 1.0
    q = np.clip(np.rint(frames / scale * 127), -127, 127).astype(np.int8)
    np.savez_compressed(a.out, h=q, scale=scale, dt_frame=save_n * a.dt,
                        Lx=a.Lx, Ly=a.Ly)
    print(f"saved {frames.shape[0]} frames of {frames.shape[1]}x{frames.shape[2]} -> {a.out}")


if __name__ == "__main__":
    main()
