#!/usr/bin/env python3
"""
Terminal player for precomputed equatorial waves (pipes.sh-style screensaver).

Uses truecolor ANSI + the upper-half-block '▀' so every terminal cell shows
two vertical pixels. Rescales to the current terminal size on the fly.

Keys:  q quit | space pause | + / - speed | e toggle equator line | h toggle HUD
"""
import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np

LEVELS = 48  # colour quantisation per sign


def colormap():
    """Diverging map for dark terminals: blue (low) - near-black (0) - orange (high)."""
    t = np.linspace(0, 1, LEVELS + 1)[:, None]
    base = np.array([14, 16, 24])
    neg_end = np.array([90, 170, 255])
    pos_end = np.array([255, 170, 60])
    # gamma so that small amplitudes are still visible
    g = t ** 0.6
    neg = base + (neg_end - base) * g
    pos = base + (pos_end - base) * g
    cmap = np.vstack([neg[::-1], pos[1:]]).astype(int)   # index 0..2*LEVELS
    fg = [f"\x1b[38;2;{r};{g_};{b}m" for r, g_, b in cmap]
    bg = [f"\x1b[48;2;{r};{g_};{b}m" for r, g_, b in cmap]
    return fg, bg


def resample_idx(n_src, n_dst):
    return np.minimum((np.arange(n_dst) + 0.5) * n_src / n_dst, n_src - 1).astype(int)


def render(frame, cols, rows, fg, bg, equator):
    py = rows * 2
    iy = resample_idx(frame.shape[0], py)[::-1]   # north at the top
    ix = resample_idx(frame.shape[1], cols)
    q = np.clip(np.rint(frame[np.ix_(iy, ix)] * LEVELS), -LEVELS, LEVELS).astype(int) + LEVELS
    if equator:
        q[py // 2, ::3] = 2 * LEVELS  # dotted equator in bright colour
    out = []
    for r in range(rows):
        top, bot = q[2 * r], q[2 * r + 1]
        line = [f"\x1b[{r + 1};1H"]
        lt = lb = -1
        for c in range(cols):
            t, b = top[c], bot[c]
            if t != lt:
                line.append(fg[t]); lt = t
            if b != lb:
                line.append(bg[b]); lb = b
            line.append("▀")
        out.append("".join(line))
    return "".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", default="waves.npz")
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--gain", type=float, default=1.0, help="amplitude multiplier")
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--snapshot", type=int, metavar="FRAME",
                   help="print one frame and exit (for testing)")
    a = p.parse_args()

    d = np.load(a.file)
    H = d["h"].astype(np.float32) * (a.gain / 127.0)   # int8, already normalised
    dt_frame = float(d["dt_frame"])
    fg, bg = colormap()

    if a.snapshot is not None:
        cols, rows = os.get_terminal_size() if sys.stdout.isatty() else (100, 30)
        sys.stdout.write(render(H[a.snapshot], cols, rows, fg, bg, False) + "\x1b[0m\n")
        return

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    w = sys.stdout.write
    w("\x1b[?1049h\x1b[?25l\x1b[2J")
    try:
        tty.setcbreak(fd)
        i, speed, paused = 0, 1, False
        hud, equator = not a.no_hud, True
        while True:
            t0 = time.monotonic()
            cols, rows = os.get_terminal_size()
            body_rows = rows - 1 if hud else rows
            w(render(H[int(i) % len(H)], cols, body_rows, fg, bg, equator))
            if hud:
                msg = (f" t = {(int(i) % len(H)) * dt_frame:7.1f}   speed x{speed}"
                       f"   Kelvin → east   ← Rossby west   [q space + - e h]")
                w(f"\x1b[{rows};1H\x1b[0m\x1b[2K{msg[:cols]}")
            sys.stdout.flush()
            if not paused:
                i += speed

            timeout = max(0.0, 1 / a.fps - (time.monotonic() - t0))
            if select.select([sys.stdin], [], [], timeout)[0]:
                k = sys.stdin.read(1)
                if k == "q":
                    break
                elif k == " ":
                    paused = not paused
                elif k in "+=":
                    speed = min(speed + 1, 10)
                elif k in "-_":
                    speed = max(speed - 1, 1)
                elif k == "e":
                    equator = not equator
                elif k == "h":
                    hud = not hud
                    w("\x1b[0m\x1b[2J")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        w("\x1b[0m\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
