#!/usr/bin/env python3
"""Turn captured wavy terminal output into PNG images (for checking visuals
without a real terminal, e.g. from an agent or CI).

The input is raw terminal output: a `--snapshot` dump, or a `script` log of a
live run (the last complete frame before the screen is restored is used). The
parser understands the subset of ANSI that wavy emits: cursor positioning
ESC[r;cH, truecolor ESC[38;2;r;g;bm / ESC[48;2;r;g;bm, reset ESC[0m, the
upper half block '▀' (fg = top pixel, bg = bottom pixel) and spaces.
Default-background cells come out white.

usage:
    uv run --with matplotlib python tools/screen2png.py OUT.png COLS ROWS FILE...

example (snapshot in a pseudo-terminal of known size):
    script -qfec "stty cols 160 rows 50; wavy/target/release/wavy \\
        --mode wbc --seed 1 --snapshot 35" /tmp/wbc.raw
    uv run --with matplotlib python tools/screen2png.py /tmp/wbc.png 160 50 /tmp/wbc.raw
"""
import re
import sys

import numpy as np

TOKEN = re.compile(
    r"\x1b\[(\d+);(\d+)H"                    # cursor position
    r"|\x1b\[(38|48);2;(\d+);(\d+);(\d+)m"   # truecolor fg/bg
    r"|\x1b\[0m"                              # reset
    r"|▀"                                     # half block
    r"|(.)",                                  # anything else (spaces count)
    re.S,
)


def last_frame(txt: str) -> str:
    """For `script` logs of live runs: keep only the last frame before exit."""
    end = txt.rfind("\x1b[?1049l")
    if end < 0:
        return txt
    start = txt.rfind("\x1b[1;", 0, end)
    return txt[max(start, 0):end]


def to_image(txt: str, cols: int, rows: int) -> np.ndarray:
    white = (255, 255, 255)
    img = np.full((2 * rows, cols, 3), 255, np.uint8)
    row = col = 0
    fg = bg = white
    for m in TOKEN.finditer(txt):
        tok = m.group(0)
        if m.group(1):
            row, col = int(m.group(1)) - 1, int(m.group(2)) - 1
        elif m.group(3):
            c = tuple(int(x) for x in m.groups()[3:6])
            if m.group(3) == "38":
                fg = c
            else:
                bg = c
        elif tok == "\x1b[0m":
            fg = bg = white
        elif tok == "▀":
            if row < rows and col < cols:
                img[2 * row, col], img[2 * row + 1, col] = fg, bg
            col += 1
        elif tok == " ":
            if row < rows and col < cols:
                img[2 * row, col] = img[2 * row + 1, col] = bg
            col += 1
        elif tok == "\n":
            row, col = row + 1, 0
        elif tok.isprintable():
            col += 1
    return img


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(sys.argv) < 5:
        sys.exit(__doc__)
    out, cols, rows, paths = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 3.4), squeeze=False)
    for ax, p in zip(axs.flat, paths):
        txt = last_frame(open(p, encoding="utf-8", errors="replace").read())
        ax.imshow(to_image(txt, cols, rows), aspect="equal", interpolation="nearest")
        ax.set_title(p.rsplit("/", 1)[-1], fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=70)


if __name__ == "__main__":
    main()
