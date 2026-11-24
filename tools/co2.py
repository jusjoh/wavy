"""Estimate the CO2 emitted by the Claude Code sessions spent on this project.

    python3 tools/co2.py [--grid 370] [--by-day] [--also DIR ...]

Token counts are read from Claude Code's local logs (~/.claude/projects), which
it may delete after a while. Nothing is sent anywhere.

The CO2 figure is an order-of-magnitude estimate. Anthropic publishes no energy
per token; the rates are third-party guesses for another model (GPT-4o) scaled by
price ratios, applied to every model in the logs:
https://www.simonpcouch.com/blog/2026-01-20-cc-impact/
"high" is the set used there; "low" is its medium-context variant for input and
output, with the cache rates scaled the same way. Not included: model training,
hardware manufacturing, your own computer.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# the project's folder before it was renamed
OLD_DIRS = [ROOT.parent / "terminal-wave"]

# kind: (field in the log, low, high) in Wh per million tokens
KINDS = {
    "cache reads": ("cache_read_input_tokens", 20, 39),
    "cache writes": ("cache_creation_input_tokens", 250, 490),
    "output": ("output_tokens", 990, 1950),
    "uncached input": ("input_tokens", 200, 390),
}


def log_dir(path):
    """Claude Code's log folder for a project folder."""
    base = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
    return base / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(path))


def read_calls(folders):
    """{message id: (day, {kind: tokens})}. A call is logged several times while
    it streams; the entry with the most output tokens is the final one."""
    calls = {}
    for folder in folders:
        for file in folder.rglob("*.jsonl"):  # includes subagent logs
            with file.open(errors="replace") as lines:
                for n, line in enumerate(lines):
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(entry, dict) or entry.get("type") != "assistant":
                        continue
                    msg = entry.get("message")
                    usage = msg.get("usage") if isinstance(msg, dict) else None
                    if not usage:
                        continue
                    tokens = {k: usage.get(f) or 0 for k, (f, _, _) in KINDS.items()}
                    key = msg.get("id") or f"{file}:{n}"
                    if key not in calls or tokens["output"] > calls[key][1]["output"]:
                        calls[key] = ((entry.get("timestamp") or "")[:10], tokens)
    return calls


def energy(tokens):
    """(low, high) in Wh."""
    return tuple(
        sum(tokens[k] * KINDS[k][i] for k in KINDS) / 1e6 for i in (1, 2)
    )


def add_up(token_dicts):
    total = dict.fromkeys(KINDS, 0)
    for tokens in token_dicts:
        for k in KINDS:
            total[k] += tokens[k]
    return total


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grid", type=float, default=370, metavar="G",
                    help="grid intensity in g CO2/kWh (default: 370, about the US average)")
    ap.add_argument("--by-day", action="store_true", help="also list each day")
    ap.add_argument("--also", type=Path, nargs="+", default=[], metavar="DIR",
                    help="count the sessions of these project folders too")
    args = ap.parse_args()

    folders = {log_dir(p) for p in [ROOT, *OLD_DIRS, *(d.resolve() for d in args.also)]}
    calls = read_calls(folders)
    if not calls:
        raise SystemExit("no Claude Code logs found for this project")

    days = defaultdict(list)
    for day, tokens in calls.values():
        days[day or "unknown"].append(tokens)

    def kg(wh):
        return wh / 1000 * args.grid / 1000

    total = add_up(t for _, t in calls.values())
    print(f"{len(calls)} model calls, {min(days)} to {max(days)}\n")
    print(f"{'':16}{'tokens':>14}{'low Wh':>10}{'high Wh':>10}")
    for k, (_, lo, hi) in KINDS.items():
        n = total[k]
        print(f"{k:16}{n:>14,}{n * lo / 1e6:>10.1f}{n * hi / 1e6:>10.1f}")
    lo, hi = energy(total)
    print(f"{'total':16}{sum(total.values()):>14,}{lo:>10.1f}{hi:>10.1f}")

    if args.by_day:
        print(f"\n{'day':16}{'tokens':>14}{'low kg':>10}{'high kg':>10}")
        for day in sorted(days):
            t = add_up(days[day])
            dlo, dhi = energy(t)
            print(f"{day:16}{sum(t.values()):>14,}{kg(dlo):>10.3f}{kg(dhi):>10.3f}")

    print(f"\nCO2 at {args.grid:g} g/kWh: {kg(lo):.2f} to {kg(hi):.2f} kg "
          "(rough: could be several times off)")


if __name__ == "__main__":
    main()
