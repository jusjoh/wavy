#!/usr/bin/env python3
"""Parallel parameter scan for the ENSO model.

    python -m wavy.enso_scan OUTDIR '{"mu":[0.38,0.42],"years":[40]}'

Runs every combination, prints diagnostics, saves run_*.npz + run_*.png into OUTDIR.
"""
import sys, json, itertools
from multiprocessing import Pool
from wavy.enso import Config, Model, diagnostics, plot
import numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

def job(kw):
    c = Config(**kw)
    out = Model(c).run()
    d = diagnostics(out)
    tag = "_".join(f"{k}{v}" for k, v in kw.items() if k not in ("years",))
    np.savez(f"{OUT}/run_{tag}.npz", **out)
    plot(out, f"{OUT}/run_{tag}.png", title=str(kw))
    return kw, d

if __name__ == "__main__":

    grid = json.loads(sys.argv[2])
    keys = list(grid)
    jobs = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]
    with Pool(min(len(jobs), 16)) as p:
        for kw, d in p.imap_unordered(job, jobs):
            print(json.dumps(kw), {k: round(v, 3) for k, v in d.items()}, flush=True)
