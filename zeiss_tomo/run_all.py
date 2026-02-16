#!/usr/bin/env python3
import multiprocessing as mp
import subprocess
import numpy as np
from pathlib import Path

# -----------------------------
# SETTINGS
# -----------------------------
NPROJ = 360
START = 0
STOP = 359
NWORKERS = 8        # adjust to CPU cores

# optional simulation parameters
KVP = 120
PRIMARIES = 2_000_000

OUTDIR = "proj_data"


def run_angle(angle):
    cmd = [
        "python",
        "sim_proj.py",
        "--angle", f"{angle}",
        "--kvp", str(KVP),
        "--primaries", str(PRIMARIES),
        "--outdir", OUTDIR
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    angles = np.linspace(START, STOP, NPROJ)

    Path(OUTDIR).mkdir(exist_ok=True)

    with mp.Pool(processes=NWORKERS) as pool:
        pool.map(run_angle, angles)

    print("All projections completed.")
