#!/usr/bin/env python3
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import rotate
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="proj_data")
    ap.add_argument("--outfile", default="accumulated.h5")
    ap.add_argument("--order", type=int, default=1,
                    help="Interpolation order (0=nearest,1=linear,3=cubic)")
    args = ap.parse_args()

    data_dir = Path(args.indir)
    files = sorted(data_dir.glob("dose_angle_*.h5"))

    if len(files) == 0:
        print("No files found.")
        return

    sum_dose = None

    for f in files:
        with h5py.File(f, "r") as h:
            dose = h["dose_Gy"][:]
            angle = float(h["angle_deg"][()])

        # rotate back into reference orientation
        # rotation axis = Z → rotate in X-Y plane
        dose_rot = rotate(
            dose,
            angle=-angle,
            axes=(2, 1),      # (x,y) plane
            reshape=False,
            order=args.order
        )

        if sum_dose is None:
            sum_dose = np.zeros_like(dose_rot)

        sum_dose += dose_rot

        print(f"Accumulated angle {angle:.2f}°")

    with h5py.File(args.outfile, "w") as f:
        f["dose_Gy"] = sum_dose

    print(f"\nSaved accumulated dose to {args.outfile}")


if __name__ == "__main__":
    main()
