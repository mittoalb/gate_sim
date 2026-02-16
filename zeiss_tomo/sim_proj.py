#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import h5py
import spekpy as sp
import opengate as gate
from scipy.spatial.transform import Rotation as R
import itk


# -----------------------------
# Safe ITK → NumPy (no numpy 2 issues)
# -----------------------------
def itk_to_numpy(img):
    return itk.array_from_image(img)


# -----------------------------
# Build ZEISS-like spectrum
# -----------------------------
def build_zeiss_spectrum(kvp, be_mm, al_mm, cu_mm):
    s = sp.Spek(
        kvp=kvp,
        th=12,      # microfocus typical target angle
        dk=0.5      # energy bin size keV
    )

    # inherent filtration
    if be_mm > 0:
        s.filter("Be", be_mm)
    if al_mm > 0:
        s.filter("Al", al_mm)
    if cu_mm > 0:
        s.filter("Cu", cu_mm)

    energies, fluence = s.get_spectrum()

    fluence = fluence / fluence.sum()

    return energies, fluence


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--angle", type=float, required=True)
    ap.add_argument("--outdir", default="proj_data")

    ap.add_argument("--kvp", type=float, default=120)
    ap.add_argument("--be-mm", type=float, default=0.3)
    ap.add_argument("--al-mm", type=float, default=1.0)
    ap.add_argument("--cu-mm", type=float, default=0.0)

    ap.add_argument("--sod-cm", type=float, default=40.0)
    ap.add_argument("--field-mm", type=float, default=120.0)

    ap.add_argument("--primaries", type=int, default=2_000_000)

    ap.add_argument("--grid", type=int, default=200)
    ap.add_argument("--vox-mm", type=float, default=1.0)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    m = gate.g4_units.m
    keV = gate.g4_units.keV

    # -----------------------------
    # Build spectrum
    # -----------------------------
    energies_keV, weights = build_zeiss_spectrum(
        args.kvp,
        args.be_mm,
        args.al_mm,
        args.cu_mm,
    )

    # -----------------------------
    # Create Simulation
    # -----------------------------
    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.visu = False
    sim.number_of_threads = 1
    sim.random_seed = 1234 + int(args.angle * 1000)

    sim.world.size = [1*m, 1*m, 1*m]
    sim.world.material = "G4_AIR"

    # -----------------------------
    # Scoring volume (fixed lab frame)
    # -----------------------------
    scoring = sim.add_volume("Box", "scoring_box")
    scoring.material = "G4_AIR"
    L = args.grid * args.vox_mm * mm
    scoring.size = [L, L, L]

    # -----------------------------
    # Water cylinder (vertical axis = Z)
    # -----------------------------
    cyl = sim.add_volume("Tubs", "water_cyl")
    cyl.mother = scoring.name
    cyl.material = "G4_WATER"
    cyl.rmin = 0*mm
    cyl.rmax = 50*mm
    cyl.dz = 50*mm

    # rotate cylinder
    rot = R.from_euler("z", args.angle, degrees=True).as_matrix()
    cyl.rotation = rot

    # -----------------------------
    # Physics
    # -----------------------------
    sim.physics_manager.physics_list_name = "G4EmLivermorePhysics"
    sim.physics_manager.enable_decay = False

    # -----------------------------
    # Source (side incidence)
    # -----------------------------
    src = sim.add_source("GenericSource", "xray")
    src.particle = "gamma"
    src.n = args.primaries

    src.position.type = "point"
    src.position.translation = [-args.sod_cm*cm, 0, 0]

    src.direction.type = "momentum"
    src.direction.momentum = [1, 0, 0]

    # optional rectangular field (if beam2d supported)
    try:
        src.direction.type = "beam2d"
        src.direction.momentum = [1, 0, 0]
        if hasattr(src.direction, "sigma"):
            src.direction.sigma = [
                args.field_mm*mm/4,
                args.field_mm*mm/4
            ]
    except Exception:
        pass

    # energy histogram
    src.energy.type = "histogram"
    src.energy.histogram_energy = (energies_keV * keV).tolist()
    src.energy.histogram_weight = weights.tolist()

    # -----------------------------
    # Dose Actor
    # -----------------------------
    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to = scoring.name
    dose.size = [args.grid, args.grid, args.grid]
    dose.spacing = [args.vox_mm*mm]*3
    dose.dose.active = True

    # optional TLE
    try:
        tle = sim.add_actor("TLEDoseActor", "tle")
        tle.attached_to = scoring.name
        tle.size = [args.grid]*3
        tle.spacing = [args.vox_mm*mm]*3
        tle.dose.active = True
    except Exception:
        tle = None

    # -----------------------------
    # Run
    # -----------------------------
    sim.run()

    # -----------------------------
    # Save HDF5
    # -----------------------------
    fname = outdir / f"dose_angle_{args.angle:07.2f}.h5"

    with h5py.File(fname, "w") as f:
        f["angle_deg"] = args.angle

        img = dose.dose.get_data()
        arr = itk_to_numpy(img)
        f["dose_Gy"] = arr

        if tle is not None:
            timg = tle.dose.get_data()
            f["tle_dose_Gy"] = itk_to_numpy(timg)

        f["spectrum_energy_keV"] = energies_keV
        f["spectrum_weight"] = weights

    print(f"[OK] Projection {args.angle:.2f}° saved to {fname}")


if __name__ == "__main__":
    main()
