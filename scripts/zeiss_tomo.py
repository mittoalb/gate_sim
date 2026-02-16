#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

import opengate as gate
from scipy.spatial.transform import Rotation as R


# -----------------------------
# NumPy 2.0-safe ITK->NumPy
# -----------------------------
def itk_to_numpy(itk_img) -> np.ndarray:
    import itk  # type: ignore
    return itk.array_from_image(itk_img)  # <- avoids __array__(copy=...) issues


def plot_spectrum(energies_keV: np.ndarray, weights: np.ndarray, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    w = weights / weights.sum()
    cdf = np.cumsum(w)

    fig = plt.figure()
    plt.plot(energies_keV, w, marker="o")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Relative weight")
    plt.title("X-ray spectrum")
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(outdir / "spectrum_weights.png", dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(energies_keV, cdf, marker="o")
    plt.xlabel("Energy (keV)")
    plt.ylabel("CDF")
    plt.title("X-ray spectrum (CDF)")
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(outdir / "spectrum_cdf.png", dpi=150)
    plt.close(fig)


def configure_beam2d_or_fallback(src, fy, fz) -> str:
    # Your build: ['iso','histogram','momentum','focused','beam2d']
    try:
        src.direction.type = "beam2d"
        if hasattr(src.direction, "momentum"):
            src.direction.momentum = [1, 0, 0]  # +x

        for attr in ("field_size", "size", "aperture", "rectangle"):
            if hasattr(src.direction, attr):
                setattr(src.direction, attr, [fy, fz])
                return f"beam2d:{attr}"

        if hasattr(src.direction, "sigma"):
            src.direction.sigma = [fy / 4.0, fz / 4.0]
            return "beam2d:sigma"

        raise AttributeError("beam2d parameters not found")
    except Exception:
        src.direction.type = "momentum"
        src.direction.momentum = [1, 0, 0]
        return "momentum"


def set_first_existing(obj, candidates: list[tuple[str, object]]) -> str:
    """
    Try setting (attr=value) for the first attribute that exists.
    Returns the attribute name that worked; raises if none match.
    """
    for attr, value in candidates:
        if hasattr(obj, attr):
            try:
                setattr(obj, attr, value)
                return attr
            except Exception:
                pass
    raise AttributeError(
        "Could not set any of: " + ", ".join(a for a, _ in candidates)
    )


def main():
    ap = argparse.ArgumentParser()

    # tomography
    ap.add_argument("--nproj", type=int, default=360)
    ap.add_argument("--start-deg", type=float, default=0.0)
    ap.add_argument("--step-deg", type=float, default=1.0)

    # “per projection” exposure control
    ap.add_argument("--primaries-per-proj", type=float, default=2e6,
                    help="Target mean primaries per run (Poisson around this).")
    ap.add_argument("--run-dt-s", type=float, default=1.0,
                    help="Each run duration (s). primaries/run ~ activity * dt.")

    # geometry
    ap.add_argument("--sod-cm", type=float, default=40.0)
    ap.add_argument("--field-y-mm", type=float, default=120.0)
    ap.add_argument("--field-z-mm", type=float, default=120.0)

    # cylinder (vertical axis = Z)
    ap.add_argument("--cyl-radius-mm", type=float, default=50.0)
    ap.add_argument("--cyl-height-mm", type=float, default=100.0)
    ap.add_argument("--cyl-offset-y-mm", type=float, default=0.0)
    ap.add_argument("--cyl-offset-z-mm", type=float, default=0.0)

    # scoring
    ap.add_argument("--grid", type=int, default=200)
    ap.add_argument("--vox-mm", type=float, default=1.0)

    # spectrum placeholder (swap later for real Versa spectrum)
    ap.add_argument("--kvp", type=float, default=120.0)

    # output
    ap.add_argument("--outdir", default="output_tomo")

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # units
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    m = gate.g4_units.m
    keV = gate.g4_units.keV
    s = gate.g4_units.s

    # ------------- spectrum (placeholder shape) -------------
    kvp = float(args.kvp)
    energies_keV = np.arange(5.0, kvp + 0.001, 1.0)
    raw = energies_keV * np.maximum(kvp - energies_keV, 0.0)
    raw *= np.exp(-0.02 * energies_keV)
    raw[raw < 0] = 0
    weights = raw / raw.sum()

    plot_spectrum(energies_keV, weights, outdir)
    print(f"[OK] Spectrum plots: {(outdir/'spectrum_weights.png').resolve()}")

    # ------------- single simulation, multiple runs -------------
    sim = gate.Simulation()
    sim.output_dir = str(outdir)
    sim.g4_verbose = False
    sim.visu = False
    sim.number_of_threads = 1
    sim.random_seed = 12345

    # multiple runs via timing intervals (fast; init once) :contentReference[oaicite:3]{index=3}
    dt = float(args.run_dt_s) * s
    sim.run_timing_intervals = [[i * dt, (i + 1) * dt] for i in range(int(args.nproj))]

    # world
    sim.world.size = [1 * m, 1 * m, 1 * m]
    sim.world.material = "G4_AIR"

    # scoring volume (fixed lab frame)
    scoring = sim.add_volume("Box", "scoring_box")
    scoring.material = "G4_AIR"
    box_len = int(args.grid) * float(args.vox_mm) * mm
    scoring.size = [box_len, box_len, box_len]
    scoring.translation = [0, 0, 0]

    # cylinder is a DAUGHTER of scoring_box => no overlaps :contentReference[oaicite:4]{index=4}
    cyl = sim.add_volume("Tubs", "water_cyl")
    cyl.mother = scoring.name
    cyl.material = "G4_WATER"
    cyl.rmin = 0 * mm
    cyl.rmax = float(args.cyl_radius_mm) * mm
    cyl.dz = (float(args.cyl_height_mm) * mm) / 2.0
    cyl.translation = [0, float(args.cyl_offset_y_mm) * mm, float(args.cyl_offset_z_mm) * mm]

    # physics
    sim.physics_manager.physics_list_name = "G4EmLivermorePhysics"
    sim.physics_manager.enable_decay = False

    # source: side incidence along +x
    x_source = -float(args.sod_cm) * cm
    src = sim.add_source("GenericSource", "xray")
    src.particle = "gamma"
    src.position.type = "point"
    src.position.translation = [x_source, 0, 0]

    dir_mode = configure_beam2d_or_fallback(
        src,
        float(args.field_y_mm) * mm,
        float(args.field_z_mm) * mm,
    )
    print(f"[INFO] direction = {dir_mode}")

    src.energy.type = "histogram"
    src.energy.histogram_energy = (energies_keV * keV).tolist()
    src.energy.histogram_weight = weights.tolist()

    # Multi-run support: use activity (docs warn fixed n + multi-run is limited) :contentReference[oaicite:5]{index=5}
    # Mean primaries per run ≈ activity * dt
    src.activity = float(args.primaries_per_proj) / float(args.run_dt_s)

    # dose actor attached to scoring_box (lab frame)
    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to = scoring.name
    dose.size = [int(args.grid), int(args.grid), int(args.grid)]
    vox = float(args.vox_mm) * mm
    dose.spacing = [vox, vox, vox]
    if hasattr(dose, "dose") and hasattr(dose.dose, "active"):
        dose.dose.active = True

    # optional TLE
    tle = None
    try:
        tle = sim.add_actor("TLEDoseActor", "tle_dose")
        tle.attached_to = scoring.name
        tle.size = [int(args.grid), int(args.grid), int(args.grid)]
        tle.spacing = [vox, vox, vox]
        if hasattr(tle, "dose") and hasattr(tle.dose, "active"):
            tle.dose.active = True
        print("[OK] TLEDoseActor enabled")
    except Exception as e:
        print(f"[SKIP] TLEDoseActor not available: {e}")

    # ---------- Dynamic geometry: rotate cylinder each run ----------
    # MotionVolumeActor is not in your build; DynamicGeometryActor is.
    dyn = sim.add_actor("DynamicGeometryActor", "dyn_geom")

    rotations = [
        R.from_euler("z", float(args.start_deg) + i * float(args.step_deg), degrees=True).as_matrix()
        for i in range(int(args.nproj))
    ]

    translations = [cyl.translation for _ in range(int(args.nproj))]

    # Different OpenGATE versions expose different parameter names; we set whatever exists.
    # 1) which volumes?
    vol_attr = set_first_existing(
        dyn,
        [
            ("volumes", [cyl.name]),
            ("volume_names", [cyl.name]),
            ("volumes_to_move", [cyl.name]),
            ("targets", [cyl.name]),
        ],
    )

    # 2) per-run transforms
    rot_attr = set_first_existing(
        dyn,
        [
            ("rotations", rotations),
            ("rotation_matrices", rotations),
            ("rotation", rotations),
        ],
    )
    trans_attr = set_first_existing(
        dyn,
        [
            ("translations", translations),
            ("translation", translations),
        ],
    )

    # 3) which run/time index? Some versions want the list of run intervals or time points.
    # If this fails, it often still works because DynamicGeometryActor defaults to "each run".
    try:
        time_attr = set_first_existing(
            dyn,
            [
                ("run_timing_intervals", sim.run_timing_intervals),
                ("timing_intervals", sim.run_timing_intervals),
                ("times", [float(i) * float(args.run_dt_s) for i in range(int(args.nproj))]),
                ("time_points", [float(i) * float(args.run_dt_s) for i in range(int(args.nproj))]),
            ],
        )
    except Exception:
        time_attr = "(default)"

    print(f"[OK] DynamicGeometryActor configured: {vol_attr=}, {rot_attr=}, {trans_attr=}, {time_attr=}")

    # ---------- run ONCE ----------
    sim.run()

    # ---------- export HDF5 ----------
    h5_path = outdir / "tomo_accumulated.h5"
    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["nproj"] = int(args.nproj)
        meta.attrs["start_deg"] = float(args.start_deg)
        meta.attrs["step_deg"] = float(args.step_deg)
        meta.attrs["primaries_per_proj_target"] = float(args.primaries_per_proj)
        meta.attrs["run_dt_s"] = float(args.run_dt_s)
        meta.attrs["direction_mode"] = dir_mode
        meta.create_dataset("spectrum_energy_keV", data=energies_keV)
        meta.create_dataset("spectrum_weight", data=weights)

        g = f.create_group("DoseActor")
        edep_img = dose.edep.get_data()
        edep = itk_to_numpy(edep_img)
        g.create_dataset("edep", data=edep, compression="gzip", compression_opts=4, shuffle=True)

        if hasattr(dose, "dose"):
            dimg = dose.dose.get_data()
            darr = itk_to_numpy(dimg)
            g.create_dataset("dose_Gy", data=darr, compression="gzip", compression_opts=4, shuffle=True)

        if tle is not None and hasattr(tle, "dose"):
            gt = f.create_group("TLEDoseActor")
            timg = tle.dose.get_data()
            tarr = itk_to_numpy(timg)
            gt.create_dataset("dose_Gy", data=tarr, compression="gzip", compression_opts=4, shuffle=True)

    print("\nDone.")
    print(f"HDF5: {h5_path.resolve()}")


if __name__ == "__main__":
    main()
