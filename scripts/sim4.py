#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import opengate as gate


def try_add_actor(sim: gate.Simulation, actor_type: str, name: str):
    try:
        a = sim.add_actor(actor_type, name)
        print(f"[OK] added actor: {actor_type} -> {name}")
        return a
    except Exception as e:
        print(f"[SKIP] cannot add actor {actor_type}: {e}")
        return None


def itk_to_numpy(itk_img) -> np.ndarray:
    """
    NumPy 2.0-safe conversion. Avoid np.asarray(itk_img) which can pass copy=False.
    """
    try:
        return np.array(itk_img)
    except Exception:
        if hasattr(gate, "image_to_numpy"):
            return gate.image_to_numpy(itk_img)
        import itk  # type: ignore
        return itk.array_from_image(itk_img)


def write_dataset_with_meta(group: h5py.Group, name: str, itk_img):
    arr = itk_to_numpy(itk_img)
    if name in group:
        del group[name]
    ds = group.create_dataset(
        name,
        data=arr,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    ds.attrs["origin"] = tuple(float(x) for x in itk_img.GetOrigin())
    ds.attrs["spacing"] = tuple(float(x) for x in itk_img.GetSpacing())
    d = np.array(itk_img.GetDirection())
    if d.size == 9:
        d = d.reshape(3, 3)
    ds.attrs["direction"] = d
    ds.attrs["axis_order"] = "z,y,x"
    ds.attrs["dtype"] = str(arr.dtype)
    return arr


def plot_spectrum(energies_keV: np.ndarray, weights: np.ndarray, outdir: Path, show: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    w = weights / weights.sum()
    cdf = np.cumsum(w)

    fig = plt.figure()
    plt.plot(energies_keV, w, marker="o")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Relative weight (normalized)")
    plt.title("X-ray spectrum")
    plt.grid(True)
    plt.tight_layout()
    p1 = outdir / "spectrum_weights.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(energies_keV, cdf, marker="o")
    plt.xlabel("Energy (keV)")
    plt.ylabel("CDF")
    plt.title("X-ray spectrum (CDF)")
    plt.grid(True)
    plt.tight_layout()
    p2 = outdir / "spectrum_cdf.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    print(f"[OK] Saved spectrum plots:\n  {p1.resolve()}\n  {p2.resolve()}")

    if show:
        # Interactive view (needs GUI backend)
        plt.figure()
        plt.plot(energies_keV, w, marker="o")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Relative weight")
        plt.title("X-ray spectrum (interactive)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-plot", action="store_true")
    ap.add_argument("--n", type=int, default=2_000_000, help="primaries")
    ap.add_argument("--field-x-mm", type=float, default=120.0, help="beam width at isocenter (mm)")
    ap.add_argument("--field-y-mm", type=float, default=120.0, help="beam height at isocenter (mm)")
    args = ap.parse_args()

    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Spectrum (replace with your real tube spectrum)
    # -----------------------------
    energies_keV = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=float)
    weights = np.array([0.01, 0.05, 0.12, 0.22, 0.35, 0.45, 0.42, 0.35, 0.25, 0.15, 0.08, 0.03], dtype=float)
    weights /= weights.sum()

    plot_spectrum(energies_keV, weights, outdir, show=args.show_plot)

    # -----------------------------
    # Simulation
    # -----------------------------
    sim = gate.Simulation()
    sim.output_dir = str(outdir)
    sim.g4_verbose = False
    sim.visu = False
    sim.random_seed = 12345
    sim.number_of_threads = 1

    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    m = gate.g4_units.m
    keV = gate.g4_units.keV

    # World
    sim.world.size = [1 * m, 1 * m, 1 * m]
    sim.world.material = "G4_AIR"

    # Water cylinder (radius 5 cm, height 10 cm)
    cyl = sim.add_volume("Tubs", "water_cyl")
    cyl.material = "G4_WATER"
    cyl.rmin = 0 * mm
    cyl.rmax = 50 * mm
    cyl.dz = 50 * mm
    cyl.translation = [0, 0, 0]

    # Aluminum filter slab (1 mm Al)
    filt = sim.add_volume("Box", "Al_filter")
    filt.material = "G4_Al"
    filt.size = [12 * cm, 12 * cm, 1 * mm]
    filt.translation = [0, 0, -20 * cm]

    # Physics
    sim.physics_manager.physics_list_name = "G4EmLivermorePhysics"
    sim.physics_manager.enable_decay = False

    # -----------------------------
    # Source
    # -----------------------------
    src = sim.add_source("GenericSource", "xray")
    src.particle = "gamma"
    src.n = int(args.n)

    # point source 40 cm upstream
    src.position.type = "point"
    src.position.translation = [0, 0, -40 * cm]

    # Beam direction: beam2d (AVAILABLE in your build)
    # This produces a rectangular field centered on the +z axis.
    src.direction.type = "beam2d"

    # Parameter names can vary slightly across opengate versions.
    # We'll set the common ones, and fall back if a name is different.
    fx = float(args.field_x_mm) * mm
    fy = float(args.field_y_mm) * mm

    # Try common field names for beam2d:
    set_ok = False
    for (attrx, attry) in [
        ("field_size", None),          # some versions: field_size = [x,y]
        ("size", None),                # size = [x,y]
        ("sigma", None),               # sigma = [sx,sy] (Gaussian beam)
        ("aperture", None),            # aperture = [x,y]
        ("rectangle", None),           # rectangle = [x,y]
    ]:
        if hasattr(src.direction, attrx):
            try:
                if attrx in ("field_size", "size", "aperture", "rectangle"):
                    setattr(src.direction, attrx, [fx, fy])
                    set_ok = True
                    break
                if attrx == "sigma":
                    # if it's Gaussian, use ~field/4 as sigma (rough)
                    setattr(src.direction, attrx, [fx / 4.0, fy / 4.0])
                    set_ok = True
                    break
            except Exception:
                pass

    # Also set axis if supported
    if hasattr(src.direction, "momentum"):
        src.direction.momentum = [0, 0, 1]

    if not set_ok:
        # If your beam2d uses different property names, we still don't want zeros:
        # fallback to a guaranteed pencil beam.
        print("[WARN] Could not set beam2d field parameters on this build; falling back to momentum pencil beam.")
        src.direction.type = "momentum"
        src.direction.momentum = [0, 0, 1]

    # Energy histogram
    src.energy.type = "histogram"
    src.energy.histogram_energy = (energies_keV * keV).tolist()
    src.energy.histogram_weight = weights.tolist()

    # -----------------------------
    # DoseActor
    # -----------------------------
    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to = cyl.name
    dose.size = [160, 160, 160]
    dose.spacing = [1 * mm, 1 * mm, 1 * mm]

    # activate outputs
    if hasattr(dose, "dose") and hasattr(dose.dose, "active"):
        dose.dose.active = True
    if hasattr(dose, "dose_uncertainty") and hasattr(dose.dose_uncertainty, "active"):
        dose.dose_uncertainty.active = True

    # native .mhd/.raw
    if hasattr(dose, "edep") and hasattr(dose.edep, "output_filename"):
        dose.edep.output_filename = "dose-edep.mhd"
    if hasattr(dose, "dose") and hasattr(dose.dose, "output_filename"):
        dose.dose.output_filename = "dose-dose_Gy.mhd"
    if hasattr(dose, "dose_uncertainty") and hasattr(dose.dose_uncertainty, "output_filename"):
        dose.dose_uncertainty.output_filename = "dose-dose_uncertainty.mhd"

    # TLE
    tle = try_add_actor(sim, "TLEDoseActor", "tle_dose")
    if tle is not None:
        tle.attached_to = cyl.name
        tle.size = [160, 160, 160]
        tle.spacing = [1 * mm, 1 * mm, 1 * mm]
        if hasattr(tle, "dose") and hasattr(tle.dose, "active"):
            tle.dose.active = True
        if hasattr(tle, "dose_uncertainty") and hasattr(tle.dose_uncertainty, "active"):
            tle.dose_uncertainty.active = True

    # -----------------------------
    # Run
    # -----------------------------
    sim.run()

    # -----------------------------
    # Export HDF5
    # -----------------------------
    h5_path = outdir / "results.h5"
    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["opengate_version"] = getattr(gate, "__version__", "unknown")
        meta.attrs["n_primaries"] = int(src.n)
        meta.attrs["direction_type"] = getattr(src.direction, "type", "unknown")
        meta.create_dataset("spectrum_energy_keV", data=energies_keV)
        meta.create_dataset("spectrum_weight", data=weights)

        g = f.create_group("DoseActor")

        edep_img = dose.edep.get_data()
        edep_arr = write_dataset_with_meta(g, "edep", edep_img)
        print("DoseActor EDEP sanity:",
              "sum=", float(edep_arr.sum()),
              "min=", float(edep_arr.min()),
              "max=", float(edep_arr.max()))

        if hasattr(dose, "dose"):
            try:
                dose_img = dose.dose.get_data()
                dose_arr = write_dataset_with_meta(g, "dose_Gy", dose_img)
                print("DoseActor DOSE sanity:",
                      "sum=", float(dose_arr.sum()),
                      "min=", float(dose_arr.min()),
                      "max=", float(dose_arr.max()))
            except Exception as e:
                print(f"[WARN] Dose export failed: {e}")

        if hasattr(dose, "dose_uncertainty"):
            try:
                unc_img = dose.dose_uncertainty.get_data()
                unc_arr = write_dataset_with_meta(g, "dose_uncertainty", unc_img)
                mask = edep_arr > 0
                if np.any(mask):
                    print("UNC (edep>0 voxels): mean=", float(unc_arr[mask].mean()),
                          "median=", float(np.median(unc_arr[mask])),
                          "max=", float(unc_arr[mask].max()))
            except Exception as e:
                print(f"[WARN] UNC export failed: {e}")

        if tle is not None:
            gt = f.create_group("TLEDoseActor")
            if hasattr(tle, "edep"):
                img = tle.edep.get_data()
                arr = write_dataset_with_meta(gt, "edep", img)
                print("TLE EDEP sanity:",
                      "sum=", float(arr.sum()),
                      "min=", float(arr.min()),
                      "max=", float(arr.max()))
            if hasattr(tle, "dose"):
                img = tle.dose.get_data()
                arr = write_dataset_with_meta(gt, "dose_Gy", img)
                print("TLE DOSE sanity:",
                      "sum=", float(arr.sum()),
                      "min=", float(arr.min()),
                      "max=", float(arr.max()))

    print("\nDone.")
    print(f"Native maps (.mhd/.raw): {outdir.resolve()}")
    print(f"HDF5: {h5_path.resolve()}")
    print(f"Spectrum plots:\n  {(outdir/'spectrum_weights.png').resolve()}\n  {(outdir/'spectrum_cdf.png').resolve()}")


if __name__ == "__main__":
    main()

