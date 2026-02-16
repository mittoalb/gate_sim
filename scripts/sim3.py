#!/usr/bin/env python3
from __future__ import annotations

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
    # OpenGATE commonly supports np.asarray(itk_image)
    try:
        return np.asarray(itk_img)
    except Exception:
        if hasattr(gate, "image_to_numpy"):
            return gate.image_to_numpy(itk_img)
        try:
            import itk  # type: ignore
            return itk.array_from_image(itk_img)
        except Exception as e:
            raise RuntimeError("Cannot convert ITK image to numpy array") from e


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


def plot_spectrum(energies_keV: np.ndarray, weights: np.ndarray, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    w = weights / weights.sum()
    cdf = np.cumsum(w)

    # Plot weights
    plt.figure()
    plt.plot(energies_keV, w, marker="o")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Relative weight (normalized)")
    plt.title("X-ray spectrum (input histogram)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "spectrum_weights.png", dpi=150)
    plt.close()

    # Plot cumulative
    plt.figure()
    plt.plot(energies_keV, cdf, marker="o")
    plt.xlabel("Energy (keV)")
    plt.ylabel("CDF")
    plt.title("X-ray spectrum cumulative distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "spectrum_cdf.png", dpi=150)
    plt.close()

    print(f"[OK] saved spectrum plots in {outdir.resolve()}")


def main():
    # -----------------------------
    # Define spectrum (replace with your real tube spectrum if you want)
    # -----------------------------
    energies_keV = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=float)
    weights = np.array([0.01, 0.05, 0.12, 0.22, 0.35, 0.45, 0.42, 0.35, 0.25, 0.15, 0.08, 0.03], dtype=float)
    weights /= weights.sum()

    # Plot first (as you requested)
    plot_spectrum(energies_keV, weights, Path("output"))

    # -----------------------------
    # Create simulation
    # -----------------------------
    sim = gate.Simulation()
    sim.output_dir = "output"
    Path(sim.output_dir).mkdir(parents=True, exist_ok=True)

    sim.g4_verbose = False
    sim.visu = False
    sim.random_seed = 12345
    sim.number_of_threads = 1

    # -----------------------------
    # Units
    # -----------------------------
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    m = gate.g4_units.m
    keV = gate.g4_units.keV

    # -----------------------------
    # World
    # -----------------------------
    sim.world.size = [1 * m, 1 * m, 1 * m]
    sim.world.material = "G4_AIR"

    # -----------------------------
    # Water cylinder phantom (Geant4 Tubs)
    # radius = 5 cm, height = 10 cm (along z)
    # -----------------------------
    cyl = sim.add_volume("Tubs", "water_cyl")
    cyl.material = "G4_WATER"

    # OpenGATE Tubs typically uses: rmin, rmax, dz (half-length)
    cyl.rmin = 0 * mm
    cyl.rmax = 50 * mm          # 5 cm radius
    cyl.dz = 50 * mm            # half-length => total height 10 cm
    cyl.translation = [0, 0, 0]

    # -----------------------------
    # Physics
    # -----------------------------
    sim.physics_manager.physics_list_name = "G4EmLivermorePhysics"
    sim.physics_manager.enable_decay = False

    # -----------------------------
    # Source: polychromatic X-ray beam
    # Put source upstream, shoot toward +z
    # -----------------------------
    src = sim.add_source("GenericSource", "xray")
    src.particle = "gamma"
    src.n = 2_000_000

    src.position.type = "point"
    src.position.translation = [0, 0, -40 * cm]

    # Force direction: +z
    src.direction.type = "momentum"
    src.direction.momentum = [0, 0, 1]

    src.energy.type = "histogram"
    src.energy.histogram_energy = (energies_keV * keV).tolist()
    src.energy.histogram_weight = weights.tolist()

    # -----------------------------
    # DoseActor on cylinder
    # -----------------------------
    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to = cyl.name

    # Use a box-shaped scoring grid that encloses the cylinder.
    # If your version supports "size/spacing" like this, it will work.
    dose.size = [128, 128, 128]
    dose.spacing = [1 * mm, 1 * mm, 1 * mm]

    # Activate dose + uncertainty (often OFF by default)
    if hasattr(dose, "dose") and hasattr(dose.dose, "active"):
        dose.dose.active = True
    if hasattr(dose, "dose_uncertainty") and hasattr(dose.dose_uncertainty, "active"):
        dose.dose_uncertainty.active = True

    # Native output (.mhd/.raw)
    if hasattr(dose, "edep") and hasattr(dose.edep, "output_filename"):
        dose.edep.output_filename = "dose-edep.mhd"
    if hasattr(dose, "dose") and hasattr(dose.dose, "output_filename"):
        dose.dose.output_filename = "dose-dose_Gy.mhd"
    if hasattr(dose, "dose_uncertainty") and hasattr(dose.dose_uncertainty, "output_filename"):
        dose.dose_uncertainty.output_filename = "dose-dose_uncertainty.mhd"

    # Optional TLE dose (if available)
    tle = try_add_actor(sim, "TLEDoseActor", "tle_dose")
    if tle is not None:
        tle.attached_to = cyl.name
        tle.size = [128, 128, 128]
        tle.spacing = [1 * mm, 1 * mm, 1 * mm]
        if hasattr(tle, "dose") and hasattr(tle.dose, "active"):
            tle.dose.active = True
        if hasattr(tle, "dose_uncertainty") and hasattr(tle.dose_uncertainty, "active"):
            tle.dose_uncertainty.active = True
        if hasattr(tle, "edep") and hasattr(tle.edep, "output_filename"):
            tle.edep.output_filename = "tle-edep.mhd"
        if hasattr(tle, "dose") and hasattr(tle.dose, "output_filename"):
            tle.dose.output_filename = "tle-dose_Gy.mhd"

    # -----------------------------
    # Run
    # -----------------------------
    sim.run()

    # -----------------------------
    # Export to HDF5 (results.h5)
    # -----------------------------
    h5_path = Path(sim.output_dir) / "results.h5"
    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["opengate_version"] = getattr(gate, "__version__", "unknown")
        meta.attrs["n_primaries"] = int(src.n)
        meta.create_dataset("spectrum_energy_keV", data=energies_keV)
        meta.create_dataset("spectrum_weight", data=weights)

        g = f.create_group("DoseActor")

        # EDEP sanity + save
        edep_img = dose.edep.get_data()
        edep_arr = write_dataset_with_meta(g, "edep", edep_img)
        print("DoseActor EDEP sanity:",
              "sum=", float(edep_arr.sum()),
              "min=", float(edep_arr.min()),
              "max=", float(edep_arr.max()))

        # Dose sanity + save (if active/available)
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
                print("DoseActor UNC sanity:",
                      "sum=", float(unc_arr.sum()),
                      "min=", float(unc_arr.min()),
                      "max=", float(unc_arr.max()))
            except Exception as e:
                print(f"[WARN] Uncertainty export failed: {e}")

        if tle is not None:
            gt = f.create_group("TLEDoseActor")
            if hasattr(tle, "edep"):
                try:
                    img = tle.edep.get_data()
                    arr = write_dataset_with_meta(gt, "edep", img)
                    print("TLE EDEP sanity:",
                          "sum=", float(arr.sum()),
                          "min=", float(arr.min()),
                          "max=", float(arr.max()))
                except Exception as e:
                    print(f"[WARN] TLE edep export failed: {e}")
            if hasattr(tle, "dose"):
                try:
                    img = tle.dose.get_data()
                    arr = write_dataset_with_meta(gt, "dose_Gy", img)
                    print("TLE DOSE sanity:",
                          "sum=", float(arr.sum()),
                          "min=", float(arr.min()),
                          "max=", float(arr.max()))
                except Exception as e:
                    print(f"[WARN] TLE dose export failed: {e}")

    print("\nDone.")
    print(f"Native maps (.mhd/.raw): {Path(sim.output_dir).resolve()}")
    print(f"HDF5: {h5_path.resolve()}")
    print(f"Spectrum plots: {Path(sim.output_dir).resolve()}/spectrum_weights.png and spectrum_cdf.png")


if __name__ == "__main__":
    main()

