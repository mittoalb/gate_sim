#!/usr/bin/env python3
"""
OpenGATE (GATE 10+) Python simulation:
- X-ray tube spectrum source (polychromatic)
- Water phantom
- DoseActor + TLEDoseActor (if available)
- Save results to:
    1) native MetaImage: .mhd + .raw (OpenGATE default)
    2) a single HDF5 file (results.h5) with arrays + geometry metadata

Key fixes for "all zeros":
- Use direction.type="momentum" to force beam towards +z.
- Explicitly activate dose and uncertainty outputs (they're off by default).
- Export from actor.get_data() and np.asarray(itk_img) (robust).

Run:
  python sim1_fixed.py

Outputs in ./output/
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py

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
    Convert ITK image to numpy array.
    OpenGATE docs show using np.asarray(image). If it fails, fall back.
    """
    try:
        return np.asarray(itk_img)
    except Exception:
        # fallback if needed
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

    # geometry metadata
    ds.attrs["origin"] = tuple(float(x) for x in itk_img.GetOrigin())
    ds.attrs["spacing"] = tuple(float(x) for x in itk_img.GetSpacing())

    d = np.array(itk_img.GetDirection())
    if d.size == 9:
        d = d.reshape(3, 3)
    ds.attrs["direction"] = d
    ds.attrs["axis_order"] = "z,y,x"
    ds.attrs["dtype"] = str(arr.dtype)

    return arr


def main():
    sim = gate.Simulation()

    # -----------------------------
    # Output directory
    # -----------------------------
    sim.output_dir = "output"
    Path(sim.output_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # General settings
    # -----------------------------
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
    # Water phantom
    # -----------------------------
    phantom = sim.add_volume("Box", "phantom")
    phantom.size = [50 * cm, 50 * cm, 50 * cm]
    phantom.translation = [0, 0, 0]
    phantom.material = "G4_WATER"

    # -----------------------------
    # Physics
    # -----------------------------
    sim.physics_manager.physics_list_name = "G4EmLivermorePhysics"
    sim.physics_manager.enable_decay = False

    # -----------------------------
    # Source: polychromatic X-ray spectrum
    # -----------------------------
    energies_keV = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=float)
    weights = np.array([0.01, 0.05, 0.12, 0.22, 0.35, 0.45, 0.42, 0.35, 0.25, 0.15, 0.08, 0.03], dtype=float)
    weights /= weights.sum()

    src = sim.add_source("GenericSource", "xray")
    src.particle = "gamma"

    # Place source upstream (negative z), shoot toward +z
    src.position.type = "point"
    src.position.translation = [0, 0, -40 * cm]

    # IMPORTANT: force beam direction (avoids sign/convention ambiguity)
    src.direction.type = "momentum"
    src.direction.momentum = [0, 0, 1]

    # Energy histogram (polychromatic)
    src.energy.type = "histogram"
    src.energy.histogram_energy = (energies_keV * keV).tolist()
    src.energy.histogram_weight = weights.tolist()

    n_primaries = 2_000_000
    src.n = int(n_primaries)

    # -----------------------------
    # DoseActor (voxelized)
    # -----------------------------
    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to = phantom.name
    dose.size = [128, 128, 128]
    dose.spacing = [2 * mm, 2 * mm, 2 * mm]

    # Activate outputs (dose + uncertainty are off by default in many builds)
    # Keep robust: check attribute existence.
    if hasattr(dose, "dose") and hasattr(dose.dose, "active"):
        dose.dose.active = True
    if hasattr(dose, "dose_uncertainty") and hasattr(dose.dose_uncertainty, "active"):
        dose.dose_uncertainty.active = True

    # Set explicit filenames (so you get separate .mhd/.raw)
    if hasattr(dose, "edep") and hasattr(dose.edep, "output_filename"):
        dose.edep.output_filename = "dose-edep.mhd"
    if hasattr(dose, "dose") and hasattr(dose.dose, "output_filename"):
        dose.dose.output_filename = "dose-dose_Gy.mhd"
    if hasattr(dose, "dose_uncertainty") and hasattr(dose.dose_uncertainty, "output_filename"):
        dose.dose_uncertainty.output_filename = "dose-dose_uncertainty.mhd"

    # -----------------------------
    # TLE Dose Actor (optional)
    # -----------------------------
    tle = try_add_actor(sim, "TLEDoseActor", "tle_dose")
    if tle is not None:
        tle.attached_to = phantom.name
        tle.size = [128, 128, 128]
        tle.spacing = [2 * mm, 2 * mm, 2 * mm]

        # Activate if available
        if hasattr(tle, "dose") and hasattr(tle.dose, "active"):
            tle.dose.active = True
        if hasattr(tle, "dose_uncertainty") and hasattr(tle.dose_uncertainty, "active"):
            tle.dose_uncertainty.active = True

        # Filenames if the fields exist
        if hasattr(tle, "edep") and hasattr(tle.edep, "output_filename"):
            tle.edep.output_filename = "tle-edep.mhd"
        if hasattr(tle, "dose") and hasattr(tle.dose, "output_filename"):
            tle.dose.output_filename = "tle-dose_Gy.mhd"
        if hasattr(tle, "dose_uncertainty") and hasattr(tle.dose_uncertainty, "output_filename"):
            tle.dose_uncertainty.output_filename = "tle-dose_uncertainty.mhd"

    # -----------------------------
    # Run (newer OpenGATE: no sim.initialize())
    # -----------------------------
    sim.run()

    # -----------------------------
    # Export to HDF5
    # -----------------------------
    h5_path = Path(sim.output_dir) / "results.h5"
    with h5py.File(h5_path, "w") as f:
        # metadata
        meta = f.create_group("meta")
        meta.attrs["opengate_version"] = getattr(gate, "__version__", "unknown")
        meta.attrs["n_primaries"] = int(n_primaries)
        meta.create_dataset("spectrum_energy_keV", data=energies_keV)
        meta.create_dataset("spectrum_weight", data=weights)
        meta.attrs["source_position_cm"] = (0.0, 0.0, -40.0)
        meta.attrs["source_direction_momentum"] = (0.0, 0.0, 1.0)
        meta.attrs["phantom_size_cm"] = (50.0, 50.0, 50.0)

        # DoseActor datasets
        g = f.create_group("DoseActor")

        # edep is generally available
        edep_img = dose.edep.get_data() if hasattr(dose, "edep") else None
        if edep_img is not None:
            edep_arr = write_dataset_with_meta(g, "edep", edep_img)
            print("DoseActor EDEP sanity:",
                  "sum=", float(edep_arr.sum()),
                  "min=", float(edep_arr.min()),
                  "max=", float(edep_arr.max()))
        else:
            print("[WARN] DoseActor edep not found; cannot export edep.")

        # dose (Gy) if active/available
        if hasattr(dose, "dose"):
            try:
                dose_img = dose.dose.get_data()
                dose_arr = write_dataset_with_meta(g, "dose_Gy", dose_img)
                print("DoseActor DOSE sanity:",
                      "sum=", float(dose_arr.sum()),
                      "min=", float(dose_arr.min()),
                      "max=", float(dose_arr.max()))
            except Exception as e:
                print(f"[WARN] DoseActor dose export failed: {e}")

        # dose uncertainty if active/available
        if hasattr(dose, "dose_uncertainty"):
            try:
                unc_img = dose.dose_uncertainty.get_data()
                unc_arr = write_dataset_with_meta(g, "dose_uncertainty", unc_img)
                print("DoseActor UNC sanity:",
                      "sum=", float(unc_arr.sum()),
                      "min=", float(unc_arr.min()),
                      "max=", float(unc_arr.max()))
            except Exception as e:
                print(f"[WARN] DoseActor uncertainty export failed: {e}")

        # TLE datasets (best-effort)
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

            if hasattr(tle, "dose_uncertainty"):
                try:
                    img = tle.dose_uncertainty.get_data()
                    arr = write_dataset_with_meta(gt, "dose_uncertainty", img)
                    print("TLE UNC sanity:",
                          "sum=", float(arr.sum()),
                          "min=", float(arr.min()),
                          "max=", float(arr.max()))
                except Exception as e:
                    print(f"[WARN] TLE uncertainty export failed: {e}")

    print("\nDone.")
    print(f"Native maps (.mhd/.raw): {Path(sim.output_dir).resolve()}")
    print(f"HDF5: {h5_path.resolve()}")


if __name__ == "__main__":
    main()

