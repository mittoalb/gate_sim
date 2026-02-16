#!/usr/bin/env python3
"""
OpenGATE / GATE10+ example:
- X-ray spectrum source
- Water phantom
- DoseActor (always)
- Optional TLE dose actor (if your build has it)
- Save all voxel maps into ONE HDF5 file (plus normal .mhd/.raw outputs)

Run:
  python xray_water_h5.py

Outputs:
  output/dose_actor*.mhd/.raw   (native MetaImage)
  output/tle_dose*.mhd/.raw     (if available)
  output/results.h5             (HDF5 bundle of arrays + metadata)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py

import opengate as gate


def _image_to_numpy(img):
    """Convert ITK image to numpy array with best available backend."""
    # Preferred helper in OpenGATE (exists in many versions)
    if hasattr(gate, "image_to_numpy"):
        return gate.image_to_numpy(img)

    # Fallback to itk if present
    try:
        import itk  # type: ignore
        return itk.array_from_image(img)
    except Exception as e:
        raise RuntimeError(
            "Cannot convert ITK image to numpy. "
            "Your install lacks gate.image_to_numpy and itk.array_from_image."
        ) from e


def _itk_meta(img):
    """Extract origin/spacing/direction (and size) from an ITK image."""
    origin = tuple(float(x) for x in img.GetOrigin())
    spacing = tuple(float(x) for x in img.GetSpacing())
    # direction is a 3x3 (or NxN) matrix
    d = np.array(img.GetDirection())
    # robust reshape for 3D
    if d.size == 9:
        d = d.reshape(3, 3)
    size = tuple(int(x) for x in img.GetLargestPossibleRegion().GetSize())
    return origin, spacing, d, size


def try_add_actor(sim: gate.Simulation, actor_type: str, name: str):
    """Add an actor if available; otherwise return None."""
    try:
        a = sim.add_actor(actor_type, name)
        print(f"[OK] added actor: {actor_type} -> {name}")
        return a
    except Exception as e:
        print(f"[SKIP] {actor_type} not available: {e}")
        return None


def save_actor_images_to_h5(h5: h5py.File, group_name: str, images: dict):
    """
    Save a dict of {dataset_name: itk_image} into HDF5
    with gzip compression + ITK metadata as attrs.
    """
    g = h5.require_group(group_name)

    for ds_name, img in images.items():
        if img is None:
            continue

        arr = _image_to_numpy(img)
        origin, spacing, direction, size = _itk_meta(img)

        # Replace existing dataset if rerun
        if ds_name in g:
            del g[ds_name]

        ds = g.create_dataset(
            ds_name,
            data=arr,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        ds.attrs["origin"] = origin
        ds.attrs["spacing"] = spacing
        ds.attrs["direction"] = direction
        ds.attrs["itk_size"] = size
        ds.attrs["axis_order"] = "z,y,x"  # OpenGATE/ITK numpy is typically z,y,x

        # Convenience: store dtype
        ds.attrs["dtype"] = str(arr.dtype)


def main():
    sim = gate.Simulation()

    # ---- output dir ----
    sim.output_dir = "output"
    Path(sim.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- general ----
    sim.g4_verbose = False
    sim.visu = False
    sim.random_seed = 12345
    sim.number_of_threads = 1

    # ---- units ----
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    m = gate.g4_units.m
    keV = gate.g4_units.keV
    deg = gate.g4_units.deg

    # ---- world ----
    sim.world.size = [1 * m, 1 * m, 1 * m]
    sim.world.material = "G4_AIR"

    # ---- phantom ----
    phantom = sim.add_volume("Box", "phantom")
    phantom.size = [50 * cm, 50 * cm, 50 * cm]
    phantom.translation = [0, 0, 0]
    phantom.material = "G4_WATER"

    # ---- physics ----
    sim.physics_manager.physics_list_name = "G4EmLivermorePhysics"
    sim.physics_manager.enable_decay = False

    # ---- source: X-ray spectrum ----
    energies_keV = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=float)
    weights = np.array([0.01, 0.05, 0.12, 0.22, 0.35, 0.45, 0.42, 0.35, 0.25, 0.15, 0.08, 0.03], dtype=float)
    weights /= weights.sum()

    src = sim.add_source("GenericSource", "xray")
    src.particle = "gamma"

    src.position.type = "point"
    src.position.translation = [0, 0, -40 * cm]

    src.direction.type = "iso"
    src.direction.theta = [0 * deg, 10 * deg]

    src.energy.type = "histogram"
    src.energy.histogram_energy = (energies_keV * keV).tolist()
    src.energy.histogram_weight = weights.tolist()

    n_primaries = 2_000_000
    src.n = n_primaries

    # ============================================================
    # Actors
    # ============================================================

    # 1) DoseActor (native outputs are MetaImage .mhd/.raw)
    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to = phantom.name
    dose.size = [128, 128, 128]
    dose.spacing = [2 * mm, 2 * mm, 2 * mm]
    dose.output_filename = "dose_actor.mhd"  # will land in output/

    # Some builds expose toggles; keep it robust
    for attr in ("dose", "edep", "dose_uncertainty"):
        if hasattr(dose, attr):
            try:
                setattr(dose, attr, True)
            except Exception:
                pass

    # 2) Optional: TLE dose (if your build has it)
    # Actor name might differ in some versions; we attempt common ones.
    tle = try_add_actor(sim, "TLEDoseActor", "tle_dose")
    if tle is None:
        tle = try_add_actor(sim, "SETLEDoseActor", "tle_dose")  # some builds expose seTLE separately
    if tle is not None:
        tle.attached_to = phantom.name
        tle.size = [128, 128, 128]
        tle.spacing = [2 * mm, 2 * mm, 2 * mm]
        tle.output_filename = "tle_dose.mhd"

    # ============================================================
    # Run
    # ============================================================
    # sim.initialize()
    sim.run()

    # ============================================================
    # Save to HDF5 (in addition to .mhd/.raw already saved)
    # ============================================================
    h5_path = Path(sim.output_dir) / "results.h5"
    with h5py.File(h5_path, "w") as h5:
        # Save run metadata
        meta = h5.require_group("meta")
        meta.attrs["opengate_version"] = getattr(gate, "__version__", "unknown")
        meta.attrs["n_primaries"] = int(n_primaries)
        meta.create_dataset("spectrum_energy_keV", data=energies_keV)
        meta.create_dataset("spectrum_weight", data=weights)

        # DoseActor images (these properties are present in most OpenGATE builds)
        dose_images = {}
        if hasattr(dose, "dose"):
            try:
                dose_images["dose_Gy"] = dose.dose.image
            except Exception:
                pass
        if hasattr(dose, "edep"):
            try:
                dose_images["edep"] = dose.edep.image
            except Exception:
                pass
        if hasattr(dose, "dose_uncertainty"):
            try:
                dose_images["dose_uncertainty"] = dose.dose_uncertainty.image
            except Exception:
                pass

        save_actor_images_to_h5(h5, "DoseActor", dose_images)

        # TLE images (best-effort; property names vary)
        if tle is not None:
            tle_images = {}
            for cand in ("dose", "edep", "dose_uncertainty"):
                if hasattr(tle, cand):
                    try:
                        tle_images[cand] = getattr(tle, cand).image
                    except Exception:
                        pass
            # If we couldn't find fields, still write a marker
            g = h5.require_group("TLEDoseActor")
            g.attrs["present"] = True
            if tle_images:
                save_actor_images_to_h5(h5, "TLEDoseActor", tle_images)
            else:
                g.attrs["note"] = "TLE actor present, but image fields not found via common names."

    print("\nDone.")
    print(f"Native voxel outputs (MetaImage): {Path(sim.output_dir).resolve()}/*.mhd + *.raw")
    print(f"HDF5 bundle: {h5_path.resolve()}")


if __name__ == "__main__":
    main()

