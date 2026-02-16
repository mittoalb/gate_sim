#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import opengate as gate


# -----------------------------
# Helpers
# -----------------------------
def itk_to_numpy(itk_img) -> np.ndarray:
    """
    NumPy 2.0-safe conversion. Avoid np.asarray(itk_img) which may pass copy=False.
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
    plt.title("Tube spectrum (after filtration model)")
    plt.grid(True)
    plt.tight_layout()
    p1 = outdir / "spectrum_weights.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(energies_keV, cdf, marker="o")
    plt.xlabel("Energy (keV)")
    plt.ylabel("CDF")
    plt.title("Tube spectrum CDF")
    plt.grid(True)
    plt.tight_layout()
    p2 = outdir / "spectrum_cdf.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    print(f"[OK] Saved spectrum plots:\n  {p1.resolve()}\n  {p2.resolve()}")
    if show:
        plt.figure()
        plt.plot(energies_keV, w, marker="o")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Relative weight")
        plt.title("Tube spectrum (interactive)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def try_add_actor(sim: gate.Simulation, actor_type: str, name: str):
    try:
        a = sim.add_actor(actor_type, name)
        print(f"[OK] added actor: {actor_type} -> {name}")
        return a
    except Exception as e:
        print(f"[SKIP] cannot add actor {actor_type}: {e}")
        return None


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # "Versa-like" tube settings (defaults are a reasonable starting point)
    ap.add_argument("--kvp", type=float, default=120.0, help="Tube voltage (kVp). Versa is commonly 30–160 kV range.")
    ap.add_argument("--n", type=int, default=500_000_000, help="Number of primaries (increase for smoother dose).")

    # geometry distances
    ap.add_argument("--sod-cm", type=float, default=20.0, help="Source-to-object distance (cm).")
    ap.add_argument("--source-z-cm", type=float, default=None, help="Override source z (cm). If None, uses -SOD.")

    # focal spot (Versa 520-class reported ~2–4 µm; use ~3 µm default)
    ap.add_argument("--spot-um", type=float, default=3.0, help="Effective focal spot diameter (microns).")

    # beam field at object plane (rectangular)
    ap.add_argument("--field-x-mm", type=float, default=30.0, help="Beam width at object plane (mm).")
    ap.add_argument("--field-y-mm", type=float, default=30.0, help="Beam height at object plane (mm).")

    # filter stack (explicit geometry)
    ap.add_argument("--be-window-mm", type=float, default=0.25, help="Be window thickness (mm) (typical order ~0.1–0.5).")
    ap.add_argument("--al-mm", type=float, default=1.0, help="Added Al filtration thickness (mm).")
    ap.add_argument("--cu-mm", type=float, default=0.0, help="Added Cu filtration thickness (mm).")

    # phantom cylinder
    ap.add_argument("--cyl-radius-mm", type=float, default=50.0, help="Water cylinder radius (mm).")
    ap.add_argument("--cyl-height-mm", type=float, default=100.0, help="Water cylinder height (mm).")

    # scoring grid
    ap.add_argument("--vox-mm", type=float, default=1.0, help="Voxel size (mm).")
    ap.add_argument("--grid", type=int, default=160, help="Grid size per dimension (Nx=Ny=Nz).")

    ap.add_argument("--show-plot", action="store_true")
    args = ap.parse_args()

    outdir = Path("output")
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Spectrum: placeholder “tube spectrum”
    # -----------------------------
    # IMPORTANT:
    # This is NOT a full physical bremsstrahlung model. It’s a reasonable placeholder shape.
    # For a real Versa, you should generate the spectrum with TASMICS/TASMIP or SpekPy
    # and apply proper inherent filtration and target angle.
    #
    # Here we provide a simple filtered-like spectrum shape as a stand-in.
    kvp = float(args.kvp)
    energies_keV = np.arange(5.0, kvp + 0.001, 1.0)
    # crude shape: E*(kVp-E) with an exponential “soft” attenuation term
    raw = energies_keV * np.maximum(kvp - energies_keV, 0.0)
    raw *= np.exp(-0.02 * energies_keV)  # soft suppression
    raw[raw < 0] = 0
    weights = raw / raw.sum()

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
    um = gate.g4_units.um

    # World
    sim.world.size = [1 * m, 1 * m, 1 * m]
    sim.world.material = "G4_AIR"

    # -----------------------------
    # Water cylinder phantom (Tubs)
    # -----------------------------
    cyl = sim.add_volume("Tubs", "water_cyl")
    cyl.material = "G4_WATER"
    cyl.rmin = 0 * mm
    cyl.rmax = float(args.cyl_radius_mm) * mm
    cyl.dz = (float(args.cyl_height_mm) * mm) / 2.0  # half-length
    cyl.translation = [0, 0, 0]

    # -----------------------------
    # Filter stack as explicit geometry
    # Be window + Al + Cu (optional)
    # -----------------------------
    # Place filters between source and object plane.
    # Keep them large enough to cover the field.
    filter_xy = max(float(args.field_x_mm), float(args.field_y_mm), 50.0) * mm

    # Z positions: source at z0, filters closer to source; cylinder centered at 0.
    sod_cm = float(args.sod_cm)
    src_z = (-sod_cm if args.source_z_cm is None else float(args.source_z_cm)) * cm

    z_cursor = src_z + 5 * cm  # start 5 cm downstream from source

    def add_filter(name, material, thick_mm):
        nonlocal z_cursor
        thick = float(thick_mm) * mm
        if thick <= 0:
            return None
        v = sim.add_volume("Box", name)
        v.material = material
        v.size = [filter_xy, filter_xy, thick]
        v.translation = [0, 0, z_cursor]
        z_cursor += (thick + 2 * mm)  # spacing
        return v

    add_filter("Be_window", "G4_Be", args.be_window_mm)
    add_filter("Al_filter", "G4_Al", args.al_mm)
    add_filter("Cu_filter", "G4_Cu", args.cu_mm)

    # -----------------------------
    # Physics
    # -----------------------------
    sim.physics_manager.physics_list_name = "G4EmLivermorePhysics"
    sim.physics_manager.enable_decay = False

    # -----------------------------
    # Source: "Versa-like" approximation
    # - finite focal spot (disk-ish) implemented as a small box source region
    # - beam2d rectangular field (your build supports it)
    # - histogram energy spectrum
    # -----------------------------
    src = sim.add_source("GenericSource", "xray")
    src.particle = "gamma"
    src.n = int(args.n)

    # Finite spot: use a small square area in x/y at src_z
    spot_d = float(args.spot_um) * um
    spot_half = spot_d / 2.0

    # Many OpenGATE builds support 'box' position; if not, fallback to point.
    # This gives an effective spot size (order of microns).
    try:
        src.position.type = "box"
        src.position.translation = [0, 0, src_z]
        src.position.size = [spot_d, spot_d, 1 * um]
        print(f"[OK] Focal spot: box {args.spot_um:.3f} µm")
    except Exception as e:
        print(f"[WARN] position.type='box' not available; using point source. ({e})")
        src.position.type = "point"
        src.position.translation = [0, 0, src_z]

    # Direction: beam2d (rectangular field) towards +z
    # NOTE: exact attribute names for beam2d vary across versions; we try common ones.
    src.direction.type = "beam2d"
    if hasattr(src.direction, "momentum"):
        src.direction.momentum = [0, 0, 1]

    fx = float(args.field_x_mm) * mm
    fy = float(args.field_y_mm) * mm

    set_ok = False
    for attr in ("field_size", "size", "aperture", "rectangle"):
        if hasattr(src.direction, attr):
            try:
                setattr(src.direction, attr, [fx, fy])
                set_ok = True
                print(f"[OK] beam2d {attr} = [{args.field_x_mm} mm, {args.field_y_mm} mm]")
                break
            except Exception:
                pass
    if not set_ok and hasattr(src.direction, "sigma"):
        # Gaussian-like beam; rough mapping
        src.direction.sigma = [fx / 4.0, fy / 4.0]
        set_ok = True
        print(f"[OK] beam2d sigma ≈ [{float(args.field_x_mm)/4:.2f} mm, {float(args.field_y_mm)/4:.2f} mm]")
    if not set_ok:
        # Guaranteed non-zero fallback
        print("[WARN] Could not set beam2d field params on this build; falling back to momentum pencil beam.")
        src.direction.type = "momentum"
        src.direction.momentum = [0, 0, 1]

    # Energy histogram (in keV)
    src.energy.type = "histogram"
    src.energy.histogram_energy = (energies_keV * keV).tolist()
    src.energy.histogram_weight = weights.tolist()

    # -----------------------------
    # Actors: Dose + TLE
    # -----------------------------
    grid = int(args.grid)
    vox = float(args.vox_mm) * mm

    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to = cyl.name
    dose.size = [grid, grid, grid]
    dose.spacing = [vox, vox, vox]

    # Activate outputs (dose & uncertainty often off by default)
    if hasattr(dose, "dose") and hasattr(dose.dose, "active"):
        dose.dose.active = True
    if hasattr(dose, "dose_uncertainty") and hasattr(dose.dose_uncertainty, "active"):
        dose.dose_uncertainty.active = True

    # Native output filenames (.mhd/.raw)
    if hasattr(dose, "edep") and hasattr(dose.edep, "output_filename"):
        dose.edep.output_filename = "dose-edep.mhd"
    if hasattr(dose, "dose") and hasattr(dose.dose, "output_filename"):
        dose.dose.output_filename = "dose-dose_Gy.mhd"
    if hasattr(dose, "dose_uncertainty") and hasattr(dose.dose_uncertainty, "output_filename"):
        dose.dose_uncertainty.output_filename = "dose-dose_uncertainty.mhd"

    tle = try_add_actor(sim, "TLEDoseActor", "tle_dose")
    if tle is not None:
        tle.attached_to = cyl.name
        tle.size = [grid, grid, grid]
        tle.spacing = [vox, vox, vox]
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
        meta.attrs["tube_kvp"] = kvp
        meta.attrs["n_primaries"] = int(src.n)
        meta.attrs["sod_cm"] = sod_cm
        meta.attrs["spot_um"] = float(args.spot_um)
        meta.attrs["field_x_mm"] = float(args.field_x_mm)
        meta.attrs["field_y_mm"] = float(args.field_y_mm)
        meta.attrs["be_window_mm"] = float(args.be_window_mm)
        meta.attrs["al_mm"] = float(args.al_mm)
        meta.attrs["cu_mm"] = float(args.cu_mm)

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
    print(f"Outputs:\n  {outdir.resolve()}")
    print(f"HDF5:\n  {h5_path.resolve()}")
    print(f"Spectrum plots:\n  {(outdir/'spectrum_weights.png').resolve()}\n  {(outdir/'spectrum_cdf.png').resolve()}")


if __name__ == "__main__":
    main()
