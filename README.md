# GATE Simulation - ZEISS Tomography

Monte Carlo simulations for ZEISS X-ray tomography systems using OpenGATE.

------------------------------------------------------------------------

## 📦 Requirements

-   Python ≥ 3.10
-   OpenGATE (Monte Carlo simulation toolkit)
-   spekpy (X-ray spectrum generation)
-   ITK (image processing)
-   NumPy, SciPy, h5py

------------------------------------------------------------------------

## 🛠 Setup

Install dependencies:

``` bash
pip install opengate spekpy itk h5py numpy scipy
```

------------------------------------------------------------------------

## 🚀 Usage

### Run Single Projection Simulation

``` bash
python zeiss_tomo/sim_proj.py
```

### Run Full Tomography Acquisition

``` bash
python zeiss_tomo/run_all.py
```

### Accumulate Results

``` bash
python zeiss_tomo/accumulate.py
```

------------------------------------------------------------------------

## 📁 Project Structure

    gate_sim/
    │
    ├── zeiss_tomo/           # Main package
    │   ├── sim_proj.py       # Projection simulation
    │   ├── run_all.py        # Full acquisition runner
    │   └── accumulate.py     # Results accumulator
    │
    ├── scripts/              # Additional simulation scripts
    │   ├── sim1-4.py
    │   ├── zeiss_versa.py
    │   └── zeiss_tomo.py
    │
    └── README.md

------------------------------------------------------------------------

## 📊 Simulation Features

-   ZEISS-like X-ray spectrum generation (with Be/Al/Cu filtration)
-   Dose and energy deposition tracking
-   Projection data acquisition
-   ITK integration for image processing

------------------------------------------------------------------------

## 💡 Notes

-   Large result files (.h5, .raw, .mhd) are excluded from git
-   Projection data directories are excluded from version control
