# Project Name

Short description of what this project does.

------------------------------------------------------------------------

## 📦 Requirements

-   Miniconda or Anaconda
-   Python ≥ 3.10
-   (Optional) CUDA for GPU acceleration

Check that conda is installed:

``` bash
conda --version
```

------------------------------------------------------------------------

## 🛠 Create Conda Environment

### Option 1 --- Using `environment.yml` (Recommended)

Create the environment:

``` bash
conda env create -f environment.yml
```

Activate the environment:

``` bash
conda activate myenv
```

Update the environment later if needed:

``` bash
conda env update -f environment.yml --prune
```

------------------------------------------------------------------------

### Option 2 --- Manual Creation

Create a new environment:

``` bash
conda create -n myenv python=3.10 -y
```

Activate it:

``` bash
conda activate myenv
```

Install base dependencies:

``` bash
conda install numpy scipy matplotlib -y
```

Install additional pip dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 Install This Package

From the project root directory:

Standard install:

``` bash
pip install .
```

Development install (editable mode):

``` bash
pip install -e .
```

------------------------------------------------------------------------

## 🔬 GPU Setup (Optional)

If using CUDA (example: CUDA 12):

``` bash
conda install -c conda-forge cudatoolkit=12.0
pip install cupy-cuda12x
```

Verify GPU access:

``` bash
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

------------------------------------------------------------------------

## ✅ Verify Installation

``` bash
python -c "import your_package_name; print('Installation successful')"
```

------------------------------------------------------------------------

## 📁 Project Structure

    project/
    │
    ├── your_package/
    ├── environment.yml
    ├── requirements.txt
    ├── pyproject.toml
    └── README.md

------------------------------------------------------------------------

## 🧹 Remove Environment

Deactivate:

``` bash
conda deactivate
```

Remove completely:

``` bash
conda remove -n myenv --all
```

------------------------------------------------------------------------

## 💡 Notes

-   Replace `myenv` with your desired environment name.
-   Replace `your_package_name` with your actual Python package name.
-   If GPU is not required, skip the CUDA section.
