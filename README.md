<div align="center">
  <h1 style="color: #ffffff; font-size: 2em; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">DAC-SIM</h1>
  <p style="color: #dddddd; font-size: 1.2em; font-weight: bold;">
    A molecular simulation package integrating a transferable Machine Learning Force Field in Metal-Organic Frameworks for CO<sub>2</sub> Direct Air Capture
  </p>
  <p>
    <img src="./images/logo.jpg" width="300" style="border-radius: 10px;">
  </p>
</div>

<p align="center">
 <a href="https://hspark1212.github.io/DAC-SIM-project/">
     <img alt="project-page" src="https://img.shields.io/badge/Project-Page-orange.svg?style=plastic">
 </a>
 <a href="https://hspark1212.github.io/DAC-SIM/">
     <img alt="Docs" src="https://img.shields.io/badge/Docs-v0.0.3-brightgreen.svg?style=plastic">
 </a>
  <a href="https://pypi.org/project/dac-sim">
     <img alt="PyPI" src="https://img.shields.io/badge/PyPI-v0.0.3-blue.svg?style=plastic&logo=PyPI">
 </a>
 <a href="https://doi.org/10.26434/chemrxiv-2024-7w6g6">
     <img alt="DOI" src="https://img.shields.io/badge/DOI-doi-organge.svg?style=plastic">
 </a>
 <a href="https://github.com/hspark1212/DAC-SIM/blob/main/LICENSE">
     <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey.svg?style=plastic">
 </a>
</p>

This package provides a simulation tool for Direct Air Capture (DAC) in Metal-Organic Frameworks (MOFs) using Widom insertion and Molecular Dynamics (MD) simulations. The molecular simulations are processed using the finetuning of the Universal Machine-learning Potentials for the gas molecules in MOFs.

## Features

- **Widom Insertion Monte Carlo Simulation**: Computes Henry's law coefficients (K<sub>H</sub>), averaged interaction energy, and heat of adsorption in zero loading (Q<sub>st</sub>) of gases with the machine learning force fields.
- **Molecular Dynamics Simulation**: Calculates the diffusion coefficients of gas molecules with the machine learning force fields.
- **Geometry Optimization**: Relax the structure with the machine learning force fields.
- **High-Throughput Screening**: Efficient processing of multiple structures for large-scale simulations.
- **Support for various gas molecules**: CO<sub>2</sub>, H<sub>2</sub>O, and more.
- **Flexible usage**: Customizable command line interface and Python API.

## Installation

### Requirements

- Python 3.10 or later

```bash
conda create -n dac-sim python=3.10
conda activate dac-sim
```

- Pytorch >= 1.12,<=2.5 (install from the [official website](https://pytorch.org/) suitable for your environment)

>[!NOTE]
> It is recommended to install PyTorch prior to installing DAC-SIM to avoid potential issues with GPU support.

`DAC-SIM` can be installed from PyPI or the source code.

### Install from PyPI

To install the latest version from [PyPI](https://pypi.org/project/dac-sim/):

```bash
pip install dac-sim
```

### Install from source code

To install the latest version from the source code:

```bash
git clone https://github.com/hspark1212/DAC-SIM.git
cd DAC-SIM
pip install -e . 
```

>[!NOTE]
> Once the package is installed, you can check the installation correctly by running `pytest` in the terminal. The test cases are located in the `tests` directory.

## Usage

`DAC-SIM` supports both Python API and command line interface (CLI) for running Widom insertion, molecular dynamics, and geometry optimization simulations.
The detailed description of the command line options can be found by running `dac-sim --help`.

### 1. Python API

---

#### (1) Widom insertion Monte Carlo simulation

Perform Widom insertion Monte Carlo (MC) simulation to calculate K<sub>H</sub>, averaged interaction energy, and Q<sub>st</sub> of gas molecules. The `WidomInsertion` class inherits from [`Dynamics`](https://github.com/qsnake/ase/blob/master/ase/optimize/optimize.py), an **ASE** class that manages the simulation of molecular dynamics. That is, it works in a same way of the **ASE Calculator**.

```python
from ase.io import read
from ase.build import molecule

from dac_sim.widom_insertion import WidomInsertion

# Load the structure and build the gas
structure = read("examples/mg-mof-74.cif")
gas = molecule("CO2")

# Create the WidomInsertion object
temperature = 300  # [K]
trajectory = "widom_co2_mg-mof-74.traj"
logfile = "widom_co2_mg-mof-74.log"
widom_insertion = WidomInsertion(
    structure,
    gas=gas,
    temperature=temperature,
    trajectory=trajectory,
    logfile=logfile,
    device="cuda",
    default_dtype="float32",
    dispersion=True,
)
result = widom_insertion.run(num_insertions=5000, random_seed=0, fold=2)
print(result)
```

---

#### (2) Molecular dynamics simulation

Perform molecular dynamics simulation to calculate the diffusion coefficient and transport properties of gas molecules using [ASE molecular dynamics](https://wiki.fysik.dtu.dk/ase/ase/md/md.html).

The `MolecularDyamic` class involves setting intial configuration of gas molecules in the accessible sites of the structure using [`add_gas_in_accessible_positions`](https://github.com/hspark1212/DAC-SIM/blob/main/dac_sim/grid.py#L85) function.

```python
from ase.io import read
from ase.build import molecule
from ase.visualize import view
from dac_sim.molecule_dynamic import MolecularDynamic

# Load the structure and build the gas
structure = read("examples/mg-mof-74.cif")
gas_list = [molecule("CO2"), molecule("H2O")]

# Run the molecular dynamics simulation
timestep = 1.0  # [fs]
temperature = 300  # [K]
trajectory = "md_co2_h2o_mg-mof-74.traj"
logfile = "md_co2_h2o_mg-mof-74.log"
md = MolecularDynamic(
    structure,
    gas_list=gas_list,
    timesteps=timestep,
    temperature=temperature,
    trajectory=trajectory,
    logfile=logfile,
    loginterval=10,
    device="cuda",
    default_dtype="float32",
    dispersion=True,
)
md.run(num_init_steps=5000, num_md_steps=10000)

# Calculate diffusion coefficient
result = md.calculate_diffusion_coefficient(show_plot=True)
print(result)

# Visualize the trajectory
traj = read("md_co2_h2o_mg-mof-74.traj", ":")
view(traj, viewer="ngl")
```

---

#### (3) Geometry optimization

Perform geometry optimization to relax the structure with internal and cell optimization steps. The `GeometryOptimization` class utilizs a joint [`FIRE`](https://wiki.fysik.dtu.dk/ase/ase/optimize.html#ase.optimize.FIRE) (optimizer) and [`FrechetCellFilter`](https://wiki.fysik.dtu.dk/ase/ase/filters.html#the-frechetcellfilter-class) (cell optimzation) from the [ASE](https://wiki.fysik.dtu.dk/ase/ase/optimize/optimize.html) package to optimize the structure.

```python
from ase.io import read
from dac_sim.optimize import GeometryOptimization

# Load the structure
structure = read("examples/mg-mof-74.cif")

# Run geometry optimization
go = GeometryOptimization(
    num_total_optimization=30,
    num_internal_steps=50,
    num_cell_steps=50,
    fmax=0.05,
    cell_relax=True,
    device="cuda",
    default_dtype="float32",
    dispersion=True,
)
opt_structure = go.run(structure)
```

### 2. command line interface (CLI)

---

`DAC-SIM` provides CLI commands for running Widom insertion MC, molecular dynamics, and geometry optimization simulations. The CLI commands can be used to perform simulations with a single CIF file or a directory containing multiple CIF files.

The detailed description of the command line options can be found by running

- `dac-sim widom --help`
- `dac-sim md --help`
- `dac-sim opt --help`

#### (1) Widom insertion Monte Carlo simulation

The following command performs Widom insertion MC simulation to calculate K<sub>H</sub>, averaged interaction energy and Q<sub>st</sub> of gas molecules on CIF files in the `examples` directory. The results are saved in the `results` directory.

```bash
dac-sim widom examples/ --gas=CO2 --temperature=300 --num_insertions=5000 --fold=2 --save_dir=results
```

---

#### (2) Molecular dynamics simulation

The following command performs molecular dynamics simulation to calculate the diffusion coefficient and transport properties of gas molecules on CIF files in the `examples` directory. The results are saved in the `results` directory.

```bash
dac-sim md examples --gas_list="CO2,H2O" --timesteps=1.0 --temperature=300 --num_md_steps=1000 --save_dir=results
```

#### (3) Geometry optimization

The following command performs geometry optimization to relax the structure with internal and cell optimization steps on CIF files in the `examples` directory. The results are saved in the `results` directory.

```bash
dac-sim opt examples --num_total_optimization=100 --num_internal_steps=50 --num_cell_steps=50 --save_dir=results
```

</details>

## Citation

If you find this package useful, please consider citing our work:
["Accelerating CO<sub>2</sub> Direct Air Capture Screening for Metal-Organic Frameworks with a Transferable Machine Learning Force Field"](https://doi.org/10.26434/chemrxiv-2024-7w6g6)

```bibtex
@article{Lim2024accelerating,
    title={Accelerating CO2 Direct Air Capture Screening for Metal-Organic Frameworks with a Transferable Machine Learning Force Field},
    author={Lim, Yunsung and Park, Hyunsoo and Walsh, Aron and Kim, Jihan},
    journal={ChemRxiv},
    doi={10.26434/chemrxiv-2024-7w6g6},
    year={2024},
}
```

## Contributing 🙌

Contributions are welcome! If you have any suggestions or find any issues, please open an issue or a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more information.
