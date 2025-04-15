from typing import IO, Union, Literal, Optional, Dict, Any
from pathlib import Path
from collections import defaultdict
import time

from tqdm import tqdm
import numpy as np
from ase import Atoms
from ase import units
from ase.optimize.optimize import Dynamics
from ase.io.trajectory import Trajectory
from mace.calculators import mace_mp

import torch
from dac_sim.grid import get_accessible_positions
from dac_sim.molecule import add_molecule
from dac_sim.optimize import optimize_atoms
from dac_sim import DEFAULT_MODEL_PATH


class WidomInsertion(Dynamics):
    def __init__(
        self,
        structure: Atoms,
        gas: Atoms,
        temperature: float = 300.0,
        init_structure_optimize: bool = True,
        init_gas_optimize: bool = True,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        append_trajectory: bool = False,
        model_path: Optional[str] = None,
        device: Literal["cuda", "cpu"] = "cuda",
        default_dtype: Literal["float32", "float64"] = "float32",
        dispersion: bool = True,
    ):
        """
        Widom insertion algorithm to calculate the Henry coefficient and heat of adsorption.

        Parameters
        ----------
        structure : Atoms
            Atoms object for structure (or Framework).
        gas : Atoms
            Atoms object for gas molecule.
        temperature : float, default=300.0
            Simulation temperature, in Kelvin.
        init_structure_optimize : bool, default=True
            If True, optimize the structure before the Widom insertion simulation.
        init_gas_optimize : bool, default=True
            If True, optimize the gas molecule before the Widom insertion simulation.
        trajectory : str, optional
            Path to the trajectory file. If None, the trajectory will be saved in the same directory as the CIF file.
        logfile : Union[IO, str], optional
            Path to the log file or file-like object. If None, no log file is created.
        append_trajectory : bool, default=False
            If True, append to the existing trajectory file, otherwise overwrite.
        model_path : str, optional
            Path to the MACE model file. If None, the default model is used.
        device : {"cuda", "cpu"}, default="cuda"
            Computational device for running simulations, either "cuda" for GPU or "cpu".
        default_dtype : {"float32", "float64"}, default="float32"
            Default data type for the MACE model.
        dispersion : bool, default=True
            Whether to include dispersion correction in the energy calculations.
        """
        self.gas = gas
        self.temperature = temperature
        self.init_structure_optimize = init_structure_optimize
        self.init_gas_optimize = init_gas_optimize

        # Check device availability
        device = self._configure_device(device)

        # Set up calculator
        self.calculator = self._initialize_calculator(
            model_path, device, default_dtype, dispersion
        )

        super().__init__(
            structure,
            logfile=logfile,
            trajectory=None,
            append_trajectory=append_trajectory,
        )

        if trajectory is not None:
            if isinstance(trajectory, str):
                mode = "a" if append_trajectory else "w"
                self.trajectory = self.closelater(Trajectory(trajectory, mode=mode))
            else:
                self.trajectory = trajectory

    def _configure_device(self, device: str) -> str:
        if device not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'")
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead")
            device = "cpu"
        return device

    def _initialize_calculator(
        self,
        model_path: Optional[str],
        device: Literal["cuda", "cpu"],
        default_dtype: Literal["float32", "float64"],
        dispersion: bool,
    ):
        model_path = Path(model_path) if model_path else Path(DEFAULT_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Using model at: {model_path}")
        return mace_mp(
            model=model_path,
            device=device,
            default_dtype=default_dtype,
            dispersion=dispersion,
        )

    def todict(self):
        return {
            "type": "widom-insertion",
            "md-type": self.__class__.__name__,
        }

    def log(
        self,
        interaction_energy,
        total_energy,
        energy_structure,
        energy_gas,
        boltzmann_factor,
    ):
        if self.logfile is None:
            return

        if self.nsteps == 0:
            header = (
                f"{'Step':>6} {'Time':>8} {'Interaction E':>15} "
                f"{'Total E':>15} {'Struct E':>15} {'Gas E':>15} {'Boltzmann F':>15}\n"
            )
            self.logfile.write(header)

        current_time = time.strftime("%H:%M:%S", time.localtime())
        log_entry = (
            f"{self.nsteps:6d} {current_time:>8} "
            f"{interaction_energy:15.6f} {total_energy:15.6f} "
            f"{energy_structure:15.6f} {energy_gas:15.6f} {boltzmann_factor:15.6f}\n"
        )
        self.logfile.write(log_entry)
        self.logfile.flush()

    def log_results(self, results: Dict[str, Any]) -> None:
        if self.logfile is None:
            return

        if self.nsteps == 0:
            header = "Total Results:\n"
            self.logfile.write(header)

        for k, v in results.items():
            if not v:
                continue
            mean = np.mean(v)
            std = np.std(v)
            log_entry = f"{k:28}: {mean:15.6f} ± {std:10.6f} | {str(v)}\n"
            self.logfile.write(log_entry)
        self.logfile.flush()

    def run(
        self,
        num_insertions: int = 5000,
        grid_spacing: float = 0.15,
        cutoff_distance: float = 1.50,
        min_interplanar_distance: float = 6.0,
        fold: int = 2,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the Widom insertion algorithm to calculate the Henry coefficient and heat of adsorption.

        Parameters
        ----------
        num_insertions : int, default=5000
            Number of random insertions of the gas molecule during simulation.
        grid_spacing : float, default=0.15
            Spacing of the grid for possible gas insertion points, in angstroms.
        cutoff_distance : float, default=1.50
            When the distance between framework atoms and the gas molecule is less than this value, the insertion is rejected. In angstroms.
        min_interplanar_distance : float, default=6.0
            When the interplanar distance of the framework is less than this value, a supercell is constructed. In angstroms.
        fold : int, default=2
            Number of repetitions of Widom insertion to improve statistics.
        random_seed : int, optional
            Seed for the random number generator for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the calculated Henry coefficient (mol/kg Pa), averaged interaction energy (eV), and heat of adsorption (kJ/mol) over the number of folds.
        """
        structure = self.atoms.copy()
        gas = self.gas.copy()

        # Optimize structure and gas molecule
        if self.init_structure_optimize:
            print("Optimizing the structure...")
            structure = optimize_atoms(calculator=self.calculator, atoms=structure)
            if structure is None:
                raise ValueError("Failed to optimize structure")
        if self.init_gas_optimize:
            print("Optimizing the gas molecule...")
            gas = optimize_atoms(
                calculator=self.calculator, atoms=gas, cell_relax=False
            )
            if gas is None:
                raise ValueError("Failed to optimize gas molecule")

        # Calculate accessible positions
        ret = get_accessible_positions(
            structure=structure,
            grid_spacing=grid_spacing,
            cutoff_distance=cutoff_distance,
            min_interplanar_distance=min_interplanar_distance,
        )
        pos_grid = ret["pos_grid"]
        idx_accessible_pos = ret["idx_accessible_pos"]
        structure = ret["structure"]  # supercell structure if necessary
        print(
            f"Number of accessible positions: {len(idx_accessible_pos)} out of total {len(pos_grid)}"
        )
        # Calculate energies for structure and gas
        energy_structure = self.calculator.get_potential_energy(structure)
        energy_gas = self.calculator.get_potential_energy(gas)

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            print(f"Setting random seed: {random_seed}")

        # Run Widom insertion algorithm
        results = defaultdict(list)
        for i in range(fold):
            random_indices = np.random.choice(
                len(pos_grid), size=num_insertions, replace=True
            )
            interaction_energies = np.zeros(num_insertions)
            for i, rand_idx in enumerate(tqdm(random_indices)):
                if rand_idx not in idx_accessible_pos:
                    self.log(
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )
                    interaction_energies[i] = 1e10  # lead to zero boltzmann factor
                    self.nsteps += 1
                    continue

                # Add gas molecule to the accessible position
                pos = pos_grid[rand_idx]
                added_gas = add_molecule(gas, rotate=True, translate=pos)
                structure_with_gas = structure + added_gas
                structure_with_gas.wrap()  # wrap atoms to unit cell

                # Calculate interaction energy
                structure_with_gas.calc = self.calculator
                total_energy = structure_with_gas.get_potential_energy()  # [eV]
                interaction_energy = (
                    total_energy - energy_structure - energy_gas
                )  # [eV]

                # Handle invalid interaction energy
                if interaction_energy < -1.25:
                    interaction_energy = 1e10  # lead to zero boltzmann factor

                interaction_energies[i] = interaction_energy
                boltzmann_factor = np.exp(
                    -interaction_energy / (self.temperature * units._k / units._e)
                )

                # Log results
                self.log(
                    interaction_energy,
                    total_energy,
                    energy_structure,
                    energy_gas,
                    boltzmann_factor,
                )
                self.nsteps += 1

                # Write trajectory
                if self.trajectory is not None:
                    self.trajectory.write(structure_with_gas)

            # Calculate ensemble averages properties
            # units._e [J/eV], units._k [J/K], units._k / units._e # [eV/K]
            boltzmann_factor = np.exp(
                -interaction_energies / (self.temperature * units._k / units._e)
            )

            # KH = <exp(-E/RT)> / (R * T)
            atomic_density = self._calculate_atomic_density(structure)  # [kg / m^3]
            kh = (
                boltzmann_factor.sum()
                / num_insertions
                / (units._k * units._Nav)  # R = [J / mol K] = [Pa m^3 / mol K]
                / self.temperature  # T = [K] -> [mol/ m^3 Pa]
                / atomic_density  #  = [kg / m^3] -> [mol / kg Pa]
            )  # [mol/kg Pa]

            # U = < E * exp(-E/RT) > / <exp(-E/RT)> # [eV]
            u = (interaction_energies * boltzmann_factor).sum() / boltzmann_factor.sum()

            # Qst = U - RT # [kJ/mol]
            qst = (u * units._e - units._k * self.temperature) * units._Nav * 1e-3

            results["henry_coefficient"].append(kh)
            results["averaged_interaction_energy"].append(u)
            results["heat_of_adsorption"].append(qst)
            self.log_results(results)
        return results

    def _calculate_atomic_density(self, atoms: Atoms) -> float:
        """
        Calculate atomic density of the atoms.

        Parameters
        ----------
        atoms : Atoms
            The Atoms object to operate on.

        Returns
        -------
        float
            Atomic density of the atoms in kg/m³.
        """
        volume = atoms.get_volume() * 1e-30  # Convert Å³ to m³
        total_mass = sum(atoms.get_masses()) * units._amu  # Convert amu to kg
        return total_mass / volume
