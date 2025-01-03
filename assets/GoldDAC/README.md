# GolDDAC dataset

The GoldDAC dataset is designed for fine-tuning foundation models on MOFs for Direct Air Capture (DAC) applications. This dataset provides a curated selection of MOF structures across various metal types, specifically targeted at enhancing the predictive capabilities of models in CO₂ capture scenarios under humid conditions.

## Dataset Description

### Train/Validation

- Total Structures: 60 MOFs, with three unique structures per metal type (20 metal types in total).
- Split Strategy: 2 MOFs for training, 1 MOF for validation.
- Sampling Details:
  - 2 configurations from repulsion, 1 configuration from equilibrium, and 3 configurations from weak-attraction for each MOF structure.
  - 2 gas configurations (CO₂ and H₂O) for each MOF configuration.

### Test

- Total Structures: 26 MOFs, identified in the literature as promising for CO2 capture.
- Sampling Details:
  - 2 configurations from repulsion, 1 configuration from equilibrium, and 3 configurations from weak-attraction for each MOF structure.
  - 2 gas configurations (CO₂ and H₂O) for each MOF configuration.

| Split      | MOF structures | Repulsion   | Equilibrium | Weak-attraction | Total        |
|------------|-------------|-------------|-----------------|-------------|----------------------------------|
| Train      | 40          | 40 \* 2 \* 2  | 40 \* 1 \*2  | 40 \* 3 \* 2 | 480 (system) + 40 (framework) + 2 (gas) |
| Val        | 20          | 20 \* 2 \* 2  | 20 \* 1 \*2  | 20 \* 3 \* 2 | 240 (system) + 20 (framework) + 2 (gas) |
| Test       | 26          | 26 \* 2 \* 2  | 26 \* 1 \*2  | 26 \* 3 \* 2 | 312 (system) |

>[!NOTE]
> system: framework + gas | framework: MOF structure | gas: CO₂ and H₂O

## Dataset Structure

The dataset is provided in an ASE-compatible format, with each configuration stored as an ASE `Atoms` object. The following keys are included:

- `atoms.info["REF_energy"]`: Reference energy for each configuration.
- `atoms.arrays["REF_forces"]`: Reference forces for each configuration.
- `atoms.info["name"]`: Name of the data.
- `atoms.info["group"]`: MOF structure name.
- `atoms.info["metal"]`: Metal type.

For the test set, additional information is provided in `atoms.info`:

- `atoms.info["DFT_E_int"]`: Internal energy of the system.
- `atoms.info["DFT_E_total"]`: Total energy of the system.
- `atoms.info["DFT_E_mof"]`: Energy of the MOF framework.
- `atoms.info["DFT_E_gas"]`: Energy of the gas molecules.

The XYZ files can be read using ASE, and the reference energies and forces can be accessed as shown below:

```python
from ase.io import read

train = read("train.xyz", ":")
val = read("val.xyz", ":")
test = read("test.xyz", ":")
```

>[!TIP]
> The tutorial notebook for `GoldDAC` is available at [`turorial_golddac.ipynb`(tutorial_golddac.ipynb).
