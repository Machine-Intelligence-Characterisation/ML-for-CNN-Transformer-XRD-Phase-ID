## XRD Phase Classification using CNN with transformer – Training & Validation

This repository contains code to **train and validate deep learning models for X-ray diffraction (XRD)–based crystal structure classification**, with a workflow optimized for the **Monash M3 cluster**. The project replicates the CPICANN-based workflow reported in *IUCrJ* (2024) by **Gandhi et al.**, “Classifying crystal structures and phases from X-ray diffraction patterns using convergent pathways in neural networks” ([link](https://journals.iucr.org/m/issues/2024/04/00/fc5077/)).

You can:
- **Quickly validate pretrained models** (most common workflow, via `quick_test.py`)
- **Train models from scratch or fine-tune** (CNN, VIT/Attention-only, CPICANN)
- Work with both the **D1_full** dataset (paper setup) and **custom SimXRD-4M–style datasets**

---

## 1. Quick Start – Validate Pretrained Models

This is the **fastest way to use the repo**.

### 1.1 On M3 (recommended)

1. **Load modules & activate environment** (adjust to your M3 setup):

```bash
module load cuda   # or appropriate CUDA module
module load anaconda
conda activate cpicann  # or your env
cd /home/<user>/.../Ankita_CPICANN_Phase
```

2. **Run the quick validation script** (uses pretrained models in `trained_models/`):

```bash
python quick_test.py
```

This will:
- Load one or more **pretrained models** (CNN, VIT, CPICANN)
- Run a **small validation/evaluation** pipeline
- Print key metrics (accuracy, confusion summaries, etc.)

3. (Optional) **Bi-phase / extra checks**:

```bash
python quick_test_bi_phase.py
```

### 1.2 Locally (single GPU workstation)

1. Create/activate a Python environment (see Section 2).
2. From the repo root:

```bash
python quick_test.py
```

Use this when you just want to **verify the models and pipeline work** without full training.

---

## 2. Environment Setup

You can use the same environment on both **M3** and **local GPU** machines.

### 2.1 Create a Python environment

Using `conda`:

```bash
conda create -n cpicann python=3.10 -y
conda activate cpicann
```

Or with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2.2 Install dependencies

Install core packages (versions can be tuned to your system):

```bash
pip install torch torchvision torchaudio \
    numpy pandas tqdm scikit-learn \
    matplotlib tensorboard
```

On **M3**, also ensure you load the usual modules for your account (e.g. `module load cuda`, `module load anaconda`) as per standard setup.

---

## 3. Repository Structure (High-Level)

- **Top-level scripts**
  - `quick_test.py`, `quick_test_bi_phase.py` – **Quick sanity checks / validation** using pretrained models.
  - `test_pretrained_models.py` – More systematic evaluation of all pretrained models.
  - `train_cnn_d1.py`, `train_cpicann_d1.py`, `train_vit_d1.py` – Entry points for **single-phase D1_full training**.
  - `train_simxrd.py` – Training CPICANN on **custom SimXRD-4M–style data**.
  - `analyze_single_phase.py`, `analyze_bi_phase.py` – Deeper analysis & plots.
  - `prepare_simxrd_data.py` – Preprocessing script for custom XRD datasets.

- **Core code (`src/`)**
  - `src/models/`
    - `CNNonly.py` – CNN baseline model.
    - `ATTENTIONonly.py` – VIT/attention-only model.
    - `CPICANN.py` – Hybrid CNN + transformer model (main method).
  - `src/data_loading/`
    - `dataset.py`, `datasetCPICANN_loader.py`, `data_format.py` – Dataset definitions and format helpers.
    - `streaming_loader.py` – Streaming data loading for large datasets.
    - `auth_helper.py` – Helpers for authenticated / remote data access (if used).
  - `src/training/`
    - `train_single-phase.py`, `train_bi-phase.py` – Training scripts for single-phase and bi-phase.
    - `val_single-phase.py`, `val_bi-phase.py` – Validation scripts for single-phase and bi-phase.
  - `src/utils/`
    - `focal_loss.py` – Focal loss (for class imbalance).
    - `logger.py` – Logging utilities.

- **HPC / SLURM (`monash_HPC_commands/`)**
  - `sh/*.sh` – SLURM job scripts:
    - Single-phase: `train_cnn_d1.sh`, `train_vit_d1.sh`, `train_cpicann_d1.sh`, `train_cpicann_original.sh`
    - Bi-phase: `train_bi-phase_cnn_d1.sh`, `train_bi-phase_vit_d1.sh`, `train_bi-phase_cpicann_d1.sh`
  - `slurm_outputs/*.out`, `*.err` – Job logs (training output and errors).

- **Guides / analysis**
  - `TRAINING_GUIDE.md` – Summary of training setup for CNN/VIT/CPICANN on D1_full.
  - `TRAINING_D1_FULL_GUIDE.md` – Detailed D1_full training guide.
  - `SIMXRD_TRAINING_GUIDE.md` – Training CPICANN on SimXRD-4M/custom datasets.
  - `analysis/` – Scripts & markdown for plots, tables, and results.

- **Models & outputs**
  - `trained_models/` – Pretrained models (e.g. `CPICANNsingle_phase_D1.pth`, `CNNonlyD1.pth`, `ATTENTIONonlyD1.pth`).

---

## 4. Data Layouts

### 4.1 D1_full Dataset (Paper Setup)

The referenced IUCrJ paper used the **D1_full** dataset (not included in this repository).  
For the exact directory structure and annotation format expected by the training scripts, see:
- `TRAINING_GUIDE.md`
- `TRAINING_D1_FULL_GUIDE.md`

### 4.2 Custom / SimXRD-4M Datasets

For a custom SimXRD-style dataset, see `SIMXRD_TRAINING_GUIDE.md`, which explains:
- How to organise your raw data (patterns, metadata, labels)
- How to run `prepare_simxrd_data.py` to generate train/val splits and annotations
- How that prepared data is consumed by `train_simxrd.py`

---

## 5. Validating Models in Detail

### 5.1 `quick_test.py` (Most Common Workflow)

This script is designed for **fast validation of pretrained models**:

- Loads one or more pretrained `.pth` files from `trained_models/`
- Runs inference on a small/standard validation set
- Prints aggregate metrics and possibly example predictions

Typical usage:

```bash
python quick_test.py
```

You can modify or extend it to:
- Change which model is loaded
- Point to different data/annotation files
- Save plots or confusion matrices

### 5.2 Other Validation Scripts

- `test_pretrained_models.py` – Compares all available pretrained models in a more systematic way.
- `src/training/val_single-phase.py` – Flexible single-phase validation script.
- `src/training/val_bi-phase.py` – Validation for bi-phase models.

---

## 6. Training on M3 – Single-Phase

If you move beyond quick validation and want to **train models from scratch or fine-tune**:

### 6.1 Single-Phase D1_full Training via SLURM (Recommended)

From the repo root on M3:

```bash
# CNN baseline
sbatch monash_HPC_commands/sh/train_cnn_d1.sh

# VIT / Attention-only
sbatch monash_HPC_commands/sh/train_vit_d1.sh

# CPICANN (main hybrid model)
sbatch monash_HPC_commands/sh/train_cpicann_d1.sh
```

Each script:
- Sets up modules/environment
- Calls the appropriate top-level training script (`train_cnn_d1.py`, `train_vit_d1.py`, `train_cpicann_d1.py`)
- Writes logs to `monash_HPC_commands/slurm_outputs/`
- Saves models to `trained_models/`

Monitor jobs:

```bash
squeue -u $USER
tail -f monash_HPC_commands/slurm_outputs/cnn_d1_train_*.out
tail -f monash_HPC_commands/slurm_outputs/cpicann_d1_train_*.out
```

### 6.2 Run Training Directly (Local or Interactive M3)

Example (CPICANN single-phase training):

```bash
python train_cpicann_d1.py \
    --epochs 150 \
    --batch_size 64 \
    --lr 8e-5
```

Most training scripts support:
- `--epochs`
- `--batch_size`
- `--lr`
- `--embed_dim`
- `--device` (e.g. `cuda:0`)

See `TRAINING_GUIDE.md` and `TRAINING_D1_FULL_GUIDE.md` for recommended defaults and rationale.

---

## 7. Training on M3 – Bi-Phase

Bi-phase training fine-tunes a **pretrained single-phase model** to handle mixtures of phases.

### 7.1 SLURM Scripts

From the repo root:

```bash
# Bi-phase CNN
sbatch monash_HPC_commands/sh/train_bi-phase_cnn_d1.sh

# Bi-phase VIT
sbatch monash_HPC_commands/sh/train_bi-phase_vit_d1.sh

# Bi-phase CPICANN
sbatch monash_HPC_commands/sh/train_bi-phase_cpicann_d1.sh
```

These expect:
- A suitable pretrained single-phase `.pth` in `trained_models/`
- D1_full data in `training_data/D1_full/`

### 7.2 Flexible Direct Script

You can also call the flexible script directly:

```bash
python src/training/train_bi-phase_flexible.py \
    --model_type cpicann \
    --load_path trained_models/CPICANNsingle_phase_D1.pth \
    --data_dir_train training_data/D1_full/train \
    --data_dir_val training_data/D1_full/val \
    --anno_struc training_data/D1_full/anno_struc.csv \
    --epochs 150 \
    --batch_size 64 \
    --lr 8e-4
```

Change `--model_type` to `cnn` or `vit` and `--load_path` to the corresponding pretrained file as needed.

---

## 8. Training on Custom / SimXRD-4M Data

See `SIMXRD_TRAINING_GUIDE.md` for full details. Basic workflow:

1. **Prepare data**:

```bash
python prepare_simxrd_data.py \
    --input_dir /path/to/simxrd_data \
    --output_dir /path/to/prepared_data
```

2. **Train CPICANN**:

```bash
python train_simxrd.py \
    --data_dir /path/to/prepared_data \
    --output_dir /path/to/output_dir \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4
```

3. **Validate** with `validate_model.py` or a custom script, following `SIMXRD_TRAINING_GUIDE.md`.

---

## 9. Monitoring, Outputs, and TensorBoard

- **SLURM logs (M3)**:
  - `monash_HPC_commands/slurm_outputs/*.out` – Standard output.
  - `monash_HPC_commands/slurm_outputs/*.err` – Errors and tracebacks.

- **Model files**:
  - `trained_models/` – Main directory for saved checkpoints and best models.
  - Some workflows also use `training_outputs/checkpoints/`.

- **TensorBoard** (if enabled in scripts):

```bash
tensorboard --logdir /path/to/output_dir
```

Open `http://localhost:6006` in a browser (via SSH tunnelling if on M3).

---

## 10. Troubleshooting

- **CUDA out of memory**:
  - Reduce `--batch_size`.
  - Lower `--embed_dim` or select a smaller model.

- **Data not found / path errors**:
  - Confirm dataset structure under `training_data/D1_full/` or your prepared data directory.
  - Check that SLURM scripts (`monash_HPC_commands/sh/*.sh`) use the correct absolute paths.

- **Slow training**:
  - Try a larger batch size (if GPU memory allows).
  - Ensure you’re actually using GPU (`--device cuda:0`).

- **Poor convergence**:
  - Adjust learning rate (`--lr`).
  - Increase epochs.
  - Inspect annotations and label distributions.

For M3-specific issues, inspect the `.out`/`.err` files in `monash_HPC_commands/slurm_outputs/`.

---

If you mainly want to **quickly validate models**, focus on:
- Setting up the environment (Section 2)
- Running `quick_test.py` (Section 1)
- Optionally exploring `test_pretrained_models.py` and the analysis scripts for deeper evaluation.

