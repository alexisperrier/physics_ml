# Local Development Setup Guide

This guide walks you through setting up the Physics ML project on your local machine.

## Prerequisites

- **Python 3.10 or higher** (check with `python3 --version`)
- **Git** (for version control)
- **uv** (fast Python package manager - install with `pip install uv` or see [uv installation](https://docs.astral.sh/uv/getting-started/installation/))
- **~2GB disk space** (for dependencies and generated data)
- Optional: **CUDA 11.8+** (for GPU acceleration with PyTorch)

## Quick Setup (Automated)

If you're on macOS or Linux, run the automated setup script:

```bash
cd /path/to/physics_ml
chmod +x setup.sh
./setup.sh
```

This will handle steps 1-3 below. Then skip to **Step 4: Generate Data**.

## Manual Setup (5 Steps)

### Step 1: Create Virtual Environment

```bash
cd /path/to/physics_ml
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR on Windows:
# .venv\Scripts\activate
```

Verify activation (you should see `(.venv)` in your terminal prompt):
```bash
which python  # Should show path ending in .venv/bin/python
```

### Step 2: Install Dependencies

```bash
uv pip install -e .
```

This installs all dependencies specified in `pyproject.toml`. The `uv` tool is significantly faster than `pip` and handles dependency resolution more efficiently.

Verify installation:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import pytorch_lightning; print(f'PyTorch Lightning installed')"
python -c "import mlflow; print(f'MLflow {mlflow.__version__}')"
```

### Step 3: Create Directory Structure

```bash
mkdir -p raw_data
mkdir -p artifacts
mkdir -p plots
```

### Step 4: Generate Training Data

The project requires ODE trajectory data for training. Generate it with:

```bash
python data_processing/generate_data.py
```

This will create Parquet files in `data_processing/data/lotka_volterra_trajectories/`.

Then organize the data into the structure expected by training configs:

```bash
# Create target directories
mkdir -p raw_data/lotka_volterra_trajectories_T50_beta05_20_IC/{train_val,test}

# Move generated parquet files (adjust paths as needed)
# Assuming 80% train_val, 20% test split
cp data_processing/data/lotka_volterra_trajectories/*.parquet raw_data/lotka_volterra_trajectories_T50_beta05_20_IC/train_val/
# Move ~20% of files to test directory for a test split
```

**Note:** You can adjust the data generation parameters in `data_processing/generate_data.py` if needed (T_max, beta range, number of trajectories, etc.).

### Step 5: Start MLflow Server

Open a new terminal window and start the MLflow tracking server:

```bash
cd /path/to/physics_ml
source .venv/bin/activate
mlflow ui
```

The MLflow dashboard will be available at **http://localhost:5000**. Leave this running while training models.

## Running Training

In another terminal, train a model:

```bash
cd /path/to/physics_ml
source .venv/bin/activate
python main.py
```

This trains using the default config: `config/LV/train_RNN.yaml`

To train with a different config:

```bash
# Edit main.py line where train_from_config is called, or create a new script:
python -c "from utils import train_from_config; train_from_config('config/LV/train_PINN.yaml')"
```

## Verifying Your Setup

After following the steps above, verify everything works:

```bash
# 1. Check Python packages
python -c "import torch, pytorch_lightning, mlflow, scipy, pandas; print('✓ All packages installed')"

# 2. Check data exists
ls raw_data/lotka_volterra_trajectories_T50_beta05_20_IC/train_val/*.parquet | head -5

# 3. Check MLflow is running (in separate terminal)
curl http://localhost:5000  # Should return HTML (MLflow UI)

# 4. Start a simple training
python main.py --help  # Check if main.py is executable
```

## GPU Support (Optional)

If you have a CUDA-capable GPU and want to accelerate training:

1. **Install CUDA toolkit**: Follow [NVIDIA CUDA installation guide](https://developer.nvidia.com/cuda-downloads)
2. **Reinstall PyTorch with CUDA support**:
   ```bash
   pip uninstall torch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify CUDA is detected:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

The code automatically detects and uses CUDA if available.

## Project Structure

```
.
├── main.py                          # Training/evaluation entry point
├── utils.py                         # Utility functions
├── pyproject.toml                  # Project metadata and dependencies
├── SETUP.md                        # This file
├── setup.sh                        # Automated setup script
├── config/                         # Training configs (YAML)
│   ├── LV/
│   │   ├── train_RNN.yaml
│   │   ├── train_PINN.yaml
│   │   └── eval_*.yaml
│   └── ...
├── models/                         # Neural network architectures
│   ├── MLP.py
│   ├── PINN.py
│   ├── Seq2SeqRNN.py
│   └── seq2seqmodule.py
├── data_processing/               # Data generation and datasets
│   ├── generate_data.py
│   ├── datasets.py
│   └── data/                      # Generated data (created at runtime)
├── raw_data/                      # Training/test data (created by you)
├── artifacts/                     # Model checkpoints (created at runtime)
├── plots/                         # Visualizations (created at runtime)
└── mlflow.db                      # Experiment tracking database
```

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Ensure virtual environment is activated (`source .venv/bin/activate`)

### Issue: "Connection refused" when accessing MLflow at http://localhost:5000
**Solution:** Start MLflow server in a separate terminal: `mlflow ui`

### Issue: Data directory not found when training
**Solution:** Verify data was generated and organized:
```bash
ls raw_data/lotka_volterra_trajectories_T50_beta05_20_IC/train_val/
# Should show .parquet files
```

### Issue: Out of memory errors
**Solution:** Reduce `batch_size` in config YAML file (try 16 or 8 instead of 32)

### Issue: Slow training on CPU
**Solution:** Either:
1. Set up CUDA support (see GPU Support section above)
2. Reduce `num_workers` in config (try 0 or 1)
3. Reduce dataset size or `max_epochs`

## Next Steps

1. Read the **CLAUDE.md** file for architecture overview and development guidance
2. Check **README.md** for project goals and model descriptions
3. Explore configs in `config/LV/` to understand training parameters
4. View training progress in MLflow UI (http://localhost:5000)

## Need Help?

- **PyTorch**: https://pytorch.org/docs/
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/stable/
- **MLflow**: https://mlflow.org/docs/latest/
- **SciPy ODE**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
