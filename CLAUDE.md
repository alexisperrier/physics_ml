# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project comparing different neural network approaches for learning and forecasting dynamics of physical systems governed by ordinary differential equations (ODEs). The primary focus is on the Lotka-Volterra predator-prey equations. The codebase implements three main model architectures:

1. **MLP** - Data-driven windowed input-to-output mapping
2. **RNN Encoder-Decoder** - Sequence-to-sequence learning with variable horizons
3. **PINNs** - Physics-Informed Neural Networks that incorporate ODE constraints into the loss function

## Key Development Commands

### Training Models
```bash
python main.py  # Trains using the default config (config/LV/train_RNN.yaml)
```

### Viewing Experiments
```bash
mlflow ui  # Opens MLflow dashboard at http://localhost:5000 to view training metrics and results
```

### Data Generation
```bash
python data_processing/generate_data.py  # Generates ODE trajectories and stores as Parquet files
```

### Running Evaluations
To evaluate a trained model, modify `main.py` or create a new evaluation config YAML file and call:
```python
eval_from_config("config/path/to/eval_config.yaml")
```

## Code Architecture

### Core Training Pipeline

**Entry Point**: `main.py`
- `train_from_config(config_path)` - Trains a model from YAML configuration
- `eval_from_config(config_path)` - Evaluates a trained model
- Uses MLflow for logging metrics and hyperparameters

**Utilities**: `utils.py`
- `load_trajectories_parquet()` - Loads trajectory data from Parquet files
- `build_model_from_config()` - Instantiates model architectures from config
- `autoregressive_forecast()` - Performs iterative forecasting on test trajectories
- `plot_predictions()` - Visualizes model predictions vs ground truth

### Model Architectures

**Location**: `models/` directory

- **MLP** (`models/MLP.py`):
  - Flattened windowed input → output mapping
  - Supports autoregressive forecasting
  - Simple baseline for data-driven approaches

- **RNN Encoder-Decoder** (`models/Seq2SeqRNN.py`):
  - Encodes history into a context vector
  - Decoder MLP expands context to forecast multiple timesteps
  - Handles variable forecast horizons
  - Autoregressive mode available

- **PINN Variants** (`models/PINN.py`):
  - `OdePINN` - Base single-trajectory PINN
  - `LotkaVolterraOdePINN` - Fixed LV parameters
  - `ParameterAgnosticOdePINN` - Learns across different parameter sets
  - `LVOdePINN` - Parameter-aware LV equations
  - `ParameterICAgnosticOdePINN` - Conditions on initial conditions
  - `NormedPINN` / `NormedLVOdePINN` - Batch-normalized variants with softplus outputs

  All PINNs use a weighted loss combining:
  - Residual loss (PDE constraint from ODE equations)
  - Initial condition loss
  - Data/regression loss

### Data Handling

**Location**: `data_processing/` directory

- **Data Generation** (`data_processing/generate_data.py`):
  - Simulates ODE trajectories using SciPy's Radau solver
  - Stores trajectories as Parquet files for efficient access
  - Supports multiple ODE systems (Lotka-Volterra, Lorenz, Chen)

- **Datasets** (`data_processing/datasets.py`):
  - `TrajectoryWindowDataset` - Sliding windows for RNN/MLP training
  - `OdePINNDataset` - Collocation points for PINN physics loss
  - Supports decimation (downsampling) for multi-scale learning

### Configuration System

**Location**: `config/` directory

All training and evaluation is controlled via YAML configs with these sections:

```yaml
data:
  data_root: "path/to/trajectories"  # Directory with Parquet trajectory files
  decimation: 10                     # Downsampling factor
  val_ratio: 0.2                     # Validation split
  input_len: 10                      # Context window
  target_len: 50                     # Forecast horizon

model:
  type: "RNN"                        # Model type: RNN, MLP, or PINN variant
  hidden_size: 64                    # Hidden dimension
  num_layers: 2                      # Number of layers
  # ... model-specific hyperparameters

training:
  batch_size: 32
  max_epochs: 100
  learning_rate: 0.001
  num_workers: 4
  mlflow_uri: "http://localhost:5000"
```

## Important Patterns and Conventions

### Dataset Windows

The codebase uses sliding windows for training:
- **Input length** (`input_len`): Past states used for prediction
- **Target length** (`target_len`): Forecast horizon
- Different models interpret these differently:
  - MLP: Flattens input_len states to predict next target_len states
  - RNN: Uses input_len states → hidden state → target_len predictions
  - PINN: Uses continuous-time formulation with ODE constraints

### Autoregressive Forecasting

For long-horizon forecasting, use `autoregressive_forecast()` from `utils.py`:
- Iteratively applies the model
- Each prediction becomes next input (for RNNs)
- For MLPs, uses sliding window approach
- Returns full trajectory predictions

### PINN Physics Loss

PINNs compute residuals by:
1. Taking derivatives of network outputs w.r.t. time using autograd
2. Computing ODE residuals: `du/dt - f(u)` where f is the ODE function
3. Combining residuals with data and IC losses

The `t_max` parameter normalizes time for numerical stability.

### Data Storage

Trajectories are stored as Parquet files with columns:
- `t` - time values
- `y0`, `y1`, ... - state variables (e.g., predator/prey populations)
- Optional: `params` - ODE parameters for parameter-conditioned models

## Common Development Tasks

### Adding a New ODE System

1. Implement ODE function in `data_processing/generate_data.py`
2. Create a PINN variant in `models/PINN.py` that computes the correct physics loss
3. Add data generation config and training configs

### Training a PINN vs RNN

Compare approaches by:
1. Creating matching configs in `config/LV/` (or similar)
2. Training both with `python main.py` (updating default config path)
3. Viewing results in MLflow UI
4. Using autoregressive evaluation from `utils.py` for long-horizon assessment

### Debugging Model Behavior

- Check `main.py` for loaded config and model instantiation
- Use `build_model_from_config()` to inspect architecture
- Plot training curves in MLflow UI (loss convergence, metrics over time)
- Use `plot_predictions()` from `utils.py` to visualize on test set

## Tech Stack

- **PyTorch + PyTorch Lightning**: Model training with GPU support
- **MLflow**: Experiment tracking, hyperparameter logging, artifact storage
- **SciPy**: ODE simulation (Radau method for stiff systems)
- **PyArrow/Parquet**: Efficient trajectory storage
- **Matplotlib**: Result visualization
- **NumPy/Pandas**: Numerical computation and data manipulation
