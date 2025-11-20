# Physics ML - Neural Network Models for ODE Systems

This is a machine learning research project that trains and compares different neural
network architectures to learn and forecast the dynamics of physical systems governed
by ordinary differential equations (ODEs).

## Main Goal

Learn dynamics of ODE systems (particularly Lotka-Volterra predator-prey equations)
from trajectory data and make predictions on unseen trajectories.

## Key Components

1. ODE Systems Implemented:
- Lotka-Volterra (predator-prey) - primary focus
- Lorenz system
- Chen system

2. Model Architectures:
- MLP (Multi-layer Perceptron) - data-driven approach
- RNN (Encoder-Decoder) - sequence-to-sequence learning
- PINNs (Physics-Informed Neural Networks) - incorporates physics equations into loss
function
- Multiple variants: parameter-agnostic, IC-conditioned, normalized versions

3. Tech Stack:
- PyTorch + PyTorch Lightning for training
- MLflow for experiment tracking
- SciPy for ODE integration (Radau solver)
- Parquet for efficient trajectory storage

4. Directory Structure:
├── config/           # YAML configs for experiments
├── data_processing/  # Data generation & PyTorch datasets
├── models/           # Neural network architectures
├── plots/            # Visualization outputs
├── main.py          # Training/evaluation entry point
└── mlflow.db        # Experiment tracking database

## How It Works

1. Generate data (python data_processing/generate_data.py): Simulate ODE trajectories
with varying parameters
2. Split data (python data_processing/split_data.py): Divide trajectories into
train_val (80%) and test (20%) sets with fixed random seed for reproducibility
3. Train models (python main.py): Compare data-driven (RNN/MLP) vs physics-informed
(PINN) approaches
4. Evaluate: Autoregressive forecasting on test trajectories
5. Track: MLflow logs all experiments, metrics, and artifacts

## Quick Start

### Set up environment and dependencies
make setup

### Generate synthetic trajectory data
make generate-data

### Split data into train/test (80/20)
python data_processing/split_data.py

### Start MLflow tracking server (in separate terminal)
make mlflow

### Start Jupyter for exploration (in separate terminal)
make jupyter

### Train a model
make train

### Or use direct command
python main.py --config config/alexis/train_MLP.yaml

### Resume training from latest checkpoint
make train-resume

### Evaluate model
make eval

## Training Features

- Autoregressive training: Model learns to handle its own prediction errors
- Teacher forcing: Configurable blend of ground truth (80%) and predictions (20%)
- Checkpoint resumption: Train for N epochs, evaluate, then continue if results look good
- MLflow integration: All metrics logged automatically

## Example Workflow for Iterative Training

### Train for first batch of epochs
make train

### Check results in MLflow UI
mlflow ui

### View evaluation plots
make eval

### If results promising, continue training
make train-resume

## Model Architectures

This codebase implements multiple neural network models for learning dynamical systems. Choose based on your use case, computational budget, and whether you have physics equations available.

### 1. WindowMLP - Data-Driven Baseline

**Best for:** Fast inference, fixed forecast horizons, baseline comparisons

**Architecture:**
- Flattens sliding window of past states: `[batch, state_dim, input_len]` → `[batch, state_dim*input_len]`
- Passes through configurable hidden layers with activation function (default: tanh)
- Linear output layer produces future states: `[batch, state_dim*target_len]`
- Reshapes output to `[batch, state_dim, target_len]`

**Key Hyperparameters:**
- `state_dim`: Number of state variables (e.g., 2 for predator-prey)
- `input_len`: Context window size (how many past timesteps to use)
- `target_len`: Forecast horizon (fixed number of future timesteps)
- `hidden_sizes`: Tuple of hidden layer widths (e.g., `[256, 512, 256]`)
- `activation`: Activation function ('tanh', 'relu', 'gelu', 'elu', 'silu')

**Strengths:**
- Simplest and fastest model
- No recurrence = single forward pass
- Effective baseline for short-horizon forecasting

**Limitations:**
- Fixed receptive field (only sees `input_len` timesteps)
- No temporal structure modeling
- Cannot capture long-range dependencies

**Training:**
- Supervised learning with MSE loss
- Optional: Autoregressive training exposes model to its own predictions

**Example Configuration:**
```yaml
model:
  class: models.MLP.WindowMLP
  params:
    state_dim: 2
    input_len: 100
    target_len: 10
    hidden_sizes: [256, 512, 256]
    activation: tanh
    lr: 1e-5
```

---

### 2. EncoderDecoderRNN - Sequence-to-Sequence Learning

**Best for:** When temporal structure matters, variable forecast horizons, compact latent representations

**Architecture:**
- **Encoder:** RNN (LSTM/GRU variant) processes past sequence → hidden state `h`
  - Input: `[batch, input_len, state_dim]`
  - Output: Context vector `h_n` of shape `[batch, hidden_dim]`
- **Decoder:** MLP expands hidden state to future predictions
  - Input: `h_n` from encoder
  - Output: `[batch, state_dim, target_len]` (parallel prediction)

**Key Hyperparameters:**
- `state_dim`: Number of state variables
- `input_length`: Sequence length for encoder
- `target_length`: Number of future timesteps (fixed during training)
- `hidden_dim`: Size of RNN hidden state (latent representation)
- `num_layers`: Number of stacked RNN layers

**Strengths:**
- Captures temporal dependencies via RNN memory
- Learns compressed latent representation
- More expressive than MLP

**Limitations:**
- Slower inference than MLP (sequential RNN operations)
- Still produces fixed-length outputs (but works better with autoregressive extension)
- More parameters → higher memory usage

**Training:**
- Supervised learning with MSE loss
- Encoder compresses entire history into context vector
- Decoder generates all future steps in parallel (can use autoregressive training for improvement)

**Example Configuration:**
```yaml
model:
  class: models.Seq2SeqRNN.EncoderDecoderRNN
  params:
    state_dim: 2
    input_length: 100
    target_length: 10
    hidden_dim: 64
    num_layers: 2
    lr: 1e-5
```

---

### 3. Physics-Informed Neural Networks (PINNs)

PINNs embed knowledge of physics equations into the learning process. Unlike data-driven models, PINNs:
- Learn continuous functions `u(t)` instead of discrete predictions
- Enforce ODE residuals: `∂u/∂t - f(u, θ) = 0` via automatic differentiation
- Generalize better to unseen parameter values and initial conditions
- Work with sparse, noisy data

**Training Approach:**
All PINNs combine three loss components:
1. **Residual Loss:** `||∂u/∂t - f(u, θ)||²` - enforces physics
2. **Initial Condition Loss:** `||u(t₀) - u₀||²` - correct starting point
3. **Data Loss:** `||u(tᵢ) - yᵢ||²` - fit to observations

#### 3.1 Basic OdePINN

**Best for:** Learning a single trajectory with known ODE equations

**Architecture:**
- Fully connected network `t → u(t)`
- Takes normalized time as input
- Outputs system state

**Limitations:**
- Can only learn one specific trajectory
- Must know exact ODE equations
- Not generalizable to different parameters

#### 3.2 LotkaVolterraOdePINN

**Best for:** Lotka-Volterra systems with fixed parameters

Hardcoded Lotka-Volterra dynamics:
```
x' = α*x - β*x*y
y' = δ*x*y - γ*y
```

#### 3.3 ParameterAgnosticOdePINN

**Best for:** Training on data with varying ODE parameters

**Key Enhancement:**
- Takes parameter vector `θ` as input alongside time `t`
- Network learns `u(t, θ)` - continuous function of both time and parameters
- Can interpolate/extrapolate to unseen parameter values

**Use Cases:**
- Ecosystem with varying α, β, γ, δ
- Chemical reactions at different temperatures
- Physical systems at different conditions

#### 3.4 LVOdePINN

**Best for:** Lotka-Volterra with varying parameters

Combines parameter-aware learning with Lotka-Volterra dynamics.

#### 3.5 ParameterICAgnosticOdePINN

**Best for:** Maximum generalization across parameters AND initial conditions

**Key Enhancement:**
- Network learns `u(t, θ, u₀)`
- Conditions on both ODE parameters and initial conditions
- Single model covers entire phase space

#### 3.6 LVOdePICPINN

**Best for:** Complete Lotka-Volterra model

Combines full generalization with hardcoded Lotka-Volterra dynamics.

#### 3.7 NormedPINN & NormedLVOdePINN (Production-Ready)

**Best for:** Numerical stability, production deployments

**Key Enhancements:**
- **Batch-wise normalization:** Normalizes inputs (`t`, `θ`, `u₀`) by their statistics
- **Tanh activations:** More stable than ReLU
- **Softplus output:** Enforces positivity `u_positive = log(1 + exp(u))`

**When to use:**
- Inputs/outputs have different scales
- Physical constraint: quantities must be non-negative
- Dealing with stiff systems or numerical instability
- Production/publication quality results

**Example Configuration:**
```yaml
model:
  class: models.PINN.NormedLVOdePINN
  params:
    input_size: 1
    theta_dim: 4
    hidden_size: 256
    output_size: 2
    n_layer: 4
    lr: 1e-5
    t_max: 50
    loss_weights:
      residual: 1.0
      ic: 0.1
      data: 2.0
```

---

### Quick Comparison

| Aspect | WindowMLP | RNN | Basic PINN | Param-Aware PINN | Normed PINN |
|--------|-----------|-----|-----------|------------------|------------|
| **Paradigm** | Data-driven | Data-driven | Physics-informed | Physics-informed | Physics-informed |
| **Inference Speed** | Fastest ⚡ | Medium ⏱️ | Slow 🐌 | Slow 🐌 | Slow 🐌 |
| **Memory Usage** | Lowest | Medium | Low | Low | Low |
| **Generalization** | Fixed window | Variable (via AR) | Single trajectory | Cross-parameter | Cross everything |
| **Data Efficiency** | Needs lots | Moderate | Few samples OK | Few samples OK | Few samples OK |
| **Requires Physics Eqs** | ❌ No | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Extrapolation** | Poor | Moderate | Good | Excellent | Excellent |
| **Numerical Stability** | Good | Good | Fair | Fair | Excellent |
| **Best Use Case** | Baseline | Temporal structure | Specific trajectory | Parameter sweep | Production |

---

### Decision Tree: Which Model to Use?

```
Do you have physics equations?
├─ NO → Use data-driven models
│  ├─ Need fast inference?
│  │  └─ YES → WindowMLP ⚡
│  └─ NO → Need temporal structure?
│     └─ YES → EncoderDecoderRNN
│
└─ YES → Use Physics-Informed Models
   ├─ Learning single trajectory?
   │  └─ YES → OdePINN
   │
   └─ NO, multiple scenarios
      ├─ Varying parameters only?
      │  └─ YES → ParameterAgnosticOdePINN / LVOdePINN
      │
      ├─ Varying parameters AND initial conditions?
      │  └─ YES → ParameterICAgnosticOdePINN / LVOdePICPINN
      │
      └─ Production deployment? Numerical issues?
         └─ YES → NormedPINN / NormedLVOdePINN ✅
```

---

### Training Tips by Model

**WindowMLP:**
- Increase `hidden_sizes` if underfitting
- Decrease `input_len` if overfitting to short-term patterns
- Tanh works well for oscillatory systems

**RNN:**
- Use `num_layers=2-3` for good balance
- Larger `hidden_dim` if capturing complex dynamics
- Monitor for gradient explosion with RNNs

**PINNs:**
- Use `loss_weights` to balance physics vs data:
  - High `residual`: Strict physics enforcement
  - High `data`: Trust observations more
- Normalize time: `t_max` should be maximum time in dataset
- Use normalized variants for robustness

