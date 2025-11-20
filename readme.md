Physics ML - Neural Network Models for ODE Systems

  This is a machine learning research project that trains and compares different neural
  network architectures to learn and forecast the dynamics of physical systems governed
  by ordinary differential equations (ODEs).

  Main Goal

  Learn dynamics of ODE systems (particularly Lotka-Volterra predator-prey equations)
  from trajectory data and make predictions on unseen trajectories.

  Key Components

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

  How It Works

  1. Generate data (python data_processing/generate_data.py): Simulate ODE trajectories
  with varying parameters
  2. Split data (python data_processing/split_data.py): Divide trajectories into
  train_val (80%) and test (20%) sets with fixed random seed for reproducibility
  3. Train models (python main.py): Compare data-driven (RNN/MLP) vs physics-informed
  (PINN) approaches
  4. Evaluate: Autoregressive forecasting on test trajectories
  5. Track: MLflow logs all experiments, metrics, and artifacts

  Quick Start

  # Set up environment and dependencies
  make setup

  # Generate synthetic trajectory data
  make generate-data

  # Split data into train/test (80/20)
  python data_processing/split_data.py

  # Start MLflow tracking server (in separate terminal)
  make mlflow

  # Start Jupyter for exploration (in separate terminal)
  make jupyter

  # Train a model
  make train

  # Or use direct command
  python main.py --config config/alexis/train_MLP.yaml

  # Resume training from latest checkpoint
  make train-resume

  # Evaluate model
  make eval

  Training Features

  - Autoregressive training: Model learns to handle its own prediction errors
  - Teacher forcing: Configurable blend of ground truth (80%) and predictions (20%)
  - Checkpoint resumption: Train for N epochs, evaluate, then continue if results look good
  - MLflow integration: All metrics logged automatically

  Example Workflow for Iterative Training

  # Train for first batch of epochs
  make train

  # Check results in MLflow UI
  mlflow ui

  # View evaluation plots
  make eval

  # If results promising, continue training
  make train-resume

  Current Focus

  Based on recent commits and config files, you're working with Lotka-Volterra systems
  with varying β parameters (0.5-2.0) and comparing RNN vs PINN performance.

  This is a well-structured research codebase for comparing different ML approaches to
  learning dynamical systems!