import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import decimate

from .generate_data import read_trajectories_parquet_as_dicts, as_torch as sample_to_torch

class TrajectoryWindowDataset(Dataset):
    """
    Sliding window dataset for RNN/MLP seq2seq models.

    Creates (past_window, future_window) pairs by sliding across trajectories.

    Used by: MLP (WindowMLP), RNN (AutoencoderRNN) models

    Example with input_length=100, target_length=10, step=10:
        Trajectory: [y_0, y_1, ..., y_N]
        Window 1: past=[y_0:y_100] → future=[y_100:y_110]
        Window 2: past=[y_10:y_110] → future=[y_110:y_120]
        Window 3: past=[y_20:y_120] → future=[y_120:y_130]
        ...

    Each sample returns:
        - past: Tensor of shape [state_dim, input_length] - context window
        - future: Tensor of shape [state_dim, target_length] - target to predict
    """

    def __init__(
        self,
        samples: Sequence[Dict],
        input_length: int,
        target_length: int,
        *,
        step: int = 1,
    ) -> None:
        if input_length <= 0 or target_length <= 0:
            raise ValueError("Window lengths must be positive.")
        if step <= 0:
            raise ValueError("Step must be positive.")
        
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.step = int(step)

        self.sequences: List[torch.Tensor] = []
        self.index: List[Tuple[int, int]] = []

        required = self.input_length + self.target_length

        for sample in samples:
            if isinstance(sample, torch.Tensor):
                seq = sample
            elif isinstance(sample, Dict) and "y" in sample:
                seq = sample["y"]
            else:
                raise TypeError("Each sample must be a tensor or dict containing 'y'.")

            if not isinstance(seq, torch.Tensor):
                raise TypeError("Sequences must be torch.Tensor instances.")
            if seq.ndim != 2:
                raise ValueError("Expected sequences shaped [time, states].")

            seq = seq.to(torch.float32).contiguous()
            total = seq.size(0)
            if total < required:
                continue
            seq_idx = len(self.sequences)
            self.sequences.append(seq)
            max_start = total - required
            for start in range(0, max_start + 1, self.step):
                self.index.append((seq_idx, start))

        if not self.index:
            raise ValueError("No usable windows were generated; adjust lengths or data.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start = self.index[idx]
        seq = self.sequences[seq_idx]
        past = seq[start : start + self.input_length]  # [T1, S]
        future = seq[
            start + self.input_length : start + self.input_length + self.target_length
        ]  # [T2, S]
        return past.transpose(0, 1).contiguous(), future.transpose(0, 1).contiguous()
    

class OdePINNDataset(Dataset):
    """
    Physics-Informed Neural Network (PINN) dataset with individual time points.

    One sample = one (trajectory, time_index) pair for computing physics loss.

    Used by: PINN models for physics-based training

    Why different from TrajectoryWindowDataset?
        - TrajectoryWindowDataset: Supervised learning on windows
        - OdePINNDataset: Physics-based learning at individual points

    PINNs need individual time points to:
        1. Compute derivatives ∂u/∂t using autograd
        2. Evaluate ODE residuals: ∂u/∂t - f(u, θ) = 0
        3. Enforce initial conditions and data constraints

    Each sample returns:
        - t: Collocation point (time where residual is computed)
        - theta: ODE parameters (α, β, γ, δ) for this trajectory
        - t0, u0: Initial conditions (t=0, u(0))
        - u_res: Residual target (always zero for physics loss)
        - t_regression, u_regression: Supervised data point at time t
        - traj_idx, time_idx: Trajectory and time indices for tracking
    """
    def __init__(
        self,
        trajectories: Sequence[Dict],
        device: torch.device,
    ):
        super().__init__()
        self.device = torch.device(device)

        # Store per-trajectory tensors
        self.traj_t: List[torch.Tensor] = []
        self.traj_y: List[torch.Tensor] = []
        self.traj_theta: List[torch.Tensor] = []
        self.index: List[Tuple[int, int]] = []  # (traj_idx, time_idx)

        # Assume all samples have same param keys
        param_keys = sorted(trajectories[0]["params"].keys())
        self.param_keys = param_keys

        for k, sample in enumerate(trajectories):
            t = sample["t"].to(torch.float32)           # [T]
            y = sample["y"].to(torch.float32)           # [T, D]
            theta = torch.stack(
                [sample["params"][name].to(torch.float32) for name in param_keys],
                dim=0,
            )  # [P]

            self.traj_t.append(t)
            self.traj_y.append(y)
            self.traj_theta.append(theta)

            T = t.shape[0]
            for i in range(T):
                self.index.append((k, i))  # one entry per time step

    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, time_idx = self.index[idx]

        t_series = self.traj_t[traj_idx]         # [T]
        y_series = self.traj_y[traj_idx]         # [T, D]
        theta = self.traj_theta[traj_idx]        # [P]

        # supervised sample at this time index
        t_sup = t_series[time_idx].view(1, 1)    # (1,1)
        u_sup = y_series[time_idx].view(1, -1)   # (1,D)

        # collocation point: use the same time/params
        t_colloc = t_sup.clone()

        # initial condition for this trajectory
        t0 = t_series[0].view(1, 1)              # (1,1)
        u0 = y_series[0].view(1, -1)             # (1,D)

        # residual target is zero
        u_res = torch.zeros_like(u_sup)          # (1,D)

        return {
            "t": t_colloc,
            "theta": theta,
            "t0": t0,
            "u0": u0,
            "u_res": u_res,
            "t_regression": t_sup,
            "u_regression": u_sup,
            "traj_idx": torch.tensor(traj_idx, dtype=torch.long),
            "time_idx": torch.tensor(time_idx, dtype=torch.long),
        }
    

class TrajectoryOdePINNDataset(Dataset):
    """
    PINN dataset where one sample = entire trajectory.

    Used by: PINN models that process full trajectories in one forward pass

    Difference from OdePINNDataset:
        - OdePINNDataset: One sample = one time point (T samples per trajectory)
        - TrajectoryOdePINNDataset: One sample = full trajectory (1 sample per trajectory)
        - This dataset is ~T times smaller!

    Benefit:
        - More efficient: Single forward pass processes entire trajectory
        - Better for models that can leverage temporal structure
        - Reduced dataloader overhead

    For each trajectory k, returns stacked tensors containing all time steps:
        - t:              [T, 1]  (all time points)
        - theta:          [T, P]  (same parameters repeated T times)
        - t0:             [T, 1]  (same initial time repeated T times)
        - u0:             [T, D]  (same initial state repeated T times)
        - u_res:          [T, D]  (zeros, residual target)
        - t_regression:   [T, 1]  (all time points)
        - u_regression:   [T, D]  (all state values)
        - traj_idx:       scalar  (trajectory index for tracking)

    The model sees all T times of a single trajectory in one forward pass.
    """

    def __init__(
        self,
        trajectories: Sequence[Dict],
        device: torch.device,
    ):
        super().__init__()
        self.device = torch.device(device)

        self.traj_t: List[torch.Tensor] = []
        self.traj_y: List[torch.Tensor] = []
        self.traj_theta: List[torch.Tensor] = []

        # Assume all samples have same param keys
        param_keys = sorted(trajectories[0]["params"].keys())
        self.param_keys = param_keys

        for sample in trajectories:
            t = sample["t"].to(torch.float32)           # [T]
            y = sample["y"].to(torch.float32)           # [T, D]
            theta = torch.stack(
                [sample["params"][name].to(torch.float32) for name in param_keys],
                dim=0,
            )  # [P]

            self.traj_t.append(t)
            self.traj_y.append(y)
            self.traj_theta.append(theta)

    def __len__(self) -> int:
        # one item per trajectory
        return len(self.traj_t)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t_series = self.traj_t[idx]         # [T]
        y_series = self.traj_y[idx]         # [T, D]
        theta_vec = self.traj_theta[idx]    # [P]

        T = t_series.shape[0]
        D = y_series.shape[1]
        P = theta_vec.shape[0]

        # times
        t_all = t_series.view(T, 1)          # [T,1]
        t_regression = t_all.clone()         # same times for regression

        # states
        u_all = y_series.view(T, D)          # [T,D]
        u_regression = u_all.clone()

        # parameters: repeat along time dimension
        theta_all = theta_vec.view(1, P).repeat(T, 1)  # [T,P]

        # initial condition (same for all rows of this trajectory)
        t0 = t_series[0].view(1, 1)                    # [1,1]
        u0_single = y_series[0].view(1, D)             # [1,D]
        t0_all = t0.repeat(T, 1)                       # [T,1]
        u0_all = u0_single.repeat(T, 1)                # [T,D]

        # residual target is zero
        u_res = torch.zeros_like(u_all)                # [T,D]

        return {
            "t": t_all.to(self.device),
            "theta": theta_all.to(self.device),
            "t0": t0_all.to(self.device),
            "u0": u0_all.to(self.device),
            "u_res": u_res.to(self.device),
            "t_regression": t_regression.to(self.device),
            "u_regression": u_regression.to(self.device),
            "traj_idx": torch.tensor(idx, dtype=torch.long),
        }
    
class FourierOdePINNDataset(Dataset):
    """
    OdePINNDataset with Fourier time encodings for better temporal representation.

    Used by: PINN models that benefit from periodic time features

    Why Fourier features?
        - Raw time t ∈ [0, T_max] is a poor representation for periodic systems
        - Oscillatory systems (like Lotka-Volterra) have periodic structure
        - Fourier features capture periodicity better than linear time

    Fourier encoding:
        t_norm = t / t_max              # Normalize time to [0, 1]
        features = [sin(2π·1·t_norm), cos(2π·1·t_norm),
                    sin(2π·2·t_norm), cos(2π·2·t_norm),
                    ...,
                    sin(2π·K·t_norm), cos(2π·K·t_norm)]

        where K = n_frequencies (default 4)
        Output shape: [1, 2*K] = [1, 8] by default

    Benefits:
        - Periodic patterns are more easily represented
        - Helps PINN learn oscillatory dynamics
        - Similar to positional encoding in transformers

    Each sample returns everything from OdePINNDataset plus:
        - t_fourier: [1, 2*n_frequencies] Fourier time features
    """

    def __init__(
        self,
        trajectories: Sequence[Dict],
        device: torch.device,
        t_max: float,
        n_frequencies: int = 4,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.t_max = float(t_max)
        self.n_frequencies = int(n_frequencies)

        # Store per-trajectory tensors
        self.traj_t: List[torch.Tensor] = []
        self.traj_y: List[torch.Tensor] = []
        self.traj_theta: List[torch.Tensor] = []
        self.index: List[Tuple[int, int]] = []  # (traj_idx, time_idx)

        # Assume all samples have same param keys
        param_keys = sorted(trajectories[0]["params"].keys())
        self.param_keys = param_keys

        for k, sample in enumerate(trajectories):
            t = sample["t"].to(torch.float32)           # [T]
            y = sample["y"].to(torch.float32)           # [T, D]
            theta = torch.stack(
                [sample["params"][name].to(torch.float32) for name in param_keys],
                dim=0,
            )  # [P]

            self.traj_t.append(t)
            self.traj_y.append(y)
            self.traj_theta.append(theta)

            T = t.shape[0]
            for i in range(T):
                self.index.append((k, i))  # one entry per time step

    def __len__(self) -> int:
        return len(self.index)

    def _time_fourier_features(self, t: torch.Tensor) -> torch.Tensor:
        """
        Build Fourier/sinusoidal features for time t (shape [1,1]):

        t_norm = t / t_max
        features = [sin(2π k t_norm), cos(2π k t_norm)] for k=1..n_frequencies

        Returns: [1, 2 * n_frequencies]
        """
        # ensure shape [1, 1]
        if t.ndim == 0:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t.view(-1, 1)

        t_norm = t / self.t_max  # [0,1]
        # frequencies: 1,2,...,n_frequencies
        ks = torch.arange(1, self.n_frequencies + 1, device=t.device, dtype=t.dtype).view(1, -1)  # [1, K]
        # shape broadcasting: [1,1] * [1,K] -> [1,K]
        angles = 2 * np.pi * t_norm @ ks  # [1,1] @ [1,K] -> [1,K]
        sin_feat = torch.sin(angles)      # [1,K]
        cos_feat = torch.cos(angles)      # [1,K]
        return torch.cat([sin_feat, cos_feat], dim=1)  # [1, 2K]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, time_idx = self.index[idx]

        t_series = self.traj_t[traj_idx]         # [T]
        y_series = self.traj_y[traj_idx]         # [T, D]
        theta = self.traj_theta[traj_idx]        # [P]

        t_sup = t_series[time_idx].view(1, 1)    # (1,1)
        u_sup = y_series[time_idx].view(1, -1)   # (1,D)

        # collocation point: same time
        t_colloc = t_sup.clone()

        # initial condition
        t0 = t_series[0].view(1, 1)              # (1,1)
        u0 = y_series[0].view(1, -1)             # (1,D)

        # residual target is zero
        u_res = torch.zeros_like(u_sup)          # (1,D)

        # Fourier time features
        t_fourier = self._time_fourier_features(t_colloc)  # [1, 2K]

        return {
            "t": t_colloc.to(self.device),
            "theta": theta.to(self.device),
            "t0": t0.to(self.device),
            "u0": u0.to(self.device),
            "u_res": u_res.to(self.device),
            "t_regression": t_sup.to(self.device),
            "u_regression": u_sup.to(self.device),
            "t_fourier": t_fourier.to(self.device),
            "traj_idx": torch.tensor(traj_idx, dtype=torch.long, device=self.device),
            "time_idx": torch.tensor(time_idx, dtype=torch.long, device=self.device),
        }