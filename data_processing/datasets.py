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
    Provides (states, future states) supervision windows.
    """

    def __init__(
        self,
        samples: Sequence[Dict],
        input_length: int,
        target_length: int,
        *,
        step: int = 1,
        decimation: int = 1,
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
    Dataset where each sample corresponds to one (trajectory, time_index) pair.

    For each sample we return:
    - collocation point (t, theta)
    - supervised point (t_sup, u_sup, theta_sup) with SAME trajectory/parameters
    - initial condition (t0, u0) for that trajectory
    """
    def __init__(
        self,
        trajectories: Sequence[Dict],
        device: torch.device,
    ):
        super().__init__()
        self.device = device

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

        t_series = self.traj_t[traj_idx]                 # [T]
        y_series = self.traj_y[traj_idx]                 # [T, D]
        theta = self.traj_theta[traj_idx].view(1, -1)    # [1, P]

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
            "t": t_colloc.to(self.device),
            "theta": theta.to(self.device),
            "t0": t0.to(self.device),
            "u0": u0.to(self.device),
            "u_res": u_res.to(self.device),
            "t_regression": t_sup.to(self.device),
            "u_regression": u_sup.to(self.device),
        }