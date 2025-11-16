import datetime
import json
from pathlib import Path
from typing import Dict, List
# from transformers import set_seed
import os
import uuid
import numpy as np
import torch
import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from scipy.signal import decimate
import matplotlib.pyplot as plt


from data_processing.datasets import TrajectoryWindowDataset
from data_processing.generate_data import read_trajectories_parquet_as_dicts, as_torch as sample_to_torch
from models.MLP import WindowMLP
from utils import _build_model, build_loader, forecast_full_trajectory, load_data, plot_test_series_prediction, split_train_val

def eval_from_config(config: Dict) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(config["data"]["root"])
    test_dir = root / config["data"]["test_subdir"]
    test_series = load_data(test_dir)  # raw dict samples

    model = pl.LightningModule.load_from_checkpoint(config["model"]["checkpoint_path"])  # Generic load
    model.to(device)
    model.eval()

    out_cfg = config.get("output", {})
    plot_dir = out_cfg.get("plot_dir")
    plot_show = out_cfg.get("plot_show", False)
    plot_dir_path = Path(plot_dir) if plot_dir else None
    if plot_dir_path:
        plot_dir_path.mkdir(parents=True, exist_ok=True)


    per_sample = []
    results_cache = []
    for idx, sample in enumerate(test_series):
        res = forecast_full_trajectory(model, sample, config["data"]) # type: ignore
        results_cache.append(res)
        if idx < 5:
            run_name = res.get("run_id") or sample.get("run_id") or f"series_{idx:03d}"
            save_path = plot_dir_path / f"{run_name}.png" if plot_dir_path else None
            plot_test_series_prediction(
                model, # type: ignore
                sample,
                config["data"],
                forecast_result=res,
                show=plot_show,
                save_path=save_path,
            )
        if not np.isnan(res["rmse"]):
            per_sample.append(res["rmse"])
    overall_rmse = float(np.mean(per_sample)) if per_sample else float("nan")

    print(f"Eval (decimated) RMSE: {overall_rmse:.6f}")

    out_cfg = config.get("output", {})
    metrics_path = out_cfg.get("metrics_path")
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({"rmse": overall_rmse}, f, indent=2)

    forecast_dir = out_cfg.get("forecast_dir")
    if forecast_dir:
        fd = Path(forecast_dir)
        fd.mkdir(parents=True, exist_ok=True)
        for res in results_cache:
            if res["forecast"] is not None:
                torch.save(
                    {
                        "run_id": res.get("run_id"),
                        "forecast": res["forecast"],
                        "target": res["target"],
                        "rmse": res["rmse"],
                    },
                    fd / f"{res.get('run_id', 'series')}.pt",
                )
    return {"rmse": overall_rmse}


def eval_pinn(config: Dict) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(config["data"]["root"])
    test_dir = root / config["data"]["test_subdir"]
    test_series = load_data(test_dir)

    model = pl.LightningModule.load_from_checkpoint(config["model"]["checkpoint_path"])
    model.to(device)
    model.eval()

    out_cfg = config.get("output", {})
    plot_dir = out_cfg.get("plot_dir")
    plot_show = out_cfg.get("plot_show", False)
    plot_dir_path = Path(plot_dir) if plot_dir else None
    if plot_dir_path:
        plot_dir_path.mkdir(parents=True, exist_ok=True)

    per_series_rmse: List[float] = []
    max_examples = 5

    def _ensure_2d(array: np.ndarray) -> np.ndarray:
        return array.reshape(-1, 1) if array.ndim == 1 else array

    def _plot_series(times: np.ndarray,
                     target: np.ndarray,
                     prediction: np.ndarray,
                     title: str,
                     save_path: Optional[Path]):
        num_dims = prediction.shape[1]
        fig, axes = plt.subplots(num_dims, 1, sharex=True, figsize=(9, 2.4 * num_dims))
        if num_dims == 1:
            axes = [axes]
        for dim_idx, ax in enumerate(axes):
            ax.plot(times, target[:, dim_idx], label="data", color="tab:blue")
            ax.plot(times, prediction[:, dim_idx], label="pinn", color="tab:orange", linestyle="--")
            ax.set_ylabel(f"u{dim_idx}")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("t")
        axes[0].set_title(title)
        axes[0].legend()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if plot_show:
            plt.show()
        plt.close(fig)

    for idx, sample in enumerate(test_series):
        time_arr = np.asarray(sample["t"], dtype=np.float32)
        theta_arr = np.asarray(sample["theta"], dtype=np.float32)
        target_arr = _ensure_2d(np.asarray(sample["u"], dtype=np.float32))

        t_tensor = torch.as_tensor(time_arr, dtype=torch.float32, device=device)
        if t_tensor.ndim == 1:
            t_tensor = t_tensor.unsqueeze(1)
        input_dim = getattr(model, "n", t_tensor.shape[-1])
        t_tensor = t_tensor.reshape(-1, input_dim)

        theta_tensor = torch.as_tensor(theta_arr, dtype=torch.float32, device=device)
        if theta_tensor.ndim == 1:
            theta_tensor = theta_tensor.unsqueeze(0)
        if theta_tensor.shape[0] == 1:
            theta_tensor = theta_tensor.repeat(t_tensor.shape[0], 1)
        elif theta_tensor.shape[0] != t_tensor.shape[0]:
            raise ValueError("Theta samples must broadcast to time samples.")

        with torch.no_grad():
            prediction = model.u(t_tensor, theta_tensor).detach().cpu().numpy()
        prediction = _ensure_2d(prediction)

        rmse = float(np.sqrt(np.mean((prediction - target_arr) ** 2)))
        per_series_rmse.append(rmse)

        if idx < max_examples:
            run_name = sample.get("run_id") or f"series_{idx:03d}"
            save_path = (plot_dir_path / f"{run_name}.png") if plot_dir_path else None
            _plot_series(time_arr.reshape(-1), target_arr, prediction, f"PINN forecast · {run_name}", save_path)

    overall_rmse = float(np.mean(per_series_rmse)) if per_series_rmse else float("nan")
    print(f"PINN Eval RMSE: {overall_rmse:.6f}")

    metrics_path = out_cfg.get("metrics_path")
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({"rmse": overall_rmse}, f, indent=2)

    return {"rmse": overall_rmse}



def train_from_config(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_subdir = config["experiment"]["run_subdir"]
    run_dir = Path("artifacts") / run_subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(config["data"]["root"])
    train_val_dir = root / config["data"]["train_val_subdir"]
    test_dir = root / config["data"]["test_subdir"]

    decimation = int(config["data"].get("decimation", 1))
    train_val_samples = load_data(train_val_dir, decimation=decimation)
    test_series = load_data(test_dir, decimation=decimation)

    # Random split at series level (not windows)
    train_series, val_series = split_train_val(
        train_val_samples,
        val_ratio=config["data"]["val_ratio"],
        seed=config["data"]["seed"],
    )

    print(
        f"Series split -> train: {len(train_series)}, val: {len(val_series)}, test: {len(test_series)}"
    )

    # Dataloaders
    train_loader = build_loader(
        train_series, config["data"], train=True,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    val_loader = build_loader(
        val_series, config["data"], train=False,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    test_loader = build_loader(
        test_series, config["data"], train=False,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    # Model
    # model = WindowMLP(
    #     state_dim=config["model"]["state_dim"],
    #     input_len=config["data"]["input_length"],
    #     target_len=config["data"]["target_length"],
    #     hidden_sizes=tuple(config["model"]["hidden_sizes"]),
    #     lr=config["model"]["lr"],
    # )

    model = _build_model(config["model"], config["data"])

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="epoch{epoch:02d}-val_loss{val_loss:.4f}",
        save_top_k=-1,
        every_n_epochs=1,
    )

    # MLflow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment"]["name"],
        tracking_uri=config["experiment"]["tracking_uri"],
        run_name=run_subdir,
    )

    # Also log non-model data/training params
    mlflow_logger.log_hyperparams({
        **dict(model.hparams),
        **{k: config["data"][k] for k in ("step", "decimation", "val_ratio", "seed") if k in config["data"]},
        "batch_size": batch_size,
        "num_workers": num_workers,
        "checkpoint_dir": str(run_dir / "checkpoints"),
    })
    
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=mlflow_logger,
        callbacks=[checkpoint_cb],
        deterministic=True,
        default_root_dir=run_dir,
    )
    
    print(f"Train windows: {len(train_loader.dataset)}, batch_size: {train_loader.batch_size}, batches/epoch: {len(train_loader)}") # type: ignore
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    final_ckpt_path = run_dir / "final_model.ckpt"
    final_ckpt_path.parent.mkdir(exist_ok=True)
    trainer.save_checkpoint(str(final_ckpt_path))
    print(f"Saved final checkpoint to {final_ckpt_path}")

if __name__ == "__main__":

    with open("config/train_RNN.yaml", "r") as fh:
        cfg = yaml.safe_load(fh)
    train_from_config(cfg)

    # with open("config/eval.yaml", "r") as fh:
    #     cfg = yaml.safe_load(fh)
    # eval_from_config(cfg)
