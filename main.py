import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
import inspect

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
from utils import _build_model, _ensure_2d, _load_model_for_eval, _plot_series, build_loader, forecast_full_trajectory, load_data, plot_test_series_prediction, split_train_val

def eval_from_config(config: Dict) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(config["data"]["root"])
    test_dir = root / config["data"]["test_subdir"]
    test_series = load_data(test_dir)  # raw dict samples

    model = _load_model_for_eval(config["model"], config["data"], device)
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


def plot_pinn_examples(examples, plot_dir: Optional[str], show: bool) -> None:
    if not examples:
        return
    save_dir = Path(plot_dir) if plot_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    for item in examples:
        title = f"PINN forecast · {item['run_name']}"
        save_path = (save_dir / f"{item['run_name']}.png") if save_dir else None
        _plot_series(item["time"], item["target"], item["prediction"], title, save_path, show) 

def save_pinn_metrics(metrics_path: Optional[str], rmse: float) -> None:
    if not metrics_path:
        return
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rmse": rmse}, f, indent=2)

def eval_pinn(config_path: str = "config/eval_PINN.yaml") -> Dict[str, float]:
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(config["data"]["root"])
    test_dir = root / config["data"]["test_subdir"]
    decimation = int(config["data"].get("decimation", 1))
    batch_size = int(config["data"].get("batch_size", 128))
    num_workers = int(config["data"].get("num_workers", 0))

    test_samples = load_data(test_dir, decimation=decimation, to_torch=True)
    test_loader = build_loader(
        test_samples,
        config["data"],
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = _load_model_for_eval(config["model"], config["data"], device)
    model.to(device)
    model.eval()

    series_acc: Dict[int, Dict[str, List]] = {}
    with torch.no_grad():
        for batch in test_loader:
            t_reg = batch["t_regression"].to(device).view(-1, 1)
            u_reg = batch["u_regression"].to(device).view(-1, model.m)
            theta = batch["theta"].to(device).view(-1, model.p)
            u0 = batch["u0"].to(device).view(-1, model.m)

            # if theta.ndim == 1:
            #     theta = theta.unsqueeze(0)
            # if theta.shape[0] == 1 and t_reg.shape[0] > 1:
            #     theta = theta.repeat(t_reg.shape[0], 1)

            try:
                preds = model(t_reg, theta, u0)
            except TypeError as err:
                raise RuntimeError(
                    "Loaded PINN does not accept theta; ensure an LVOdePINN checkpoint."
                ) from err

            traj_idx = batch["traj_idx"].detach().cpu().numpy()
            time_idx = batch["time_idx"].detach().cpu().numpy()
            times = t_reg.detach().cpu().numpy()
            targets = u_reg.detach().cpu().numpy()
            predictions = preds.detach().cpu().numpy()

            for i in range(len(traj_idx)):
                key = int(traj_idx[i])
                acc = series_acc.setdefault(key, {"entries": []})
                acc["entries"].append(
                    (
                        int(time_idx[i]),
                        times[i].copy(),
                        targets[i].copy(),
                        predictions[i].copy(),
                    )
                )

    per_series_rmse: List[float] = []
    examples: List[Dict] = []
    max_examples = 5

    for key in sorted(series_acc.keys()):
        entries = sorted(series_acc[key]["entries"], key=lambda x: x[0])
        time_arr = np.vstack([e[1] for e in entries]).reshape(-1)
        target_arr = np.vstack([e[2] for e in entries])
        pred_arr = np.vstack([e[3] for e in entries])

        rmse = float(np.sqrt(np.mean((pred_arr - target_arr) ** 2)))
        per_series_rmse.append(rmse)

        if len(examples) < max_examples:
            examples.append(
                {
                    "time": time_arr,
                    "target": target_arr,
                    "prediction": pred_arr,
                    "run_name": f"traj_{key:03d}",
                }
            )

    overall_rmse = float(np.mean(per_series_rmse)) if per_series_rmse else float("nan")
    print(f"PINN Eval RMSE: {overall_rmse:.6f}")

    output_cfg = config.get("output", {})
    plot_pinn_examples(examples, output_cfg.get("plot_dir"), output_cfg.get("plot_show", False))
    save_pinn_metrics(output_cfg.get("metrics_path"), overall_rmse)

    return {"rmse": overall_rmse}

# def eval_pinn(config: Dict) -> Dict:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     root = Path(config["data"]["root"])
#     test_dir = root / config["data"]["test_subdir"]
#     decimation = int(config["data"].get("decimation", 1))
#     test_series = load_data(test_dir, decimation=decimation)

#     model = _load_model_for_eval(config["model"], config["data"], device)
#     model.to(device)
#     model.eval()
#     forward_sig = inspect.signature(model.forward)
#     expects_theta = len(forward_sig.parameters) >= 3  # self, t, theta

#     out_cfg = config.get("output", {})
#     plot_dir = out_cfg.get("plot_dir")
#     plot_show = out_cfg.get("plot_show", False)
#     plot_dir_path = Path(plot_dir) if plot_dir else None
#     if plot_dir_path:
#         plot_dir_path.mkdir(parents=True, exist_ok=True)

#     def _theta_from_sample(sample: Dict) -> torch.Tensor:
#         if "theta" in sample and sample["theta"] is not None:
#             return torch.as_tensor(sample["theta"], dtype=torch.float32, device=device).view(1, -1)
#         params = sample.get("params")
#         if isinstance(params, dict):
#             ordered = [params[name] for name in sorted(params.keys())]
#             theta_vals = [
#                 torch.as_tensor(val, dtype=torch.float32).view(-1)[0].item()
#                 for val in ordered
#             ]
#             return torch.tensor(theta_vals, dtype=torch.float32, device=device).unsqueeze(0)
#         raise ValueError("Test sample must include either 'params' dict or precomputed 'theta' tensor.")

#     per_series_rmse: List[float] = []
#     max_examples = 5

#     for idx, sample in enumerate(test_series):
#         t_tensor = torch.as_tensor(sample["t"], dtype=torch.float32, device=device)
#         if t_tensor.ndim == 1:
#             t_tensor = t_tensor.unsqueeze(1)

#         target = sample.get("y")
#         if target is None:
#             target = sample.get("u")
#         if target is None:
#             raise ValueError("Test sample must contain 'y' or 'u' (states over time).")
#         target_arr = _ensure_2d(torch.as_tensor(target, dtype=torch.float32).cpu().numpy())

#         theta_base = _theta_from_sample(sample)
#         theta_tensor = theta_base.repeat(t_tensor.shape[0], 1)

#         with torch.no_grad():
#             preds = model(t_tensor, theta_tensor) if expects_theta else model(t_tensor)
#         prediction = _ensure_2d(preds.detach().cpu().numpy())

#         rmse = float(np.sqrt(np.mean((prediction - target_arr) ** 2)))
#         per_series_rmse.append(rmse)

#         if idx < max_examples:
#             time_arr = t_tensor.detach().cpu().numpy().reshape(-1)
#             run_name = sample.get("run_id") or f"series_{idx:03d}"
#             save_path = (plot_dir_path / f"{run_name}.png") if plot_dir_path else None
#             _plot_series(time_arr, target_arr, prediction, f"PINN forecast · {run_name}", save_path, show=plot_show)

#     overall_rmse = float(np.mean(per_series_rmse)) if per_series_rmse else float("nan")
#     print(f"PINN Eval RMSE: {overall_rmse:.6f}")

#     metrics_path = out_cfg.get("metrics_path")
#     if metrics_path:
#         Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
#         with open(metrics_path, "w") as f:
#             json.dump({"rmse": overall_rmse}, f, indent=2)

#     return {"rmse": overall_rmse}

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

    training_cfg = config["training"]
    batch_size = training_cfg["batch_size"]
    num_workers = training_cfg["num_workers"]

    print(
        f"Series split -> train: {len(train_series)}, val: {len(val_series)}, test: {len(test_series)}"
    )

    # Dataloaders
    train_loader = build_loader(
        train_series, config["data"], train=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = build_loader(
        val_series, config["data"], train=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = build_loader(
        test_series, config["data"], train=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

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
    model_hparams = dict(getattr(model, "hparams", {}))
    data_hparams = {
        k: config["data"][k]
        for k in ("step", "decimation", "val_ratio", "seed")
        if k in config["data"]
    }
    mlflow_logger.log_hyperparams({
        **model_hparams,
        **data_hparams,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "checkpoint_dir": str(run_dir / "checkpoints"),
    })

    print("MLFlow tracking_uri:", mlflow_logger.experiment.tracking_uri)  # debug
    print("MLFlow experiment:", mlflow_logger._experiment_name, mlflow_logger.experiment_id) # type: ignore

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=mlflow_logger,
        callbacks=[checkpoint_cb],
        deterministic=True,
        default_root_dir=run_dir,
    )
    
    ds_cls_name = type(train_loader.dataset).__name__

    print(
        f"Train dataset: {ds_cls_name}, "
        f"samples: {len(train_loader.dataset)}, " # type: ignore
        f"batch_size: {train_loader.batch_size}, "
        f"batches/epoch: {len(train_loader)}"
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_ckpt_path = run_dir / "final_model.ckpt"
    final_ckpt_path.parent.mkdir(exist_ok=True)
    trainer.save_checkpoint(str(final_ckpt_path))
    print(f"Saved final checkpoint to {final_ckpt_path}")

if __name__ == "__main__":

    with open("config/train_PINN.yaml", "r") as fh:
        cfg = yaml.safe_load(fh)
    train_from_config(cfg)

    # eval_pinn("config/eval_PINN.yaml")
