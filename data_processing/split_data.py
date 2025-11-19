"""
Split trajectory dataset into train_val and test sets.

This script reads all trajectories from a parquet file and splits them into
train_val (default 80%) and test (default 20%) subsets, saving each to
separate parquet files in their respective directories.

Usage:
    python data_processing/split_data.py [--train-ratio 0.8] [--seed 42]
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import random

from .generate_data import read_trajectories_parquet_as_dicts, save_trajectories_parquet


def split_trajectories(
    data_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split trajectories from a data directory into train_val and test sets.

    Reads all trajectories from parquet files in the data directory, shuffles them
    with a fixed seed for reproducibility, and splits them into train_val and test
    sets based on the specified train_ratio.

    Args:
        data_dir: Path to directory containing parquet files with trajectories.
        train_ratio: Fraction of trajectories to use for training (default 0.8 = 80%).
        seed: Random seed for reproducible shuffling (default 42).

    Returns:
        Tuple of (train_val_trajectories, test_trajectories)
    """
    # Load all trajectories
    all_trajectories = list(read_trajectories_parquet_as_dicts(data_dir))
    total = len(all_trajectories)

    if total == 0:
        raise ValueError(f"No trajectory files found in {data_dir}")

    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(all_trajectories)

    # Split
    split_idx = int(total * train_ratio)
    train_val_trajs = all_trajectories[:split_idx]
    test_trajs = all_trajectories[split_idx:]

    print(f"Total trajectories: {total}")
    print(f"Train/val: {len(train_val_trajs)} ({train_ratio*100:.0f}%)")
    print(f"Test: {len(test_trajs)} ({(1-train_ratio)*100:.0f}%)")

    return train_val_trajs, test_trajs


def save_split_datasets(
    train_val_trajs: List[Dict],
    test_trajs: List[Dict],
    output_dir: Path,
    remove_old: bool = True,
) -> None:
    """
    Save split datasets to train_val and test subdirectories.

    Creates train_val/ and test/ subdirectories if they don't exist, and saves
    the trajectories as parquet files. Optionally removes old parquet files first.

    Args:
        train_val_trajs: List of training/validation trajectories.
        test_trajs: List of test trajectories.
        output_dir: Parent directory containing train_val/ and test/ subdirectories.
        remove_old: Whether to remove existing parquet files (default True).
    """
    output_dir = Path(output_dir)

    # Helper function to reconstruct param_keys and param_values
    def _add_param_keys_values(trajs: List[Dict]) -> List[Dict]:
        """Add param_keys and param_values to trajectories if they have params dict."""
        for traj in trajs:
            if "params" in traj and isinstance(traj["params"], dict):
                if "param_keys" not in traj:
                    traj["param_keys"] = list(traj["params"].keys())
                if "param_values" not in traj:
                    traj["param_values"] = [float(v) for v in traj["params"].values()]
        return trajs

    # Prepare trajectories for saving
    train_val_trajs = _add_param_keys_values(train_val_trajs)
    test_trajs = _add_param_keys_values(test_trajs)

    # Create subdirectories
    train_val_dir = output_dir / "train_val"
    test_dir = output_dir / "test"
    train_val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Remove old files if requested
    if remove_old:
        for old_file in train_val_dir.glob("*.parquet"):
            old_file.unlink()
            print(f"Removed: {old_file}")
        for old_file in test_dir.glob("*.parquet"):
            old_file.unlink()
            print(f"Removed: {old_file}")

    # Save train_val
    train_val_path = train_val_dir / "part-train.parquet"
    save_trajectories_parquet(train_val_path, train_val_trajs)
    print(f"✓ Saved {len(train_val_trajs)} train/val trajectories to {train_val_path}")
    print(f"  File exists: {train_val_path.exists()}, size: {train_val_path.stat().st_size if train_val_path.exists() else 'N/A'}")

    # Save test
    test_path = test_dir / "part-test.parquet"
    save_trajectories_parquet(test_path, test_trajs)
    print(f"✓ Saved {len(test_trajs)} test trajectories to {test_path}")
    print(f"  File exists: {test_path.exists()}, size: {test_path.stat().st_size if test_path.exists() else 'N/A'}")


def main():
    """Command-line interface for splitting trajectory data."""
    parser = argparse.ArgumentParser(
        description="Split trajectory dataset into train_val and test sets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(Path(__file__).parent / "data" / "lotka_volterra_trajectories").resolve(),
        help="Directory containing trajectory data (default: data_processing/data/lotka_volterra_trajectories)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--keep-old",
        action="store_true",
        help="Keep old parquet files instead of removing them",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    print(f"Splitting data from: {args.data_dir}")
    print(f"Train/val ratio: {args.train_ratio}")
    print(f"Random seed: {args.seed}")
    print()

    # Split
    train_val_trajs, test_trajs = split_trajectories(
        args.data_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    # Save (to subdirectories within data_dir)
    print()
    save_split_datasets(
        train_val_trajs,
        test_trajs,
        args.data_dir,  # Save train_val/ and test/ as subdirs of data_dir
        remove_old=not args.keep_old,
    )

    print()
    print("✓ Data split complete!")


if __name__ == "__main__":
    main()
