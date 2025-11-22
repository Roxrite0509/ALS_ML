#!/usr/bin/env python3
# hyper_smote_tune.py (fixed: use --smote_k, don't pass unknown flags)
"""
Grid launcher for smote_train.py. Runs multiple SMOTE-MLP experiments in parallel.

This version:
 - Passes --smote_k (expected by smote_train.py) instead of --k_neighbors
 - Does NOT pass --hidden to smote_train.py (smote_train.py doesn't accept it)
 - Keeps 'hidden' in run-name so you can still track intended hidden sizes.
 - Launches child processes with sys.executable for robust invocation.
"""

from __future__ import annotations
import argparse
import itertools
import subprocess
import shlex
import time
import sys
from pathlib import Path
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smote_script", required=True,
                   help="Path to smote_train.py (e.g. ./smote_train.py)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--train_xlsx", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--out_root", default="./hp_smote_runs")
    # Grid params (keep hidden in grid for bookkeeping but not passed if smote_train lacks it)
    p.add_argument("--lr", nargs="+", type=float, default=[1e-3])
    p.add_argument("--hidden", nargs="+", type=int, default=[512],
                   help="Hidden sizes (kept for run-name bookkeeping). If you want smote_train to receive it, add --hidden to smote_train.py.")
    p.add_argument("--batch_size", nargs="+", type=int, default=[64])
    p.add_argument("--k", nargs="+", type=int, default=[5],
                   help="k for SMOTE (will be passed as --smote_k)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--concurrency", type=int, default=2)
    return p.parse_args()


def build_child_args(params: dict, out_dir: Path) -> List[str]:
    """
    Build the list of CLI args to append to the smote script invocation.
    Note: we pass --smote_k (expected by smote_train.py). We DO NOT pass --hidden here
    because the user's smote_train.py doesn't accept it.
    """
    args = [
        "--data_dir", str(params["data_dir"]),
        "--train_xlsx", str(params["train_xlsx"]),
        "--train_sheet", "Training Baseline - Task 1",
        "--val_sheet", "Validation Baseline - Task 1",
    ]
    if params.get("cache_dir"):
        args += ["--cache_dir", str(params["cache_dir"])]

    args += [
        "--pretrained",
        "--device", str(params["device"]),
        "--epochs", str(params["epochs"]),
        "--batch_size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
        "--weight_decay", str(params.get("weight_decay", 1e-4)),
        # pass SMOTE k using the correct flag name used by smote_train.py:
        "--smote_k", str(params["k"]),
        "--seed", str(params["seed"]),
        "--out_dir", str(out_dir)
    ]
    return args


def normalize_script_path(script: str) -> List[str]:
    # allow user to pass "python ./smote_train.py" or "./smote_train.py"
    parts = shlex.split(script)
    if len(parts) == 0:
        raise ValueError("smote_script cannot be empty")
    if parts[0] in ("python", "python3"):
        parts = parts[1:]
    return parts


def launch(cmd_list: List[str]) -> subprocess.Popen:
    return subprocess.Popen(cmd_list)


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    script_parts = normalize_script_path(args.smote_script)

    combos = list(itertools.product(
        args.lr,
        args.hidden,
        args.batch_size,
        args.k,
        args.seeds
    ))
    print(f"Total trials: {len(combos)}\n")

    running: list[tuple[subprocess.Popen, str]] = []

    for i, (lr, hidden, bs, k_val, seed) in enumerate(combos, start=1):
        run_name = f"run_{i:03d}_lr{lr}_h{hidden}_bs{bs}_k{k_val}_s{seed}"
        out_dir = out_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "data_dir": args.data_dir,
            "train_xlsx": args.train_xlsx,
            "cache_dir": args.cache_dir or "",
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": bs,
            "lr": lr,
            "hidden": hidden,  # only for bookkeeping / run_name
            "k": k_val,
            "seed": seed,
            "weight_decay": 1e-4
        }

        child_args = build_child_args(params, out_dir)
        # call with the current Python interpreter
        cmd_list = [sys.executable] + script_parts + child_args

        cmd_str = " ".join(shlex.quote(x) for x in cmd_list)
        print(f"[Launching Trial {i}] {cmd_str}\n")

        proc = launch(cmd_list)
        running.append((proc, run_name))

        # concurrency control
        while True:
            active = [p for p, _ in running if p.poll() is None]
            if len(active) < args.concurrency:
                break
            time.sleep(1)

    print("\nWaiting for all jobs to complete...\n")
    for p, name in running:
        p.wait()
        rc = p.returncode
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[Completed] {name} -> {status}")

    print("\n=== All SMOTE-Hyperparameter runs finished ===")


if __name__ == "__main__":
    main()
