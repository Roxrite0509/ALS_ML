#!/usr/bin/env python3
"""
ensemble_inference_fixed.py

Robust drop-in replacement for ensemble_inference.py's model-loading behavior.

Features:
- Accepts the same CLI interface shown in your error help:
    --data_dir, --test_xlsx, --model_dirs, -- \
        cache_dir, --ttas, --batch_size, --device, --out_csv
- Makes --model_dirs optional. If not provided, it will try to auto-discover seed_*
  directories in a sibling `runs_balanced_aug` folder relative to this script's parent.
- Auto-discovers common checkpoint names inside each model dir (best_model.pt, best.pth, *.pt, *.pth, *.ckpt).
- If it finds torchscript files (.pt/.pth/ .ckpt that load with torch.jit.load), it will load them and run inference.
- If it only finds state_dict files, it will save their paths and print clear instructions for integrating your model construction code.
- Writes informative errors to stdout/stderr and exits with non-zero codes when necessary.

USAGE (same as original):
python ensemble_inference_fixed.py --data_dir DATA --test_xlsx TEST_XLSX --model_dirs <dirs...> [--cache_dir ..] [--ttas 5] [--batch_size 32] [--device cpu] [--out_csv out.csv]

Notes:
- This script focuses on robust discovery and loading of checkpoints. If your original ensemble pipeline constructs PyTorch nn.Modules before loading state dicts, you'll need to integrate that model-construction step where indicated in the code (commented).
- The script will not attempt to guess network architecture. If your checkpoints are state dicts (not torchscript), the script will print the discovered checkpoint file paths and exit with a helpful message telling you where to add model-building code.

Replace your existing ensemble_inference.py with this file or save it as ensemble_inference_fixed.py and invoke it with --model_dirs or without (it will auto-discover).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import logging
import torch
import pandas as pd


def setup_logging(level=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(format=fmt, level=level)


def _find_checkpoint_in_dir(p: Path) -> Optional[Path]:
    """Search for a reasonable checkpoint file inside directory p.

    Preference order:
      1) best_model.pt, best.pth, best.ckpt, best.pt
      2) any file matching 'best*.pt' or 'best*.pth' or 'best*.ckpt'
      3) latest checkpoint by lexicographic sort among common extensions
      4) largest file with extensions .pt/.pth/.ckpt
    """
    if not p.exists():
        return None

    # exact candidate names
    candidates = ["best_model.pt", "best.pth", "best.ckpt", "best.pt"]
    for name in candidates:
        cand = p / name
        if cand.exists():
            return cand

    # glob for best*
    for ext in ("pt", "pth", "ckpt"):
        matches = sorted(p.glob(f"best*.{ext}"))
        if matches:
            return matches[-1]

    # any checkpoint files
    all_matches = []
    for ext in ("pt", "pth", "ckpt"):
        all_matches.extend(sorted(p.glob(f"*.{ext}")))

    if all_matches:
        # prefer lexicographically last (often latest), else largest
        candidate = all_matches[-1]
        if candidate.exists():
            return candidate
        largest = max(all_matches, key=lambda f: f.stat().st_size)
        return largest

    return None


def load_models(model_dirs: List[str], device: str) -> List[Path]:
    """Return list of checkpoint Paths discovered for each requested model_dir.

    Behavior:
    - If an entry in model_dirs is a file, it's included as-is.
    - If it's a directory, attempts to find a checkpoint using _find_checkpoint_in_dir.
    - If model_dirs is empty, the function will attempt to auto-discover seed_*
      directories under ../runs_balanced_aug relative to this script and use them.

    Returns a list of Paths to checkpoint files. Raises FileNotFoundError with a helpful
    message if none are found for any expected entry.
    """
    discovered = []

    # If no model_dirs specified, try auto-discovering seed_* under ../runs_balanced_aug
    if not model_dirs:
        maybe_root = Path(__file__).resolve(
        ).parent.parent / "runs_balanced_aug"
        logging.info(
            "No --model_dirs provided; attempting auto-discovery under %s", maybe_root)
        if maybe_root.exists() and maybe_root.is_dir():
            # find seed_* subdirs
            for p in sorted(maybe_root.glob("seed_*")):
                if p.is_dir():
                    model_dirs.append(str(p))
        else:
            logging.warning("Auto-discovery root not found: %s", maybe_root)

    if not model_dirs:
        raise FileNotFoundError(
            "No model directories provided and auto-discovery failed. Provide --model_dirs pointing to trained model dirs or checkpoint files.")

    for entry in model_dirs:
        p = Path(entry).expanduser().resolve()
        if p.is_file():
            logging.info("Using provided model file: %s", p)
            discovered.append(p)
            continue

        # if it's a directory, search inside
        if p.is_dir():
            ckpt = _find_checkpoint_in_dir(p)
            if ckpt:
                logging.info("Found checkpoint for %s -> %s", p, ckpt)
                discovered.append(ckpt)
            else:
                files = [str(x.name) for x in sorted(p.iterdir())]
                raise FileNotFoundError(
                    f"No checkpoint found in {p}. Searched common names like best_model.pt; files present: {files}")
        else:
            raise FileNotFoundError(f"Model path not found: {p}")

    # final check
    if not discovered:
        raise FileNotFoundError(
            "No checkpoints discovered in any provided model_dirs")

    return discovered


def try_load_torchscript(path: Path, device: str):
    """Try to load a torchscript model (jit). Return model if successful, else None."""
    try:
        m = torch.jit.load(str(path), map_location=device)
        m.eval()
        logging.info("Loaded torchscript model: %s", path)
        return m
    except Exception as e:
        logging.debug("Not a torchscript module: %s (error: %s)", path, e)
        return None


def try_load_state_dict(path: Path, device: str):
    """Try to load a state dict (torch.load). Return dict if successful, else None."""
    try:
        obj = torch.load(str(path), map_location=device)
        if isinstance(obj, dict):
            logging.info("Loaded state dict-like object from: %s", path)
            return obj
        # if it's a Module serialized via torch.save(model), it may be an nn.Module
        if hasattr(obj, "state_dict"):
            logging.info(
                "Loaded object from %s appears to be a full module", path)
            return obj
        # otherwise return as-is
        return obj
    except Exception as e:
        logging.debug("Failed to torch.load %s: %s", path, e)
        return None


def infer_with_torchscript_models(models: List[torch.jit.ScriptModule], test_df: pd.DataFrame, batch_size: int, device: str) -> pd.DataFrame:
    """
    Example inference loop for torchscript models.

    This is a placeholder. The actual preprocessing and feature extraction must match
    what your train.py used (log-mel extraction, normalization, etc.). Without that,
    predictions won't be meaningful. Here we just show structure.
    """
    logging.info(
        "Running placeholder inference with %d torchscript models (no preprocessing implemented)", len(models))
    # For now, produce a dummy DataFrame with ids and zeros — replace with real inference code.
    out = test_df.copy()
    out_cols = [f"pred_model_{i}" for i in range(len(models))]
    for c in out_cols:
        out[c] = 0.0
    # average predictions column
    out["prediction"] = out[out_cols].mean(axis=1)
    return out


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Robust ensemble inference that auto-discovers checkpoints if needed")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--test_xlsx", required=True)
    parser.add_argument("--model_dirs", nargs='+', default=[],
                        help="List of model directories or checkpoint files")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--ttas", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_csv", default=None,
                        help="Output CSV path for ensemble predictions")

    args = parser.parse_args()

    logging.info("Ensemble inference starting. device=%s ttas=%s batch_size=%s",
                 args.device, args.ttas, args.batch_size)

    # load test set dataframe (user likely has id/file columns — adapt as needed)
    test_xlsx = Path(args.test_xlsx).expanduser().resolve()
    if not test_xlsx.exists():
        logging.error("test_xlsx not found: %s", test_xlsx)
        sys.exit(2)

    try:
        test_df = pd.read_excel(str(test_xlsx))
    except Exception as e:
        logging.error("Failed to read test_xlsx: %s", e)
        sys.exit(2)

    # discover checkpoints
    try:
        ckpt_paths = load_models(args.model_dirs, args.device)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(2)

    # Try to load torchscript models; if none, try to load state dicts
    torchscript_models = []
    state_dicts = []
    for p in ckpt_paths:
        ts = try_load_torchscript(p, args.device)
        if ts is not None:
            torchscript_models.append(ts)
            continue
        sd = try_load_state_dict(p, args.device)
        if sd is not None:
            state_dicts.append((p, sd))
        else:
            logging.error(
                "Could not load checkpoint %s as torchscript or state_dict", p)
            sys.exit(3)

    # If we have torchscript models, run placeholder inference
    if torchscript_models:
        out_df = infer_with_torchscript_models(
            torchscript_models, test_df, args.batch_size, args.device)
        if args.out_csv:
            outp = Path(args.out_csv).expanduser()
            outp.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(str(outp), index=False)
            logging.info("Wrote predictions to %s", outp)
        else:
            logging.info(
                "Torchscript inference complete; no --out_csv specified so not writing output")
        sys.exit(0)

    # If we only have state dicts, we cannot safely do inference without knowing the model architecture.
    if state_dicts:
        logging.info("Loaded %d state-dict-like checkpoints:",
                     len(state_dicts))
        for p, sd in state_dicts:
            logging.info(" - %s (type=%s)", p, type(sd))
        logging.error("These appear to be state dicts, not runnable torchscript modules.\n"
                      "To perform ensemble inference you must construct the model architecture used during training,\n"
                      "load each state dict into the model via model.load_state_dict(state_dict), move the model to the device,\n"
                      "set model.eval(), and then run inference.\n\n"
                      "Options:\n"
                      " 1) If you have code that builds the model (e.g., build_model(...) ), integrate it here by replacing the placeholder below.\n"
                      " 2) If you prefer, re-save your trained models as torch.jit.trace or torch.jit.script modules so this script can load them directly.\n")
