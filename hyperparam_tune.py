#!/usr/bin/env python3
"""
hyperparam_tune.py

Grid / random hyperparameter launcher for train.py.
This version WILL append --pretrained and --freeze_backbone to every generated train.py command.

Usage example (similar to what you've used):
python hyperparam_tune.py \
  --train_py ./train.py \
  --data_dir /Users/pranav/Desktop/Task1 \
  --train_xlsx /Users/pranav/Desktop/Task1/training/sand_task_1.xlsx \
  --cache_dir /Users/pranav/Desktop/Task1/cache_logmel_N256 \
  --out_root ./hp_runs_v2 \
  --mode grid \
  --stage1_epochs 1 \
  --stage2_epochs 5 \
  --lr_ft 5e-5 3e-5 1e-5 \
  --lr_head 1e-3 3e-3 \
  --batch_size 8 12 \
  --optimizer lion adamw \
  --unfreeze_k 2 4 \
  --weight_decay 1e-5 1e-3 \
  --use_sampler \
  --use_class_weights \
  --max_trials 12
"""

import argparse
import itertools
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_py', type=str, default='./train.py',
                   help='Path to train.py')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--train_xlsx', type=str, required=True)
    p.add_argument('--train_sheet', type=str,
                   default='Training Baseline - Task 1')
    p.add_argument('--val_sheet', type=str,
                   default='Validation Baseline - Task 1')
    p.add_argument('--cache_dir', type=str, default=None)
    p.add_argument('--out_root', type=str, default='./hp_runs')
    p.add_argument('--mode', choices=['grid', 'random'], default='grid')
    p.add_argument('--stage1_epochs', type=int, default=1)
    p.add_argument('--stage2_epochs', type=int, default=5)
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--max_trials', type=int, default=20)
    p.add_argument('--lr_ft', nargs='+', type=float, default=[5e-5])
    p.add_argument('--lr_head', nargs='+', type=float, default=[1e-3])
    p.add_argument('--batch_size', nargs='+', type=int, default=[8])
    p.add_argument('--optimizer', nargs='+', type=str, default=['adamw'])
    p.add_argument('--unfreeze_k', nargs='+', type=int, default=[2])
    p.add_argument('--weight_decay', nargs='+', type=float, default=[1e-5])
    p.add_argument('--use_sampler', action='store_true')
    p.add_argument('--use_class_weights', action='store_true')
    p.add_argument('--loss', choices=['ce', 'focal'], default='focal')
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--device', type=str, default='mps')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--dry_run', action='store_true',
                   help='Only print the generated commands')
    p.add_argument('--seed_shuffle', action='store_true',
                   help='Randomize trials order')
    return p.parse_args()


def build_candidates(args):
    grid_items = {
        'lr_ft': args.lr_ft,
        'lr_head': args.lr_head,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'unfreeze_k': args.unfreeze_k,
        'weight_decay': args.weight_decay,
        'seed': args.seeds,
    }
    names = list(grid_items.keys())
    values = [grid_items[k] for k in names]
    all_combos = list(itertools.product(*values))
    # convert combos to dicts
    combos = []
    for combo in all_combos:
        d = dict(zip(names, combo))
        combos.append(d)
    return combos


def sample_random(candidates, max_trials):
    if len(candidates) <= max_trials:
        return candidates
    return random.sample(candidates, max_trials)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_outdir(base: Path, idx: int, cfg: dict):
    name_elems = []
    name_elems.append(f"run_{idx:03d}")
    # short tag of key params
    name_elems.append(f"lrft{cfg['lr_ft']}")
    name_elems.append(f"lrh{cfg['lr_head']}")
    name_elems.append(f"bs{cfg['batch_size']}")
    name_elems.append(f"opt{cfg['optimizer']}")
    name_elems.append(f"k{cfg['unfreeze_k']}")
    name_elems.append(f"wd{cfg['weight_decay']}")
    name = "_".join(name_elems)
    return base / name


def build_cmd(train_py: str, cfg: dict, global_args: argparse.Namespace, out_dir: Path):
    """
    Build the train.py command for a trial.
    THIS FUNCTION AUTOMATICALLY APPENDS:
      --pretrained
      --freeze_backbone
    to the command (as requested).
    """
    cmd = [sys.executable, str(Path(train_py).resolve())]
    # required args
    cmd += ['--data_dir', str(Path(global_args.data_dir).resolve())]
    cmd += ['--train_xlsx', str(Path(global_args.train_xlsx).resolve())]
    cmd += ['--train_sheet', global_args.train_sheet]
    cmd += ['--val_sheet', global_args.val_sheet]
    # cache dir
    if global_args.cache_dir:
        cmd += ['--cache_dir', str(Path(global_args.cache_dir).resolve())]
    # stage epochs
    cmd += ['--stage1_epochs', str(global_args.stage1_epochs)]
    cmd += ['--stage2_epochs', str(global_args.stage2_epochs)]
    # hyperparams from cfg
    cmd += ['--lr_ft', str(cfg['lr_ft'])]
    cmd += ['--lr_head', str(cfg['lr_head'])]
    cmd += ['--batch_size', str(cfg['batch_size'])]
    cmd += ['--optimizer', str(cfg['optimizer'])]
    cmd += ['--unfreeze_k', str(cfg['unfreeze_k'])]
    cmd += ['--weight_decay', str(cfg['weight_decay'])]
    # add flags requested
    if global_args.use_sampler:
        cmd.append('--use_sampler')
    if global_args.use_class_weights:
        cmd.append('--use_class_weights')
    # loss & gamma
    cmd += ['--loss', global_args.loss]
    cmd += ['--gamma', str(global_args.gamma)]
    # device, workers, save_every
    cmd += ['--device', global_args.device]
    cmd += ['--num_workers', str(global_args.num_workers)]
    cmd += ['--save_every', str(global_args.save_every)]
    # seed
    cmd += ['--seed', str(cfg['seed'])]
    # output dir
    cmd += ['--out_dir', str(out_dir)]
    # IMPORTANT: append the two flags to force pretrained & frozen backbone
    cmd += ['--pretrained', '--freeze_backbone']
    return cmd


def run_cmd_stream(cmd, cwd=None):
    """
    Run a command (list) and stream stdout / stderr to console.
    Returns completed process returncode.
    """
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True, cwd=cwd)
    try:
        for line in proc.stdout:
            print(line, end='')
    except KeyboardInterrupt:
        proc.kill()
        raise
    proc.wait()
    return proc.returncode


def main():
    args = parse_args()
    # build candidate list
    candidates = build_candidates(args)
    # optionally shuffle
    if args.seed_shuffle:
        random.Random(0).shuffle(candidates)
    # limit by max_trials (grid vs random)
    if args.mode == 'grid':
        chosen = candidates[:args.max_trials]
    else:  # random
        chosen = sample_random(candidates, args.max_trials)
    # expand with seeds if not already included in combos
    # Note: build_candidates already included seeds, but user might pass single seed; keep as-is.
    out_root = Path(args.out_root)
    ensure_dir(out_root)
    summary = []
    idx = 0
    for cfg in chosen:
        idx += 1
        out_dir = make_outdir(out_root, idx, cfg)
        ensure_dir(out_dir)
        # build command
        cmd = build_cmd(args.train_py, cfg, args, out_dir)
        if args.dry_run:
            print("[dry_run] CMD:", " ".join(shlex.quote(x) for x in cmd))
            summary.append({'idx': idx, 'cmd': " ".join(shlex.quote(x)
                           for x in cmd), 'out_dir': str(out_dir)})
            continue
        # run
        t0 = datetime.now()
        rc = run_cmd_stream(cmd, cwd=str(Path(args.train_py).parent))
        t1 = datetime.now()
        dt = (t1 - t0).total_seconds()
        summary.append({'idx': idx, 'rc': rc, 'out_dir': str(
            out_dir), 'cfg': cfg, 'time_s': dt})
        # quick status
        print(
            f"=== Trial {idx} finished (rc={rc}) time={dt:.1f}s out_dir={out_dir} ===\n")
    # write summary json
    summary_path = out_root / "hp_summary.json"
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print("All trials done. Summary written to", summary_path)


if __name__ == '__main__':
    main()
