#!/usr/bin/env python3
"""
multi_seed_runner_fixed.py

Updated droppable replacement that:
- Accepts all flags you used in your run command (base args, ensemble args, wait_between, etc.)
- Builds commands safely with sys.executable so .py files don't need execute bit
- Deduplicates repeated flags in base_args (preserves last occurrence)
- Provides a compatibility check option --validate that runs `python train.py --help`
  and verifies the base flags appear in the help text (warns if not)
- Supports dry-run, concurrency, wait_between (seconds between task launches),
  ensemble invocation with extra args, logging per-seed

Compatibility note (important):
- The runner will not change or interpret the semantics of your train.py flags.
  It simply forwards them. If you pass the same flag twice in base_args, the
  runner will keep the last occurrence (dedupe behavior) and warn about duplicates.
- Use `--validate` to check that each flag you supplied in --base_args is mentioned
  in `train.py --help`. This is a *best-effort* check (it looks for the flag string in
  the help text) and may produce false positives/negatives for very custom parsers.

Usage (example):
python multi_seed_runner_fixed.py \
  --train_script /Users/pranav/Desktop/Task1/sand_vit_baseline/train.py \
  --base_args "--data_dir /Users/pranav/Desktop/Task1 --train_xlsx /Users/pranav/Desktop/Task1/training/sand_task_1.xlsx --cache_dir /Users/pranav/Desktop/Task1/cache_logmel_N256 --train_sheet 'Training Baseline - Task 1' --val_sheet 'Validation Baseline - Task 1' --pretrained --freeze_backbone --unfreeze_k 8 --stage1_epochs 3 --stage2_epochs 50 --batch_size 32 --use_sampler --loss focal --optimizer adamw --lr_head 1e-3 --lr_ft 4e-5 --weight_decay 0.01 --device mps --save_every 5" \
  --out_root /Users/pranav/Desktop/Task1/sand_vit_baseline/runs_balanced_aug \
  --seeds 44 2 563 \
  --concurrency 1 \
  --wait_between 1 \
  --ensemble_script /Users/pranav/Desktop/Task1/sand_vit_baseline/ensemble_inference.py \
  --ensemble_out /Users/pranav/Desktop/Task1/sand_vit_baseline/ensemble/submission_balanced_aug.csv \
  --ensemble_cmd_extra "--data_dir /Users/pranav/Desktop/Task1 --test_xlsx /Users/pranav/Desktop/Task1/test/sand_task1_test.xlsx --cache_dir /Users/pranav/Desktop/Task1/cache_logmel_N256 --ttas 5 --device mps --batch_size 32" \
  --validate

"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import threading
import os
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import queue
import time


def setup_logging(level=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(format=fmt, level=level)


def _split_and_dedupe_args(base_args: str) -> Tuple[List[str], List[str]]:
    """Split a base_args string into tokens and deduplicate flags.

    Deduplication policy:
    - Walk tokens left->right.
    - For a flag token starting with '--', if the next token exists and does not start
      with '--' then it's considered the flag's value and stored as (flag, value)
      otherwise it's a boolean flag with no value.
    - If a flag repeats, the LAST occurrence wins (we keep the later value).

    Returns (deduped_tokens_list, list_of_duplicate_flag_names) where deduped_tokens_list
    is suitable for passing to subprocess.
    """
    tokens = shlex.split(base_args)
    i = 0
    parsed = []  # list of (flag, value_or_None)
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith('--'):
            # check if next token is a value
            if i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
                parsed.append((tok, tokens[i + 1]))
                i += 2
            else:
                parsed.append((tok, None))
                i += 1
        else:
            # bare token without leading -- (positional) - keep as-is
            parsed.append((None, tok))
            i += 1

    # dedupe: preserve order but last occurrence wins for flags
    flag_map = {}
    pos_tokens = []  # list of positional tokens (None, value)
    for flag, val in parsed:
        if flag is None:
            pos_tokens.append(val)
        else:
            flag_map[flag] = val

    deduped_tokens = []
    # preserve the original order of first appearance of flags as they appear in parsed,
    # but because last occurrence wins we will iterate parsed and build an ordered unique list
    seen = set()
    for flag, val in parsed:
        if flag is None:
            # include positional tokens in the order they appeared
            deduped_tokens.append(val)
        else:
            if flag in seen:
                continue
            # find the final value for this flag from flag_map
            final_val = flag_map[flag]
            deduped_tokens.append(flag)
            if final_val is not None:
                deduped_tokens.append(final_val)
            seen.add(flag)

    # compute duplicates list for warning: any flag that had multiple occurrences
    flag_counts = {}
    for flag, val in parsed:
        if flag is not None:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    duplicates = [f for f, cnt in flag_counts.items() if cnt > 1]

    return deduped_tokens, duplicates


def build_train_cmd(train_script: str, base_args: str, seed: int, out_dir: Path) -> Tuple[List[str], List[str]]:
    """Build a command list that runs the script with the current Python interpreter.

    Returns (cmd_list, duplicates_list) where duplicates_list are duplicated flags discovered
    in base_args.
    """
    parts, duplicates = _split_and_dedupe_args(base_args)
    script_path = str(Path(train_script).expanduser().resolve())
    cmd = [sys.executable, script_path] + parts + \
        ["--seed", str(seed), "--out_dir", str(out_dir)]
    return cmd, duplicates


def run_process(cmd: List[str], cwd: Optional[Path], log_path: Path, dry_run: bool = False) -> Optional[subprocess.Popen]:
    """Run cmd (list) and stream output to log_path. Returns Popen if started, else None on dry-run."""
    logging.info("Running: %s", shlex.join(cmd))
    logging.info("Log: %s", log_path)

    if dry_run:
        logging.info("Dry run enabled — not executing.")
        return None

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the log file and spawn the subprocess (append mode)
    f = open(log_path, "ab")
    proc = subprocess.Popen(cmd, cwd=str(
        cwd) if cwd else None, stdout=f, stderr=subprocess.STDOUT)
    return proc


def _validate_flags_with_help(train_script: str, flags: List[str]) -> Tuple[bool, List[str]]:
    """Run `python train_script --help` and check presence of each flag (best-effort).

    Returns (all_found_bool, list_of_missing_flags)
    """
    script_path = str(Path(train_script).expanduser().resolve())
    try:
        proc = subprocess.Popen([sys.executable, script_path, "--help"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate(timeout=10)
        help_text = out.decode(errors='ignore')
    except Exception as e:
        logging.warning(
            "Could not run train_script --help for validation: %s", e)
        return False, flags

    missing = []
    for flag in flags:
        # only check flags starting with --
        if not flag.startswith('--'):
            continue
        if flag not in help_text:
            missing.append(flag)
    ok = len(missing) == 0
    return ok, missing


def worker(worker_id: int, task_q: "queue.Queue[dict]", result_q: "queue.Queue[dict]", dry_run: bool):
    logging.info("Worker %d starting", worker_id)
    while True:
        try:
            task = task_q.get(timeout=1)
        except queue.Empty:
            break

        cmd = task["cmd"]
        cwd = task.get("cwd")
        log_path = task["log_path"]
        seed = task.get("seed")

        try:
            proc = run_process(cmd, cwd, log_path, dry_run=dry_run)
            if proc is None:
                result_q.put(
                    {"seed": seed, "returncode": None, "log": str(log_path)})
            else:
                rc = proc.wait()
                result_q.put(
                    {"seed": seed, "returncode": rc, "log": str(log_path)})
        except Exception as e:
            logging.exception("Exception while running seed %s", seed)
            result_q.put({"seed": seed, "error": str(e), "log": str(log_path)})
        finally:
            task_q.task_done()
    logging.info("Worker %d exiting", worker_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust multi-seed runner that executes train.py with the current Python interpreter.")

    parser.add_argument("--train_script", required=True,
                        help="Path to the training script (train.py)")
    parser.add_argument("--base_args", required=True,
                        help="All args to pass to train.py except --seed and --out_dir. Provide as a single quoted string.")
    parser.add_argument("--seeds", required=True, nargs='+', type=int,
                        help="Space-separated list of seeds to run (e.g. --seeds 44 2 563)")
    parser.add_argument("--out_root", required=True,
                        help="Root directory where per-seed out_dir will be created")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of parallel workers to run (default: 1)")
    parser.add_argument("--cwd", default=None,
                        help="Optional working directory for spawned processes")
    parser.add_argument("--dry_run", action='store_true',
                        help="Print commands but do not execute")
    parser.add_argument("--save_logs", action='store_true',
                        help="Save logs under each out_dir/log.txt (default True)")
    parser.add_argument("--wait", action='store_true',
                        help="Wait for all processes to finish before exiting (default True)")

    # New args requested by the user
    parser.add_argument("--wait_between", type=float, default=0.0,
                        help="Seconds to wait between launching tasks (default 0)")
    parser.add_argument("--ensemble_script", default=None,
                        help="Optional path to an ensemble/inference script to run after seeds complete")
    parser.add_argument("--ensemble_out", default=None,
                        help="Output path (CSV) for ensemble script")
    parser.add_argument("--ensemble_cmd_extra", default="",
                        help="Extra arguments (single-quoted string) to pass to the ensemble script")

    # Validation flag
    parser.add_argument("--validate", action='store_true',
                        help="Run train_script --help and check that base flags appear in help text")

    # Backwards-compatible flags from your earlier runner
    parser.add_argument("--save_every", type=int,
                        default=5, help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    train_script = args.train_script
    base_args = args.base_args
    seeds: List[int] = args.seeds
    out_root = Path(args.out_root).expanduser().resolve()
    cwd = Path(args.cwd).expanduser().resolve() if args.cwd else None
    concurrency = max(1, args.concurrency)
    dry_run = args.dry_run
    wait_between = max(0.0, float(args.wait_between))

    ensemble_script = args.ensemble_script
    ensemble_out = args.ensemble_out
    ensemble_cmd_extra = args.ensemble_cmd_extra or ""
    validate = args.validate

    logging.info("Using Python interpreter: %s", sys.executable)
    logging.info("Train script: %s", train_script)
    logging.info("Out root: %s", out_root)
    logging.info("Seeds: %s", seeds)
    logging.info("Concurrency: %d", concurrency)
    logging.info("Dry run: %s", dry_run)
    logging.info("Wait between launches: %s seconds", wait_between)
    logging.info("Ensemble script: %s", ensemble_script)
    logging.info("Validate flags via --help: %s", validate)

    # Basic checks
    train_script_path = Path(train_script).expanduser()
    if not train_script_path.exists():
        logging.error("train_script does not exist: %s", train_script_path)
        sys.exit(2)
    if train_script_path.suffix != '.py':
        logging.warning(
            "train_script does not have .py suffix (still allowed): %s", train_script_path)

    # Create output root
    out_root.mkdir(parents=True, exist_ok=True)

    # Parse and dedupe base args once (for validation and for building each command)
    deduped_parts, duplicates = _split_and_dedupe_args(base_args)
    if duplicates:
        logging.warning(
            "Detected duplicate flags in --base_args; the last occurrence will be used: %s", duplicates)

    # If validate requested, run help and check flags
    if validate:
        # collect only the flags (tokens starting with --) from deduped_parts
        flags_to_check = [tok for tok in deduped_parts if tok.startswith('--')]
        ok, missing = _validate_flags_with_help(train_script, flags_to_check)
        if ok:
            logging.info(
                "Validation passed: all provided flags were found in train_script --help")
        else:
            logging.warning(
                "Validation found missing flags in train_script --help: %s", missing)
            logging.info(
                "Continuing run despite missing flags — if these are custom flags, ignore this warning")

    task_q: "queue.Queue[dict]" = queue.Queue()
    result_q: "queue.Queue[dict]" = queue.Queue()

    # Build tasks and enqueue, optionally waiting between enqueues
    for seed in seeds:
        seed_out = out_root / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        log_path = seed_out / "run.log"
        cmd, dup = build_train_cmd(train_script, base_args, seed, seed_out)
        if dup:
            logging.warning(
                "(seed %s) Duplicate flags detected and resolved: %s", seed, dup)
        task_q.put({"cmd": cmd, "cwd": cwd, "log_path": log_path, "seed": seed})
        if wait_between > 0:
            logging.info("Sleeping %.2fs between job enqueues", wait_between)
            time.sleep(wait_between)

    # Start workers
    workers = []
    for i in range(concurrency):
        t = threading.Thread(target=worker, args=(
            i + 1, task_q, result_q, dry_run), daemon=True)
        t.start()
        workers.append(t)

    # Wait for queue to be processed
    try:
        while any(t.is_alive() for t in workers):
            time.sleep(0.5)
            # drain and print some results as they come
            while not result_q.empty():
                res = result_q.get()
                if "error" in res:
                    logging.error("Seed %s failed with error: %s (log: %s)", res.get(
                        "seed"), res.get("error"), res.get("log"))
                else:
                    logging.info("Seed %s finished with returncode=%s (log: %s)", res.get(
                        "seed"), res.get("returncode"), res.get("log"))
                result_q.task_done()
    except KeyboardInterrupt:
        logging.warning(
            "KeyboardInterrupt received — exiting and not launching new tasks")

    # Final drain of results
    while not result_q.empty():
        res = result_q.get()
        if "error" in res:
            logging.error("Seed %s failed with error: %s (log: %s)", res.get(
                "seed"), res.get("error"), res.get("log"))
        else:
            logging.info("Seed %s finished with returncode=%s (log: %s)", res.get(
                "seed"), res.get("returncode"), res.get("log"))
        result_q.task_done()

    logging.info("All workers done.")

    # Optionally run ensemble script
    if ensemble_script:
        logging.info("Running ensemble script: %s", ensemble_script)
        ensemble_parts = shlex.split(
            ensemble_cmd_extra) if ensemble_cmd_extra else []
        ensemble_script_path = str(
            Path(ensemble_script).expanduser().resolve())
        ensemble_cmd = [sys.executable, ensemble_script_path] + ensemble_parts
        if ensemble_out:
            # append ensemble_out with a flag name --ensemble_out (see notes at top)
            ensemble_cmd += ["--ensemble_out",
                             str(Path(ensemble_out).expanduser().resolve())]
        logging.info("Ensemble command: %s", shlex.join(ensemble_cmd))
        if dry_run:
            logging.info("Dry run — not executing ensemble script")
        else:
            ensemble_log = out_root / "ensemble.log"
            ensemble_log.parent.mkdir(parents=True, exist_ok=True)
            with open(ensemble_log, "ab") as f:
                rc = subprocess.call(ensemble_cmd, stdout=f,
                                     stderr=subprocess.STDOUT)
            if rc != 0:
                logging.error(
                    "Ensemble script exited with return code %d (see %s)", rc, ensemble_log)
            else:
                logging.info(
                    "Ensemble script completed successfully (see %s)", ensemble_log)


if __name__ == "__main__":
    main()
