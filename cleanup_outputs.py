#!/usr/bin/env python3
"""
cleanup_outputs.py

Safely clean up training/output directories inside a project (e.g. sand_vit_baseline).
By default this script runs in DRY-RUN mode and only prints what it *would* move.
To actually move files to a recoverable trash directory, run with --apply.
To permanently delete the moved items, run with --apply --purge (use with CAUTION).

Behavior (conservative):
 - Scans immediate subdirectories of the provided root (default: current working dir).
 - For any directory that looks like an experiment/output directory (name starts with
   "outputs", "hp_", "hp-smote", "hp-smote", "hp", "run_", "hp_runs", "hp_smote", "smote", etc.)
   it will:
     - Keep (leave in place) files/dirs that match KEEP_PATTERNS:
         * "best_model.pt"
         * any file/dir containing the substring "best" (case-insensitive)
         * "hp_summary.json", "results.csv", "hp_summary*.json"
         * "outputs_best_run" (dir)
     - Move all other files & directories from that experiment dir into a timestamped
       trash folder under root (default: ./cleaned_trash/<timestamp>/...) so you can recover later.
 - Leaves other top-level project files and folders untouched.
 - Creates a log file in the trash folder describing moves.

Usage:
  # dry run (default) -> prints actions
  python cleanup_outputs.py --root /path/to/sand_vit_baseline

  # actually perform move (safe - moved to trash folder)
  python cleanup_outputs.py --root /path/to/sand_vit_baseline --apply

  # perform move and then permanently delete moved items from trash
  python cleanup_outputs.py --root /path/to/sand_vit_baseline --apply --purge

Notes:
 - This is intentionally conservative: it MOVES files to a trash folder (not immediate rm)
   unless you pass --purge. Always inspect the trash folder before purging.
 - If you want different keep rules, edit KEEP_PATTERNS below.
"""

from pathlib import Path
import argparse
import fnmatch
import shutil
import datetime
import json
import sys

# --- Customize keep/match rules here ---
# Keep anything that matches one of these glob patterns (relative to the experiment dir)
KEEP_PATTERNS = [
    "best_model.pt",
    "*best*",
    "*Best*",
    "hp_summary.json",
    "hp_summary*.json",
    "results.csv",
    "outputs_best_run",
    "outputs_best_run.*",
    "README*",
    "*.md",
    "*.txt",
]

# Directory-name patterns considered to be "experiment/output" directories to clean
EXP_DIR_PREFIXES = (
    "outputs",    # outputs_*
    "hp_",        # hp_*
    "hp",         # hp*
    "run_",       # run_*
    "hp_runs",
    "hp-smote",
    "hp_smote",
    "smote",
    "outputs_",
    "outputs",
)

# Files/dirs to ignore entirely (never move), relative to project root
ROOT_PROTECT = {
    ".git", ".venv", "venv", "env", ".env", "requirements.txt", "dataset.py", "model.py",
    "train.py", "smote_train.py", "hyperparam_tune.py", "hyper_smote_tune.py",
    "ensemble_inference.py", "README.md", "README", "README.txt",
}

# --------------------------------------------------


def is_experiment_dir(name: str) -> bool:
    name_lower = name.lower()
    for p in EXP_DIR_PREFIXES:
        if name_lower.startswith(p.lower()):
            return True
    return False


def matches_keep(name: str) -> bool:
    for pat in KEEP_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def scan_and_plan(root: Path):
    """
    Returns a plan dict:
      { exp_dir_path: { "keep": [paths], "move": [paths] } }
    """
    plan = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in ROOT_PROTECT:
            continue
        if not is_experiment_dir(child.name):
            continue

        keep = []
        move = []
        # Walk only one level deep (items inside the experiment dir)
        for item in sorted(child.iterdir()):
            nm = item.name
            if matches_keep(nm):
                keep.append(item)
            else:
                move.append(item)
        # If nothing to move, skip
        if not move:
            continue
        plan[str(child)] = {"keep": [str(p)
                                     for p in keep], "move": [str(p) for p in move]}
    return plan


def perform_move(plan: dict, trash_root: Path, dry_run: bool = True):
    """
    Move planned items into trash_root/<experiment-name>/
    Returns list of moved paths
    """
    moved = []
    trash_root.mkdir(parents=True, exist_ok=True)
    log = []
    for exp_dir, info in plan.items():
        exp_path = Path(exp_dir)
        dest_dir = trash_root / exp_path.name
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
        for p_str in info["move"]:
            src = Path(p_str)
            dest = dest_dir / src.name
            if dry_run:
                print(f"[DRY] Would move: {src} -> {dest}")
                log.append(
                    {"action": "move(dry)", "src": str(src), "dest": str(dest)})
            else:
                try:
                    shutil.move(str(src), str(dest))
                    print(f"Moved: {src} -> {dest}")
                    moved.append(str(dest))
                    log.append(
                        {"action": "move", "src": str(src), "dest": str(dest)})
                except Exception as e:
                    print(f"Failed to move {src} -> {dest}: {e}")
                    log.append({"action": "move_failed", "src": str(
                        src), "dest": str(dest), "error": str(e)})
    # write log
    log_path = trash_root / "cleanup_log.json"
    try:
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
    except Exception:
        pass
    return moved


def purge_trash(trash_root: Path):
    """Permanently delete trash_root (careful)"""
    if not trash_root.exists():
        print("Trash root does not exist:", trash_root)
        return
    print("Purging (permanent delete) trash folder:", trash_root)
    # Try to delete, printing large ops
    try:
        shutil.rmtree(str(trash_root))
        print("Purge successful.")
    except Exception as e:
        print("Purge failed:", e)


def main():
    p = argparse.ArgumentParser(
        description="Cleanup experiment/output folders, keep only best models.")
    p.add_argument("--root", type=str, default=".",
                   help="Project root path (default CWD)")
    p.add_argument("--apply", action="store_true",
                   help="Actually move files (default: dry-run).")
    p.add_argument("--purge", action="store_true",
                   help="If set with --apply, permanently delete the trash (USE WITH CAUTION).")
    p.add_argument("--trash", type=str, default=None,
                   help="Explicit trash folder path (default: ./cleaned_trash/<ts>)")
    p.add_argument("--dry-run", dest="dry_run",
                   action="store_true", help="Force dry-run (no moves).")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print("Root not found:", root)
        sys.exit(1)

    # generate default trash folder with timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_trash = root / "cleaned_trash" / ts
    trash_root = Path(args.trash).expanduser(
    ).resolve() if args.trash else default_trash

    plan = scan_and_plan(root)
    if not plan:
        print("No candidate experiment/output dirs found for cleanup under", root)
        return

    # Print summary
    print("Planned cleanup for project root:", root)
    total_moves = sum(len(v["move"]) for v in plan.values())
    print(f"Found {len(plan)} experiment dirs with {total_moves} items to MOVE (keep patterns: {KEEP_PATTERNS})")
    if args.verbose:
        for exp, info in plan.items():
            print("\nExp dir:", exp)
            print("  Keep:")
            for k in info["keep"]:
                print("    ", k)
            print("  Move:")
            for m in info["move"]:
                print("    ", m)

    # If user didn't pass --apply or passed --dry-run, perform dry-run behavior
    do_dry = (not args.apply) or args.dry_run
    if do_dry:
        print("\nDRY RUN (no files moved). To actually move items, re-run with --apply.")
        _ = perform_move(plan, trash_root, dry_run=True)
        print(
            f"\nDry-run log written to: {trash_root} (cleanup_log.json created if possible).")
        return

    # Actual apply: move files
    moved = perform_move(plan, trash_root, dry_run=False)
    print(f"\nMoved {len(moved)} items into trash folder: {trash_root}")
    print("Inspect the trash folder before purging. If everything looks good you can re-run with --purge to delete permanently.")

    if args.purge:
        ans = input(
            f"\nYou requested --purge. This will PERMANENTLY DELETE {trash_root}. Type 'YES' to confirm: ")
        if ans.strip() == "YES":
            purge_trash(trash_root)
        else:
            print("Purge cancelled. Trash still at:", trash_root)


if __name__ == "__main__":
    main()
