#!/usr/bin/env python3
# precompute_mels.py
"""
Precompute log-mel + delta1 + delta2 stacks and save as .npy files.

Each output file:
    shape = (3, N_MELS, T)
    dtype = float32
    channels = [mel_db, delta1, delta2]

Output structure:
    <out_dir>/<sheet_name_sanitized>/<ID>.npy

Example:
    python precompute_mels.py \
      --data_dir /Users/pranav/Desktop/task \
      --xlsx /Users/pranav/Desktop/task/training/sand_task_1.xlsx \
      --sheets "Training Baseline - Task 1" "Validation Baseline - Task 1" \
      --out_dir /Users/pranav/Desktop/task/cache_logmel_N256 \
      --workers 6
"""

import argparse
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import traceback
import pandas as pd

# Must match dataset.py
SAMPLE_RATE = 8000
DURATION_SEC = 5
TARGET_LEN = SAMPLE_RATE * DURATION_SEC
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = None
TOP_DB = 80.0

# Folder/priority order
DEFAULT_PRIORITY = [
    'phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
    'rhythmKA', 'rhythmPA', 'rhythmTA'
]


def find_wav_for_subject(root: Path, sid: str, recording_priority=DEFAULT_PRIORITY):
    """Try preferred folders first, then fallback."""
    for folder in recording_priority:
        p1 = root / folder / f"{sid}_{folder}.wav"
        p2 = root / folder / f"{sid}_{folder.lower()}.wav"
        if p1.exists():
            return p1
        if p2.exists():
            return p2

    # fallback search
    for folder in root.iterdir():
        if folder.is_dir():
            for f in folder.iterdir():
                if f.is_file() and f.name.startswith(f"{sid}_") and f.suffix.lower() == ".wav":
                    return f
    return None


def load_and_process(wav_path: Path):
    """Load wav → pad/truncate → mel → deltas → stacked numpy."""
    data, sr = sf.read(str(wav_path))

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != SAMPLE_RATE:
        data = librosa.resample(data.astype('float32'),
                                orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # pad/trim to exactly 5 sec
    if len(data) < TARGET_LEN:
        data = np.pad(data, (0, TARGET_LEN - len(data)))
    else:
        data = data[:TARGET_LEN]

    data = data.astype('float32')

    # mel-power → mel-dB
    S = librosa.feature.melspectrogram(
        y=data,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)
    d1 = librosa.feature.delta(mel_db, order=1)
    d2 = librosa.feature.delta(mel_db, order=2)

    stack = np.stack([mel_db, d1, d2], axis=0).astype('float32')
    return stack


def process_one(subject_id, root_dir, out_dir, sheet_name, force=False):
    """Process one subject ID."""
    sid = str(subject_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sid}.npy"

    if out_path.exists() and not force:
        return sid, True, "exists"

    wav_path = find_wav_for_subject(root_dir, sid)
    if wav_path is None:
        return sid, False, "wav_not_found"

    try:
        arr = load_and_process(wav_path)
        np.save(str(out_path), arr)
        return sid, True, "ok"
    except Exception as e:
        tb = traceback.format_exc()
        return sid, False, f"error: {e}\n{tb}"


def process_sheet(task_root, xlsx_path, sheet_name, out_root, workers, force=False):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if "ID" not in df.columns:
        raise ValueError(f"Sheet {sheet_name} missing 'ID' column.")

    ids = df["ID"].astype(str).tolist()

    out_dir = out_root / sheet_name.replace(" ", "_")
    root_dir = task_root / "training"

    worker_fn = partial(
        process_one,
        root_dir=root_dir,
        out_dir=out_dir,
        sheet_name=sheet_name,
        force=force
    )

    results = []
    if workers == 0:
        for sid in tqdm(ids, desc=f"Processing {sheet_name}"):
            results.append(worker_fn(sid))
    else:
        with Pool(workers) as pool:
            for res in tqdm(pool.imap_unordered(worker_fn, ids), total=len(ids), desc=f"Processing {sheet_name}"):
                results.append(res)

    # write missing list
    missing = [r for r in results if not r[1]]
    with open(out_root / f"missing_{sheet_name.replace(' ', '_')}.txt", "w") as f:
        for sid, ok, msg in missing:
            f.write(f"{sid}\t{msg}\n")

    print(f"[{sheet_name}] total={len(ids)} saved={len(results) - len(missing)} missing={len(missing)}")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--xlsx", required=True)
    p.add_argument("--sheets", nargs="+", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    xlsx = Path(args.xlsx)

    out_root = Path(args.out_dir) if args.out_dir else data_dir / \
        "cache_logmel_N256"
    out_root.mkdir(parents=True, exist_ok=True)

    print("Precomputing mel stacks…")
    print("Output directory:", out_root)

    for sheet in args.sheets:
        print(f"\n======= Processing sheet: {sheet} =======")
        process_sheet(data_dir, xlsx, sheet, out_root,
                      args.workers, force=args.force)


if __name__ == "__main__":
    main()
