#!/usr/bin/env python3
# smote_train.py
"""
SMOTE-based training script for SAND ViT baseline.

Example:
python smote_train.py \
  --data_dir /Users/pranav/Desktop/Task1 \
  --train_xlsx /Users/pranav/Desktop/Task1/training/sand_task_1.xlsx \
  --cache_dir /Users/pranav/Desktop/Task1/cache_logmel_N256 \
  --train_sheet "Training Baseline - Task 1" \
  --val_sheet "Validation Baseline - Task 1" \
  --pretrained \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --device mps \
  --out_dir ./smote_run
"""

import argparse
from pathlib import Path
import sys
import os
import json
import time
import logging
from typing import Tuple, List

import numpy as np
import soundfile as sf
import librosa
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm

# import user modules (assumes these exist in same folder)
try:
    from model import build_vit
except Exception as e:
    raise ImportError("Could not import build_vit from model.py: " + str(e))
try:
    from utils import averaged_f1_from_preds, ensure_dir
except Exception:
    # if utils isn't present, provide a minimal averaged_f1 fallback (micro F1 average across classes)
    def averaged_f1_from_preds(y_true, y_pred, classes):
        # simple f1 per class
        from sklearn.metrics import f1_score
        per = []
        for c in classes:
            y_t = [1 if y == c else 0 for y in y_true]
            y_p = [1 if y == c else 0 for y in y_pred]
            per.append(float(f1_score(y_t, y_p, zero_division=0)))
        return float(np.mean(per)), per

    def ensure_dir(p):
        Path(p).mkdir(parents=True, exist_ok=True)

# imblearn SMOTE
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

# try torch_optimizer for Lion etc.
try:
    import torch_optimizer as topt
except Exception:
    topt = None

# default mel parameters must match your dataset.py
N_MELS = 256
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = None
TOP_DB = 80.0

# helper: build optimizer by name (supports adamw + torch_optimizer.Lion if installed)


def get_optimizer_by_name(name: str, params, lr: float, weight_decay: float):
    n = name.lower()
    if n in ("adamw", "adam-w", "adam"):
        if n == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if n == "lion":
        if topt is not None and hasattr(topt, "Lion"):
            return topt.Lion(params, lr=lr, weight_decay=weight_decay)
        # try lion-pytorch if installed (different package)
        try:
            import lion_pytorch
            return lion_pytorch.Lion(params, lr=lr)
        except Exception:
            logging.warning(
                "Lion optimizer not available; falling back to AdamW")
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if n == "radam":
        if topt is not None and hasattr(topt, "RAdam"):
            return topt.RAdam(params, lr=lr, weight_decay=weight_decay)
        try:
            return optim.RAdam(params, lr=lr, weight_decay=weight_decay)
        except Exception:
            logging.warning("RAdam not available; falling back to AdamW")
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    logging.warning("Unknown optimizer '%s', using AdamW", name)
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# ---------- utilities for loading mel stacks ----------
def load_cached_stack(cache_dir: Path, sid: str):
    p = cache_dir / f"{sid}.npy"
    if not p.exists():
        return None
    arr = np.load(str(p))
    return arr.astype("float32")


def compute_stack_from_wav(wav_path: Path, sample_rate=8000, duration=5):
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sample_rate:
        data = librosa.resample(data.astype('float32'),
                                orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    target_len = sample_rate * duration
    if len(data) < target_len:
        data = np.concatenate(
            [data, np.zeros(target_len - len(data), dtype=data.dtype)])
    elif len(data) > target_len:
        data = data[:target_len]
    if data.dtype.kind == 'i':
        info = np.iinfo(data.dtype)
        data = data.astype('float32') / max(abs(info.min), info.max)
    else:
        data = data.astype('float32')
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                       win_length=WIN_LENGTH, n_mels=N_MELS, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)
    d1 = librosa.feature.delta(S_db, order=1)
    d2 = librosa.feature.delta(S_db, order=2)
    return np.stack([S_db.astype('float32'), d1.astype('float32'), d2.astype('float32')], axis=0)


def flatten_stack(stack: np.ndarray) -> np.ndarray:
    # flatten to 1D row (SMOTE expects 2D array)
    return stack.ravel()


def unflatten_stack(flat: np.ndarray, shape: Tuple[int, int, int]):
    return flat.reshape(shape)


# dataset class for SMOTE-produced tensors
class SmoteStackDataset(Dataset):
    def __init__(self, stacks: np.ndarray, labels: np.ndarray, resize_fn=None):
        """
        stacks: numpy array shape (N, C, H, W) float32
        labels: numpy array shape (N,)
        """
        self.stacks = stacks.astype('float32')
        self.labels = labels.astype(int)
        self.resize_fn = resize_fn  # optional function to convert stack->torch tensor
        # default resize: normalize per channel to [0,1] and convert to torch.float32
        import torchvision.transforms as T
        from PIL import Image
        self.to_tensor = T.Compose(
            [T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])

    def __len__(self):
        return len(self.stacks)

    def __getitem__(self, idx):
        stack = self.stacks[idx]  # (C,H,W)
        # convert to uint8-like image per-channel normalization similar to dataset._stack_to_tensor
        chans, H, W = stack.shape
        stack_norm = np.zeros_like(stack, dtype=np.uint8)
        for c in range(chans):
            s = stack[c]
            m = s.mean()
            sd = s.std() if s.std() > 0 else 1.0
            sn = (s - m) / sd
            smin, smax = sn.min(), sn.max()
            if smax - smin > 0:
                ssc = (sn - smin) / (smax - smin)
            else:
                ssc = sn - smin
            stack_norm[c] = (ssc * 255.0).astype('uint8')
        img = np.transpose(stack_norm, (1, 2, 0)).astype('uint8')
        # use PIL + torchvision Resize + ToTensor
        tensor = self.to_tensor(img)  # float tensor in [0,1]
        return tensor.to(torch.float32), int(self.labels[idx])


# ---------- parse CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--train_xlsx", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--train_sheet", default="Training Baseline - Task 1")
    p.add_argument("--val_sheet", default="Validation Baseline - Task 1")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "lion", "radam", "adam"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out_dir", type=str, default="./smote_out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smote_k", type=int, default=5,
                   help="k_neighbors for SMOTE")
    p.add_argument("--smote_sampling_strategy", type=str, default="not majority",
                   help="SMOTE sampling_strategy (e.g. 'not majority' or dict or 'auto')")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------- main ----------
def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("SMOTE training script")
    cfg = vars(args).copy()
    logging.info(json.dumps({k: str(v) for k, v in cfg.items()}, indent=2))

    # read sheets (IDs + classes)
    import pandas as pd
    train_df = pd.read_excel(args.train_xlsx, sheet_name=args.train_sheet)
    val_df = pd.read_excel(args.train_xlsx, sheet_name=args.val_sheet)
    if 'ID' not in train_df.columns:
        raise RuntimeError("train sheet missing ID")
    if 'Class' not in train_df.columns:
        raise RuntimeError("train sheet missing Class")
    train_ids = train_df['ID'].astype(str).tolist()
    train_labels = train_df['Class'].fillna(-1).astype(int).tolist()
    val_ids = val_df['ID'].astype(str).tolist()
    val_labels = val_df['Class'].fillna(-1).astype(int).tolist()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    data_root = Path(args.data_dir) / 'training'

    # load stacks for training ids (cached or compute)
    stacks = []
    labels = []
    missing = []
    shape0 = None
    logging.info("Loading training stacks (cached or compute)...")
    for sid, lbl in tqdm(zip(train_ids, train_labels), total=len(train_ids)):
        st = None
        if cache_dir:
            st = load_cached_stack(
                cache_dir / args.train_sheet.replace(' ', '_'), sid)
        if st is None:
            # try find wav
            wav_path = None
            # try preferred folders first (same logic as dataset)
            priority = ['phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
                        'rhythmKA', 'rhythmPA', 'rhythmTA']
            for folder in priority:
                cand = data_root / folder / f"{sid}_{folder}.wav"
                if cand.exists():
                    wav_path = cand
                    break
                cand2 = data_root / folder / f"{sid}_{folder.lower()}.wav"
                if cand2.exists():
                    wav_path = cand2
                    break
            if wav_path is None:
                # fallback search
                for folder in data_root.iterdir():
                    if not folder.is_dir():
                        continue
                    for f in folder.iterdir():
                        if f.is_file() and f.name.startswith(f"{sid}_") and f.suffix.lower() == '.wav':
                            wav_path = f
                            break
                    if wav_path:
                        break
            if wav_path is None:
                missing.append(sid)
                continue
            st = compute_stack_from_wav(wav_path)
        stacks.append(st)
        labels.append(int(lbl))
        if shape0 is None:
            shape0 = st.shape  # (3, n_mels, T)

    if len(stacks) == 0:
        raise RuntimeError("No training stacks found!")

    stacks = np.stack(stacks, axis=0)  # (N, C, H, W)
    labels = np.array(labels, dtype=int)

    logging.info("Train label counts: %s", dict(Counter(labels)))

    # flatten for SMOTE
    N, C, H, W = stacks.shape
    X = stacks.reshape((N, C * H * W)).astype('float32')
    y = labels.copy()

    # apply SMOTE
    if SMOTE is None:
        raise RuntimeError(
            "imblearn not installed. pip install imbalanced-learn")
    logging.info("Running SMOTE (k=%d, sampling_strategy=%s)...",
                 args.smote_k, args.smote_sampling_strategy)
    sm = SMOTE(k_neighbors=args.smote_k,
               sampling_strategy=args.smote_sampling_strategy, random_state=args.seed)
    X_res, y_res = sm.fit_resample(X, y)
    logging.info("After SMOTE label counts: %s", dict(Counter(y_res)))
    # for reporting, print distribution percentages
    total_before = float(len(y))
    total_after = float(len(y_res))
    before_pct = {k: v / total_before * 100.0 for k, v in Counter(y).items()}
    after_pct = {k: v / total_after * 100.0 for k, v in Counter(y_res).items()}
    logging.info("Before pct: %s", {
                 k: f"{v:.2f}%" for k, v in before_pct.items()})
    logging.info("After pct: %s", {
                 k: f"{v:.2f}%" for k, v in after_pct.items()})

    # unflatten back to stacks
    stacks_res = X_res.reshape((-1, C, H, W)).astype('float32')
    labels_res = y_res.astype(int)

    # build datasets
    train_dataset = SmoteStackDataset(stacks_res, labels_res)
    # validation: load val stacks similarly (no smote)
    logging.info("Loading validation stacks...")
    val_stacks = []
    val_lbls = []
    missing_val = []
    for sid, lbl in tqdm(zip(val_ids, val_labels), total=len(val_ids)):
        st = None
        if cache_dir:
            st = load_cached_stack(
                cache_dir / args.val_sheet.replace(' ', '_'), sid) if cache_dir else None
        if st is None:
            # try compute from wav (same search)
            wav_path = None
            priority = ['phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
                        'rhythmKA', 'rhythmPA', 'rhythmTA']
            for folder in priority:
                cand = data_root / folder / f"{sid}_{folder}.wav"
                if cand.exists():
                    wav_path = cand
                    break
                cand2 = data_root / folder / f"{sid}_{folder.lower()}.wav"
                if cand2.exists():
                    wav_path = cand2
                    break
            if wav_path is None:
                for folder in data_root.iterdir():
                    if not folder.is_dir():
                        continue
                    for f in folder.iterdir():
                        if f.is_file() and f.name.startswith(f"{sid}_") and f.suffix.lower() == '.wav':
                            wav_path = f
                            break
                    if wav_path:
                        break
            if wav_path is None:
                missing_val.append(sid)
                continue
            st = compute_stack_from_wav(wav_path)
        val_stacks.append(st)
        val_lbls.append(int(lbl))
    if len(val_stacks) == 0:
        raise RuntimeError("No validation stacks found!")
    val_stacks = np.stack(val_stacks, axis=0)
    val_lbls = np.array(val_lbls, dtype=int)
    val_dataset = SmoteStackDataset(val_stacks, val_lbls)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=False)

    # model
    device = torch.device(args.device)
    model = build_vit(
        num_classes=5, pretrained=args.pretrained, device=args.device)
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model params total=%d, trainable=%d", total, trainable)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optim = get_optimizer_by_name(args.optimizer, [p for p in model.parameters(
    ) if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    best_f1 = -1.0
    best_path = None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device).long()
            optim.zero_grad()
            logits = model(xb)
            # labels are 1..5 -> convert to 0..4
            loss = criterion(logits, yb - 1)
            loss.backward()
            optim.step()
            bs = xb.size(0)
            running_loss += loss.item() * bs
            n += bs
        train_loss = running_loss / (n if n else 1.0)

        # validation
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="val", leave=False):
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy() + 1
                y_pred.extend(preds.tolist())
                y_true.extend(yb.numpy().tolist())
        avg_f1, per_class = averaged_f1_from_preds(
            y_true, y_pred, classes=[1, 2, 3, 4, 5])
        per_class_str = ";".join([f"{x:.4f}" for x in per_class])
        t = time.time() - t0
        logging.info(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_f1={avg_f1:.4f} | per_class=[{per_class_str}] | time={t:.1f}s")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_path = Path(args.out_dir) / "best_model.pt"
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict()}, best_path)

    logging.info("Training finished. best_f1=%f best_model=%s",
                 best_f1, str(best_path))
    print(f"Averaged validation F1 (best): {best_f1:.6f}")


if __name__ == "__main__":
    main()
