#!/usr/bin/env bash
set -euo pipefail

# Change to the directory containing this script (assumes you drop it into sand_vit_baseline)
cd "$(dirname "$0")"

TS=$(date +%Y%m%d_%H%M%S)
files=("dataset.py" "model.py" "smote_train.py" "ensemble_inference.py")

echo "Backing up and attempting restore for: ${files[*]}"

# backup existing files
for f in "${files[@]}"; do
  if [ -f "$f" ]; then
    cp "$f" "${f}.bak.${TS}"
    echo "Backed up $f -> ${f}.bak.${TS}"
  fi
done

# If this is a git repo and files are tracked, try to restore from last commit
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Git repo detected — attempting git restore for tracked files..."
  for f in "${files[@]}"; do
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git checkout -- "$f" && echo "Restored $f from git HEAD"
    fi
  done
  echo "Git restore attempt finished."
fi

# For any file that is missing or still not the original, overwrite with known original content.
# We'll write the original contents (as provided earlier) into the files.

echo "Writing original contents to files (overwrites only if file differs from expected original)."

# dataset.py original (from earlier conversation)
cat > dataset.py <<'PYDATA'
# dataset.py
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import librosa
import random
from collections import Counter

DEFAULT_SAMPLE_RATE = 8000
DEFAULT_DURATION_SEC = 5
TARGET_LENGTH = DEFAULT_SAMPLE_RATE * DEFAULT_DURATION_SEC

N_MELS = 256
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = None
TOP_DB = 80.0

OUT_SIZE = (224, 224)

resize = transforms.Compose([
    transforms.Resize(OUT_SIZE),
    transforms.ToTensor()
])


def add_noise(wav, noise_level_db=-30):
    rms = np.sqrt(np.mean(wav**2)) + 1e-9
    noise_rms = rms * (10 ** (noise_level_db / 20.0))
    noise = np.random.normal(0, noise_rms, size=wav.shape).astype('float32')
    return wav + noise


def random_time_stretch(wav, low=0.9, high=1.1):
    rate = float(np.random.uniform(low, high))
    try:
        return librosa.effects.time_stretch(wav, rate)
    except Exception:
        new_len = int(len(wav) / max(1e-6, rate))
        return librosa.resample(wav, orig_sr=DEFAULT_SAMPLE_RATE, target_sr=DEFAULT_SAMPLE_RATE)[:new_len]


def random_pitch_shift(wav, sr, n_steps_low=-2, n_steps_high=2):
    n_steps = float(np.random.uniform(n_steps_low, n_steps_high))
    try:
        return librosa.effects.pitch_shift(wav, sr, n_steps)
    except Exception:
        return wav


def random_augment(wav, sr, strong=False):
    """
    strong=True produces more aggressive augmentations for very rare classes.
    """
    out = wav
    # noise
    if np.random.rand() < (0.9 if strong else 0.7):
        level = np.random.uniform(-45, -
                                  10) if strong else np.random.uniform(-35, -20)
        out = add_noise(out, noise_level_db=level)
    # time stretch
    if np.random.rand() < (0.7 if strong else 0.5):
        low, high = (0.8, 1.2) if strong else (0.92, 1.08)
        out = random_time_stretch(out, low=low, high=high)
    # pitch shift
    if np.random.rand() < (0.7 if strong else 0.5):
        n_low, n_high = (-4, 4) if strong else (-2, 2)
        out = random_pitch_shift(
            out, sr, n_steps_low=n_low, n_steps_high=n_high)
    # ensure length
    if len(out) < TARGET_LENGTH:
        out = np.concatenate(
            [out, np.zeros(TARGET_LENGTH - len(out), dtype=out.dtype)])
    elif len(out) > TARGET_LENGTH:
        out = out[:TARGET_LENGTH]
    return out.astype('float32')


def spec_augment(S, num_masks=2, freq_mask_param=20, time_mask_param=40):
    S_aug = S.copy()
    n_mels, t = S_aug.shape
    for _ in range(num_masks):
        f = np.random.randint(0, min(freq_mask_param, n_mels))
        f0 = np.random.randint(0, max(1, n_mels - f))
        S_aug[f0:f0+f, :] = 0.0
    for _ in range(num_masks):
        tt = np.random.randint(0, min(time_mask_param, t))
        t0 = np.random.randint(0, max(1, t - tt))
        S_aug[:, t0:t0+tt] = 0.0
    return S_aug


DEFAULT_PRIORITY = [
    'phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
    'rhythmKA', 'rhythmPA', 'rhythmTA'
]


def find_wav_for_subject(root: Path, sid: str, recording_priority=DEFAULT_PRIORITY):
    for folder in recording_priority:
        candidate = root / folder / f"{sid}_{folder}.wav"
        if candidate.exists():
            return candidate
        candidate2 = root / folder / f"{sid}_{folder.lower()}.wav"
        if candidate2.exists():
            return candidate2
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        for f in folder.iterdir():
            if f.is_file() and f.name.startswith(f"{sid}_") and f.suffix.lower() == '.wav':
                return f
    return None


class SandSubjectDataset(Dataset):
    def __init__(self, root_dir: str, xlsx_path: str, sheet_name: str,
                 cache_dir: str = None, recording_priority=None,
                 sample_rate=DEFAULT_SAMPLE_RATE, duration=DEFAULT_DURATION_SEC,
                 train_mode=False):
        self.root = Path(root_dir)
        self.xlsx_path = Path(xlsx_path)
        self.sheet_name = sheet_name
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = sample_rate * duration
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.recording_priority = recording_priority or DEFAULT_PRIORITY
        df = pd.read_excel(self.xlsx_path, sheet_name=self.sheet_name)
        if 'ID' not in df.columns:
            raise ValueError("Excel sheet must contain 'ID' column")
        self.df = df
        self.ids = df['ID'].astype(str).tolist()
        self.labels = None
        if 'Class' in df.columns:
            self.labels = df['Class'].fillna(-1).astype(int).tolist()
        self.train_mode = bool(train_mode)
        if self.labels is not None:
            cnt = Counter(self.labels)
            self.class_freq = {k: cnt.get(k, 0) for k in cnt}
        else:
            self.class_freq = {}
        if self.cache_dir:
            self.cache_subdir = self.cache_dir / \
                self.sheet_name.replace(' ', '_')
        else:
            self.cache_subdir = None

    def __len__(self):
        return len(self.ids)

    def _load_cached_stack(self, sid: str):
        if self.cache_subdir is None:
            return None
        p = self.cache_subdir / f"{sid}.npy"
        if not p.exists():
            return None
        try:
            arr = np.load(str(p))
            if arr.dtype != np.float32:
                arr = arr.astype('float32')
            return arr
        except Exception:
            return None

    def _load_wav(self, wav_path: Path):
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.sample_rate:
            data = librosa.resample(data.astype(
                'float32'), orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
        if len(data) < self.target_length:
            pad_len = self.target_length - len(data)
            data = np.concatenate([data, np.zeros(pad_len, dtype=data.dtype)])
        elif len(data) > self.target_length:
            data = data[:self.target_length]
        if data.dtype.kind == 'i':
            info = np.iinfo(data.dtype)
            data = data.astype('float32') / max(abs(info.min), info.max)
        else:
            data = data.astype('float32')
        return data, sr

    def _wave_to_mel_and_deltas(self, waveform: np.ndarray, sr: int):
        S = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH, n_mels=N_MELS, power=2.0
        )
        S_db = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)
        d1 = librosa.feature.delta(S_db, order=1)
        d2 = librosa.feature.delta(S_db, order=2)
        return np.stack([S_db, d1, d2], axis=0)

    def _stack_to_tensor(self, stack: np.ndarray):
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
        pil = Image.fromarray(img)
        tensor = resize(pil)
        return tensor

    def __getitem__(self, idx):
        sid = self.ids[idx]
        label = int(self.labels[idx]) if self.labels is not None else -1
        cached = self._load_cached_stack(sid)
        if cached is not None:
            stack = cached
            if self.train_mode and np.random.rand() < 0.9:
                mel = stack[0]
                mel = spec_augment(
                    mel, num_masks=2, freq_mask_param=20, time_mask_param=40)
                d1 = librosa.feature.delta(mel, order=1)
                d2 = librosa.feature.delta(mel, order=2)
                stack = np.stack([mel, d1, d2], axis=0)
            tensor = self._stack_to_tensor(stack)
            return tensor.to(torch.float32), label, sid
        wav_path = find_wav_for_subject(
            self.root, sid, recording_priority=self.recording_priority)
        if wav_path is None:
            raise FileNotFoundError(
                f"Could not find WAV for subject {sid} under {self.root}")
        waveform, sr = self._load_wav(wav_path)
        if self.train_mode:
            freq = self.class_freq.get(label, 1)
            # aggressive augmentation for very rare classes
            if freq <= 5:
                p_aug = min(0.99, 0.4 + (2.0 / (freq + 1.0)))
                strong = True
            else:
                p_aug = min(0.9, 0.1 + (1.0 / (freq + 1.0)))
                strong = False
            if np.random.rand() < p_aug:
                waveform = random_augment(waveform, sr, strong=strong)
        stack = self._wave_to_mel_and_deltas(waveform, sr)
        if self.train_mode and np.random.rand() < (0.9 if (self.class_freq.get(label, 1) <= 5) else 0.8):
            mel = stack[0]
            mel = spec_augment(
                mel, num_masks=2, freq_mask_param=20, time_mask_param=40)
            d1 = librosa.feature.delta(mel, order=1)
            d2 = librosa.feature.delta(mel, order=2)
            stack = np.stack([mel, d1, d2], axis=0)
        tensor = self._stack_to_tensor(stack)
        return tensor.to(torch.float32), label, sid
PYDATA

# model.py original
cat > model.py <<'PYMODEL'
# model.py
from typing import Optional
import torch
import torch.nn as nn

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except Exception:
    vit_b_16 = None
    ViT_B_16_Weights = None


def _find_transformer_blocks(model):
    candidates = []
    if hasattr(model, "encoder"):
        enc = getattr(model, "encoder")
        if hasattr(enc, "layers"):
            candidates.append((enc, enc.layers))
        if hasattr(enc, "blocks"):
            candidates.append((enc, enc.blocks))
        if hasattr(enc, "encoder"):
            enc2 = getattr(enc, "encoder")
            if hasattr(enc2, "layers"):
                candidates.append((enc2, enc2.layers))
            if hasattr(enc2, "blocks"):
                candidates.append((enc2, enc2.blocks))
    if hasattr(model, "transformer"):
        tr = getattr(model, "transformer")
        if hasattr(tr, "encoder"):
            enc = getattr(tr, "encoder")
            if hasattr(enc, "layers"):
                candidates.append((enc, enc.layers))
            if hasattr(enc, "blocks"):
                candidates.append((enc, enc.blocks))
    if len(candidates) == 0:
        return None, None
    return candidates[0]


def _get_head_in_features(head_module: nn.Module) -> Optional[int]:
    if isinstance(head_module, nn.Linear):
        return head_module.in_features
    if hasattr(head_module, "in_features"):
        try:
            return int(getattr(head_module, "in_features"))
        except Exception:
            pass
    for c in reversed(list(head_module.children())):
        if isinstance(c, nn.Linear):
            return c.in_features
        for sc in reversed(list(c.children())):
            if isinstance(sc, nn.Linear):
                return sc.in_features
    return None


def _replace_head_with_linear(model: nn.Module, num_classes: int, head_dropout: float = 0.0):
    if hasattr(model, "heads"):
        head = model.heads
        in_features = _get_head_in_features(head)
        if in_features is None:
            if hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
                in_features = model.classifier.in_features
        if in_features is None:
            raise RuntimeError(
                "Cannot determine classifier in_features for ViT model.")
        if head_dropout is not None and head_dropout > 0.0:
            new_head = nn.Sequential(nn.Dropout(
                head_dropout), nn.Linear(in_features, num_classes))
        else:
            new_head = nn.Linear(in_features, num_classes)
        model.heads = new_head
        return model
    if hasattr(model, "classifier"):
        clf = model.classifier
        in_features = getattr(clf, "in_features", None)
        if in_features is not None:
            if head_dropout is not None and head_dropout > 0.0:
                new_head = nn.Sequential(nn.Dropout(
                    head_dropout), nn.Linear(in_features, num_classes))
            else:
                new_head = nn.Linear(in_features, num_classes)
            model.classifier = new_head
            return model
    default_feat = 768
    if head_dropout is not None and head_dropout > 0.0:
        model.heads = nn.Sequential(nn.Dropout(
            head_dropout), nn.Linear(default_feat, num_classes))
    else:
        model.heads = nn.Linear(default_feat, num_classes)
    return model


def freeze_backbone(model: nn.Module, freeze: bool = True, exclude_head: bool = True):
    head_params = set()
    if exclude_head and hasattr(model, "heads"):
        for p in model.heads.parameters():
            head_params.add(p)
    if exclude_head and hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            head_params.add(p)
    for p in model.parameters():
        if exclude_head and p in head_params:
            p.requires_grad = True
        else:
            p.requires_grad = not freeze
    return model


def unfreeze_last_transformer_blocks(model: nn.Module, k: int = 1):
    enc_parent, blocks = _find_transformer_blocks(model)
    if blocks is None:
        return model
    total = len(blocks)
    k = min(max(int(k), 0), total)
    for p in model.parameters():
        p.requires_grad = False
    for i in range(total - k, total):
        for p in blocks[i].parameters():
            p.requires_grad = True
    if hasattr(model, "heads"):
        for p in model.heads.parameters():
            p.requires_grad = True
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    return model


def build_vit(num_classes: int = 5, pretrained: bool = False, device: str = "cpu", head_dropout: float = 0.0, freeze_backbone_flag: bool = False):
    if vit_b_16 is None:
        raise ImportError("torchvision ViT (vit_b_16) not available.")
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT if ViT_B_16_Weights is not None else None
        model = vit_b_16(weights=weights)
    else:
        model = vit_b_16(weights=None)
    model = _replace_head_with_linear(
        model, num_classes, head_dropout=head_dropout)
    if freeze_backbone_flag:
        model = freeze_backbone(model, freeze=True, exclude_head=True)
    return model.to(torch.device(device))
PYMODEL

# smote_train.py original (from earlier)
cat > smote_train.py <<'PYSM'
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
PYSM

# ensemble_inference.py original
cat > ensemble_inference.py <<'PYENS'
# ensemble_inference.py
"""
Ensemble + TTA inference for SAND task1 (fixed).
Usage example:
python ensemble_inference.py \
  --data_dir /Users/pranav/Desktop/Task1 \
  --test_xlsx /Users/pranav/Desktop/Task1/test/sand_task1_test.xlsx \
  --model_dirs ./outputs_vit_onecycle_focal ./outputs_vit_ft_unfreeze8 \
  --cache_dir /Users/pranav/Desktop/Task1/cache_logmel \
  --ttas 5 \
  --device mps \
  --out_csv ./ensemble/submission.csv
"""
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import tqdm
from dataset import SandSubjectDataset, find_wav_for_subject, random_augment
from model import build_vit
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--test_xlsx', type=str, required=True)
    p.add_argument('--model_dirs', nargs='+', required=True,
                   help='Paths with best_model.pt inside')
    p.add_argument('--cache_dir', type=str, default=None)
    p.add_argument('--ttas', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--out_csv', type=str, default='./submission.csv')
    return p.parse_args()


def load_models(model_dirs, device):
    models = []
    for d in model_dirs:
        p = Path(d)
        ck = p / "best_model.pt"
        if not ck.exists():
            raise FileNotFoundError(f"No best_model.pt in {p}")
        # build model architecture (no pretrained weights needed)
        m = build_vit(num_classes=5, pretrained=False, device=device)
        state = torch.load(str(ck), map_location=device)
        # handle different checkpoint formats
        if isinstance(state, dict) and 'model_state' in state:
            st = state['model_state']
        else:
            st = state
        m.load_state_dict(st)
        m.eval()
        models.append(m)
    return models


def make_tta_stack(wav_path: Path, ds_inst: SandSubjectDataset, tta: int):
    """
    Returns a tensor of shape (TTA, C, H, W) on CPU.
    Uses dataset methods to compute mel+deltas and convert to tensor.
    TTA[0] is clean (no augment); others are mild random_augment variants.
    """
    wav, sr = ds_inst._load_wav(wav_path)
    stacks = []
    # clean
    stack_clean = ds_inst._wave_to_mel_and_deltas(wav, sr)
    stacks.append(ds_inst._stack_to_tensor(stack_clean).unsqueeze(0))
    # TTA augmentations (mild)
    for i in range(tta - 1):
        try:
            wav_aug = random_augment(wav.copy(), sr, strong=False)
        except Exception:
            wav_aug = wav
        stack = ds_inst._wave_to_mel_and_deltas(wav_aug, sr)
        stacks.append(ds_inst._stack_to_tensor(stack).unsqueeze(0))
    return torch.cat(stacks, dim=0)  # shape (TTA, C, H, W)


def main():
    args = parse_args()
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    test_xlsx = Path(args.test_xlsx)
    if not test_xlsx.exists():
        raise FileNotFoundError(f"Test xlsx not found: {test_xlsx}")
    df_test = pd.read_excel(str(test_xlsx))
    if 'ID' not in df_test.columns:
        raise ValueError("Test xlsx must contain column 'ID'")
    ids = df_test['ID'].astype(str).tolist()

    # dataset instance for helper methods (point root to test folder)
    test_root = data_dir / 'test'
    # We don't need sheet_name for test stack helper, but SandSubjectDataset expects one.
    # Create a minimal dummy dataset object by passing the test xlsx and first sheet name.
    sheet_name = pd.ExcelFile(str(test_xlsx)).sheet_names[0]
    ds_inst = SandSubjectDataset(root_dir=str(test_root), xlsx_path=str(test_xlsx),
                                 sheet_name=sheet_name, cache_dir=args.cache_dir, train_mode=False)

    # load models
    models = load_models(args.model_dirs, device)
    print(f"Loaded {len(models)} models.")

    preds = []
    for sid in tqdm.tqdm(ids, desc="Inferring"):
        wav_path = find_wav_for_subject(test_root, sid)
        if wav_path is None:
            # if not found in subfolders, also try direct file in test root
            cand = test_root / f"{sid}.wav"
            wav_path = cand if cand.exists() else None
        if wav_path is None:
            print(f"Missing WAV for {sid}, skipping.")
            preds.append((sid, -1))
            continue

        # build TTA stacks (on CPU)
        stack_batch = make_tta_stack(
            wav_path, ds_inst, args.ttas)  # (TTA, C, H, W)
        stack_batch = stack_batch.to(device)

        # accumulate averaged logits across models
        logits_sum = None
        for m in models:
            with torch.no_grad():
                out = m(stack_batch)  # (TTA, num_classes)
                # average across TTA dimension
                out_mean = out.mean(dim=0).cpu().numpy()  # (num_classes,)
                if logits_sum is None:
                    logits_sum = out_mean
                else:
                    logits_sum += out_mean
        logits_avg = logits_sum / len(models)
        pred_class = int(np.argmax(logits_avg) + 1)
        preds.append((sid, pred_class))

    # write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(preds, columns=['ID', 'Class'])
    df_out.to_csv(str(out_csv), index=False)
    print("Wrote:", out_csv)
    print(df_out['Class'].value_counts())


if __name__ == '__main__':
    main()
PYENS

echo "Restore finished. Please check the .bak files for safety. If you used git, remember to git status/diff to confirm."
echo "Done."
