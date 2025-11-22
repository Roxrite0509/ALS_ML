# dataset.py
"""
Robust dataset module for SAND ViT baseline.

Drop-in replacement for your previous dataset.py. Exposes:
 - SandSubjectDataset(root_dir, xlsx_path, sheet_name, cache_dir=None, train_mode=False, allow_missing_wav=False)
 - find_wav_for_subject(root, sid, recording_priority=DEFAULT_PRIORITY)

Behavior:
 - By default missing WAVs raise FileNotFoundError so issues are visible.
 - Set allow_missing_wav=True to return a silent (zero) waveform instead of raising.
 - Supports cached numpy stacks in cache_dir/<sheet_name>/<ID>.npy (float32).
 - Returns tensors shaped (3, H, W) after resizing to OUT_SIZE (default 224x224).
"""

from pathlib import Path
import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from collections import Counter

# ----------------------------
# Config / constants
# ----------------------------
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

# ----------------------------
# Filename priority and helpers
# ----------------------------
DEFAULT_PRIORITY = [
    'phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
    'rhythmKA', 'rhythmPA', 'rhythmTA'
]


def _safe_load_wav(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Read WAV using soundfile, convert to mono float32 and resample if needed.
    Returns (waveform, sr).
    """
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    # convert integer to float32 in [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype('float32') / max(abs(info.min), info.max)
    else:
        data = data.astype('float32')
    if int(sr) != int(target_sr):
        try:
            data = librosa.resample(
                data, orig_sr=int(sr), target_sr=int(target_sr))
            sr = int(target_sr)
        except Exception:
            # fallback: simple numpy resize (not ideal)
            target_len = int(len(data) * (target_sr / sr))
            data = np.interp(
                np.linspace(0, len(data) - 1, target_len),
                np.arange(len(data)), data).astype('float32')
            sr = int(target_sr)
    return data, sr


def _ensure_length(wav: np.ndarray, target_length: int) -> np.ndarray:
    if len(wav) < target_length:
        pad_len = target_length - len(wav)
        return np.concatenate([wav, np.zeros(pad_len, dtype=wav.dtype)])
    elif len(wav) > target_length:
        return wav[:target_length]
    return wav


# ----------------------------
# Augmentation helpers
# ----------------------------
def add_noise(wav: np.ndarray, noise_level_db: float = -30.0) -> np.ndarray:
    wav = wav.astype('float32')
    rms = np.sqrt(np.mean(wav**2)) + 1e-9
    noise_rms = rms * (10 ** (noise_level_db / 20.0))
    noise = np.random.normal(0, noise_rms, size=wav.shape).astype('float32')
    return wav + noise


def random_time_stretch(wav: np.ndarray, low: float = 0.9, high: float = 1.1) -> np.ndarray:
    rate = float(np.random.uniform(low, high))
    try:
        return librosa.effects.time_stretch(wav.astype('float32'), rate)
    except Exception:
        # fallback: quick resample-like behavior
        try:
            target_len = int(len(wav) / max(1e-6, rate))
            resampled = librosa.resample(wav.astype(
                'float32'), orig_sr=DEFAULT_SAMPLE_RATE, target_sr=DEFAULT_SAMPLE_RATE)
            return resampled[:target_len]
        except Exception:
            return wav


def random_pitch_shift(wav: np.ndarray, sr: int, n_steps_low: float = -2.0, n_steps_high: float = 2.0) -> np.ndarray:
    n_steps = float(np.random.uniform(n_steps_low, n_steps_high))
    try:
        return librosa.effects.pitch_shift(wav.astype('float32'), sr=sr, n_steps=n_steps)
    except Exception:
        return wav


def polarity_invert(wav: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        return -wav
    return wav


def random_gain(wav: np.ndarray, min_db: float = -6.0, max_db: float = 6.0) -> np.ndarray:
    db = float(np.random.uniform(min_db, max_db))
    gain = 10 ** (db / 20.0)
    return (wav * gain).astype('float32')


def time_shift(wav: np.ndarray, max_shift_seconds: float = 0.2, sr: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    max_shift = int(max_shift_seconds * sr)
    if max_shift <= 0:
        return wav
    shift = np.random.randint(-max_shift, max_shift)
    if shift == 0:
        return wav
    return np.roll(wav, shift)


def bandpass_like_filter_in_spectrogram(S_db: np.ndarray, low_frac: float = 0.05, high_frac: float = 0.95) -> np.ndarray:
    n_mels, t = S_db.shape
    band = np.random.randint(max(1, int(0.05 * n_mels)),
                             max(2, int(0.3 * n_mels)))
    start = np.random.randint(0, max(1, n_mels - band))
    S_db_cp = S_db.copy()
    S_db_cp[start:start + band, :] *= np.random.uniform(0.0, 0.4)
    return S_db_cp


def random_augment(wav: np.ndarray, sr: int, strong: bool = False) -> np.ndarray:
    out = wav.astype('float32')

    # random gain
    if np.random.rand() < (0.9 if strong else 0.7):
        out = random_gain(out, min_db=(-10 if strong else -6),
                          max_db=(10 if strong else 6))

    # additive noise
    if np.random.rand() < (0.95 if strong else 0.75):
        level = np.random.uniform(-45, -
                                  10) if strong else np.random.uniform(-35, -18)
        out = add_noise(out, noise_level_db=level)

    # time stretch
    if np.random.rand() < (0.85 if strong else 0.6):
        low, high = (0.8, 1.25) if strong else (0.92, 1.08)
        out = random_time_stretch(out, low=low, high=high)

    # pitch shift
    if np.random.rand() < (0.75 if strong else 0.5):
        n_low, n_high = (-5, 5) if strong else (-2, 2)
        out = random_pitch_shift(
            out, sr, n_steps_low=n_low, n_steps_high=n_high)

    # time shift / roll
    if np.random.rand() < (0.6 if strong else 0.35):
        out = time_shift(out, max_shift_seconds=(
            0.35 if strong else 0.12), sr=sr)

    # polarity invert occasionally
    if np.random.rand() < (0.2 if strong else 0.05):
        out = polarity_invert(out)

    # ensure length
    out = _ensure_length(out, TARGET_LENGTH)

    return out.astype('float32')


def spec_augment(S: np.ndarray, num_masks: int = 3, freq_mask_param: int = 30, time_mask_param: int = 60) -> np.ndarray:
    S_aug = S.copy()
    n_mels, t = S_aug.shape
    for _ in range(num_masks):
        f = np.random.randint(0, min(freq_mask_param, n_mels) + 1)
        if f <= 0:
            continue
        f0 = np.random.randint(0, max(1, n_mels - f))
        S_aug[f0:f0 + f, :] = 0.0
    for _ in range(num_masks):
        tt = np.random.randint(0, min(time_mask_param, t) + 1)
        if tt <= 0:
            continue
        t0 = np.random.randint(0, max(1, t - tt))
        S_aug[:, t0:t0 + tt] = 0.0
    return S_aug


# ----------------------------
# WAV finding helper
# ----------------------------
def find_wav_for_subject(root: Path, sid: str, recording_priority: Optional[List[str]] = None) -> Optional[Path]:
    """
    Try a number of filename patterns to find a WAV for subject `sid` under `root`.
    Returns Path or None.
    Search strategy (in order):
      1) root/<priority_folder>/<SID>_<folder>.wav  and case variants
      2) root/<priority_folder>/<sid>_<folder>.wav (lowercase)
      3) root/<priority_folder>/<SID>.wav  (if user used simple names)
      4) recursive glob search for {sid}* .wav under root
    """
    root = Path(root)
    if recording_priority is None:
        recording_priority = DEFAULT_PRIORITY

    sid = sid.strip()
    candidates = []

    # 1) Priority-based exact patterns
    for folder in recording_priority:
        base = root / folder
        # a few common filename shapes
        candidates.extend([
            base / f"{sid}_{folder}.wav",
            base / f"{sid}_{folder.lower()}.wav",
            base / f"{sid}.wav",
            base / f"{sid.lower()}_{folder}.wav",
            base / f"{sid.replace('-', '_')}_{folder}.wav",
            base / f"{sid.replace('_', '-')}_{folder}.wav",
        ])

    # 2) check common alternate folders at top level
    for alt in ("phonationA", "phonationE", "phonationI", "phonationO", "phonationU",
                "rhythmKA", "rhythmPA", "rhythmTA"):
        base = root / alt
        candidates.append(base / f"{sid}.wav")
        candidates.append(base / f"{sid.lower()}.wav")

    # 3) try direct file in root / root/<somefolder>
    candidates.append(root / f"{sid}.wav")
    candidates.append(root / f"{sid.lower()}.wav")

    # 4) try all candidates and return first existing
    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c
        except Exception:
            continue

    # 5) recursive fallback: find a file starting with sid_ or sid anywhere under root
    try:
        pattern1 = f"**/{sid}_*.wav"
        for p in root.glob(pattern1):
            if p.is_file():
                return p
        pattern2 = f"**/{sid}*.wav"
        for p in root.glob(pattern2):
            if p.is_file():
                return p
        # as last resort, any file that contains the id substring and ends with wav
        for p in root.rglob("*.wav"):
            name = p.name.lower()
            if sid.lower() in name:
                return p
    except Exception:
        pass

    return None


# ----------------------------
# Dataset class
# ----------------------------
class SandSubjectDataset(Dataset):
    """
    Returns (tensor, label (int), sid (str))
    - root_dir: path with training/test subfolders (see find_wav_for_subject)
    - xlsx_path: path to excel with sheet containing columns 'ID' and optionally 'Class'
    - sheet_name: sheet to use
    - cache_dir: optional dir with precomputed numpy stacks named <SID>.npy
    - train_mode: whether to apply augmentations
    - allow_missing_wav: if True, missing WAVs are replaced by silence (zeros) instead of raising
    """

    def __init__(self, root_dir: str, xlsx_path: str, sheet_name: str,
                 cache_dir: Optional[str] = None, recording_priority: Optional[List[str]] = None,
                 sample_rate: int = DEFAULT_SAMPLE_RATE, duration: int = DEFAULT_DURATION_SEC,
                 train_mode: bool = False, allow_missing_wav: bool = False):
        self.root = Path(root_dir)
        self.xlsx_path = Path(xlsx_path)
        self.sheet_name = sheet_name
        self.sample_rate = int(sample_rate)
        self.duration = int(duration)
        self.target_length = int(self.sample_rate * self.duration)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.recording_priority = recording_priority or DEFAULT_PRIORITY
        self.train_mode = bool(train_mode)
        self.allow_missing_wav = bool(allow_missing_wav)

        if not self.xlsx_path.exists():
            raise FileNotFoundError(f"XLSX not found: {self.xlsx_path}")

        df = pd.read_excel(self.xlsx_path, sheet_name=self.sheet_name)
        if 'ID' not in df.columns:
            raise ValueError("Excel sheet must contain column 'ID'")
        self.df = df
        self.ids = df['ID'].astype(str).tolist()
        self.labels = None
        if 'Class' in df.columns:
            # ensure ints
            self.labels = df['Class'].fillna(-1).astype(int).tolist()

        if self.labels is not None:
            cnt = Counter(self.labels)
            self.class_freq = {int(k): int(v) for k, v in cnt.items()}
        else:
            self.class_freq = {}

        # cache subdir per sheet (avoid collisions)
        if self.cache_dir:
            self.cache_subdir = self.cache_dir / \
                self.sheet_name.replace(' ', '_')
        else:
            self.cache_subdir = None

    def __len__(self):
        return len(self.ids)

    def _load_cached_stack(self, sid: str) -> Optional[np.ndarray]:
        if self.cache_subdir is None:
            return None
        p = self.cache_subdir / f"{sid}.npy"
        if not p.exists():
            return None
        try:
            arr = np.load(str(p))
            if arr.dtype != np.float32:
                arr = arr.astype('float32')
            # expected shape (3, n_mels, T)
            if arr.ndim == 3:
                return arr
            # try to reshape if needed
            return arr.astype('float32')
        except Exception:
            return None

    def _stack_to_tensor(self, stack: np.ndarray) -> torch.Tensor:
        """
        stack: (3, H, W) float32 - mel, d1, d2 in log scale
        Normalize per-channel to 0..255, convert to PIL and transform to tensor.
        """
        chans, H, W = stack.shape
        # avoid division by zero
        stack_norm = np.zeros((chans, H, W), dtype=np.uint8)
        for c in range(chans):
            s = stack[c]
            m = float(np.mean(s))
            sd = float(np.std(s)) if float(np.std(s)) > 0 else 1.0
            sn = (s - m) / sd
            smin, smax = float(sn.min()), float(sn.max())
            if (smax - smin) > 0:
                ssc = (sn - smin) / (smax - smin)
            else:
                ssc = sn - smin
            # 0..255
            stack_norm[c] = (np.clip(ssc, 0.0, 1.0) * 255.0).astype('uint8')

        img = np.transpose(stack_norm, (1, 2, 0)).astype('uint8')  # H W C
        pil = Image.fromarray(img)
        tensor = resize(pil)  # returns float tensor 0..1
        return tensor

    def _wave_to_mel_and_deltas(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                           win_length=WIN_LENGTH, n_mels=N_MELS, power=2.0)
        S_db = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)
        d1 = librosa.feature.delta(S_db, order=1)
        d2 = librosa.feature.delta(S_db, order=2)
        # ensure shapes (n_mels, T)
        return np.stack([S_db.astype('float32'), d1.astype('float32'), d2.astype('float32')], axis=0)

    def __getitem__(self, idx):
        sid = str(self.ids[idx]).strip()
        label = int(self.labels[idx]) if self.labels is not None else -1

        # 1) try cached stack
        cached = self._load_cached_stack(sid)
        if cached is not None:
            stack = cached.copy()
            # optional stronger spec augment on cached stacks when training
            if self.train_mode and np.random.rand() < 0.95:
                mel = stack[0]
                mel = spec_augment(
                    mel, num_masks=3, freq_mask_param=30, time_mask_param=60)
                mel = bandpass_like_filter_in_spectrogram(mel)
                d1 = librosa.feature.delta(mel, order=1)
                d2 = librosa.feature.delta(mel, order=2)
                stack = np.stack([mel.astype('float32'), d1.astype(
                    'float32'), d2.astype('float32')], axis=0)
            tensor = self._stack_to_tensor(stack)
            return tensor.to(torch.float32), label, sid

        # 2) find WAV on disk
        wav_path = find_wav_for_subject(
            self.root, sid, recording_priority=self.recording_priority)
        if wav_path is None:
            msg = f"Could not find WAV for subject {sid} under {self.root}"
            if self.allow_missing_wav:
                warnings.warn(
                    msg + " — returning silence because allow_missing_wav=True")
                waveform = np.zeros(self.target_length, dtype='float32')
                sr = self.sample_rate
            else:
                raise FileNotFoundError(msg)
        else:
            waveform, sr = _safe_load_wav(wav_path, target_sr=self.sample_rate)
            waveform = _ensure_length(waveform, self.target_length)

        # 3) augment waveform if training
        if self.train_mode and len(waveform) > 0:
            freq = self.class_freq.get(label, 1)
            strong = freq <= 5
            p_aug = min(0.99, 0.5 + (1.5 / (freq + 1.0))
                        ) if strong else min(0.9, 0.15 + (1.0 / (freq + 1.0)))
            if np.random.rand() < p_aug:
                waveform = random_augment(waveform, sr, strong=strong)

        # 4) compute mel + deltas
        stack = self._wave_to_mel_and_deltas(waveform, sr)

        # 5) spectrogram augment (stronger for rare classes)
        if self.train_mode:
            prob_spec = 0.95 if (self.class_freq.get(label, 1) <= 5) else 0.85
            if np.random.rand() < prob_spec:
                mel = stack[0]
                mel = spec_augment(
                    mel, num_masks=3, freq_mask_param=30, time_mask_param=60)
                mel = bandpass_like_filter_in_spectrogram(mel)
                d1 = librosa.feature.delta(mel, order=1)
                d2 = librosa.feature.delta(mel, order=2)
                stack = np.stack([mel.astype('float32'), d1.astype(
                    'float32'), d2.astype('float32')], axis=0)

        tensor = self._stack_to_tensor(stack)
        return tensor.to(torch.float32), label, sid


# ----------------------------
# Utility: list missing ids (handy quick-check)
# ----------------------------
def find_missing_ids(root_dir: str, xlsx_path: str, sheet_name: str = "Training Baseline - Task 1",
                     recording_priority: Optional[List[str]] = None) -> List[str]:
    root = Path(root_dir)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if 'ID' not in df.columns:
        raise ValueError("Excel must contain ID column")
    ids = df['ID'].astype(str).tolist()
    missing = []
    for sid in ids:
        if find_wav_for_subject(root, sid, recording_priority=recording_priority) is None:
            missing.append(sid)
    return missing
