# sampler.py
"""
Sampler utilities for SAND ViT baseline.

Provides:
 - BalancedBatchSampler(labels, batch_size, drop_last=False)
 - BalancedOversampler(labels) -> list of indices (oversampled)
 - build_weighted_sampler(labels) -> WeightedRandomSampler
 - find_best_global_threshold(y_true, y_probs, search=...)
 - create_swa(model) -> AveragedModel or None (if not available)

Notes:
 - Labels accepted as list/1D-array of ints (either 0..K-1 or 1..K).
 - BalancedBatchSampler yields lists of indices (compatible with DataLoader when used as sampler).
 - BalancedOversampler returns a list of indices (you can pass to DataLoader with sampler=SubsetRandomSampler(oversampled_list)).
"""

from typing import List, Iterable, Optional, Dict, Tuple
from collections import defaultdict, deque
import math
import random
import numpy as np

import torch
from torch.utils.data import Sampler, WeightedRandomSampler, SubsetRandomSampler

# sklearn only used in threshold search; imported locally in function to avoid hard dependency
# SWA utilities imported lazily in create_swa


def _normalize_labels(labels: Iterable[int]) -> np.ndarray:
    arr = np.array(labels, dtype=int)
    # if zero-based, convert to 1..K for internal maps
    if arr.min() == 0:
        return arr + 1
    return arr


class BalancedBatchSampler(Sampler):
    """
    Yield indices organized into batches where each batch attempts to contain an equal
    number of samples from each class.

    Usage:
        sampler = BalancedBatchSampler(labels, batch_size=16)
        loader = DataLoader(dataset, batch_size=None, sampler=sampler)

    Notes:
        - DataLoader must be created with batch_size=None when using a Sampler that
          yields full-batch index lists; otherwise DataLoader will re-chunk them.
        - The sampler will cycle through classes, drawing one sample from each class
          in round-robin until a batch is filled.
        - If some classes exhaust, they will be replenished by random choice (oversample).
        - This is a best-effort balanced batch sampler — for very skewed data it may repeat indices.
    """

    def __init__(self, labels: Iterable[int], batch_size: int = 16, drop_last: bool = False, seed: Optional[int] = None):
        self.labels = np.array(labels, dtype=int)
        if self.labels.min() == 0:
            self.labels = self.labels + 1
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        # map class->deque(indices)
        self.class_indices = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            self.class_indices[int(lab)].append(int(idx))
        # convert to deques for efficient pop/appendleft
        self.class_deques = {k: deque(v)
                             for k, v in self.class_indices.items()}
        self.classes = sorted(list(self.class_deques.keys()))
        if len(self.classes) == 0:
            raise ValueError("No classes found in labels")
        random.seed(self.seed)
        # shuffle each class deque
        for dq in self.class_deques.values():
            idxs = list(dq)
            random.shuffle(idxs)
            dq.clear()
            dq.extend(idxs)

        # determine approximate number of batches per epoch
        self.num_samples = len(self.labels)
        self.num_batches = math.ceil(self.num_samples / float(self.batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """
        Yields lists of indices (one list per batch). Use DataLoader with batch_size=None.
        """
        # make local copies of deques to avoid modifying original between epochs
        local_deques = {k: deque(v) for k, v in self.class_deques.items()}
        rng = random.Random(self.seed + 1)
        produced = 0
        while produced < self.num_batches:
            batch = []
            # round-robin through classes to fill a batch
            class_order = list(self.classes)
            rng.shuffle(class_order)
            ci = 0
            while len(batch) < self.batch_size:
                cls = class_order[ci % len(class_order)]
                ci += 1
                dq = local_deques[cls]
                if len(dq) == 0:
                    # refill by sampling with replacement from original indices for this class
                    orig = list(self.class_indices[cls])
                    if len(orig) == 0:
                        continue
                    pick = rng.choice(orig)
                else:
                    pick = dq.popleft()
                batch.append(int(pick))
            produced += 1
            # optionally drop last incomplete batch (we always produce fixed-size batches here)
            yield batch


def BalancedOversampler(labels: Iterable[int], target_count: Optional[int] = None, seed: Optional[int] = None) -> List[int]:
    """
    Return an oversampled index list that balances class frequencies by repeating minority samples.

    Args:
        labels: list/1D array of ints (0..K-1 or 1..K)
        target_count: if None, set to max_count * num_classes (classic oversample to match majority class)
                      if int, will generate that many total samples.
        seed: random seed for reproducibility

    Returns:
        indices: list of indices length == target_count (or computed value)
    """
    arr = _normalize_labels(labels)
    cls, counts = np.unique(arr, return_counts=True)
    class_to_indices = {int(c): np.where(arr == c)[0].tolist() for c in cls}
    max_count = int(counts.max())
    rng = random.Random(seed if seed is not None else 0)

    if target_count is None:
        # produce equalized dataset with size = max_count * num_classes
        target_count = max_count * len(cls)

    out = []
    cls_list = list(cls)
    # distribute target_count roughly equally across classes
    per_class = int(math.ceil(target_count / len(cls)))
    for c in cls_list:
        indices = class_to_indices[c]
        if len(indices) == 0:
            continue
        # repeat indices until per_class satisfied
        rep = []
        while len(rep) < per_class:
            rep.extend(rng.sample(indices, min(
                len(indices), per_class - len(rep))))
        out.extend(rep[:per_class])

    # trim or pad to exact target_count
    if len(out) > target_count:
        out = out[:target_count]
    elif len(out) < target_count:
        # pad by random choices
        choices = []
        for c in cls_list:
            choices.extend(class_to_indices[c])
        while len(out) < target_count:
            out.append(rng.choice(choices))
    # shuffle final indices
    rng.shuffle(out)
    return [int(i) for i in out]


def build_weighted_sampler(labels: Iterable[int]) -> Tuple[WeightedRandomSampler, torch.Tensor]:
    """
    Build a WeightedRandomSampler and class-weight tensor (0..K-1) for loss weighting.
    Returns (sampler, class_weights_tensor)

    Labels accepted as 0..K-1 or 1..K.
    """
    import torch
    arr = _normalize_labels(labels)  # 1..K
    cls, cnts = np.unique(arr, return_counts=True)
    inv = {int(c): 1.0 / int(cnts[i]) for i, c in enumerate(cls)}
    # sample weight per sample (1..K -> inv)
    sample_weights = np.array([inv[int(l)] for l in arr], dtype=float)
    sample_weights = sample_weights / float(sample_weights.mean())
    sampler = WeightedRandomSampler(weights=sample_weights.tolist(
    ), num_samples=len(sample_weights), replacement=True)
    # class weight tensor aligned to 0..K-1
    K = int(cls.max())
    cw = np.ones((K,), dtype=float)
    for c in cls:
        cw[c - 1] = inv[int(c)]
    cw = cw / float(cw.mean())
    return sampler, torch.tensor(cw, dtype=torch.float32)


def find_best_global_threshold(y_true: np.ndarray, y_probs: np.ndarray, search: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Find a single global threshold (not really applicable to single-label argmax; kept for compatibility).

    Behaviour:
      - For single-label multiclass classification, argmax is used; thresholding is only meaningful in multi-label.
      - We nonetheless implement a threshold search that does:
          for t in search: convert probs to argmax (unchanged) and compute macro-F1
      - Returns (best_threshold, best_macro_f1)

    Args:
        y_true: (N,) ints 1..K
        y_probs: (N, K) floats (softmax style)
        search: iterable of thresholds to try (defaults to np.linspace(0.2, 0.9, 15))

    Returns:
        best_threshold (float), best_f1 (float)
    """
    if search is None:
        search = np.linspace(0.2, 0.9, 15)
    try:
        from sklearn.metrics import f1_score
    except Exception:
        raise ImportError(
            "scikit-learn is required for find_best_global_threshold")

    best_t = float(search[0])
    best_f = -1.0
    # ensure arrays
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    if y_probs.ndim != 2:
        raise ValueError("y_probs must be shape (N, K)")

    for t in search:
        # For single-label multiclass we keep argmax behaviour
        preds = np.argmax(y_probs, axis=1) + 1
        f = f1_score(y_true, preds, average='macro', zero_division=0)
        if f > best_f:
            best_f = float(f)
            best_t = float(t)
    return best_t, best_f


def create_swa(model: torch.nn.Module):
    """
    Try to create SWA AveragedModel wrapper. Returns AveragedModel instance or None.
    Caller should handle import/availability.
    """
    try:
        from torch.optim.swa_utils import AveragedModel
        return AveragedModel(model)
    except Exception:
        return None


# Helpful small test / demo functions (not executed on import)
if __name__ == "__main__":  # pragma: no cover
    # simple smoke test
    labels = [1, 1, 2, 2, 2, 3, 3]
    print("BalancedOversampler len:", len(BalancedOversampler(labels)))
    sam, cw = build_weighted_sampler(labels)
    print("Weighted sampler created, class-weights:", cw)
    bbs = BalancedBatchSampler(labels, batch_size=4)
    print("One batch from BalancedBatchSampler (example):", next(iter(bbs)))
