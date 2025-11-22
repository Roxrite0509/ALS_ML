# smote_utils.py
"""
Helpers to compute embeddings for each training subject and run SMOTE to generate synthetic embeddings.

Usage examples (from project root):
python -c "from smote_utils import build_embeddings; build_embeddings(...)"  # or call from train.py
python -c "from smote_utils import run_smote; run_smote(...)" 
"""

import os
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def compute_embeddings_for_dataset(model_backbone, dataloader, device='cpu', out_path: str = None):
    """
    Given a backbone (torch.nn.Module) that returns feature vectors for inputs,
    compute and return arrays: embeddings (N x D), labels (N,), ids (N,).
    `dataloader` yields (x, label, sid) same as your dataset.
    If out_path provided, saves npy files: embeddings.npy, labels.npy, sids.npy
    """
    model_backbone.eval()
    embeddings = []
    labels = []
    sids = []
    with torch.no_grad():
        for xb, yb, sid_batch in tqdm(dataloader, desc="Embeddings"):
            xb = xb.to(device)
            # model_backbone should return (N, D) embeddings
            emb = model_backbone(xb)
            if isinstance(emb, tuple):
                emb = emb[0]
            emb = emb.cpu().numpy()
            embeddings.append(emb)
            labels.extend(yb.numpy().tolist())
            sids.extend([str(s) for s in sid_batch])
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels, dtype=int)
    sids = np.array(sids, dtype=str)
    if out_path:
        out_dir = Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", embeddings)
        np.save(out_dir / "labels.npy", labels)
        np.save(out_dir / "sids.npy", sids)
        logger.info(f"Saved embeddings to {out_dir}")
    return embeddings, labels, sids


def run_smote(embeddings, labels, sampling_strategy='not_majority', k_neighbors=5, random_state=42):
    """
    Run SMOTE on embeddings+labels.
    Returns: (X_resampled, y_resampled), both numpy arrays.

    sampling_strategy:
      - float in (0,1]: proportion of minority to majority after resampling
      - 'auto', 'not_majority', 'all', 'minority', etc. (see imblearn docs)
    """
    sm = SMOTE(sampling_strategy=sampling_strategy,
               k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = sm.fit_resample(embeddings, labels)
    return X_res, y_res


def save_smote_results(out_dir, X_res, y_res):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "emb_smote.npy", X_res)
    np.save(out_dir / "labels_smote.npy", y_res)
    logging.info(f"Saved SMOTE outputs to {out_dir}")
    return out_dir


def reduce_dimensionality(X, n_components=256):
    """
    Optional: reduce embedding dimensionality before SMOTE (sometimes helps).
    Returns (X_reduced, pca_obj) where pca_obj can inverse_transform if needed.
    """
    pca = PCA(n_components=min(n_components, X.shape[1]))
    Xr = pca.fit_transform(X)
    return Xr, pca
