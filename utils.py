# utils.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import torch
from collections import defaultdict

def read_ids_from_excel(xlsx_path: str, sheet_name: str) -> List[str]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if 'ID' not in df.columns:
        raise ValueError("Sheet must contain 'ID' column")
    return df['ID'].astype(str).tolist(), df

def averaged_f1_from_preds(y_true: list, y_pred: list, classes: list):
    """
    Compute Averaged F1 per the formula in the brief:
    For each class c compute: TP_c / (TP_c + 1/2*(FP_c + FN_c))
    Then average over |C|
    y_true and y_pred are lists or 1D arrays of ints (1..5)
    classes is list of classes e.g. [1,2,3,4,5]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = []
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        denom = tp + 0.5 * (fp + fn)
        if denom == 0:
            # If no instances of this class in true and predicted, define score as 0 (or handle as needed)
            score = 0.0
        else:
            score = tp / denom
        scores.append(score)
    return float(np.mean(scores)), scores

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
