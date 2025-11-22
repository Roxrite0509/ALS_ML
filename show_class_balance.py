# show_class_balance.py
import numpy as np
import argparse
from collections import Counter
from pathlib import Path


def pct_counts(arr):
    c = Counter(arr)
    total = len(arr)
    out = {k: (v, 100.0 * v / total) for k, v in sorted(c.items())}
    return out


def print_counts(label_arr, title=""):
    print(f"\n-- {title} --")
    out = pct_counts(label_arr)
    for k, (v, pct) in out.items():
        print(f"class {k}: {v} ({pct:.2f}%)")
    print("total:", len(label_arr))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--emb_dir', required=True,
                   help='embedding cache dir (contains labels.npy and/or labels_smote.npy)')
    args = p.parse_args()
    d = Path(args.emb_dir)
    labs = None
    if (d / "labels.npy").exists():
        labs = np.load(d / "labels.npy")
        print_counts(labs, "Original training labels")
    if (d / "labels_smote.npy").exists():
        labs2 = np.load(d / "labels_smote.npy")
        print_counts(labs2, "After SMOTE labels")
    else:
        print("No labels_smote.npy found")


if __name__ == "__main__":
    main()
