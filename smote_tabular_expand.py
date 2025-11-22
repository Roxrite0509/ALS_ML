#!/usr/bin/env python3
"""
Generate SMOTE-synthesized training rows (NOT spectrograms).
Produces a new training excel with more samples for rare class.
"""

import argparse
import pandas as pd
from imblearn.over_sampling import SMOTE
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_xlsx", required=True)
    p.add_argument("--train_sheet", default="Training Baseline - Task 1")
    p.add_argument("--out_xlsx", default="./training_smote_expanded.xlsx")

    # How many synthetic samples?
    p.add_argument("--target_class", type=int, default=1)
    p.add_argument("--target_size", type=int, default=80,
                   help="final desired count for class 1")

    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_excel(args.train_xlsx, sheet_name=args.train_sheet)

    if "Class" not in df.columns or "ID" not in df.columns:
        raise RuntimeError("Sheet must have ID and Class columns")

    df["Class"] = df["Class"].astype(int)
    class1_df = df[df["Class"] == args.target_class]

    cur = len(class1_df)
    if cur >= args.target_size:
        print("Already enough class 1 samples.")
        return

    need = args.target_size - cur
    print(f"Generating {need} synthetic rows for class {args.target_class}")

    # Minimal SMOTE variable: only label and a noise column
    X = class1_df[["ID"]].copy()
    X["dummy"] = 0.0
    y = class1_df["Class"]

    sm = SMOTE(sampling_strategy={
               args.target_class: args.target_size}, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X, y)

    new_rows = X_res.iloc[cur:][["ID"]].copy()
    new_rows["Class"] = args.target_class

    # Generate new synthetic IDs
    new_rows["ID"] = [f"SYN_{i}" for i in range(len(new_rows))]

    df2 = pd.concat([df, new_rows], ignore_index=True)
    df2.to_excel(args.out_xlsx, index=False)

    print("Wrote:", args.out_xlsx)
    print(df2["Class"].value_counts())


if __name__ == "__main__":
    main()
