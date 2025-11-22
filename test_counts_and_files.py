# test_counts_and_files.py
import argparse
from pathlib import Path
import pandas as pd


def find_sheet_by_keyword(xlsx_path: Path, keyword: str):
    sheets = pd.ExcelFile(xlsx_path).sheet_names
    for s in sheets:
        if keyword.lower() in s.lower():
            return s
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task_dir', required=True)
    p.add_argument('--train_sheet', default=None,
                   help='(optional) exact training sheet name')
    p.add_argument('--val_sheet', default=None,
                   help='(optional) exact validation sheet name')
    args = p.parse_args()

    task = Path(args.task_dir)
    train_xlsx = task/'training'/'sand_task_1.xlsx'
    test_xlsx = task/'test'/'sand_task1_test.xlsx'
    print("Checking:", train_xlsx, test_xlsx)

    if not train_xlsx.exists():
        raise FileNotFoundError(f"Training xlsx not found: {train_xlsx}")
    if not test_xlsx.exists():
        raise FileNotFoundError(f"Test xlsx not found: {test_xlsx}")

    # detect sheet names if not provided
    train_sheet_name = args.train_sheet or find_sheet_by_keyword(
        train_xlsx, 'train')
    val_sheet_name = args.val_sheet or find_sheet_by_keyword(
        train_xlsx, 'valid')
    sand_sheet_name = None
    # try to find the full SAND sheet (could contain 'SAND - TRAINING' or similar)
    for s in pd.ExcelFile(train_xlsx).sheet_names:
        if 'sand' in s.lower() and 'train' in s.lower():
            sand_sheet_name = s
            break
    if sand_sheet_name is None:
        # fallback to first sheet
        sand_sheet_name = pd.ExcelFile(train_xlsx).sheet_names[0]

    print("Detected sheets:")
    print("  SAND sheet:", sand_sheet_name)
    print("  Train sheet:", train_sheet_name)
    print("  Val sheet:  ", val_sheet_name)

    train_df = pd.read_excel(train_xlsx, sheet_name=sand_sheet_name)
    tr_base = pd.read_excel(
        train_xlsx, sheet_name=train_sheet_name) if train_sheet_name else pd.DataFrame()
    val_base = pd.read_excel(
        train_xlsx, sheet_name=val_sheet_name) if val_sheet_name else pd.DataFrame()
    test_df = pd.read_excel(test_xlsx)

    print("Counts - SAND training total:", len(train_df))
    print("Training Baseline:", len(tr_base))
    print("Validation Baseline:", len(val_base))
    print("Test file rows:", len(test_df))

    # ensure disjointness if baseline sheets found
    if not tr_base.empty and not val_base.empty:
        inter = set(tr_base['ID']).intersection(set(val_base['ID']))
        print("Intersection train/val (should be 0):", len(inter))
    else:
        print("Training Baseline or Validation Baseline sheet not detected; skipping intersection check.")

    # Check that IDs in Training Baseline have at least one wav
    missing = []
    if not tr_base.empty:
        for sid in tr_base['ID'].astype(str):
            found = False
            for folder in (task/'training').iterdir():
                if not folder.is_dir():
                    continue
                for f in folder.iterdir():
                    if f.is_file() and f.name.startswith(f"{sid}_"):
                        found = True
                        break
                if found:
                    break
            if not found:
                missing.append(sid)
    print("Missing WAVs for Training Baseline IDs (count):", len(missing))
    if missing:
        print("Example missing IDs:", missing[:10])


if __name__ == '__main__':
    main()
