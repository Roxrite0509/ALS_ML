# check_data.py
import argparse
from pathlib import Path
from dataset import SandSubjectDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--xlsx', required=True)
    p.add_argument('--sheet', default='Training Baseline')
    p.add_argument('--n', type=int, default=3)
    args = p.parse_args()

    ds = SandSubjectDataset(root_dir=str(Path(args.data_dir)/'training'),
                            xlsx_path=args.xlsx,
                            sheet_name=args.sheet)
    print(f"Dataset length: {len(ds)}")
    for i in range(min(args.n, len(ds))):
        x, y, sid = ds[i]
        print(f"{i}: ID={sid} label={y} tensor_shape={tuple(x.shape)}")


if __name__ == '__main__':
    main()
