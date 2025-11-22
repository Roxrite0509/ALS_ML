# smoke_run.py
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import SandSubjectDataset
from model import build_vit
import torch.nn as nn
import torch.optim as optim


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--xlsx', required=True)
    p.add_argument('--train_sheet', default='Training Baseline')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    ds = SandSubjectDataset(root_dir=str(Path(args.data_dir)/'training'),
                            xlsx_path=args.xlsx,
                            sheet_name=args.train_sheet)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    device = torch.device(args.device)
    model = build_vit(num_classes=5, pretrained=False, device=device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    xb, yb, sids = next(iter(loader))
    xb = xb.to(device)
    yb = yb.to(device).long()
    print("Batch shapes:", xb.shape, yb.shape)
    logits = model(xb)
    print("Logits shape:", logits.shape)
    loss = criterion(logits, yb - 1)
    print("Loss:", loss.item())
    loss.backward()
    opt.step()
    preds = logits.argmax(dim=1).cpu().tolist()
    print("Preds:", [p+1 for p in preds])


if __name__ == '__main__':
    from pathlib import Path
    main()
