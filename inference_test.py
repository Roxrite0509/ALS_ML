# inference_test.py
import argparse
from pathlib import Path
import torch
from dataset import SandSubjectDataset
from model import build_vit
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--xlsx', required=True)
    p.add_argument('--sheet', required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    device = torch.device(args.device)

    # load dataset
    ds = SandSubjectDataset(
        root_dir=str(Path(args.data_dir)/"training"),
        xlsx_path=args.xlsx,
        sheet_name=args.sheet
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    # load model
    model = build_vit(num_classes=5, pretrained=False, device=device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # run inference
    for i, (x, y, sid) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
            preds = torch.argmax(logits, dim=1) + 1
        print(list(zip(sid, preds.tolist())))
        if i == 3:  # just first few batches
            break


if __name__ == "__main__":
    main()
