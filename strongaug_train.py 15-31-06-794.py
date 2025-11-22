#!/usr/bin/env python3
import argparse
import json
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import SandSubjectDataset, random_augment
from model import build_vit
from utils import averaged_f1_from_preds, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--train_xlsx", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--train_sheet", default="Training Baseline - Task 1")
    p.add_argument("--val_sheet", default="Validation Baseline - Task 1")

    p.add_argument("--out_dir", default="./strongaug_run")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="mps")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def strong_audio_aug(wav, sr):
    """Applies strong augmentations in audio domain."""
    wav = random_augment(wav, sr, strong=True)
    return wav


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    set_seed(args.seed)

    print(json.dumps(vars(args), indent=2))

    df_train = pd.read_excel(args.train_xlsx, sheet_name=args.train_sheet)
    df_val = pd.read_excel(args.train_xlsx, sheet_name=args.val_sheet)

    train_ids = df_train["ID"].astype(str).tolist()
    train_labels = df_train["Class"].astype(int).tolist()
    val_ids = df_val["ID"].astype(str).tolist()
    val_labels = df_val["Class"].astype(int).tolist()

    # compute sampling weights
    class_counts = {c: train_labels.count(c) for c in set(train_labels)}
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights, num_samples=len(train_labels) * 4, replacement=True)

    # dataset
    train_ds = SandSubjectDataset(
        root_dir=str(Path(args.data_dir) / "training"),
        xlsx_path=args.train_xlsx,
        sheet_name=args.train_sheet,
        cache_dir=args.cache_dir,
        train_mode=True,
        use_aug=True,
        strong_aug=True
    )

    val_ds = SandSubjectDataset(
        root_dir=str(Path(args.data_dir) / "training"),
        xlsx_path=args.train_xlsx,
        sheet_name=args.val_sheet,
        cache_dir=args.cache_dir,
        train_mode=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = build_vit(num_classes=5, pretrained=args.pretrained,
                      device=args.device).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs)

    best_f1 = -1
    best_path = Path(args.out_dir) / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n = 0, 0

        for xb, yb in tqdm(train_loader, desc=f"train {epoch}", leave=False):
            xb = xb.to(device)
            yb = (yb - 1).to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n += xb.size(0)

        scheduler.step()
        train_loss = running_loss / n

        # validation
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="val", leave=False):
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(1).cpu().numpy() + 1
                y_pred.extend(preds)
                y_true.extend(yb.numpy())

        f1, per_class = averaged_f1_from_preds(y_true, y_pred, [1, 2, 3, 4, 5])
        print(
            f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state": model.state_dict()}, best_path)
            print("Saved new best model.")

    print("Done. Best F1:", best_f1)
