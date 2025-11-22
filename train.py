#!/usr/bin/env python3
# train.py (droppable - robust, full-featured replacement)

"""
Robust training script for SAND ViT baseline (drop-in replacement).

Features:
- Two-stage training (head-only then unfreeze last K ViT blocks)
- LLRD (layer-wise lr decay) support
- SWA support
- MixUp, focal/CE loss, class weights, balanced sampler support
- Robust DataLoader + collate to handle various sampler/dataset batch formats
- Works on mps/cuda/cpu (auto-detect)
"""

from pathlib import Path
import argparse
import json
import time
import gc
import traceback
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# Optional sampler helpers if present
try:
    from sampler import build_weighted_sampler, BalancedBatchSampler, BalancedOversampler, find_best_global_threshold
except Exception:
    BalancedBatchSampler = None
    BalancedOversampler = None
    find_best_global_threshold = None

# Local modules: dataset and model (must exist)
try:
    from dataset import SandSubjectDataset
except Exception as e:
    raise ImportError(
        "Could not import SandSubjectDataset from dataset.py: " + str(e))

try:
    from model import build_vit, unfreeze_last_transformer_blocks, freeze_backbone
except Exception as e:
    raise ImportError(
        "Could not import required functions from model.py: " + str(e))

# fallback threshold finder
if find_best_global_threshold is None:
    def find_best_global_threshold(y_true, y_probs, search=np.linspace(0.2, 0.9, 15)):
        from sklearn.metrics import f1_score
        preds = (np.argmax(y_probs, axis=1) + 1).astype(int)
        best_t, best_f = 0.5, -1.0
        for t in search:
            f = f1_score(y_true, preds, average="macro", zero_division=0)
            if f > best_f:
                best_f = float(f)
                best_t = float(t)
        return best_t, best_f

# ----------------------------
# Helpers
# ----------------------------


def set_seed(s):
    import random
    random.seed(int(s))
    np.random.seed(int(s))
    torch.manual_seed(int(s))
    try:
        torch.cuda.manual_seed_all(int(s))
    except Exception:
        pass


def save_json(pth: Path, obj):
    with open(str(pth), "w") as f:
        json.dump(obj, f, indent=2)


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def focal_loss_logits(logits, targets, gamma=1.0, weight=None):
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


def get_vit_block_count_from_named_params(named_params):
    max_idx = -1
    for n, _ in named_params:
        if "blocks." in n:
            try:
                part = n.split("blocks.")[1]
                idx = int(part.split(".")[0])
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                pass
    return max_idx + 1 if max_idx >= 0 else 0


def build_optimizer_with_llrd(model, base_lr, weight_decay, use_llrd=False, llrd_decay=0.9, opt_name="adamw"):
    named = list(model.named_parameters())
    if not use_llrd:
        params = [p for n, p in named if p.requires_grad]
        opt_name = (opt_name or "adamw").lower()
        if opt_name == "adam":
            return optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
        if opt_name == "radam":
            try:
                return optim.RAdam(params, lr=base_lr, weight_decay=weight_decay)
            except Exception:
                return optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
        if opt_name == "lion":
            try:
                import torch_optimizer as topt
                return topt.Lion(params, lr=base_lr, weight_decay=weight_decay)
            except Exception:
                return optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
        return optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    # LLRD grouping
    n_blocks = get_vit_block_count_from_named_params(named)
    layer_ids = {}
    for n, p in named:
        if "head" in n or "classifier" in n or "fc" in n:
            lid = n_blocks + 2
        elif "blocks." in n:
            try:
                idx = int(n.split("blocks.")[1].split(".")[0])
                lid = idx + 1
            except Exception:
                lid = 1
        elif "patch_embed" in n or "pos_embed" in n:
            lid = 0
        else:
            lid = 0
        layer_ids[n] = lid

    max_lid = max(layer_ids.values()) if len(layer_ids) else 0
    groups = {}
    for n, p in named:
        if not p.requires_grad:
            continue
        lid = layer_ids.get(n, 0)
        exponent = max_lid - lid
        scale = (llrd_decay ** exponent) if max_lid > 0 else 1.0
        lr = float(base_lr * scale)
        key = f"{lr:.12f}"
        if key not in groups:
            groups[key] = {"params": [], "lr": lr}
        groups[key]["params"].append(p)

    param_groups = list(groups.values())
    opt_name = (opt_name or "adamw").lower()
    if opt_name == "adam":
        return optim.Adam(param_groups, weight_decay=weight_decay)
    if opt_name == "radam":
        try:
            return optim.RAdam(param_groups, weight_decay=weight_decay)
        except Exception:
            return optim.AdamW(param_groups, weight_decay=weight_decay)
    if opt_name == "lion":
        try:
            import torch_optimizer as topt
            return topt.Lion(param_groups, weight_decay=weight_decay)
        except Exception:
            return optim.AdamW(param_groups, weight_decay=weight_decay)
    return optim.AdamW(param_groups, weight_decay=weight_decay)


def build_sampler_and_class_weights(labels, batch_size: int = 8):
    labels = np.array(labels, dtype=int)
    if labels.min() == 0:
        labels_1 = labels + 1
    else:
        labels_1 = labels
    cls, cnts = np.unique(labels_1, return_counts=True)
    inv = {int(c): 1.0 / int(cnts[i]) for i, c in enumerate(cls)}
    K = int(cls.max())
    cw = np.ones((K,), dtype=float)
    for c in cls:
        cw[c - 1] = inv.get(int(c), 1.0)
    cw = cw / float(cw.mean())

    sampler = None
    try:
        import sampler as custom_sampler_module  # noqa
        if hasattr(custom_sampler_module, "BalancedBatchSampler"):
            try:
                sampler = custom_sampler_module.BalancedBatchSampler(
                    labels_1.tolist(), batch_size=batch_size)
            except Exception:
                sampler = None
        elif hasattr(custom_sampler_module, "BalancedOversampler"):
            try:
                overs = custom_sampler_module.BalancedOversampler(
                    labels_1.tolist())
                sampler = overs.sampler if hasattr(overs, "sampler") else overs
            except Exception:
                sampler = None
    except Exception:
        sampler = None

    if sampler is None:
        sample_weights = np.array([inv[int(l)] for l in labels_1], dtype=float)
        sample_weights = sample_weights / float(sample_weights.mean())
        sampler = WeightedRandomSampler(weights=sample_weights.tolist(
        ), num_samples=len(sample_weights), replacement=True)
    return sampler, torch.tensor(cw, dtype=torch.float32)

# ----------------------------
# CLI
# ----------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--train_xlsx", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--train_sheet", default="Training Baseline - Task 1")
    p.add_argument("--val_sheet", default="Validation Baseline - Task 1")

    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--pretrained_weights_path", default=None)
    p.add_argument("--head_dropout", type=float, default=0.4)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--stage1_epochs", type=int, default=3)
    p.add_argument("--stage2_epochs", type=int, default=50)
    p.add_argument("--unfreeze_k", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_ft", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument(
        "--optimizer", choices=["adamw", "adam", "radam", "lion"], default="adamw")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="./run_out")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--l1_lambda", type=float, default=0.0)
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--use_sampler", action="store_true")
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--loss", choices=["ce", "focal"], default="ce")
    p.add_argument("--gamma", type=float, default=2.0)

    p.add_argument("--use_llrd", action="store_true")
    p.add_argument("--llrd_decay", type=float, default=0.9)
    p.add_argument("--use_swa", action="store_true")
    p.add_argument("--swa_start", type=int, default=30)
    p.add_argument("--swa_lr", type=float, default=1e-5)

    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_val_probs", action="store_true")
    p.add_argument("--imagenet_norm", action="store_true")
    return p.parse_args()

# ----------------------------
# Robust collate and loader creation
# ----------------------------


def robust_collate(batch):
    """
    Normalize many batch shapes into (xb, yb, sid)
    Acceptable batch elements:
     - Tensor (xb only) -> returns dummy labels -1 and sid None
     - (xb, yb) or (xb, yb, sid)
     - pre-collated tensors
    """
    # direct tensor case: pre-collated xb-only
    if isinstance(batch, torch.Tensor):
        xb = batch
        B = xb.shape[0]
        yb = torch.full((B,), -1, dtype=torch.long)
        sid = [None] * B
        return xb, yb, sid

    # normal list/tuple-of-elements
    if isinstance(batch, (list, tuple)):
        # sometimes DataLoader wraps the real batch inside a single-element list
        if len(batch) == 1 and isinstance(batch[0], (list, tuple, torch.Tensor)):
            return robust_collate(batch[0])

        xb_list = []
        y_list = []
        sid_list = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                xb_list.append(item)
                y_list.append(-1)
                sid_list.append(None)
                continue
            if isinstance(item, (list, tuple)):
                if len(item) == 3:
                    xi, yi, sidi = item
                elif len(item) == 2:
                    xi, yi = item
                    sidi = None
                else:
                    xi = item[0]
                    yi = item[1] if len(item) > 1 else -1
                    sidi = item[2] if len(item) > 2 else None
                xb_list.append(xi)
                try:
                    y_list.append(int(yi))
                except Exception:
                    try:
                        y_list.append(int(yi.item()))
                    except Exception:
                        y_list.append(-1)
                sid_list.append(str(sidi) if sidi is not None else None)
                continue
            raise RuntimeError(
                f"robust_collate: unsupported batch item type {type(item)}")

        # stack xb_list into tensor
        try:
            xb = torch.stack(xb_list, dim=0)
        except Exception:
            xb = torch.stack([torch.as_tensor(x) for x in xb_list], dim=0)
        yb = torch.tensor(y_list, dtype=torch.long)
        return xb, yb, sid_list

    raise RuntimeError(
        "robust_collate: unexpected batch format: " + str(type(batch)))

# ----------------------------
# Training routine
# ----------------------------


def train_main(args):
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    dev = args.device.lower()
    if dev == "mps":
        device = torch.device("mps" if getattr(
            torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    elif dev == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}; torch {torch.__version__}")

    # read sheets
    df_train = pd.read_excel(args.train_xlsx, sheet_name=args.train_sheet)
    df_val = pd.read_excel(args.train_xlsx, sheet_name=args.val_sheet)
    if 'ID' not in df_train.columns or 'Class' not in df_train.columns:
        raise RuntimeError("train sheet must contain ID and Class columns")
    if 'ID' not in df_val.columns or 'Class' not in df_val.columns:
        raise RuntimeError("val sheet must contain ID and Class columns")

    train_labels = df_train['Class'].fillna(-1).astype(int).tolist()
    val_labels = df_val['Class'].fillna(-1).astype(int).tolist()

    train_root = Path(args.data_dir)
    val_root = Path(args.data_dir)

    train_ds = SandSubjectDataset(root_dir=str(train_root),
                                  xlsx_path=str(args.train_xlsx),
                                  sheet_name=args.train_sheet,
                                  cache_dir=args.cache_dir,
                                  train_mode=True)
    val_ds = SandSubjectDataset(root_dir=str(val_root),
                                xlsx_path=str(args.train_xlsx),
                                sheet_name=args.val_sheet,
                                cache_dir=args.cache_dir,
                                train_mode=False)

    sampler = None
    class_weights = None
    if args.use_sampler:
        sampler, class_weights = build_sampler_and_class_weights(
            train_labels, batch_size=args.batch_size)
    else:
        _, class_weights = build_sampler_and_class_weights(
            train_labels, batch_size=args.batch_size)

    if class_weights is None:
        class_weights = torch.ones((5,), dtype=torch.float32)
    cw = class_weights.to(device) if args.use_class_weights else None

    # decide if sampler is a batch-sampler
    sampler_is_batch = False
    if sampler is not None:
        sname = type(sampler).__name__.lower()
        if hasattr(sampler, "batch_size") or ("batch" in sname) or ("batchsampler" in sname):
            sampler_is_batch = True

    def make_train_loader(num_workers):
        if sampler_is_batch:
            return DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=False, collate_fn=robust_collate)
        else:
            return DataLoader(train_ds, batch_size=args.batch_size, sampler=(sampler if sampler is not None else None),
                              shuffle=(sampler is None), num_workers=num_workers, pin_memory=False, collate_fn=robust_collate)

    try:
        train_loader = make_train_loader(num_workers=args.num_workers)
    except Exception:
        print("Warning: DataLoader creation failed with num_workers=",
              args.num_workers, " — falling back to num_workers=0")
        traceback.print_exc()
        train_loader = make_train_loader(num_workers=0)

    try:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=max(1, args.num_workers // 2), pin_memory=False, collate_fn=robust_collate)
    except Exception:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=False, collate_fn=robust_collate)

    # build model
    model = build_vit(num_classes=5, pretrained=args.pretrained, head_dropout=args.head_dropout,
                      device=str(device), freeze_backbone_flag=args.freeze_backbone,
                      pretrained_weights_path=args.pretrained_weights_path)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Model params total={total_params}, trainable={trainable_params}")

    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = None  # will call focal helper when needed

    best_macro_f1 = -1.0
    best_info = None
    best_model_path = None

    def validate_and_save(epoch_idx):
        nonlocal best_macro_f1, best_model_path, best_info
        model.eval()
        all_probs = []
        all_true = []
        with torch.no_grad():
            for xb, yb, sid in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_true.append(yb.numpy())
        if len(all_probs) == 0:
            return -1.0, None, None
        all_probs = np.vstack(all_probs)
        all_true = np.concatenate(all_true).astype(int)
        preds_arg = np.argmax(all_probs, axis=1) + 1
        try:
            from sklearn.metrics import f1_score
            f1_arg = f1_score(all_true, preds_arg,
                              average='macro', zero_division=0)
        except Exception:
            f1_arg = -1.0
        t, f1_t = find_best_global_threshold(
            all_true, all_probs, search=np.linspace(0.2, 0.9, 15))
        final_f1 = max(f1_arg, f1_t)
        used_t = float(t)
        if args.save_val_probs:
            np.savez_compressed(str(Path(args.out_dir) / f"val_probs_epoch_{epoch_idx:03d}.npz"),
                                probs=all_probs.astype(np.float32), true=all_true.astype(np.int32))
        if final_f1 > best_macro_f1:
            best_macro_f1 = float(final_f1)
            best_model_path = Path(args.out_dir) / "best_model.pt"
            torch.save({'epoch': int(epoch_idx), 'model_state': model.state_dict(
            ), 'best_macro_f1': float(final_f1)}, str(best_model_path))
            np.savez_compressed(str(Path(args.out_dir) / "best_val_probs.npz"),
                                probs=all_probs.astype(np.float32), true=all_true.astype(np.int32))
            best_info = {"best_macro_f1": float(final_f1), "epoch": int(
                epoch_idx), "used_threshold": used_t}
            save_json(Path(args.out_dir) / "best_info.json", best_info)
        return final_f1, all_probs, all_true

    def run_epochs(num_epochs, start_epoch, optimizer, scheduler=None, swa_info=None):
        nonlocal best_macro_f1, best_model_path, best_info
        for e in range(num_epochs):
            epoch_idx = start_epoch + e
            t0 = time.time()
            model.train()
            running_loss = 0.0
            n_samples = 0
            optimizer.zero_grad()
            try:
                batch_iter = enumerate(train_loader)
            except Exception:
                print("DataLoader iteration failed: falling back to num_workers=0")
                traceback.print_exc()
                safe_train_loader = make_train_loader(num_workers=0)
                batch_iter = enumerate(safe_train_loader)

            for step, batch in batch_iter:
                # normalize batch to (xb, yb, sid)
                try:
                    if isinstance(batch, (list, tuple)) and len(batch) == 1 and isinstance(batch[0], (list, tuple, torch.Tensor)):
                        batch = batch[0]
                    if isinstance(batch, torch.Tensor):
                        xb = batch
                        B = xb.shape[0]
                        yb = torch.full((B,), -1, dtype=torch.long)
                        sid = [None] * B
                    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        xb, yb = batch[0], batch[1]
                        sid = batch[2] if len(batch) > 2 else None
                    else:
                        raise RuntimeError("Unexpected batch format from DataLoader. Batch type: {}. Value: {}".format(
                            type(batch), str(type(batch))))
                except Exception:
                    print("ERROR unpacking batch. Summary:")
                    try:
                        print("type(batch)=", type(batch))
                    except Exception:
                        pass
                    traceback.print_exc()
                    raise

                xb = xb.to(device)
                yb0 = (yb.long() - 1).to(device) if isinstance(yb,
                                                               torch.Tensor) else torch.tensor(yb, dtype=torch.long, device=device) - 1

                # MixUp
                if args.mixup_alpha > 0:
                    xb, ya, yb_mix, lam = mixup_data(
                        xb, yb0, alpha=args.mixup_alpha)
                    ya = ya.to(device)
                    yb_mix = yb_mix.to(device)
                else:
                    ya = yb0
                    yb_mix = None
                    lam = 1.0

                logits = model(xb)
                if criterion is not None:
                    if yb_mix is None:
                        loss = criterion(logits, ya)
                    else:
                        loss = lam * criterion(logits, ya) + \
                            (1.0 - lam) * criterion(logits, yb_mix)
                else:
                    if yb_mix is None:
                        loss = focal_loss_logits(
                            logits, ya, gamma=args.gamma, weight=cw)
                    else:
                        loss = (lam * focal_loss_logits(logits, ya, gamma=args.gamma, weight=cw) +
                                (1.0 - lam) * focal_loss_logits(logits, yb_mix, gamma=args.gamma, weight=cw))

                if args.l1_lambda and args.l1_lambda > 0.0:
                    l1 = torch.tensor(0.0, device=device)
                    for p in model.parameters():
                        if p.requires_grad:
                            l1 = l1 + p.abs().sum()
                    loss = loss + args.l1_lambda * l1

                loss = loss / float(args.accumulation_steps)
                loss.backward()

                if (step + 1) % args.accumulation_steps == 0:
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        try:
                            scheduler.step()
                        except Exception:
                            pass
                    optimizer.zero_grad()

                bs = xb.size(0)
                running_loss += float(loss.item()) * bs * \
                    float(args.accumulation_steps)
                n_samples += bs

            train_loss = running_loss / max(1, n_samples)
            val_f1, val_probs, val_true = validate_and_save(epoch_idx)

            # SWA updates
            if args.use_swa and swa_info is not None and swa_info.get("enabled", False):
                base_epoch = swa_info.get("base_epoch", 0)
                swa_start_local = swa_info.get("swa_start_local", 999999)
                stage2_epoch_idx = epoch_idx - base_epoch + 1
                if stage2_epoch_idx >= swa_start_local:
                    try:
                        swa_model = swa_info["swa_model"]
                        swa_model.update_parameters(model)
                    except Exception as ex:
                        print("SWA update error:", ex)

            t_elapsed = time.time() - t0
            print(
                f"Epoch {epoch_idx} | train_loss={train_loss:.4f} | val_f1={val_f1:.4f} | time={t_elapsed:.1f}s")

            if epoch_idx % args.save_every == 0:
                pth = Path(args.out_dir) / \
                    f"checkpoint_epoch_{epoch_idx:03d}.pt"
                torch.save(
                    {'epoch': epoch_idx, 'model_state': model.state_dict()}, str(pth))

            gc.collect()
            try:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # Stage 1
    if args.stage1_epochs > 0:
        print("Starting stage1 (head-only) for", args.stage1_epochs, "epochs")
        try:
            model = freeze_backbone(model, freeze=True, exclude_head=True)
        except Exception:
            for n, p in model.named_parameters():
                if "head" not in n and "classifier" not in n and "fc" not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        optimizer_head = build_optimizer_with_llrd(
            model, args.lr_head, args.weight_decay, use_llrd=args.use_llrd, llrd_decay=args.llrd_decay, opt_name=args.optimizer)
        run_epochs(args.stage1_epochs, start_epoch=1, optimizer=optimizer_head)

    # Stage 2
    if args.stage2_epochs > 0:
        print("Unfreezing last", args.unfreeze_k,
              "blocks and fine-tuning for", args.stage2_epochs, "epochs")
        try:
            if args.unfreeze_k > 0:
                model = unfreeze_last_transformer_blocks(
                    model, k=args.unfreeze_k)
            else:
                for p in model.parameters():
                    p.requires_grad = True
        except Exception:
            named = list(model.named_parameters())
            n_blocks = get_vit_block_count_from_named_params(named)
            if args.unfreeze_k > 0 and n_blocks > 0:
                first_unfreeze = max(0, n_blocks - args.unfreeze_k)
                for n, p in named:
                    if "blocks." in n:
                        try:
                            idx = int(n.split("blocks.")[1].split(".")[0])
                            p.requires_grad = idx >= first_unfreeze
                        except Exception:
                            p.requires_grad = True
                    elif "head" in n or "classifier" in n or "fc" in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            else:
                for p in model.parameters():
                    p.requires_grad = True

        optimizer_ft = build_optimizer_with_llrd(
            model, args.lr_ft, args.weight_decay, use_llrd=args.use_llrd, llrd_decay=args.llrd_decay, opt_name=args.optimizer)

        swa_info = {"enabled": False}
        if args.use_swa:
            try:
                from torch.optim.swa_utils import AveragedModel, SWALR
                swa_model = AveragedModel(model)
                swa_info = {"enabled": True, "swa_model": swa_model,
                            "swa_start_local": args.swa_start, "base_epoch": 1 + args.stage1_epochs}
                try:
                    swa_scheduler = SWALR(optimizer_ft, swa_lr=args.swa_lr)
                except Exception:
                    swa_scheduler = None
            except Exception:
                print("SWA not available - proceeding without SWA.")
                swa_info = {"enabled": False}

        try:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_ft, T_max=max(1, args.stage2_epochs))
        except Exception:
            scheduler = None

        run_epochs(args.stage2_epochs, start_epoch=1 + args.stage1_epochs,
                   optimizer=optimizer_ft, scheduler=scheduler, swa_info=swa_info)

        if args.use_swa and swa_info.get("enabled", False):
            try:
                swa_model = swa_info["swa_model"]
                try:
                    torch.optim.swa_utils.update_bn(
                        train_loader, swa_model, device=device)
                except Exception:
                    pass
                final_model = swa_model.module if hasattr(
                    swa_model, "module") else swa_model
                out_path = Path(args.out_dir) / "swa_model.pt"
                torch.save({'epoch': args.stage1_epochs + args.stage2_epochs,
                           'model_state': final_model.state_dict()}, str(out_path))
                print("Saved SWA model to", out_path)
            except Exception as ex:
                print("SWA finalization error:", ex)

    print("Training finished. Best macro-F1 = ", best_macro_f1)
    if best_info is not None:
        print("Best info:", best_info)
    save_json(Path(args.out_dir) / "train_run_summary.json",
              {"best_macro_f1": best_macro_f1, "best_info": best_info})


if __name__ == "__main__":
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train_main(args)
