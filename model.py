# model.py
from typing import Optional
import torch
import torch.nn as nn

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except Exception:
    vit_b_16 = None
    ViT_B_16_Weights = None


def _find_transformer_blocks(model):
    candidates = []
    if hasattr(model, "encoder"):
        enc = getattr(model, "encoder")
        if hasattr(enc, "layers"):
            candidates.append((enc, enc.layers))
        if hasattr(enc, "blocks"):
            candidates.append((enc, enc.blocks))
        if hasattr(enc, "encoder"):
            enc2 = getattr(enc, "encoder")
            if hasattr(enc2, "layers"):
                candidates.append((enc2, enc2.layers))
            if hasattr(enc2, "blocks"):
                candidates.append((enc2, enc2.blocks))
    if hasattr(model, "transformer"):
        tr = getattr(model, "transformer")
        if hasattr(tr, "encoder"):
            enc = getattr(tr, "encoder")
            if hasattr(enc, "layers"):
                candidates.append((enc, enc.layers))
            if hasattr(enc, "blocks"):
                candidates.append((enc, enc.blocks))
    if len(candidates) == 0:
        return None, None
    return candidates[0]


def _get_head_in_features(head_module: nn.Module) -> Optional[int]:
    if isinstance(head_module, nn.Linear):
        return head_module.in_features
    if hasattr(head_module, "in_features"):
        try:
            return int(getattr(head_module, "in_features"))
        except Exception:
            pass
    for c in reversed(list(head_module.children())):
        if isinstance(c, nn.Linear):
            return c.in_features
        for sc in reversed(list(c.children())):
            if isinstance(sc, nn.Linear):
                return sc.in_features
    return None


def _replace_head_with_linear(model: nn.Module, num_classes: int, head_dropout: float = 0.0):
    if hasattr(model, "heads"):
        head = model.heads
        in_features = _get_head_in_features(head)
        if in_features is None:
            if hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
                in_features = model.classifier.in_features
        if in_features is None:
            raise RuntimeError(
                "Cannot determine classifier in_features for ViT model.")
        if head_dropout is not None and head_dropout > 0.0:
            new_head = nn.Sequential(nn.Dropout(
                head_dropout), nn.Linear(in_features, num_classes))
        else:
            new_head = nn.Linear(in_features, num_classes)
        model.heads = new_head
        return model
    if hasattr(model, "classifier"):
        clf = model.classifier
        in_features = getattr(clf, "in_features", None)
        if in_features is not None:
            if head_dropout is not None and head_dropout > 0.0:
                new_head = nn.Sequential(nn.Dropout(
                    head_dropout), nn.Linear(in_features, num_classes))
            else:
                new_head = nn.Linear(in_features, num_classes)
            model.classifier = new_head
            return model
    default_feat = 768
    if head_dropout is not None and head_dropout > 0.0:
        model.heads = nn.Sequential(nn.Dropout(
            head_dropout), nn.Linear(default_feat, num_classes))
    else:
        model.heads = nn.Linear(default_feat, num_classes)
    return model


def freeze_backbone(model: nn.Module, freeze: bool = True, exclude_head: bool = True):
    head_params = set()
    if exclude_head and hasattr(model, "heads"):
        for p in model.heads.parameters():
            head_params.add(p)
    if exclude_head and hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            head_params.add(p)
    for p in model.parameters():
        if exclude_head and p in head_params:
            p.requires_grad = True
        else:
            p.requires_grad = not freeze
    return model


def unfreeze_last_transformer_blocks(model: nn.Module, k: int = 1):
    enc_parent, blocks = _find_transformer_blocks(model)
    if blocks is None:
        return model
    total = len(blocks)
    k = min(max(int(k), 0), total)
    for p in model.parameters():
        p.requires_grad = False
    for i in range(total - k, total):
        for p in blocks[i].parameters():
            p.requires_grad = True
    if hasattr(model, "heads"):
        for p in model.heads.parameters():
            p.requires_grad = True
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    return model


# inside model.py, modify build_vit signature and add optional load from path:
def build_vit(num_classes: int = 5, pretrained: bool = False, device: str = "cpu",
              head_dropout: float = 0.0, freeze_backbone_flag: bool = False,
              pretrained_weights_path: str = None):
    if vit_b_16 is None:
        raise ImportError("torchvision ViT (vit_b_16) not available.")
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT if ViT_B_16_Weights is not None else None
        model = vit_b_16(weights=weights)
    else:
        model = vit_b_16(weights=None)

    model = _replace_head_with_linear(
        model, num_classes, head_dropout=head_dropout)

    # if user provided an external pretrained checkpoint (state_dict)
    if pretrained_weights_path:

        sd = torch.load(pretrained_weights_path, map_location="cpu")
        # allow both full state or wrapper {'model_state': ...}
        st = sd.get('model_state', sd) if isinstance(sd, dict) else sd
        # try relaxed load to tolerate small key name differences
        model.load_state_dict(st, strict=False)

    if freeze_backbone_flag:
        model = freeze_backbone(model, freeze=True, exclude_head=True)
    return model.to(torch.device(device))
