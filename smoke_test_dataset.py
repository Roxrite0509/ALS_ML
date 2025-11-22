# smoke_test_dataset.py
from PIL import Image
from collections import Counter
from dataset import SandSubjectDataset
import os
import random
import numpy as np
import torch

# adjust these paths to your environment
DATA_DIR = "/Users/pranav/Desktop/Task1/training"
XLSX = "/Users/pranav/Desktop/Task1/training/sand_task_1.xlsx"
TRAIN_SHEET = "Training Baseline - Task 1"
# set to None to force on-the-fly
CACHE_DIR = "/Users/pranav/Desktop/Task1/cache_logmel_N256"
N_CHECK = 8  # how many samples to check

# make deterministic for reproducibility of augment calls
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def print_dataset_info(ds):
    print("Dataset mode:", "train" if ds.train_mode else "eval")
    print("Number of samples:", len(ds))
    if ds.labels is not None:
        labs = [int(x) for x in ds.labels]
        cnt = Counter(labs)
        print("Label counts (1..5):", [cnt.get(i, 0) for i in [1, 2, 3, 4, 5]])
    else:
        print("No labels present in sheet.")


def sample_and_inspect(ds, n=8, save_example_png=True):
    print(f"\nSampling {n} items from dataset (random indices):")
    idxs = list(range(min(len(ds), n)))
    for i in idxs:
        tensor, label, sid = ds[i]
        # tensor from dataset._stack_to_tensor -> FloatTensor in [0,1]
        print(f"{i:02d}: sid={sid} label={label} tensor.shape={tuple(tensor.shape)} dtype={tensor.dtype} min={tensor.min().item():.4f} max={tensor.max().item():.4f}")
        if save_example_png and i == 0:
            # convert tensor (C,H,W) to PIL and save for quick visual check
            img = (tensor * 255.0).numpy().astype('uint8')
            # torchvision transforms.ToPILImage would do this, but keep local conversion
            # img shape (C,H,W) -> (H,W,C)
            img = np.transpose(img, (1, 2, 0))
            Image.fromarray(img).save("smoke_sample_0.png")
            print("Saved smoke_sample_0.png (visual check)")


if __name__ == "__main__":
    print("=== Smoke test: cached mode ===")
    ds_cache = SandSubjectDataset(root_dir=DATA_DIR,
                                  xlsx_path=XLSX,
                                  sheet_name=TRAIN_SHEET,
                                  cache_dir=CACHE_DIR,
                                  train_mode=True)
    print_dataset_info(ds_cache)
    sample_and_inspect(ds_cache, n=N_CHECK, save_example_png=True)

    print("\n=== Smoke test: no-cache (force raw wav processing) ===")
    ds_nocache = SandSubjectDataset(root_dir=DATA_DIR,
                                    xlsx_path=XLSX,
                                    sheet_name=TRAIN_SHEET,
                                    cache_dir=None,
                                    train_mode=True)
    print_dataset_info(ds_nocache)
    sample_and_inspect(ds_nocache, n=4, save_example_png=False)

    print("\nDone. If you see tensor shapes (3,224,224) and smoke_sample_0.png saved, dataset pipeline is working.")
