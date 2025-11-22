import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Config:
    data_root: str = "./data/OAIZIB-CM"
    weights_path: str = "./weights/monai_knee_ft_best.pth"
    out_pred_dir: str = "./preds"
    qc_dir: str = "./qc"

    # spacing (z, y, x) in mm
    target_spacing: tuple = (0.7, 0.365, 0.365)

    batch_size: int = 1
    epochs: int = 200
    val_split: int = 30
    base_lr: float = 1e-4


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("✅ CUDA:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠️ Running on CPU")
    return device
