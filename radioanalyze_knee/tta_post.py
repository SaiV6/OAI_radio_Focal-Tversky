import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy import ndimage


def infer_tta(x, model, inferer, device):
    """
    Test-time augmentation via axis flips; averages logits.
    """
    with autocast(enabled=(device.type == "cuda")):
        l0 = inferer(x, model)
        l1 = inferer(torch.flip(x, dims=[2]), model); l1 = torch.flip(l1, dims=[2])
        l2 = inferer(torch.flip(x, dims=[3]), model); l2 = torch.flip(l2, dims=[3])
        l3 = inferer(torch.flip(x, dims=[4]), model); l3 = torch.flip(l3, dims=[4])
        return (l0 + l1 + l2 + l3) / 4.0


def lcc_per_class(pred_np):
    """
    Keep largest connected component per class (1 & 2).
    """
    out = pred_np.copy()
    for cls in [1, 2]:
        m = (out == cls)
        if m.sum() == 0:
            continue
        lab, num = ndimage.label(m)
        if num < 2:
            continue
        sizes = ndimage.sum(m, lab, range(1, num + 1))
        keep = np.argmax(sizes) + 1
        out[(lab != keep) & (lab > 0)] = 0
    return out


def one_hot_long(t, C=3):
    return F.one_hot(t.squeeze(1).long(), num_classes=C).permute(0, 4, 1, 2, 3).float()
