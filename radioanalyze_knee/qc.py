import os

import numpy as np
import SimpleITK as sitk
import torch
import imageio.v3 as iio

from .config import Config
from .tta_post import infer_tta, lcc_per_class


def colorize(mask_zyx):
    z, y, x = mask_zyx.shape
    rgb = np.zeros((z, y, x, 3), dtype=np.uint8)
    rgb[mask_zyx == 1, 0] = 255  # femur→red
    rgb[mask_zyx == 2, 2] = 255  # tibia→blue
    return rgb


def blend(gray, overlay_rgb, alpha=0.35):
    g = gray.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    base = np.stack([g, g, g], -1) * 255.0
    out = (1 - alpha) * base + alpha * overlay_rgb.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_case_qc_dict(
    ex: dict,
    val_test,
    cfg: Config,
    device,
    model,
    inferer,
):
    """
    ex: {"image": path, "label": path} from test_files list.
    val_test: MONAI transform pipeline used for val/test.
    """
    os.makedirs(cfg.out_pred_dir, exist_ok=True)
    os.makedirs(cfg.qc_dir, exist_ok=True)

    # load transformed tensors
    data_dict = val_test({"image": ex["image"], "label": ex["label"]})
    x = data_dict["image"].unsqueeze(0).to(device)   # (1,1,D,H,W)
    vol = data_dict["image"].squeeze(0).cpu().numpy()
    gt = data_dict["label"].squeeze(0).cpu().numpy().astype(np.uint8)

    with torch.no_grad():
        logits = infer_tta(x, model, inferer, device)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

    pred = lcc_per_class(pred)

    # save NIfTI in transformed space
    pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
    sx, sy, sz = cfg.target_spacing[2], cfg.target_spacing[1], cfg.target_spacing[0]
    pred_img.SetSpacing((sx, sy, sz))
    pred_img.SetOrigin((0.0, 0.0, 0.0))
    pred_img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    sid = os.path.basename(ex["image"]).replace("_0000.nii.gz", "")
    out_nii = os.path.join(cfg.out_pred_dir, f"{sid}_pred_tta_lcc.nii.gz")
    sitk.WriteImage(pred_img, out_nii)
    print("Saved NIfTI:", out_nii)

    # overlays (GT vs pred, side-by-side video)
    gt_rgb = colorize(gt)
    pred_rgb = colorize(pred)
    Z = vol.shape[0]
    frames = []
    for z in range(Z):
        gt_overlay = blend(vol[z], gt_rgb[z])
        pr_overlay = blend(vol[z], pred_rgb[z])
        frames.append(np.concatenate([gt_overlay, pr_overlay], axis=1))

    mp4 = os.path.join(cfg.qc_dir, f"{sid}_gt_vs_pred.mp4")
    iio.imwrite(mp4, frames, fps=30, codec="libx264", quality=7)
    print("Saved video:", mp4)
