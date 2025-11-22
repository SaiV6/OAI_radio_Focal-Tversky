import os

import SimpleITK as sitk
import torch
import numpy as np

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensity,
    EnsureType,
)

from .config import Config
from .tta_post import infer_tta, lcc_per_class


def dicom_to_nifti(input_path: str, out_nii: str) -> str:
    """
    Convert a DICOM series folder or single DICOM file to NIfTI.
    If input_path is already a NIfTI, returns it unchanged.

    Args:
        input_path: folder containing DICOM series, single DICOM, or NIfTI file.
        out_nii: path to write the converted NIfTI (if needed).

    Returns:
        Path to a NIfTI file.
    """
    input_path = str(input_path)

    # Already NIfTI
    if input_path.lower().endswith((".nii", ".nii.gz")):
        return input_path

    # DICOM folder
    if os.path.isdir(input_path):
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(input_path)
        if len(files) == 0:
            raise RuntimeError(f"No DICOM slices found in folder: {input_path}")
        reader.SetFileNames(files)
        img = reader.Execute()
    else:
        # Assume single DICOM file
        img = sitk.ReadImage(input_path)

    sitk.WriteImage(img, out_nii)
    return out_nii


def get_infer_transform(target_spacing):
    """
    Build the image-only transform for inference.

    This matches the core image pipeline used in training:
    - Load
    - Channel-first
    - Orientation to RAS
    - Spacing to target_spacing (z, y, x)
    - Intensity scaling to [0,1]
    """
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=target_spacing, mode=("bilinear",)),
            ScaleIntensity(minv=0.0, maxv=1.0),
            EnsureType(),
        ]
    )


def infer_new(
    model: torch.nn.Module,
    input_path: str,
    infer_transform,
    inferer,
    device: torch.device,
    cfg: Config,
    out_pred_path: str = "./new_pred_tta_lcc.nii.gz",
    tmp_case_path: str = "./tmp_new_case.nii.gz",
) -> str:
    """
    Run inference with TTA + LCC on a new case (DICOM or NIfTI).

    - Converts input to NIfTI if needed.
    - Applies the same geometric & intensity transforms as training (image-only).
    - Uses TTA (flips) via infer_tta.
    - Applies largest-connected-component filtering per class.
    - Writes segmentation as NIfTI in model space (resampled to cfg.target_spacing).

    Args:
        model: trained UNet model.
        input_path: path to DICOM folder, single DICOM file, or NIfTI.
        infer_transform: MONAI Compose built by get_infer_transform.
        inferer: SlidingWindowInferer used during training.
        device: torch.device.
        cfg: Config object (for target_spacing).
        out_pred_path: output NIfTI path.
        tmp_case_path: temp NIfTI path for DICOM conversion.

    Returns:
        Path to output NIfTI file.
    """
    model.eval()

    # 1) Normalize input to NIfTI
    nii_path = dicom_to_nifti(input_path, tmp_case_path)

    # 2) Apply inference transforms (image-only)
    x_t = infer_transform(nii_path)        # (1, D, H, W)
    x = x_t.unsqueeze(0).to(device)        # (1, 1, D, H, W)

    # 3) TTA inference
    with torch.no_grad():
        logits = infer_tta(x, model, inferer, device)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]  # (D, H, W)

    # 4) Largest connected component per class
    pred = lcc_per_class(pred)

    # 5) Save NIfTI in model space (target spacing)
    pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
    # cfg.target_spacing is (z, y, x); SimpleITK expects (x, y, z)
    sx, sy, sz = cfg.target_spacing[2], cfg.target_spacing[1], cfg.target_spacing[0]
    pred_img.SetSpacing((sx, sy, sz))
    pred_img.SetOrigin((0.0, 0.0, 0.0))
    pred_img.SetDirection((1.0, 0.0, 0.0,
                           0.0, 1.0, 0.0,
                           0.0, 0.0, 1.0))

    sitk.WriteImage(pred_img, out_pred_path)
    print(f"[infer_new] Saved segmentation to: {out_pred_path}")

    return out_pred_path
