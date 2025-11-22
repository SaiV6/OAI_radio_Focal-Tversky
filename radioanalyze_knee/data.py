import glob
import os
import shutil
import zipfile

import numpy as np
import torch
from huggingface_hub import snapshot_download
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    Lambdad,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandAffined,
    RandBiasFieldd,
    RandGaussianNoised,
    RandAdjustContrastd,
)

from .config import Config

# try to import an elastic transform (name differs across MONAI versions)
try:
    from monai.transforms import RandElasticDeformationd as RandElasticd
except ImportError:
    try:
        from monai.transforms import Rand3DElasticd as RandElasticd
    except ImportError:
        print("⚠️ Elastic deformation transform not available; continuing without it.")
        RandElasticd = None


def download_oaizib_cm(cfg: Config):
    os.makedirs(cfg.data_root, exist_ok=True)
    print(f"Downloading OAIZIB-CM into {cfg.data_root} ...")
    snapshot_download(
        repo_id="YongchengYAO/OAIZIB-CM",
        repo_type="dataset",
        revision="load_dataset-support",
        local_dir=cfg.data_root,
        local_dir_use_symlinks=False,
    )
    print("Download complete.")


def prepare_data_folders(cfg: Config):
    data_root = cfg.data_root
    print(f"Preparing data under {data_root} ...")

    # Unzip everything into data_root
    for z in glob.glob(os.path.join(data_root, "**", "*.zip"), recursive=True):
        print("Unzipping:", z)
        with zipfile.ZipFile(z, "r") as f:
            f.extractall(data_root)

    # Canonical dirs
    IMTR = os.path.join(data_root, "imagesTr")
    LBTR = os.path.join(data_root, "labelsTr")
    IMTS = os.path.join(data_root, "imagesTs")
    LBTS = os.path.join(data_root, "labelsTs")
    for d in (IMTR, LBTR, IMTS, LBTS):
        os.makedirs(d, exist_ok=True)

    # Move any extracted NIfTIs into canonical dirs
    for f in glob.glob(os.path.join(data_root, "**", "*.nii.gz"), recursive=True):
        base = os.path.basename(f)
        parent = os.path.dirname(f).lower()
        if "_0000.nii.gz" in base:
            # images
            if "imagestr" in parent and os.path.dirname(f) != IMTR:
                shutil.move(f, os.path.join(IMTR, base))
            elif "imagests" in parent and os.path.dirname(f) != IMTS:
                shutil.move(f, os.path.join(IMTS, base))
        else:
            # labels
            if "labelstr" in parent and os.path.dirname(f) != LBTR:
                shutil.move(f, os.path.join(LBTR, base))
            elif "labelsts" in parent and os.path.dirname(f) != LBTS:
                shutil.move(f, os.path.join(LBTS, base))

    train_imgs = sorted(glob.glob(os.path.join(IMTR, "*.nii.gz")))
    train_labs = [os.path.join(LBTR, os.path.basename(p).replace("_0000", "")) for p in train_imgs]
    test_imgs = sorted(glob.glob(os.path.join(IMTS, "*.nii.gz")))
    test_labs = [os.path.join(LBTS, os.path.basename(p).replace("_0000", "")) for p in test_imgs]

    print(f"Train images: {len(train_imgs)} | Train labels: {len(train_labs)}")
    print(f"Test  images: {len(test_imgs)} | Test  labels: {len(test_labs)}")
    assert len(train_imgs) == len(train_labs) and len(test_imgs) == len(test_labs)

    return IMTR, LBTR, IMTS, LBTS, train_imgs, train_labs, test_imgs, test_labs


def remap_femur_tibia_any(lbl):
    """
    Map labels to {0:bg, 1:femur, 2:tibia} for torch.Tensor or np.ndarray.
    OAI/OAIZIB uses femur=1, tibia=3; everything else -> 0.
    """
    if isinstance(lbl, torch.Tensor):
        out = lbl.clone()
        keep = (out == 1) | (out == 3)
        out = torch.where(keep, out, torch.zeros_like(out))
        out = torch.where(out == 3, torch.tensor(2, dtype=out.dtype, device=out.device), out)
        return out.to(torch.uint8)
    else:
        a = np.array(lbl, copy=True)
        a[(a != 1) & (a != 3)] = 0
        a[a == 3] = 2
        return a.astype(np.uint8)


def build_transforms_and_datasets(
    cfg: Config,
    train_imgs,
    train_labs,
    test_imgs,
    test_labs,
):
    base_keys = ["image", "label"]
    target_spacing = cfg.target_spacing

    # Base transforms
    base_train = Compose([
        LoadImaged(keys=base_keys, image_only=False),
        EnsureChannelFirstd(keys=base_keys),
        Orientationd(keys=base_keys, axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
        Spacingd(keys=["label"], pixdim=target_spacing, mode="nearest"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0, a_max=1.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Lambdad(keys=["label"], func=remap_femur_tibia_any),
        EnsureTyped(keys=base_keys),
    ])

    elastic_aug = []
    if RandElasticd is not None:
        elastic_aug = [
            RandElasticd(
                keys=["image", "label"],
                prob=0.2,
                sigma_range=(2, 4),
                magnitude_range=(1, 2),
                mode=("bilinear", "nearest"),
            )
        ]

    train_aug = Compose([
        base_train,
        RandCropByPosNegLabeld(
            keys=base_keys,
            label_key="label",
            spatial_size=(96, 192, 192),
            pos=1,
            neg=1,
            num_samples=1,
        ),
        RandFlipd(keys=base_keys, prob=0.5, spatial_axis=2),
        RandFlipd(keys=base_keys, prob=0.3, spatial_axis=1),
        *elastic_aug,
        RandAffined(
            keys=base_keys,
            prob=0.3,
            rotate_range=(0.05, 0.05, 0.05),
            scale_range=(0.05, 0.05, 0.05),
            mode=("bilinear", "nearest"),
        ),
        RandBiasFieldd(keys=["image"], prob=0.3, coeff_range=(0.0, 0.7)),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.02),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.9, 1.1)),
        EnsureTyped(keys=base_keys),
    ])

    val_test = Compose([
        LoadImaged(keys=base_keys, image_only=False),
        EnsureChannelFirstd(keys=base_keys),
        Orientationd(keys=base_keys, axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
        Spacingd(keys=["label"], pixdim=target_spacing, mode="nearest"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0.0, a_max=1.0,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        Lambdad(keys=["label"], func=remap_femur_tibia_any),
        EnsureTyped(keys=base_keys),
    ])

    # Split
    val_split = cfg.val_split
    train_files = [{"image": i, "label": l} for i, l in zip(train_imgs[:-val_split], train_labs[:-val_split])]
    val_files = [{"image": i, "label": l} for i, l in zip(train_imgs[-val_split:], train_labs[-val_split:])]
    test_files = [{"image": i, "label": l} for i, l in zip(test_imgs, test_labs)]

    # CacheDatasets
    train_ds = CacheDataset(train_files, transform=train_aug, cache_rate=0.5, num_workers=2)
    val_ds = CacheDataset(val_files, transform=val_test, cache_rate=1.0, num_workers=2)
    test_ds = CacheDataset(test_files, transform=val_test, cache_rate=1.0, num_workers=2)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print("Dataloader sizes:",
          "train:", len(train_loader),
          "| val:", len(val_loader),
          "| test:", len(test_loader))

    return train_loader, val_loader, test_loader, train_files, val_files, test_files, val_test
