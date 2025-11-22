#!/usr/bin/env python

import os
import torch

from radioanalyze_knee import (
    Config,
    set_seed,
    get_device,
    build_model_and_training,
    get_infer_transform,
    infer_new,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run trained radioanalyze V2 model on a new MRI (DICOM or NIfTI)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to DICOM folder, single DICOM file, or NIfTI (.nii/.nii.gz)",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="./weights/monai_knee_ft_best.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--out-pred",
        type=str,
        default="./new_pred_tta_lcc.nii.gz",
        help="Output NIfTI path for segmentation",
    )
    parser.add_argument(
        "--target-spacing-z",
        type=float,
        default=0.7,
        help="Target spacing along z in mm (used during training)",
    )
    parser.add_argument(
        "--target-spacing-y",
        type=float,
        default=0.365,
        help="Target spacing along y in mm (used during training)",
    )
    parser.add_argument(
        "--target-spacing-x",
        type=float,
        default=0.365,
        help="Target spacing along x in mm (used during training)",
    )

    args = parser.parse_args()

    # Build config (we mostly care about target_spacing here)
    cfg = Config(
        weights_path=args.weights_path,
        target_spacing=(args.target_spacing_z,
                        args.target_spacing_y,
                        args.target_spacing_x),
    )

    set_seed(42)
    device = get_device()

    # Build model + inferer (reuse training utility, ignore loss/optimizer)
    model, _, _, _, _, _, inferer = build_model_and_training(cfg, device)

    if not os.path.isfile(cfg.weights_path):
        raise FileNotFoundError(
            f"Weights file not found at {cfg.weights_path}. "
            "Train the model first or point --weights-path to a valid checkpoint."
        )

    print(f"[infer_new_case] Loading weights from: {cfg.weights_path}")
    model.load_state_dict(torch.load(cfg.weights_path, map_location=device))
    model.eval()

    # Build inference transform (image-only pipeline)
    infer_xforms = get_infer_transform(cfg.target_spacing)

    # Run inference
    infer_new(
        model,
        input_path=args.input_path,
        infer_transform=infer_xforms,
        inferer=inferer,
        device=device,
        cfg=cfg,
        out_pred_path=args.out_pred,
    )


if __name__ == "__main__":
    main()
