#!/usr/bin/env python

import os
import torch

from radioanalyze_knee import (
    Config,
    set_seed,
    get_device,
    download_oaizib_cm,
    prepare_data_folders,
    build_transforms_and_datasets,
    build_model_and_training,
    train_loop,
    evaluate_test,
    save_case_qc_dict,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="radioanalyze V2 – OAIZIB-CM knee cartilage segmentation"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/OAIZIB-CM",
        help="Root directory for OAIZIB-CM dataset",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="./weights/monai_knee_ft_best.pth",
        help="Path to save / load best model weights",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size",
    )
    parser.add_argument(
        "--val-split",
        type=int,
        default=30,
        help="Last N training scans used for validation",
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=1e-4,
        help="Base learning rate",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download OAIZIB-CM via huggingface_hub (if not already present)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only evaluate / run QC using existing weights",
    )
    parser.add_argument(
        "--run-qc",
        action="store_true",
        help="Generate QC NIfTI + MP4 for first test case",
    )

    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        weights_path=args.weights_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        base_lr=args.base_lr,
    )

    os.makedirs(os.path.dirname(cfg.weights_path), exist_ok=True)
    os.makedirs(cfg.out_pred_dir, exist_ok=True)
    os.makedirs(cfg.qc_dir, exist_ok=True)

    set_seed(42)
    device = get_device()

    if args.download_data:
        download_oaizib_cm(cfg)

    IMTR, LBTR, IMTS, LBTS, train_imgs, train_labs, test_imgs, test_labs = prepare_data_folders(cfg)

    (
        train_loader,
        val_loader,
        test_loader,
        train_files,
        val_files,
        test_files,
        val_test,
    ) = build_transforms_and_datasets(cfg, train_imgs, train_labs, test_imgs, test_labs)

    model, loss_fn, optimizer, scheduler, scaler, dice_metric, inferer = build_model_and_training(cfg, device)

    if not args.skip_train:
        train_loop(
            model,
            loss_fn,
            optimizer,
            scheduler,
            scaler,
            dice_metric,
            train_loader,
            val_loader,
            inferer,
            device,
            cfg,
        )
    else:
        print("Skipping training (--skip-train).")

    if os.path.isfile(cfg.weights_path):
        print("Loading best weights from:", cfg.weights_path)
        model.load_state_dict(torch.load(cfg.weights_path, map_location=device))
    else:
        print("WARNING: weights file not found, evaluating with current model parameters.")

    evaluate_test(model, inferer, test_loader, device)

    if args.run_qc:
        if len(test_files):
            ex = test_files[0]
            print("Running QC on:", ex)
            save_case_qc_dict(ex, val_test, cfg, device, model, inferer)
        else:
            print("No test files to visualize for QC.")


if __name__ == "__main__":
    main()
