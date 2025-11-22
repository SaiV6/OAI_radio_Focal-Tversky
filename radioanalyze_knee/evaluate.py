import numpy as np
import torch
from monai.metrics import DiceMetric

from .tta_post import infer_tta, lcc_per_class, one_hot_long


def dice_channel(pred_oh, gt_oh, ch):
    return DiceMetric(include_background=False, reduction="mean")(
        pred_oh[:, ch : ch + 1], gt_oh[:, ch : ch + 1]
    ).item()


def evaluate_test(model, inferer, test_loader, device):
    d_all, df_all, dt_all = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            y = batch["label"].long().to(device)

            logits = infer_tta(x, model, inferer, device)
            pred = torch.argmax(logits, dim=1, keepdim=True)
            pred_np = pred.cpu().numpy()[0, 0]
            pred_np = lcc_per_class(pred_np)
            pred = torch.from_numpy(pred_np[None, None]).to(device)

            y_oh = one_hot_long(y)
            p_oh = one_hot_long(pred)

            d_all.append(
                DiceMetric(include_background=False, reduction="mean")(
                    p_oh[:, 1:3], y_oh[:, 1:3]
                ).item()
            )
            df_all.append(dice_channel(p_oh, y_oh, 1))
            dt_all.append(dice_channel(p_oh, y_oh, 2))

    print(f"Test Dice (femur+tibia): {np.mean(d_all):.4f}")
    print(f"  Femur Dice: {np.mean(df_all):.4f}")
    print(f"  Tibia Dice: {np.mean(dt_all):.4f}")
