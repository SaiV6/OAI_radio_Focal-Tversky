import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer

from .config import Config


class FocalTverskyLoss(torch.nn.Module):
    """
    FN-penalizing segmentation loss (Focal Tversky).
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B,C,D,H,W) logits; targets: (B,1,D,H,W) long
        num_classes = inputs.shape[1]
        targets_1h = F.one_hot(targets.squeeze(1), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        probs = F.softmax(inputs, dim=1)

        dims = (0, 2, 3, 4)
        TP = torch.sum(probs * targets_1h, dims)
        FP = torch.sum(probs * (1 - targets_1h), dims)
        FN = torch.sum((1 - probs) * targets_1h, dims)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = torch.pow((1 - tversky), self.gamma)
        return loss.mean()


def build_model_and_training(cfg: Config, device):
    # Wider UNet than the basic config
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 96, 128, 192),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))
    dice_metric = DiceMetric(include_background=False, reduction="mean")  # femur+tibia only

    inferer = SlidingWindowInferer(
        roi_size=(96, 192, 192),
        sw_batch_size=1,
        overlap=0.5,
        mode="gaussian",
    )

    return model, loss_fn, optimizer, scheduler, scaler, dice_metric, inferer
