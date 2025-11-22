import os
import time

import numpy as np
import torch
from torch.cuda.amp import autocast

from .tta_post import infer_tta, lcc_per_class, one_hot_long
from .config import Config


def train_loop(
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
    cfg: Config,
):
    best_val = 0.0
    os.makedirs(os.path.dirname(cfg.weights_path), exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for it, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            y = batch["label"].long().to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # cosine warm restarts per-iteration
            scheduler.step(epoch - 1 + it / len(train_loader))
            losses.append(loss.item())

        # validation
        model.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].long().to(device)
                logits = infer_tta(x, model, inferer, device)
                pred = torch.argmax(logits, dim=1, keepdim=True)
                pred_np = pred.cpu().numpy()[0, 0]
                pred_np = lcc_per_class(pred_np)
                pred = torch.from_numpy(pred_np[None, None]).to(device)

                dices.append(
                    dice_metric(one_hot_long(pred)[:, 1:3], one_hot_long(y)[:, 1:3]).item()
                )

        mv = float(np.mean(dices)) if dices else 0.0
        print(
            f"Epoch {epoch:03d} | train loss {np.mean(losses):.4f} "
            f"| val Dice(femur+tibia) {mv:.4f} | {time.time() - t0:.1f}s"
        )

        if mv > best_val:
            best_val = mv
            torch.save(model.state_dict(), cfg.weights_path)
            print("  ✅ Saved best weights →", cfg.weights_path)

    print(f"Best validation Dice (femur+tibia): {best_val:.4f}")
