#!/usr/bin/env python3
"""
Minimal CNN training smoke-test (YAML-aware, macOS safe)
=======================================================

Examples
--------
# 1 epoch, single worker (Apple-silicon / MPS)
python -m habs.experiments.train_cnn --epochs 1 --num_workers 0

# 20 epochs, 128-px crops, 4 workers (Linux / CUDA)
python -m habs.experiments.train_cnn --epochs 20 --crop 128 --num_workers 4
"""
# ── macOS OpenMP fix *must* be first ──────────────────────────────────────────
import os, platform
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"]      = "1"

# ── stdlib / third-party imports ──────────────────────────────────────────────
from pathlib import Path
import argparse, yaml, tqdm, torch
import torch.nn as nn
import torch.optim as optim

from habs.feature_engineering.split_dataloader import get_loaders   # local pkg

# ──────────────────────────────────────────────────────────────────────────────
def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",      type=int)
    ap.add_argument("--batch",       type=int)
    ap.add_argument("--crop",        type=int, help="random square crop (px)")
    ap.add_argument("--num_workers", type=int,
                    help="DataLoader workers (0 on macOS)")
    ap.add_argument("--config",      type=str, default="experiments/config.yml",
                    help="YAML hyper-param file")
    return ap.parse_args()

# ── tiny UNet-ish stub ────────────────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),    nn.ReLU(),
            nn.Conv2d(32, out_ch, 1)
        )

    def forward(self, x):                         # logits
        return self.net(x)

# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_cli()

    # ---------- read YAML (if present) ------------------------------------ #
    cfg = {}
    if Path(args.config).exists():
        print(f"🔹 loading hyper-params from  {args.config}")
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}   # ← ensure dict even if safe_load = None

    # command-line > YAML > hard-coded default
    epochs      = args.epochs      or cfg.get("num_epochs" ,  2)
    batch_size  = args.batch       or cfg.get("batch_size" ,  4)
    crop_sz     = args.crop        or cfg.get("crop")
    num_workers = args.num_workers or cfg.get("num_workers", 0)
    lr          = cfg.get("optimizer", {}).get("lr", 1e-3)
    pos_w       = float(cfg.get("loss", {}).get("pos_weight", 1.0))

    # ── device ----------------------------------------------------------------
    device = (
        torch.device("cuda" if torch.cuda.is_available() else
                     "mps"  if torch.backends.mps.is_available() else "cpu")
    )
    print("device:", device)
    torch.backends.cudnn.benchmark = True

    # ── data ------------------------------------------------------------------
    train_ld, val_ld, _ = get_loaders(
        batch=batch_size,
        crop=(crop_sz, crop_sz) if crop_sz else None,
        num_workers=num_workers
    )

    in_channels = next(iter(train_ld)).shape[1]
    model = TinyCNN(in_channels).to(device)

    optimiser = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=device)
    )

    # ── training loop ---------------------------------------------------------
    for epoch in range(epochs):
        # ----- train ----------------------------------------------------------
        model.train();   running = 0.0
        for xb in tqdm.tqdm(train_ld, desc=f"train {epoch}"):
            xb   = xb.to(device, non_blocking=True)
            yhat = model(xb)

            tgt  = torch.zeros_like(yhat)     # <-- placeholder labels
            loss = loss_fn(yhat, tgt)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += loss.item()

        print(f"epoch {epoch:02d}  train_loss={running/len(train_ld):.4f}")

        # ----- validation -----------------------------------------------------
        model.eval();  running = 0.0
        with torch.no_grad():
            for xb in val_ld:
                xb   = xb.to(device, non_blocking=True)
                yhat = model(xb)
                tgt  = torch.zeros_like(yhat)
                running += loss_fn(yhat, tgt).item()

        print(f"           val_loss  ={running/len(val_ld):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
