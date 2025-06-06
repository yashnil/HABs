#!/usr/bin/env python3
"""
Minimal CNN smoke-test  ✚ YAML hyper-params  ✚ macOS-safe OpenMP fix
===================================================================

Quick runs
----------
# 1 epoch, single worker (Apple Silicon / MPS)
python -m habs.experiments.train_cnn --epochs 1 --num_workers 0

# 20 epochs, 128-px crops, 4 workers (Linux / CUDA)
python -m habs.experiments.train_cnn --epochs 20 --crop 128 --num_workers 4
"""
# ──────────────────────────────────────────────────────────────────────────────
# --- macOS needs this *before* importing torch -------------------------------
import os, platform
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"]      = "1"

# --- stdlib / third-party -----------------------------------------------------
from pathlib import Path
import argparse, yaml, tqdm, torch
import torch.nn as nn
import torch.optim as optim

from habs.feature_engineering.split_dataloader import get_loaders  # local pkg
# ──────────────────────────────────────────────────────────────────────────────


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',      type=int)
    p.add_argument('--batch',       type=int)
    p.add_argument('--crop',        type=int, help='random square crop (px)')
    p.add_argument('--num_workers', type=int, help='DataLoader workers')
    p.add_argument('--config',      type=str, default='experiments/config.yml',
                   help='YAML hyper-param file')
    return p.parse_args()


class TinyCNN(nn.Module):
    """toy 3-layer CNN (UNet-ish stub)"""
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),    nn.ReLU(),
            nn.Conv2d(32, out_ch, 1)            # logits
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    args = cli()

    # ---------- YAML hyper-params -------------------------------------------
    cfg = {}
    if Path(args.config).exists():
        print(f"🔹 loading hyper-params from {args.config}")
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # command-line ⟹ YAML ⟹ hard-coded default
    epochs      = args.epochs      or cfg.get('num_epochs',   2)
    batch_size  = args.batch       or cfg.get('batch_size',   4)
    crop_sz     = args.crop        or cfg.get('crop')              # may be None
    num_workers = args.num_workers or cfg.get('num_workers', 0)
    lr          = float(cfg.get('optimizer', {}).get('lr', 1e-3))
    pos_w       = float(cfg.get('loss', {}).get('pos_weight', 1.0))

    # ---------- device -------------------------------------------------------
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))
    torch.backends.cudnn.benchmark = True
    print('device:', device)

    # ---------- data ---------------------------------------------------------
    train_ld, val_ld, _test_ld, n_ch = get_loaders(
        batch=batch_size,
        crop=(crop_sz, crop_sz) if crop_sz else None,
        num_workers=num_workers
    )

    model = TinyCNN(n_ch).to(device)

    optimiser = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=device)
    )

    # ---------- training loop -----------------------------------------------
    for epoch in range(epochs):
        # ---- train ---------------------------------------------------------
        model.train(); running = 0.0
        for xb, yb in tqdm.tqdm(train_ld, desc=f"train {epoch}"):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            yhat   = model(xb)

            loss = loss_fn(yhat, yb)
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            running += loss.item()

        print(f"epoch {epoch:02d}  train_loss={running/len(train_ld):.4f}")

        # ---- validation ----------------------------------------------------
        model.eval(); running = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                running += loss_fn(model(xb), yb).item()

        print(f"           val_loss  ={running/len(val_ld):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
