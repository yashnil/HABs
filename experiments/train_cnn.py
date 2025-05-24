#!/usr/bin/env python3
"""
Minimal CNN training smoke-test.

Run examples
------------
# 1 epoch, single-worker dataloaders (works on macOS/MPS)
python experiments/train_cnn.py --epochs 1 --num_workers 0

# small random crops & 4 workers (Linux / CUDA)
python experiments/train_cnn.py --epochs 20 --crop 128 --num_workers 4
"""
from pathlib import Path
import argparse, tqdm, torch, torch.nn as nn, torch.optim as optim
from feature_engineering.split_dataloader import get_loaders                     # pkg import

# ---------- tiny UNet-like stub just to prove the pipeline --------------------
class TinyCNN(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),    nn.ReLU(),
            nn.Conv2d(32, out_ch, 1)
        )

    def forward(self, x):          # out = logits
        return self.net(x)

# -----------------------------------------------------------------------------#
def main(args):
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    # ── data ------------------------------------------------------------------
    train_ld, val_ld, _ = get_loaders(batch=args.batch,
                                      crop=(args.crop, args.crop) if args.crop else None,
                                      num_workers=args.num_workers)

    C = next(iter(train_ld)).shape[1]     # channel count
    model = TinyCNN(C).to(device)

    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()              # placeholder

    for epoch in range(args.epochs):
        # ----- train ----------------------------------------------------------
        model.train();   tot = 0
        for xb in tqdm.tqdm(train_ld, desc=f"train {epoch}"):
            xb = xb.to(device, non_blocking=True)
            preds = model(xb)
            loss  = loss_fn(preds, torch.zeros_like(preds))
            optimiser.zero_grad(); loss.backward(); optimiser.step()
            tot += loss.item()
        print(f"epoch {epoch:02d}  train_loss={tot/len(train_ld):.4f}")

        # ----- val ------------------------------------------------------------
        model.eval();  tot = 0
        with torch.no_grad():
            for xb in val_ld:
                xb = xb.to(device, non_blocking=True)
                preds = model(xb)
                tot += loss_fn(preds, torch.zeros_like(preds)).item()
        print(f"           val_loss  ={tot/len(val_ld):.4f}")

# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",      type=int, default=2)
    ap.add_argument("--batch",       type=int, default=4)
    ap.add_argument("--crop",        type=int, help="random square crop size (pixels)")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader workers (set 0 on macOS)")
    args = ap.parse_args()

    # macOS / OpenMP duplicate runtime clash workaround
    import os, platform
    if platform.system() == "Darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["OMP_NUM_THREADS"] = "1"

    main(args)
