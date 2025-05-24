
#!/usr/bin/env python3
"""
02 – Train/Val/Test split + PyTorch Dataset / DataLoader
-------------------------------------------------------
* uses features.zarr (written by build_features.py)
* writes split indices to  split_indices.npz
* provides a small Torch Dataset that:
      - returns (channels, H, W) floats  (NaNs → 0, mask in a channel)
      - supports optional random crops for augmentation
* quick sanity-check at the bottom.

Run:
    python feature_engineering/split_dataloader.py          # just build ↩︎
    python feature_engineering/split_dataloader.py --demo   # plus demo batch
"""

from pathlib import Path
import argparse, numpy as np, xarray as xr, torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
ZARR = ROOT / "features.zarr"
SPLIT = ROOT / "split_indices.npz"          # train_idx, val_idx, test_idx

# -----------------------------------------------------------------------------#
# 0. Partition the 207 composites •••------------------------------------------#
# -----------------------------------------------------------------------------#
if not SPLIT.exists():          # create once
    all_idx   = np.arange(207)
    np.random.default_rng(42).shuffle(all_idx)

    n_train   = int(0.70 * len(all_idx))      # 70 % train
    n_val     = int(0.15 * len(all_idx))      # 15 % val
    train_idx = all_idx[:n_train]
    val_idx   = all_idx[n_train:n_train+n_val]
    test_idx  = all_idx[n_train+n_val:]

    np.savez(SPLIT, train=train_idx, val=val_idx, test=test_idx)
    print(f"wrote split file → {SPLIT.name}")
else:
    print(f"using existing split file  {SPLIT.name}")

split = np.load(SPLIT)

# -----------------------------------------------------------------------------#
# 1. Torch Dataset ------------------------------------------------------------#
# -----------------------------------------------------------------------------#
class HABDataset(Dataset):
    """
    Basic Dataset – returns a **tensor** (C,H,W) for one 8-day composite.
    By default it keeps the full 502×279 field; pass crop=(h,w) for random crops.
    """
    def __init__(self, indices, crop=None, zarr_path=ZARR):
        self.ds     = xr.open_zarr(zarr_path, chunks={"time":1})   # lazy
        self.idx    = np.array(indices)
        self.crop   = crop            # None or (h, w)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = int(self.idx[i])
        da = self.ds["features"].isel(time=t).load()      # load one slice → (lat,lon,chan)

        # replace NaNs with 0 *after* loading
        arr = np.nan_to_num(da.values.astype("float32"), nan=0.0)

        # optional random crop
        if self.crop is not None:
            H,W = arr.shape[:2]
            h,w = self.crop
            y0  = np.random.randint(0, H-h+1)
            x0  = np.random.randint(0, W-w+1)
            arr = arr[y0:y0+h, x0:x0+w, :]

        # (lat,lon,chan) → (chan,lat,lon)
        arr = np.transpose(arr, (2,0,1))
        return torch.from_numpy(arr)          # float32 tensor

# handy builders ---------------------------------------------------------------
def get_loaders(batch=8, crop=None, num_workers=2):
    train_ds = HABDataset(split["train"], crop)
    val_ds   = HABDataset(split["val"],   crop)
    test_ds  = HABDataset(split["test"],  crop)

    kw = dict(batch_size=batch, pin_memory=True, num_workers=num_workers)
    train_ld = DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw)
    val_ld   = DataLoader(val_ds,   shuffle=False, drop_last=False, **kw)
    test_ld  = DataLoader(test_ds,  shuffle=False, drop_last=False, **kw)
    return train_ld, val_ld, test_ld

# -----------------------------------------------------------------------------#
# 2. Demo ----------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--demo", action="store_true", help="show one mini-batch shape")
    args = argp.parse_args()

    if args.demo:
        train_ld, *_ = get_loaders(batch=4, crop=(128,128))
        x = next(iter(train_ld))        # (B,C,H,W)
        print("demo batch", x.shape)    # e.g. torch.Size([4, 19, 128, 128])

'''
demo batch torch.Size([4, 18, 128, 128])
'''