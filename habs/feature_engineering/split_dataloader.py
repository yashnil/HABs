#!/usr/bin/env python3
"""
02 ─ Train/Val/Test split  +  PyTorch Dataset that returns  (x, y)
==================================================================
* uses  features.zarr   (multi-channel predictor cube)
* uses  labels.zarr     (uint8 HAB mask)
* writes / re-uses  split_indices.npz
------------------------------------------------------------------
Run a quick demo
    python -m habs.feature_engineering.split_dataloader --demo
"""
from pathlib import Path
import argparse, numpy as np, xarray as xr, torch
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------#
# paths – change in ONE place if you move the data
ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
X_ZARR = ROOT / "features.zarr"
Y_ZARR = ROOT / "labels.zarr"           # **bare DataArray** (uint8, 0/1)
SPLIT  = ROOT / "split_indices.npz"     # holds np.arrays: train / val / test

# -----------------------------------------------------------------------------#
# 0. build or load the split once ---------------------------------------------#
if not SPLIT.exists():
    idx = np.arange(207)                              # 207 composites
    np.random.default_rng(42).shuffle(idx)
    n_tr   = int(0.70 * len(idx))
    n_val  = int(0.15 * len(idx))
    np.savez(SPLIT,
             train=idx[:n_tr],
             val  =idx[n_tr:n_tr+n_val],
             test =idx[n_tr+n_val:])
    print(f"wrote split file → {SPLIT.name}")
else:
    print(f"using existing split file  {SPLIT.name}")
split = np.load(SPLIT)

# -----------------------------------------------------------------------------#
# 1. Dataset ------------------------------------------------------------------#
class HABCubeDataset(Dataset):
    """
    Returns a tuple *(x, y)* for one 8-day composite

        x : float32  (C, H, W)    ← NaNs already → 0
        y : float32  (1, H, W)    ← 0 / 1 mask

    If *crop=(h,w)* is given, the SAME random crop is taken from x & y.
    """
    def __init__(self, time_indices, crop=None,
                 x_path=X_ZARR, y_path=Y_ZARR):

        # predictors (Dataset → DataArray “features”)
        self.x_da = xr.open_zarr(x_path,  chunks={"time": 1})["features"]

        # labels.zarr is a **bare DataArray**, so open_dataarray
        self.y_da = xr.open_dataarray(y_path, consolidated=False,
                                      chunks={"time": 1})

        # sanity once
        assert np.allclose(self.x_da.lat, self.y_da.lat)
        assert np.allclose(self.x_da.lon, self.y_da.lon)

        self.idxs  = np.asarray(time_indices, dtype=np.int16)
        self.crop  = crop    # None or (h, w)

    # ------------ helpers ----------------------------------------------------
    def _crop_slices(self, H, W):
        if self.crop is None:
            return slice(None), slice(None)
        h, w = self.crop
        y0 = np.random.randint(0, H - h + 1)
        x0 = np.random.randint(0, W - w + 1)
        return slice(y0, y0 + h), slice(x0, x0 + w)

    # ------------ torch Dataset API -----------------------------------------
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = int(self.idxs[i])

        # NB: *** .load() ***  (paren!) — otherwise you get a *method*
        x_np = self.x_da.isel(time=t).load().values    # (lat,lon,chan)
        y_np = self.y_da.isel(time=t).load().values    # (lat,lon)

        # NaNs → 0 in predictors
        x_np = np.nan_to_num(x_np, nan=0.0, copy=False).astype("float32")

        # aligned crop
        slc_y, slc_x = self._crop_slices(*x_np.shape[:2])
        x_np = x_np[slc_y, slc_x, :]
        y_np = y_np[slc_y, slc_x]

        # (lat,lon,chan) → (chan,lat,lon)   and add channel dim to y
        x_np = np.transpose(x_np, (2, 0, 1))
        y_np = y_np[None, ...].astype("float32")

        return torch.from_numpy(x_np), torch.from_numpy(y_np)

# -----------------------------------------------------------------------------#
# 2. convenient loader factory ------------------------------------------------#
def get_loaders(batch=8, crop=None, num_workers=2):
    tr_ds = HABCubeDataset(split["train"], crop)
    va_ds = HABCubeDataset(split["val"],   crop)
    te_ds = HABCubeDataset(split["test"],  crop)

    kw = dict(batch_size=batch, pin_memory=True, num_workers=num_workers)
    train_ld = DataLoader(tr_ds, shuffle=True,  drop_last=True,  **kw)
    val_ld   = DataLoader(va_ds, shuffle=False, drop_last=False, **kw)
    test_ld  = DataLoader(te_ds, shuffle=False, drop_last=False, **kw)

    n_channels = tr_ds[0][0].shape[0]
    return train_ld, val_ld, test_ld, n_channels

# -----------------------------------------------------------------------------#
# 3. quick sanity demo --------------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="print one mini-batch shape")
    args = parser.parse_args()

    if args.demo:
        ld, *_ = get_loaders(batch=4, crop=(128, 128))
        x, y = next(iter(ld))
        print("x", x.shape, x.dtype)   # (B, C, 128, 128)
        print("y", y.shape, y.dtype)   # (B, 1, 128, 128)
