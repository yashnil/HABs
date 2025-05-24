#!/usr/bin/env python3
"""
feature_engineering/build_features.py
------------------------------------
Create a 4-D tensor (time, lat, lon, channel) from
root_dataset_filled.nc   →   features.zarr

Channels
========
1. MODIS   : chlor_a, Kd_490, nflh, sst
2. ERA-5   : tp, avg_sdswrf, t2m, d2m, u10, v10
3. CMEMS   : so, thetao, uo, vo, zos
4. mask    : binary ocean mask   (1 = ocean / valid pixel, 0 = land or perm-NaN)
5. time    : sin( 2π DOY ∕ 366 ),  cos( 2π DOY ∕ 366 )

Normalisation
=============
Each science variable is z-scored over **finite** pixels only.
The means / stds are saved to  norm_stats.yml  for later inverse-transform.

Run
~~~
    python feature_engineering/build_features.py
       – the script will create *features.zarr* (~2 GB, chunked) and
         *norm_stats.yml* in the same folder.
"""

from pathlib import Path
import numpy as np
import xarray as xr
import yaml

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
SRC  = ROOT / "root_dataset_filled.nc"
DST  = ROOT / "features.zarr"

# ── variables to include ─────────────────────────────────────────────────────
SCI_VARS = [
    # MODIS
    "chlor_a", "Kd_490", "nflh", "sst",
    # ERA-5
    "tp", "avg_sdswrf", "t2m", "d2m", "u10", "v10",
    # CMEMS
    "so", "thetao", "uo", "vo", "zos",
]

# ── 1 · open source cube (lazy / dask) ───────────────────────────────────────
print(f"🔹 opening {SRC.name} …")
ds = xr.open_dataset(SRC, chunks={"time": 50})   # ≤50 frames per dask chunk

# ── 2 · build static ocean-mask channel ──────────────────────────────────────
print("🔹 computing ocean mask …")
mask = xr.full_like(ds[SCI_VARS[0]].isel(time=0), 1, dtype="int8")
for var in SCI_VARS:
    mask = mask.where(np.isfinite(ds[var].isel(time=0)), 0)
mask.name = "ocean_mask"
mask.attrs["long_name"] = "static mask (1=ocean, 0=land/permanent-NaN)"

# ── 3 · z-score normalisation per variable ──────────────────────────────────
print("🔹 normalising variables …")
norm_vars = {}
norm_stats = {}
for v in SCI_VARS:
    dv = ds[v]
    mu = float(dv.mean(dim=("time", "lat", "lon"), skipna=True))
    sd = float(dv.std (dim=("time", "lat", "lon"), skipna=True))
    norm_vars[v] = (dv - mu) / sd
    norm_vars[v].attrs.update({"mean": mu, "std": sd, "normalised": "z"})
    norm_stats[v] = {"mean": mu, "std": sd}
    print(f"   {v:10s} : μ={mu:7.3g}   σ={sd:7.3g}")

# ── 4 · sin / cos of day-of-year ────────────────────────────────────────────
print("🔹 generating time sin/cos …")
doy   = ds.time.dt.dayofyear.astype("float32")
angle = 2.0 * np.pi * doy / 366.0
sin_t = xr.DataArray(np.sin(angle), dims="time", coords={"time": ds.time},
                     name="doy_sin")
cos_t = xr.DataArray(np.cos(angle), dims="time", coords={"time": ds.time},
                     name="doy_cos")

# broadcast to (time, lat, lon) by matching an existing 3-D variable
tmpl  = ds[SCI_VARS[0]]                 # any (time,lat,lon) field
sin3  = sin_t.broadcast_like(tmpl)
cos3  = cos_t.broadcast_like(tmpl)

# ── 5 · assemble into single Dataset & stack channels ───────────────────────
print("🔹 stacking into channel dimension …")
all_vars = {**norm_vars, "mask": mask, "doy_sin": sin3, "doy_cos": cos3}
feat_ds  = xr.Dataset(all_vars)
feat_da  = feat_ds.to_array(dim="channel")          # (channel,time,lat,lon)
feat_da  = feat_da.transpose("time", "lat", "lon", "channel")

# ── 6 · write to Zarr (default compression/chunking) ────────────────────────
print(f"🔹 writing {DST.name} …")
feat_da.to_dataset(name="features").to_zarr(DST, mode="w")
print("✅  features.zarr written")

# ── 7 · save normalisation stats for later use ───────────────────────────────
with open(ROOT / "norm_stats.yml", "w") as f:
    yaml.safe_dump(norm_stats, f)
print("✅  norm_stats.yml written")
