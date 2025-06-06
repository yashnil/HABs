#!/usr/bin/env python3
"""
build_coastal_labels.py
-----------------------
Make coastal-only HAB labels:

    labels        (time, lat, lon)  uint8  {0,1}
        ⤷ mask with coastal_strip   (lat, lon)   uint8 {0,1}

Result →  coastal_labels.zarr  (DataArray)

Run:
    python -m habs.label_build.build_coastal_labels
"""
from pathlib import Path
import shutil, xarray as xr, numpy as np

ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
LBL_Z  = ROOT / "labels.zarr"
STRIP  = ROOT / "coastal_strip.zarr"
OUT_Z  = ROOT / "coastal_labels.zarr"

# ── 1. load ------------------------------------------------------------------
labels_da = xr.open_dataarray(LBL_Z)          # (time, lat, lon)  uint8
strip_da  = xr.open_dataarray(STRIP)          # (lat,  lon)       uint8

# ensure coords line up exactly
strip_da = strip_da.reindex_like(labels_da.isel(time=0))

# ── 2. mask : keep only strip pixels -----------------------------------------
coastal = labels_da.where(strip_da == 1, 0).astype("uint8")
coastal.name = "coastal_labels"

# ── 3. write cleanly ----------------------------------------------------------
if OUT_Z.exists():
    shutil.rmtree(OUT_Z)          # nuke old store to avoid stale vars

coastal.chunk({"time": -1, "lat": -1, "lon": -1}).to_zarr(
    OUT_Z, mode="w", consolidated=False
)
print(f"✅ wrote {OUT_Z}   shape {coastal.shape}")
