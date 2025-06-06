#!/usr/bin/env python3
"""
count_strip_hits.py  –  diagnostics after building coastal_strip.zarr

Reports how many HAB-positive pixels fall inside / outside the coastal strip.
"""

from pathlib import Path
import xarray as xr
import numpy as np

ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
LBL_Z  = ROOT / "labels.zarr"          # Dataset with var 'labels'
STRIP  = ROOT / "coastal_strip.zarr"   # DataArray uint8 (lat, lon)

# count_strip_hits.py  (only the middle changes)
labels = xr.open_dataarray(LBL_Z)          # (time, lat, lon)
strip  = xr.open_dataarray(STRIP)          # (lat, lon)

# --- make coordinates identical ---------------------------------------------
strip = strip.reindex_like(
            labels.isel(time=0),           # pick any time-slice => (lat,lon)
            method="nearest", tolerance=1e-6
        )

hits_total  = int((labels == 1).sum())
hits_strip  = int(((labels == 1) & (strip == 1)).sum())
hits_inland = hits_total - hits_strip

print(f"total HAB-positive pixels : {hits_total:,}")
print(f"⋅ inside coastal strip    : {hits_strip:,}")
print(f"⋅ inland / offshore       : {hits_inland:,}")
