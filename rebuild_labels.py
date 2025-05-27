#!/usr/bin/env python3
"""
rebuild_labels.py
-----------------
Re-write *labels.zarr* as a single uint8 array (ocean-only) with
**no filters / compression** so that later reads never trip the
VLenUTF-8 codec error.
"""

from pathlib import Path
import xarray as xr, numpy as np

ROOT = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
FEAT = xr.open_zarr(ROOT / "features.zarr")
MASK = xr.open_zarr(ROOT / "hab_mask.zarr")["hab_occurrence"].astype("uint8")

# ── keep only ocean pixels ----------------------------------------------------
ocean  = FEAT["features"].sel(channel="mask").isel(time=0)      # 1 = ocean
label  = MASK.reindex_like(ocean, fill_value=0).where(ocean == 1, 0)

# tidy coord dtypes
label = label.assign_coords(
    time=label.time.astype("datetime64[ns]"),
    lat =label.lat.astype("float32"),
    lon =label.lon.astype("float32"),
)

# wrap in Dataset so we can pass per-variable encoding
ds = label.to_dataset(name="labels").chunk({"time": -1, "lat": -1, "lon": -1})

encoding = {
    "labels": {                 # ← variable name
        "dtype"     : "uint8",
        "compressor": None,
        "filters"   : []        # disabling VLenUTF-8 does the trick
    }
}

ds.to_zarr(ROOT / "labels.zarr", mode="w", encoding=encoding)
print("✅  labels.zarr rebuilt (uint8, no filters)")
