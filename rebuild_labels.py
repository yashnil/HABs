#!/usr/bin/env python3
# rebuild_labels.py  – make *labels.zarr* a plain uint8 array — no codecs
from pathlib import Path
import xarray as xr

ROOT = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")

feat  = xr.open_zarr(ROOT / "features.zarr")
mask  = xr.open_zarr(ROOT / "hab_mask.zarr")["hab_occurrence"].astype("uint8")

# ── ocean-only label cube ----------------------------------------------------
ocean = feat["features"].sel(channel="mask").isel(time=0)           # 1 = ocean
label = mask.reindex_like(ocean, fill_value=0).where(ocean == 1, 0)

# tidy coords / attrs
label = (label
         .assign_coords(time=label.time.astype("datetime64[ns]"),
                        lat =label.lat.astype("float32"),
                        lon =label.lon.astype("float32"))
         .astype("uint8"))
label.attrs.clear()
for c in ("time", "lat", "lon"):
    label.coords[c].attrs.clear()

label.name = "labels"                           # give the variable a name
label = label.chunk({"time": -1, "lat": -1, "lon": -1})

# ── WRITE – turn **everything** off ------------------------------------------
label.to_zarr(
    ROOT / "labels.zarr",
    mode="w",
    consolidated=False,
    encoding={
        "labels": {
            "compressor": None,
            "filters"   : [],       # <- prevent VLenUTF-8
            "_FillValue": None      # <- DO NOT store 0/1 as a fill-value
        }
    }
)

print("✅  labels.zarr rebuilt – uint8, no codecs, no _FillValue")
