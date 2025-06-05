#!/usr/bin/env python3
# rebuild_labels.py  – minimal, bullet-proof version
from pathlib import Path
import xarray as xr

ROOT = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")

feat  = xr.open_zarr(ROOT / "features.zarr")
mask  = xr.open_zarr(ROOT / "hab_mask.zarr")["hab_occurrence"].astype("uint8")

# ocean-only
ocean  = feat["features"].sel(channel="mask").isel(time=0)
label  = mask.reindex_like(ocean, fill_value=0).where(ocean == 1, 0)

# tidy coords / attrs
label = (label.astype("uint8")
               .assign_coords(time=label.time.astype("datetime64[ns]"),
                              lat =label.lat.astype("float32"),
                              lon =label.lon.astype("float32")))
label.attrs.clear()
for c in ("time", "lat", "lon"):
    label[c].attrs.clear()

label.name = "labels"                      # important!
label = label.chunk({"time": -1, "lat": -1, "lon": -1})

# absolutely **no** encoding hints
label.encoding.clear()
for c in label.coords:
    label[c].encoding.clear()

# write – plain uint8 chunks, no codecs
label.to_zarr(ROOT / "labels.zarr", mode="w", consolidated=False)
print("✅  labels.zarr written – shape", label.shape)
