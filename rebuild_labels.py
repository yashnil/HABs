#!/usr/bin/env python3
# rebuild_labels.py  –  final “just-work” version
from pathlib import Path
import xarray as xr

ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
feat   = xr.open_zarr(ROOT / "features.zarr")
mask   = xr.open_zarr(ROOT / "hab_mask.zarr")["hab_occurrence"].astype("uint8")

# ── ocean-only ────────────────────────────────────────────────────────────────
ocean  = feat["features"].sel(channel="mask").isel(time=0)    # 1 = ocean
label  = mask.reindex_like(ocean, fill_value=0).where(ocean == 1, 0)

# tidy coords / strip attrs
label = label.assign_coords(
    time=label.time.astype("datetime64[ns]"),
    lat =label.lat.astype("float32"),
    lon =label.lon.astype("float32"),
)
label.attrs.clear()
for c in ("time", "lat", "lon"):
    label[c].attrs.clear()

# ── wrap & scrub ALL encodings ────────────────────────────────────────────────
ds = label.to_dataset(name="labels").chunk({"time": -1, "lat": -1, "lon": -1})

# remove any auto-generated encoding
for k in ds["labels"].encoding:               # build a fresh, empty dict
    ds["labels"].encoding[k] = None
ds["labels"].encoding.update({"dtype": "uint8"})

# ── write – no filters, no compressor, unconsolidated ────────────────────────
(ds
 .to_zarr(ROOT / "labels.zarr",
          mode="w",
          consolidated=False)                 # <- avoids metadata warnings
)

print("✅  labels.zarr rebuilt (bare uint8 array, no VLen codec)")
