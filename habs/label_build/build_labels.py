#!/usr/bin/env python3
"""
03 – Build labels.zarr
----------------------
Creates a (time, lat, lon) uint8 cube aligned to features.zarr
 0 = no report 1 = bloom reported (nearest 8-day composite)

Run
----
micromamba activate habs_env
python -m habs.label_build.build_labels --ocean_only      # coastal only
"""

from pathlib import Path
import argparse, xarray as xr, numpy as np

ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
FEAT_Z = ROOT / "features.zarr"
MASK_Z = ROOT / "hab_mask.zarr"
OUT_Z  = ROOT / "labels.zarr"

# ------------------------------------------------------------------ CLI ------
ap = argparse.ArgumentParser()
ap.add_argument("--ocean_only", action="store_true",
                help="zero inland pixels via the static ocean-mask channel")
args = ap.parse_args()

# ------------------------------------------------------------------ grid -----
feat  = xr.open_zarr(FEAT_Z)
grid  = feat["features"].isel(channel=0)                # steal axes/coords

# ------------------------------------------------------------------ mask → DA
label = (xr.open_zarr(MASK_Z)["hab_occurrence"]
           .reindex_like(grid, fill_value=0)            # align
           .astype("uint8"))

if args.ocean_only:
    ocean = feat["features"].sel(channel="mask").isel(time=0)
    label = label.where(ocean == 1, 0)

# tidy coord dtypes & **strip all attrs**
label = (
    label.assign_coords({
        "time": label.time.values.astype("datetime64[ns]"),
        "lat" : label.lat.values.astype("float32"),
        "lon" : label.lon.values.astype("float32"),
    })
)
label.attrs.clear()                           # remove variable-level strings

print("label cube :", tuple(label.shape), label.dtype,
      "(ocean-only)" if args.ocean_only else "(all water)")

# ------------------------------------------------------------------ write ----
# one big chunk/axis → fast & avoids mixed-size-chunk error
'''
label.chunk({"time": -1, "lat": -1, "lon": -1}).to_zarr(
    OUT_Z, mode="w",
    encoding={"dtype": "uint8", "compressor": None, "filters": None}
)
'''

print("✅ wrote", OUT_Z)

'''
To run:

micromamba activate habs_env
python -m habs.label_build.build_labels --ocean_only   # coastal only

'''