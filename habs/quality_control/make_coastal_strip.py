#!/usr/bin/env python3
"""
make_coastal_strip.py
---------------------
Builds a coastal-strip mask that *always* includes the shoreline row.

Output:  coastal_strip.zarr   uint8  (lat, lon)
"""
from pathlib import Path
import argparse, numpy as np, xarray as xr
from scipy.ndimage import distance_transform_edt as dist

ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
FEAT_Z = ROOT / "features.zarr"          # has static land/sea mask
OUT_Z  = ROOT / "coastal_strip.zarr"

# ── CLI ---------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--radius", type=int, default=5,
               help="strip half-width in grid cells (default 5)")
R = p.parse_args().radius

# ── load static mask (1 = ocean, 0 = land) ----------------------------------
feat   = xr.open_zarr(FEAT_Z)
mask_da = feat["features"].sel(channel="mask").isel(time=0)  # (lat,lon)
ocean   = mask_da.values.astype(bool)                        # bool array

# ── distance to land and to ocean -------------------------------------------
dist2land  = dist(~ocean)   # ocean → nearest land
dist2ocean = dist(ocean)    # land  → nearest ocean

# Pixel is inside strip if ***either***
#   (a) ocean-pixel closer than R to land   OR
#   (b) land-pixel closer than R to ocean   ← adds the shoreline row
strip_bool = ((ocean & (dist2land  <= R)) |
              (~ocean & (dist2ocean <= R)))

strip_da = xr.DataArray(
    strip_bool.astype("uint8"),
    coords=dict(lat=mask_da.lat, lon=mask_da.lon),
    dims=("lat", "lon"),
    name="coastal_strip",
    attrs={"description": f"all pixels ≤{R} cells from shoreline"}
).sortby("lat")                         # lat ascending

strip_da.chunk({"lat": -1, "lon": -1}).to_zarr(OUT_Z, mode="w")
print(f"✅ wrote {OUT_Z}  (radius = {R} cells, shoreline assured)")
