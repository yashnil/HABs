#!/usr/bin/env python3
"""
make_coastal_strip.py  –  ocean pixels within R grid-cells of land
Writes : coastal_strip.zarr  (DataArray lat, lon)   uint8 {0,1}
"""
from pathlib import Path
import argparse, numpy as np, xarray as xr
from scipy.ndimage import distance_transform_edt

ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
FEAT_Z = ROOT / "features.zarr"
OUT_Z  = ROOT / "coastal_strip.zarr"

# ── CLI ----------------------------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument("--radius", type=int, default=5,
                 help="# grid cells from shoreline (default 5)")
args = cli.parse_args()
R = int(args.radius)

# ── load static ocean mask ----------------------------------------------------
feat      = xr.open_zarr(FEAT_Z)
ocean_da  = feat["features"].sel(channel="mask").isel(time=0)   # (lat,lon)
ocean     = ocean_da.values.astype(bool)                        # (H,W)

# ── distance field ------------------------------------------------------------
dist2land = distance_transform_edt(~ocean)      # every ocean-px → nearest land (cells)
strip     = (ocean & (dist2land <= R)).astype("uint8")

# ── wrap as DataArray  --------------------------------------------------------
strip_da = xr.DataArray(
    strip,
    coords=dict(
        # *** cast coords to float32 so they match labels.zarr ***  ↓↓↓↓
        lat = ocean_da.lat.astype("float32"),    #  <── key change
        lon = ocean_da.lon.astype("float32")     #  <── key change
    ),
    dims=("lat", "lon"),
    name="coastal_strip",
    attrs=dict(description=f"ocean pixels within {R} cells of land")
)

# make sure latitude is ascending (matches labels)
if strip_da.lat[0] > strip_da.lat[-1]:
    strip_da = strip_da.sortby("lat")

# ── write ---------------------------------------------------------------------
strip_da.chunk({"lat": -1, "lon": -1}).to_zarr(OUT_Z, mode="w")
print(f"✅ wrote {OUT_Z}  (radius = {R} cells, lat ascending)")
