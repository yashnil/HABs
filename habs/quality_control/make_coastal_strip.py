
#!/usr/bin/env python3
"""
make_coastal_strip.py
=====================

Build a boolean mask that keeps only the ocean pixels lying within *N* grid
cells of the land–sea interface (“coastal strip”).  
The result is written as  ``coastal_strip.zarr``  (DataArray: lat, lon).

Typical use
-----------
# 2-pixel strip (recommended first try)
python -m habs.quality_control.make_coastal_strip --radius 2
"""
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import argparse, numpy as np, xarray as xr
from scipy.ndimage import distance_transform_edt

ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
FEAT_Z = ROOT / "features.zarr"          # built earlier
OUT_Z  = ROOT / "coastal_strip.zarr"

# ---------------------------------------------------------------- CLI ---------
cli = argparse.ArgumentParser()
cli.add_argument("--radius", type=int, default=2,
                 help="# grid cells from shoreline to keep (default 2)")
args = cli.parse_args()
R = int(args.radius)

# ---------------------------------------------------------------- load mask ---
feat = xr.open_zarr(FEAT_Z)
# static ocean mask: 1 = ocean, 0 = land    (any time-slice is fine)
ocean_da = feat["features"].sel(channel="mask").isel(time=0)       # (lat,lon)
ocean = ocean_da.values.astype(bool)                               # (H,W)

# ---------------------------------------------------------------- distance ----
# distance *from every ocean pixel* to the nearest land pixel (in grid cells)
dist2land = distance_transform_edt(~ocean)         # ocean→nearest land
# strip = ocean pixels whose dist ≤ R
strip = (ocean & (dist2land <= R)).astype("uint8") # uint8 -> tiny on disk

strip_da = xr.DataArray(
    strip,
    coords=dict(lat=ocean_da.lat, lon=ocean_da.lon),
    dims=("lat", "lon"),
    name="coastal_strip",
    attrs=dict(description=f"ocean pixels within {R} cells of land")
)

# ---------------------------------------------------------------- write -------
strip_da.chunk({"lat": -1, "lon": -1}).to_zarr(OUT_Z, mode="w")
print(f"✅ wrote {OUT_Z}  (radius = {R} cells)")
