
#!/usr/bin/env python3
"""
debug_strip_vs_labels.py
———————————————
Prints the exact lat/lon alignment status between

  • labels.zarr   (time, lat, lon)   uint8
  • coastal_strip.zarr (lat, lon)    uint8

Run:
    python -m habs.quality_control.debug_strip_vs_labels
"""
from pathlib import Path
import numpy as np, xarray as xr, pandas as pd

ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
LBL_Z  = ROOT / "labels.zarr"
STRIP  = ROOT / "coastal_strip.zarr"

lbl    = xr.open_dataarray(LBL_Z, consolidated=False)   # (time,lat,lon)
strip  = xr.open_dataarray(STRIP)                       # (lat,lon)

print("▶ coord dtypes",
      dict(lat_lbl=lbl.lat.dtype, lon_lbl=lbl.lon.dtype,
           lat_strip=strip.lat.dtype, lon_strip=strip.lon.dtype), "\n")

# 1) length check -------------------------------------------------------------
print("▶ len(lat), len(lon) :", len(lbl.lat), len(strip.lat),
      "|", len(lbl.lon), len(strip.lon))

# 2) how many coordinates match *exactly* -------------------------------------
lat_match = np.isclose(lbl.lat.values, strip.lat.values, rtol=0, atol=0).sum()
lon_match = np.isclose(lbl.lon.values, strip.lon.values, rtol=0, atol=0).sum()
print(f"▶ exact matches      : lat {lat_match}/{len(lbl.lat)}  |  "
      f"lon {lon_match}/{len(lbl.lon)}")

# 3) first few rows to see if one array is flipped ----------------------------
print("\nfirst 5 lat (labels) :", lbl.lat.values[:5])
print("first 5 lat (strip)  :", strip.lat.values[:5])
print("first 5 lon (labels) :", lbl.lon.values[:5])
print("first 5 lon (strip)  :", strip.lon.values[:5])

# 4) summary of where strip==1 and labels==1 overlap AFTER reindex -----------
aligned = strip.reindex_like(lbl.isel(time=0), method="nearest", tolerance=1e-6)
overlap = ((aligned == 1) & (lbl.isel(time=0) == 1)).sum().item()
print("\n▶ overlap in first composite :", overlap)
