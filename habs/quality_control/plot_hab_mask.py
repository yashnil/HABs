#!/usr/bin/env python3
"""
quality_control/plot_hab_mask.py
--------------------------------
Visual sanity-check of the HAB mask written by build_hab_mask.py.
All pixels that were flagged ≥ 1 time-step are plotted as red dots.

Run:
    python -m habs.quality_control.plot_hab_mask
"""
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs            # make sure cartopy is installed
import numpy as np

ROOT = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
MASK = ROOT / "hab_mask.zarr"         # written by build_hab_mask.py

# ── load mask cube ────────────────────────────────────────────────────────────
da = xr.open_zarr(MASK)["hab_occurrence"]          # (time, lat, lon)

# any pixel hit at least once?
hit_any = da.any("time")                           # bool (lat,lon)

# indices of True pixels
lat_vec = hit_any["lat"].values            # 1-D
lon_vec = hit_any["lon"].values            # 1-D
ilat, ilon = np.where(hit_any.values)      # indices where mask==True
lats = lat_vec[ilat]
lons = lon_vec[ilon]
print(f"Plotting {lats.size} bloom pixels")

# ── quick plot ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 7))
ax  = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_extent([-126, -114, 31, 51])                # CA coast bbox
ax.scatter(lons, lats, s=5, c="red", transform=ccrs.PlateCarree())
plt.title("Pixels flagged ≥1 time (2016-01-09 … 2021-06-23)")
plt.tight_layout()
plt.show()
'''
To run:

python -m habs.quality_control.plot_hab_mask

Results:


'''