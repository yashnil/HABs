#!/usr/bin/env python3
"""
Merge MODIS, CMEMS & ERA5 into one aligned 8-day, 4 km dataset.
Writes processed/HAB_dataset_root.nc
"""

import xarray as xr
import numpy as np
from pathlib import Path

# ─── paths ────────────────────────────────────────────────────────────────
ROOT      = Path("/Users/yashnilmohanty/Desktop/HABs_Research")
MODIS_NC  = ROOT/"processed"/"modis_target.nc"
CMEMS_NC  = ROOT/"Processed"/"cmems_so_8day_4km.nc"
ERA5_GLOB = str(ROOT/"Processed"/"era5_*_8day_4km.nc")
OUT_NC    = ROOT/"processed"/"HAB_dataset_root.nc"

# ─── load each source ─────────────────────────────────────────────────────
ds_modis = xr.open_dataset(MODIS_NC)
ds_cmems = xr.open_dataset(CMEMS_NC)
ds_era5  = xr.open_mfdataset(ERA5_GLOB, combine="by_coords", parallel=True)

# ─── compute common time axis ─────────────────────────────────────────────
t_modis = ds_modis.time.values
t_cmems = ds_cmems.time.values
t_era5  = ds_era5.time.values

# intersection of all three
common = np.intersect1d(t_modis, t_cmems)
common = np.intersect1d(common,  t_era5)
common = np.sort(common)

print(f"→ common time length: {len(common)}")

# ─── re-select each dataset on that common time ───────────────────────────
ds_modis = ds_modis.sel(time=common)
ds_cmems = ds_cmems.sel(time=common)
ds_era5  = ds_era5.sel(time=common)

# ─── merge them all ───────────────────────────────────────────────────────
ds = xr.merge([ds_modis, ds_cmems, ds_era5], compat="override")

# ─── write compressed NetCDF ──────────────────────────────────────────────
encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
ds.to_netcdf(OUT_NC, encoding=encoding)

print("✅ merged dataset written to", OUT_NC)
