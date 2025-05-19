#!/usr/bin/env python3
import xarray as xr
import numpy as np
import xesmf as xe
from pathlib import Path

# ── adjust these to your paths ────────────────────────────────────────────────
BASE   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
MODIS  = BASE / "modis_target.nc"
ERA5   = BASE / "era5_8day.nc"
CMEMS  = BASE / "cmems_8day.nc"
OUT    = BASE / "root_dataset.nc"

# ── 1) load all three datasets (with CF‐time decoding) ─────────────────────────
print("→ loading MODIS…")
ds_modis = xr.open_dataset(MODIS, decode_times=True)

print("→ loading ERA5…")
ds_era5 = xr.open_dataset(ERA5, decode_times=True)

print("→ loading CMEMS…")
ds_cmems = xr.open_dataset(CMEMS, decode_times=True)

# ── 2) rename the MODIS dims and give it real lon/lat coords ──────────────────
#    it was (time, y, x); CF wants (time, lat, lon)
ds_modis = ds_modis.rename({"x": "lon", "y": "lat"})

# derive the target lon/lat extents from your ERA5 file:
lon_min, lon_max = float(ds_era5.lon.min()), float(ds_era5.lon.max())
lat_min, lat_max = float(ds_era5.lat.min()), float(ds_era5.lat.max())

# get the MODIS grid size:
nlon = ds_modis.sizes["lon"]
nlat = ds_modis.sizes["lat"]

# build evenly spaced lon/lat arrays:
lon1d = np.linspace(lon_min, lon_max, nlon)
# ERA5 lat may run from north→south; we want lat max→lat min
lat1d = np.linspace(lat_max, lat_min, nlat)

# attach them
ds_modis = ds_modis.assign_coords({
    "lon": ("lon", lon1d),
    "lat": ("lat", lat1d),
})

# ── 3) build a 1D target‐grid for xESMF using the same coords ─────────────────
target_grid = xr.Dataset({
    "lon": ("lon", lon1d),
    "lat": ("lat", lat1d),
})

# ── 4) regrid ERA5 → MODIS grid (bilinear) ───────────────────────────────────
print("→ regridding ERA5 onto MODIS grid…")
re_e = xe.Regridder(ds_era5, target_grid, method="bilinear", periodic=False)
era5_on_modis = re_e(ds_era5)
# align times exactly to the MODIS time axis
era5_on_modis = era5_on_modis.reindex(time=ds_modis.time)

# ── 5) regrid CMEMS → MODIS grid (bilinear) ──────────────────────────────────
print("→ regridding CMEMS onto MODIS grid…")
re_c = xe.Regridder(ds_cmems, target_grid, method="bilinear", periodic=False)
cmems_on_modis = re_c(ds_cmems)
cmems_on_modis = cmems_on_modis.reindex(time=ds_modis.time)

# ── 6) merge all three stacks into one Dataset ────────────────────────────────
print("→ merging MODIS + ERA5 + CMEMS…")
ds_root = xr.merge([ds_modis, era5_on_modis, cmems_on_modis])

# ── 7) write out your final root_dataset.nc ───────────────────────────────────
print(f"→ writing merged dataset to {OUT!r}")
ds_root.to_netcdf(OUT)
print("✅ Done.")
