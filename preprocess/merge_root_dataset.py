#!/usr/bin/env python3
import xarray as xr
import numpy as np
import xesmf as xe
from pathlib import Path

# ── adjust these to your actual paths ─────────────────────────────────────────
BASE  = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
MODIS = BASE / "modis_target.nc"
ERA5  = BASE / "era5_8day.nc"
CMEMS = BASE / "cmems_8day.nc"
OUT   = BASE / "root_dataset.nc"

# ── 1) load in CF‐time mode ─────────────────────────────────────────────────────
print("→ loading datasets…")
ds_modis = xr.open_dataset(MODIS, decode_times=True)
ds_era5  = xr.open_dataset(ERA5,  decode_times=True)
ds_cmems = xr.open_dataset(CMEMS, decode_times=True)

# ── 2) rename MODIS dims x,y → lon,lat ─────────────────────────────────────────
ds_modis = ds_modis.rename({"x": "lon", "y": "lat"})

# ── 3) derive common lon/lat bounds & shape from ERA5 ─────────────────────────
lon_min, lon_max = ds_era5.lon.min().item(), ds_era5.lon.max().item()
lat_min, lat_max = ds_era5.lat.min().item(), ds_era5.lat.max().item()

nlat = ds_modis.sizes["lat"]
nlon = ds_modis.sizes["lon"]

# ── 4) build *ascending* coordinate vectors ────────────────────────────────────
lon1d = np.linspace(lon_min, lon_max, nlon)
lat1d = np.linspace(lat_min, lat_max, nlat)

# ── 5) assign those to MODIS ───────────────────────────────────────────────────
ds_modis = ds_modis.assign_coords(lon=("lon", lon1d),
                                  lat=("lat", lat1d))

# if lat ended up descending, force it ascending:
if ds_modis.lat.values[1] < ds_modis.lat.values[0]:
    ds_modis = ds_modis.sortby("lat")

# ── 6) build xESMF target grid ─────────────────────────────────────────────────
target_grid = xr.Dataset({
    "lon": ("lon", lon1d),
    "lat": ("lat", lat1d),
})

# ── 7) regrid ERA5 → MODIS grid (bilinear) ────────────────────────────────────
print("→ regridding ERA5 onto MODIS grid…")
re_e = xe.Regridder(ds_era5, target_grid, method="bilinear", periodic=False)
era5_on = re_e(ds_era5)
era5_on = era5_on.reindex(time=ds_modis.time)

# ── 8) regrid CMEMS → MODIS grid (bilinear) ───────────────────────────────────
print("→ regridding CMEMS onto MODIS grid…")
re_c = xe.Regridder(ds_cmems, target_grid, method="bilinear", periodic=False)
cmems_on = re_c(ds_cmems)
cmems_on = cmems_on.reindex(time=ds_modis.time)

# ── 9) merge everything & write out ────────────────────────────────────────────
print("→ merging all variables…")
ds_root = xr.merge([ds_modis, era5_on, cmems_on])

print(f"→ writing merged dataset to {OUT!r}")
ds_root.to_netcdf(OUT)
print("✅ Done.")
