#!/usr/bin/env python3
"""
Resample 6-hourly ERA-5 fields to 8-day means and re-grid to
the MODIS 4 km Plate-Carrée grid, for the 2016-2024 window.

Output → one NetCDF per variable in
  .../processed/era5_<var>_8day_4km.nc
"""

from align_utils import to_datetime, resample_8day, regrid_to_modis
import xarray as xr, os, pathlib

ERA_DIR   = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/era5")
OUT_DIR   = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_VARS = {
    "tp"         : "data_stream-oper_stepType-accum.nc",
    "avg_sdswrf" : "data_stream-oper_stepType-avg.nc",
    "u10"        : "data_stream-oper_stepType-instant copy.nc",
    "v10"        : "data_stream-oper_stepType-instant copy.nc",
    "t2m"        : "data_stream-oper_stepType-instant.nc",
    "d2m"        : "data_stream-oper_stepType-instant.nc",
}

for v, fname in ERA_VARS.items():
    out = OUT_DIR / f"era5_{v}_8day_4km.nc"
    if out.exists():
        print(f"✔︎ {out.name} exists — skip")
        continue

    src = ERA_DIR / fname
    if not src.is_file():
        raise FileNotFoundError(src)

    print(f"⏳ processing {v} from {fname}")
    ds = (xr.open_dataset(src, engine="netcdf4", decode_times=False)
            .rename({"valid_time": "time"}))

    ds = to_datetime(ds, "time").sel(time=slice("2016-01-01", "2024-12-31"))

    da8 = resample_8day(ds[v])          # 8-day mean
    da8 = da8.rename({"latitude": "lat", "longitude": "lon"})
    da4 = regrid_to_modis(da8)          # 4 km grid

    da4.to_netcdf(out)
    print(f"✅ wrote {out.name}")
