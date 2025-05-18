#!/usr/bin/env python3
"""
Concatenate mapped MODIS-Aqua 8-day L3m files (2016-2024) on a 4-km grid.
"""

import xarray as xr, pathlib, numpy as np, pandas as pd, sys, glob, datetime as dt

BASE = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed/modis_l3m")
OUT  = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/processed/modis_8day_4km_2016_2024.nc")
OUT.parent.mkdir(parents=True, exist_ok=True)

PRODUCTS = {
    "chlor_a": ("chlorophyll", "chlor_a"),
    "Kd_490":  ("kd490",       "Kd_490"),
    "nflh":    ("nFLH",        "nflh"),
    "sst":     ("seaSurfaceTemperature", "sst"),
}

T0, T1 = np.datetime64("2016-01-01"), np.datetime64("2024-12-31")

if OUT.exists():
    print("✔︎", OUT, "already exists — skipping")
    sys.exit()

all_ds = []
for var, (subdir, varname) in PRODUCTS.items():
    files = sorted((BASE / subdir).glob("*_4km_L3m.nc"))
    if not files:
        raise FileNotFoundError(f"No files in {BASE/subdir}")

    rasters = []
    for fp in files:
        ds = xr.open_dataset(fp, engine="netcdf4")
        # ---- extract composite start date from global attribute ----
        t_start_str = ds.attrs.get("time_coverage_start")
        if not t_start_str:
            print(f"  ⚠︎ no time_coverage_start in {fp.name}")
            continue
        t0 = np.datetime64(pd.to_datetime(t_start_str))
        if not (T0 <= t0 <= T1):
            continue

        if varname not in ds:
            print(f"  ⚠︎ variable {varname} not in {fp.name}")
            continue

        da = ds[varname].expand_dims(time=[t0]).astype("float32")
        rasters.append(da)

    print(f"⏳ {var}: kept {len(rasters)} composites in 2016-2024 window")
    if not rasters:
        raise RuntimeError(f"{var}: zero rasters after filtering")

    merged = xr.concat(rasters, dim="time").sortby("time")
    all_ds.append(merged.to_dataset(name=var))

print("🔗 merging 4 variables …")
xr.merge(all_ds, compat="override").to_netcdf(OUT)
print("✅ wrote", OUT)
