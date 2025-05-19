#!/usr/bin/env python3
"""
Build clean 4-var MODIS stack on the ERA5/Copernicus grid (279×502, 2016-01-01 … 2021-06-23).

• Only uses dates where all 4 variables exist.
• Caches xESMF weights in ./cache/modis_to_target_weights.nc
• Writes  processed/modis_target.nc
"""

import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
import pathlib, re, glob
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 1) Paths + constants
BASE      = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed/modis_l3m")
ERA5_FP   = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed/era5_avg_sdswrf_8day_4km.nc")
OUT_NC    = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/processed/modis_target.nc")
CACHE     = pathlib.Path("./cache");  CACHE.mkdir(exist_ok=True)
WEIGHTS   = CACHE/"modis_to_target_weights.nc"

VAR_DIR = {
    "chlor_a": "chlorophyll",
    "Kd_490" : "kd490",
    "nflh"   : "nFLH",
    "sst"    : "seaSurfaceTemperature",
}
DATE_RE = re.compile(r"(\d{8})_\d{8}")

# ──────────────────────────────────────────────────────────────────────────────
# 2) find the 4 file-lists, intersect their dates
files = {v: sorted(glob.glob(str(BASE/dirn/"*_4km_L3m.nc")))
         for v,dirn in VAR_DIR.items()}
dates_per_var = {
    v: { DATE_RE.search(p).group(1): p
         for p in lst if DATE_RE.search(p) }
    for v,lst in files.items()
}
dates_all = sorted(set.intersection(*[set(dates_per_var[v]) for v in VAR_DIR]))
# restrict to your desired range
dates_all = [d for d in dates_all if "20160101" <= d <= "20210623"]
print(f"Kept {len(dates_all)} composite dates")

# ──────────────────────────────────────────────────────────────────────────────
# 3) build the ERA5→target grid Dataset
era = xr.open_dataset(ERA5_FP)
lon2d, lat2d = np.meshgrid(era.x, era.y)
TGT = xr.Dataset({
    "lon": (("y","x"), lon2d),
    "lat": (("y","x"), lat2d),
})

# ──────────────────────────────────────────────────────────────────────────────
# 4) build one xESMF regridder (caching weights)
first_date = dates_all[0]
sample_fp  = dates_per_var["chlor_a"][first_date]
sample_da  = xr.open_dataset(sample_fp)["chlor_a"].squeeze()

reuse = WEIGHTS.exists()
regridder = xe.Regridder(
    sample_da, TGT,
    method="bilinear",
    filename=str(WEIGHTS),
    reuse_weights=reuse,
)

# ──────────────────────────────────────────────────────────────────────────────
# 5) loop & regrid with a progress bar
stacks = []
for d in tqdm(dates_all, desc="MODIS → target grid"):
    dt = pd.to_datetime(d, format="%Y%m%d").to_datetime64()
    vars_out = {}
    for v in VAR_DIR:
        da = xr.open_dataset(dates_per_var[v][d])[v].squeeze()
        da_rg = regridder(da).expand_dims(time=[dt])
        vars_out[v] = da_rg
    stacks.append(xr.Dataset(vars_out))

ds_out = xr.concat(stacks, dim="time").sortby("time")

# ──────────────────────────────────────────────────────────────────────────────
# 6) write final NetCDF (zlib/compress)
print(f"\nWriting → {OUT_NC}")
ds_out.to_netcdf(
    OUT_NC,
    encoding={v:{"zlib":True,"complevel":4}
              for v in ds_out.data_vars}
)
print("✅ done")
