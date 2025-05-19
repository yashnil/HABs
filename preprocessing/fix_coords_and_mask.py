#!/usr/bin/env python3
"""
Fix coord mismatch and HAB mask in merged cube, output *_fixed.nc*
"""

import xarray as xr, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
import pathlib, re
from scripts.align_utils import regrid_to_modis

IN  = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_dataset_8day_4km_clean.nc")
OUT = IN.with_name("HAB_dataset_8day_4km_fixed.nc")

ds = xr.open_dataset(IN)

# ------------------------------------------------------------------
# 1. Promote 2-D lon/lat to coords on y,x grid
lon2d, lat2d = np.meshgrid(ds.x, ds.y)
ds = ds.assign_coords(lon=(("y","x"), lon2d),
                      lat=(("y","x"), lat2d))

# 2. Interpolate MODIS variables (lat,lon) onto y,x grid
for var in ["chlor_a","Kd_490","nflh","sst"]:
    if {"lat","lon"}.issubset(ds[var].dims):
        print(f"regridding {var} from (lat,lon) → (y,x)")
        da = ds[var].interp(
                lon=ds.lon, lat=ds.lat,
                kwargs={"bounds_error": False})
        ds[var] = da.transpose("time","y","x")

# 3. Drop obsolete lat/lon dims & vars
ds = ds.drop_vars(["lat","lon"]).swap_dims({"y":"y","x":"x"})

# ------------------------------------------------------------------
# 4. Re-rasterise HAB mask on y,x grid
csv = "/Users/yashnilmohanty/Desktop/HABs_Research/Data/bloomReportsCA.csv"
df  = pd.read_csv(csv, low_memory=False)
df["date"] = pd.to_datetime(df["Observation_Date"], errors="coerce").dt.floor("8D")
df = df.dropna(subset=["Bloom_Latitude","Bloom Longitude","date"])
df = df[df["date"].isin(ds.time.values)]

gdf = gpd.GeoDataFrame(
        df, geometry=[Point(xy) for xy in zip(df["Bloom Longitude"], df["Bloom_Latitude"])],
        crs="EPSG:4326"
)

mask = xr.zeros_like(ds["chlor_a"], dtype=bool)
for dt, grp in gdf.groupby("date"):
    lats = grp.geometry.y.values
    lons = grp.geometry.x.values
    dist2 = (lats[:,None,None]-lat2d)**2 + (lons[:,None,None]-lon2d)**2
    hit   = dist2.min(axis=0) <= (0.02**2)
    mask.loc[dict(time=dt)] = hit

ds["hab_occurrence"] = mask.astype("bool")

# ------------------------------------------------------------------
ds.to_netcdf(OUT, encoding={v:{"zlib":True,"complevel":4} for v in ds.data_vars})
print("💾 wrote fixed cube to", OUT)
