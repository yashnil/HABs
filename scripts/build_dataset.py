#!/usr/bin/env python3
"""
Merge MODIS, ERA-5, CMEMS (all 8-day × 4 km) and rasterise HAB CSV.
Only keep dates common to *all* sources; no NaN padding.
"""

import xarray as xr, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
import pathlib, sys, warnings

PROC = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/processed")
OUT  = PROC / "HAB_dataset_8day_4km_common.nc"

if OUT.exists():
    print("✔︎", OUT, "already exists — skipping")
    sys.exit()

# ------------------------------------------------------------------
def load_and_normalise(p):
    """Open NetCDF, ensure time is datetime64[ns], snap to 8-day left edge,
    drop duplicate times."""
    ds = xr.open_dataset(p, decode_times=False)    # raw load
    if "time" not in ds:
        warnings.warn(f"{p.name} has no time coord – skipped")
        return None

    # decode CF units → datetime64
    ds = xr.decode_cf(ds)

    # snap to 8-day boundaries (left)
    ts = pd.to_datetime(ds.time.values)
    snapped = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("8D") * pd.Timedelta("8D") + pd.Timestamp("1970-01-01")
    ds = ds.assign_coords(time=snapped)

    # drop duplicates, keep first
    _, index = np.unique(ds.time.values, return_index=True)
    ds = ds.isel(time=index)

    return ds

print("⏳ loading and normalising NetCDFs …")
paths = sorted(PROC.glob("*_8day_4km*.nc"))
datasets = [d for p in paths if (d := load_and_normalise(p))]

# ------------------------------------------------------------------
# intersection of times
common_time = sorted(set.intersection(*[set(ds.time.values) for ds in datasets]))
if not common_time:
    print("❌ still no common dates after normalisation.")
    for ds, p in zip(datasets, paths):
        print(f"{p.name:32s}: {len(ds.time)} dates  "
              f"({str(ds.time.min().values)[:10]} … {str(ds.time.max().values)[:10]})")
    sys.exit(1)

t0, t1 = common_time[0], common_time[-1]
print(f"✅ common axis: {len(common_time)} dates ({str(t0)[:10]} … {str(t1)[:10]})")

merged = xr.merge([ds.sel(time=common_time) for ds in datasets],
                  compat="override")

# ------------------------------------------------------------------
# Rasterise HAB CSV  →  hab_occurrence bool
csv = "/Users/yashnilmohanty/Desktop/HABs_Research/Data/bloomReportsCA.csv"
df  = pd.read_csv(csv, low_memory=False)
df["date"] = pd.to_datetime(df["Observation_Date"], errors="coerce").dt.floor("8D")
df = df.dropna(subset=["Bloom_Latitude", "Bloom Longitude", "date"])
df = df[(df["date"].isin(common_time)) &
        df["Bloom_Latitude"].between(32, 50) &
        df["Bloom Longitude"].between(-125, -115)]

gdf = gpd.GeoDataFrame(df,
        geometry=[Point(xy) for xy in zip(df["Bloom Longitude"], df["Bloom_Latitude"])],
        crs="EPSG:4326")

lon2d, lat2d = np.meshgrid(merged.lon, merged.lat)
mask = xr.zeros_like(merged.chlor_a, dtype=bool)

for dt, grp in gdf.groupby("date"):
    lats = grp.geometry.y.values        # 1-D array of latitudes
    lons = grp.geometry.x.values        # 1-D array of longitudes
    if lats.size == 0:
        continue
    dist2 = (lats[:, None, None] - lat2d) ** 2 + (lons[:, None, None] - lon2d) ** 2
    hit   = dist2.min(axis=0) <= (0.02 ** 2)   # within half-cell (~2 km)
    mask.loc[dict(time=dt)] = hit | mask.sel(time=dt)

merged["hab_occurrence"] = mask

# ------------------------------------------------------------------
enc = {v: {"zlib": True, "complevel": 4} for v in merged.data_vars}
merged.to_netcdf(OUT, encoding=enc)
print("💾 wrote", OUT)
