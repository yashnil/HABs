#!/usr/bin/env python3
"""
quality_control/build_hab_mask.py   (writable-mask hot-fix)
"""
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr, geopandas as gpd
from shapely.geometry import Point

ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
CUBE   = ROOT / "root_dataset_filled.nc"
CSV    = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/bloomReportsCA.csv")
OUT    = ROOT / "hab_mask.zarr"
RADIUS = 0.02                                     # ≈ 2 km

def log(step, n): print(f"{step:<35s}: {n:6d}")

# ── grid & time ----------------------------------------------------------------
print("🔹 loading MODIS cube (lon, lat, time)…")
ds   = xr.open_dataset(CUBE)[["lon", "lat", "time"]]
lon1d, lat1d = ds.lon.values, ds.lat.values
lon2d, lat2d = np.meshgrid(lon1d, lat1d, indexing="xy")
time_index   = pd.DatetimeIndex(ds.time.values)           # 207 composites

# ── allocate *writable* numpy array ----------------------------------  ◀ NEW ▶
mask_np = np.zeros((len(time_index), len(lat1d), len(lon1d)), dtype="int8")

# ── CSV filters ---------------------------------------------------------------
print("🔹 reading bloom CSV …")
df0 = pd.read_csv(CSV, low_memory=False)
log("rows in raw CSV", len(df0))

df1 = df0.dropna(subset=["Bloom_Latitude", "Bloom_Longitude"])
log("after drop-na lat/lon", len(df1))

df1["date"] = pd.to_datetime(df1["Observation_Date"], errors="coerce")
df2 = df1.dropna(subset=["date"])
log("after valid date parse", len(df2))

tmin, tmax = time_index.min(), time_index.max()
df3 = df2[df2["date"].between(tmin, tmax)]
log(f"within {tmin.date()} … {tmax.date()}", len(df3))

lat_min, lat_max = lat1d.min(), lat1d.max()
lon_min, lon_max = lon1d.min(), lon1d.max()
df4 = df3[
    df3["Bloom_Latitude"].between(lat_min, lat_max) &
    df3["Bloom_Longitude"].between(lon_min, lon_max)
]
log("inside lon/lat bbox", len(df4))

print(f"\nSummary of filters → kept {len(df4)} reports\n")
if len(df4) == 0:
    raise SystemExit("🛑 0 rows left – adjust filters?")

# ── GeoFrame & time snap ------------------------------------------------------
gdf = gpd.GeoDataFrame(
        df4,
        geometry=gpd.points_from_xy(df4["Bloom_Longitude"], df4["Bloom_Latitude"]),
        crs="EPSG:4326",
)
gdf["t_index"] = time_index.get_indexer(gdf["date"], method="nearest")

# ── rasterise -------------------------------------------------------  ◀ NEW ▶
print("🔹 rasterising bloom points (≤2-km)…")
r2 = RADIUS**2
for ti, grp in gdf.groupby("t_index"):
    pts = np.vstack([grp.geometry.y.values, grp.geometry.x.values]).T
    dy  = pts[:, 0, None, None] - lat2d
    dx  = pts[:, 1, None, None] - lon2d
    hit = (dy**2 + dx**2).min(axis=0) <= r2
    mask_np[ti][hit] = 1                         # write to writable array

# ── wrap back into DataArray & save -------------------------------------------
mask = xr.DataArray(
        mask_np,
        dims=("time", "lat", "lon"),
        coords={"time": ds.time, "lat": lat1d, "lon": lon1d},
        name="hab_occurrence",
        attrs={"description": "1 if any bloom report within 2 km of pixel"},
)
print(f"✅ writing {OUT.name}   shape {mask.shape}")
mask.to_zarr(OUT, mode="w")
print("Done.")


# -----------------------------------------------------------------------------#

'''
Run Instruction: 

micromamba activate habs_env
python -m habs.quality_control.build_hab_mask
'''