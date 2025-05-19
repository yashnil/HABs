
#!/usr/bin/env python3
"""
Diagnose NaNs in MODIS layers and rebuild them with align_utils.regrid_to_modis.
Also fixes HAB mask metadata.  Writes *_fixed2.nc* if improvements found.
"""

import xarray as xr, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from scripts.align_utils import regrid_to_modis

IN  = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_dataset_8day_4km_fixed.nc")
OUT = IN.with_name("HAB_dataset_8day_4km_fixed2.nc")

ds = xr.open_dataset(IN)

# -----------------------------------------------------------
def stats(da):
    return int(np.isfinite(da).sum()), int(np.isnan(da).sum())

for var in ["chlor_a", "Kd_490", "nflh", "sst"]:
    n_good, n_nan = stats(ds[var])
    print(f"{var:9s} before  good={n_good}  NaN={n_nan}")

    # --- open source MODIS file & snap its time like in build_dataset ---
    src = xr.open_dataset("/Users/yashnilmohanty/Desktop/HABs_Research/processed/modis_8day_4km_2016_2024.nc")[var]
    t   = pd.to_datetime(src.time.values)
    snapped = (t - pd.Timestamp("1970-01-01")) // pd.Timedelta("8D") * pd.Timedelta("8D") + pd.Timestamp("1970-01-01")
    src = src.assign_coords(time=snapped).drop_duplicates("time")

    # --- re-grid and align to ds.time with tolerance 2 days ---
    da_rg = regrid_to_modis(src)
    da_rg = da_rg.reindex_like(ds, method="nearest", tolerance="2D")

    n_good2, n_nan2 = stats(da_rg)
    print(f"             after   good={n_good2}  NaN={n_nan2}")
    if n_good2 > n_good:
        ds[var] = da_rg
        print("   ↳ replaced with re-gridded version")

# -----------------------------------------------------------
# Rebuild HAB mask to be sure it matches grid
print("\nRe-rasterising HAB CSV …")
csv = "/Users/yashnilmohanty/Desktop/HABs_Research/Data/bloomReportsCA.csv"
df  = pd.read_csv(csv, low_memory=False)
df["date"] = pd.to_datetime(df["Observation_Date"], errors="coerce").dt.floor("8D")
df = df.dropna(subset=["Bloom_Latitude","Bloom Longitude","date"])
df = df[df["date"].isin(ds.time.values)]

gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["Bloom Longitude"], df["Bloom_Latitude"])],
        crs="EPSG:4326"
)

lon2d, lat2d = np.meshgrid(ds.x, ds.y)
mask = xr.zeros_like(ds["chlor_a"], dtype=bool)

for dt, grp in gdf.groupby("date"):
    lats = grp.geometry.y.values
    lons = grp.geometry.x.values
    if lats.size == 0:
        continue
    dist2 = (lats[:,None,None]-lat2d)**2 + (lons[:,None,None]-lon2d)**2
    hit   = dist2.min(axis=0) <= (0.02**2)
    mask.loc[dict(time=dt)] = hit

ds["hab_occurrence"]       = mask
ds.hab_occurrence.attrs    = {"long_name": "HAB occurrence mask", "units": ""}

print("\n💾 writing", OUT.name)
enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
ds.to_netcdf(OUT, encoding=enc)
print("✅ done")
