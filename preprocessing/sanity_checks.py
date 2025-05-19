#!/usr/bin/env python3
import xarray as xr, numpy as np, matplotlib.pyplot as plt

DATA = "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_dataset_8day_4km_fixed.nc"
ds   = xr.open_dataset(DATA)

print(ds)

# ------------------------------------------------------------------
# 0. summaries for variables that are really present
vars_to_check = ["chlor_a","Kd_490","sst","so","uo","v10"]
for v in vars_to_check:
    da = ds[v]
    print(f"{v:10s}  min={float(da.min()):.3f}  max={float(da.max()):.3f}  NaNs={int(da.isnull().sum())}")

# ------------------------------------------------------------------
def quick_map(da, t=0, title=""):
    """Plot whatever grid dims the DataArray has."""
    ax = plt.figure(figsize=(5,3.8)).gca()
    if {"lat","lon"}.issubset(da.dims):
        da.isel(time=t).plot(ax=ax, cmap="viridis",
            vmin=float(da.quantile(0.02)), vmax=float(da.quantile(0.98)),
            x="lon", y="lat", cbar_kwargs={"shrink":0.6})
    else:                       # uses y,x
        da.isel(time=t).plot(ax=ax, cmap="viridis",
            vmin=float(da.quantile(0.02)), vmax=float(da.quantile(0.98)),
            x="x", y="y", cbar_kwargs={"shrink":0.6})
    ax.set_title(title)
    plt.tight_layout()

quick_map(ds.chlor_a, 0, "chlor_a 2016-01-01")
quick_map(ds.sst,     0, "sst 2016-01-01")
quick_map(ds.so,      0, "salinity 2016-01-01")

# HAB mask overlay
plt.figure(figsize=(5,3.8))
if {"lat","lon"}.issubset(ds.hab_occurrence.dims):
    ds.hab_occurrence.isel(time=0).plot(x="lon", y="lat", cmap="Reds", add_colorbar=False)
else:
    ds.hab_occurrence.isel(time=0).plot(x="x", y="y", cmap="Reds", add_colorbar=False)
plt.title("HAB occurrence 2016-01-01")
plt.tight_layout()
plt.show()
