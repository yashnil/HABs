#!/usr/bin/env python3
"""
Quick visual sanity-check of labels.zarr

* randomly chooses N composites
* shows only the **positive pixels** (transparent elsewhere)
* optional --save flag writes PNGs next to labels.zarr
"""
from pathlib import Path
import argparse, random, numpy as np, xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
LABELS = ROOT / "labels.zarr"

# ── CLI ----------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--n",    type=int, default=4, help="# composites to plot")
ap.add_argument("--save", action="store_true", help="write PNG instead of show")
args = ap.parse_args()

da = xr.open_dataarray(LABELS, consolidated=False)          # (time,lat,lon)
N  = min(args.n, len(da.time))
times = random.sample(list(da.time.values), k=N)

# two-colour (transparent / red) map
cmap = ListedColormap([(0,0,0,0), "red"])
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

for t in times:
    slice2d  = da.sel(time=t)
    yy, xx   = np.where(slice2d.values == 1)          # indices of positives
    lats     = slice2d.lat.values[yy]
    lons     = slice2d.lon.values[xx]

    fig = plt.figure(figsize=(6,6))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_extent([-126,-114,31,51])                  # CA + surroundings

    # transparent background; red where 1
    ax.imshow(slice2d, interpolation="nearest",
              origin="lower",
              cmap=cmap, norm=norm,
              extent=[slice2d.lon.min(), slice2d.lon.max(),
                      slice2d.lat.min(), slice2d.lat.max()],
              transform=ccrs.PlateCarree())

    # extra: scatter cell centres (helps for sparse maps)
    ax.scatter(lons, lats, s=6, c="red", transform=ccrs.PlateCarree())

    ts = np.datetime_as_string(t, unit="D")
    plt.title(f"HAB labels – {ts}")
    plt.tight_layout()

    if args.save:
        out = ROOT / f"label_check_{ts}.png"
        plt.savefig(out, dpi=200)
        print("saved", out.name)
        plt.close()
    else:
        plt.show()


'''
python -m habs.quality_control.sanity_check_labels --n 5
'''