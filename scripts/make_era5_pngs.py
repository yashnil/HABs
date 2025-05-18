#!/usr/bin/env python3
import xarray as xr, cartopy.crs as ccrs, matplotlib.pyplot as plt
import numpy as np, pandas as pd, pathlib, os, sys

FILES = {
    "tp":   "/Users/yashnilmohanty/Desktop/HABs_Research/Data/era5/data_stream-oper_stepType-accum.nc",
    "avg_sdswrf": "/Users/yashnilmohanty/Desktop/HABs_Research/Data/era5/data_stream-oper_stepType-avg.nc",
    "u10":  "/Users/yashnilmohanty/Desktop/HABs_Research/Data/era5/data_stream-oper_stepType-instant copy.nc",
    "v10":  "/Users/yashnilmohanty/Desktop/HABs_Research/Data/era5/data_stream-oper_stepType-instant copy.nc",
    "t2m":  "/Users/yashnilmohanty/Desktop/HABs_Research/Data/era5/data_stream-oper_stepType-instant.nc",
    "d2m":  "/Users/yashnilmohanty/Desktop/HABs_Research/Data/era5/data_stream-oper_stepType-instant.nc",
}

START, END = "2016-01-01", "2025-01-01"
OUTROOT = pathlib.Path("/Users/yashnilmohanty/Desktop/plots_era5").expanduser()
OUTROOT.mkdir(exist_ok=True)

def save_daily_png(da, outdir, vmin=None, vmax=None, cmap="viridis"):
    outdir.mkdir(parents=True, exist_ok=True)
    for ts in pd.date_range(START, END, freq="1D"):
        if ts not in da.time:  # skip missing days
            continue
        img = outdir / f"{da.name}_{ts:%Y%m%d}.png"
        if img.exists():
            continue
        data = da.sel(time=ts).squeeze()
        plt.figure(figsize=(5,4))
        ax = plt.axes(projection=ccrs.PlateCarree())
        pcm = data.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                                   vmin=vmin, vmax=vmax, cmap=cmap,
                                   add_colorbar=False)
        ax.coastlines(resolution="10m")
        ax.set_extent([-125, -115, 32, 50])
        plt.colorbar(pcm, label=f"{da.name} ({da.attrs.get('units','')})")
        plt.title(f"{da.name}  {ts:%Y-%m-%d}")
        plt.tight_layout()
        plt.savefig(img, dpi=150)
        plt.close()

print("⏳ Loading ERA5 files …", flush=True)

for var, ncfile in FILES.items():
    if not os.path.isfile(ncfile):
        print(f"⚠️  Missing {ncfile}")
        continue

    print(f"→ opening  {var}  from  {os.path.basename(ncfile)}", flush=True)
    try:
        ds = xr.open_dataset(ncfile, engine="netcdf4", decode_times=False)
    except Exception as e:
        print(f"❌  failed to open {ncfile}: {e}", file=sys.stderr)
        continue
    print("   ✓ opened", flush=True)

    # rename dimension & convert epoch-seconds to datetime64 without decoding data
    ds = ds.rename({"valid_time": "time"})
    ds = ds.assign_coords(time=pd.to_datetime(ds.time.values, unit="s"))

    da = (ds[var]
          .sel(time=slice(START, END))
          .resample(time="1D").mean()
          .sortby("time"))

    print(f"▶︎ plotting {var}", flush=True)
    vmin, vmax = {
        "tp": (0, 0.02),
        "avg_sdswrf": (0, 400),
        "u10": (-15, 15), "v10": (-15, 15),
        "t2m": (270, 310), "d2m": (260, 300),
    }.get(var, (None, None))

    save_daily_png(da, OUTROOT/var, vmin, vmax,
                   cmap="coolwarm" if var in ("t2m", "d2m") else "viridis")

print("✅ ERA5 plotting complete")
