#!/usr/bin/env python3
"""
Plot CMEMS Global Ocean Physics Reanalysis daily fields (2016-01-01 →
2021-06-30) for:
  • salinity (so)                     • potential temperature (thetao)
  • east / north velocities (uo, vo) • SSH (zos)

Each variable is saved in   ~/Desktop/plots_cmems/<var>/<var>_YYYYMMDD.png
Run with:  python3 make_copernicus_pngs.py
"""

import xarray as xr, cartopy.crs as ccrs, matplotlib.pyplot as plt
import pandas as pd, pathlib, os

# ------------------------------------------------------------------
FILES = {
    # one file contains uo, vo, zos
    "/Users/yashnilmohanty/Desktop/HABs_Research/Data/copernicus/"
    "cmems_mod_glo_phy_my_0.083deg_P1D-m_1742773956384.nc":
        ["uo", "vo", "zos"],

    # the other contains so, thetao
    "/Users/yashnilmohanty/Desktop/HABs_Research/Data/copernicus/"
    "cmems_mod_glo_phy_my_0.083deg_P1D-m_1742774147747.nc":
        ["so", "thetao"],
}

START, END = "2016-01-01", "2021-06-30"
OUTROOT = pathlib.Path("~/Desktop/plots_cmems").expanduser()
OUTROOT.mkdir(exist_ok=True)

# ------------------------------------------------------------------
def save_daily_png(da, outdir, vmin=None, vmax=None, cmap="viridis"):
    outdir.mkdir(parents=True, exist_ok=True)
    for ts in pd.date_range(START, END, freq="1D"):
        if ts not in da.time:           # skip missing days
            continue
        img = outdir / f"{da.name}_{ts:%Y%m%d}.png"
        if img.exists():
            continue

        data = da.sel(time=ts).squeeze()       # lon × lat 2-D

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

# ------------------------------------------------------------------
print("⏳ Loading CMEMS files …")
for ncfile, varlist in FILES.items():
    if not os.path.isfile(ncfile):
        print(f"⚠️  Missing {ncfile}")
        continue

    print(f"→ opening {os.path.basename(ncfile)}")
    ds = xr.open_dataset(ncfile, engine="netcdf4", decode_times=False)

    # rename dimension + convert epoch seconds → datetime64
    ds = ds.rename({"time": "time"})                # dim already 'time'
    ds["time"] = pd.to_datetime(ds.time.values, unit="s")

    ds = ds.sel(time=slice(START, END))

    for var in varlist:
        da = ds[var]

        # if depth dimension exists (size 1), drop it
        if "depth" in da.dims:
            da = da.isel(depth=0, drop=True)

        # simple daily mean
        da_dly = (da
                  .resample(time="1D")
                  .mean()
                  .sortby("time"))

        print(f"▶︎ plotting {var}")
        vmin, vmax = {
            "so": (30, 36),
            "thetao": (4, 25),
            "uo": (-1.0, 1.0),
            "vo": (-1.0, 1.0),
            "zos": (-1.0, 1.0),
        }.get(var, (None, None))

        cmap = "coolwarm" if var in ("thetao", "so") else "viridis"
        save_daily_png(da_dly, OUTROOT/var, vmin, vmax, cmap=cmap)

print("✅ CMEMS plotting complete")
