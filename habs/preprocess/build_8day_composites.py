#!/usr/bin/env python3
"""
1) Read raw ERA5 daily/instant files and CMEMS daily files.
2) Subset to your lon/lat box and time range.
3) For each calendar year, bin into 8-day blocks starting on Jan 1, Jan 9, …
4) Compute the mean within each block.
5) Write out era5_8day.nc and cmems_8day.nc with time in days since 2016-01-01.
"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from dask.diagnostics import ProgressBar

# ── user parameters ──────────────────────────────────────────────────────────────
ERA5_DIR   = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/era5")
CMEMS_DIR  = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/copernicus")
OUT_DIR    = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
LON_SLICE  = slice(-125, -115)
# ERA5 lat runs 50 → 32; CMEMS lat runs 32 → 50
LAT_SLICE_ERA5  = slice(50, 32)
LAT_SLICE_CMEMS = slice(32, 50)
TIME_RANGE = slice("2016-01-01", "2021-06-30")
REF_DATE   = np.datetime64("2016-01-01")

# ── helper functions ─────────────────────────────────────────────────────────────
def make_8day(ds):
    """Group any ds.time into calendar‐year 8-day composite blocks."""
    doy    = ds.time.dt.dayofyear
    offset = ((doy - 1) % 8).astype("timedelta64[D]")
    block  = ds.time - offset
    block_vals = block.data.astype("datetime64[ns]")
    ds2 = ds.assign_coords(block_time=("time", block_vals))
    out = ds2.groupby("block_time").mean(dim="time")
    return out.rename({"block_time": "time"}).sortby("time")

def encode_time_as_days(ds, ref):
    """Convert a datetime64[ns] time axis into int days since `ref`."""
    times = ds.time.values.astype("datetime64[ns]")
    days  = ((times - ref) / np.timedelta64(1, "D")).astype("int32")
    ds2 = ds.assign_coords(time=("time", days))
    attrs = {
        "units":    f"days since {pd.Timestamp(ref).strftime('%Y-%m-%d')}",
        "calendar": "proleptic_gregorian",
    }
    ds2.time.attrs.update(attrs)
    ds2.time.encoding.update({
        "dtype":    "int32",
        "units":    attrs["units"],
        "calendar": attrs["calendar"],
    })
    return ds2

# ── ERA5 ─────────────────────────────────────────────────────────────────────────
def process_era5():
    print("→ opening ERA5 files lazily with Dask…")
    era5_files = {
        "data_stream-oper_stepType-accum.nc":        ["tp"],
        "data_stream-oper_stepType-avg.nc":          ["avg_sdswrf"],
        "data_stream-oper_stepType-instant.nc":      ["t2m","d2m"],
        "data_stream-oper_stepType-instant copy.nc": ["u10","v10"],
    }
    ds_list = []
    for fn, vars_ in era5_files.items():
        ds = (
            xr.open_dataset(ERA5_DIR/fn, decode_times=True, chunks={"time":1000})
              .rename({"valid_time":"time","longitude":"lon","latitude":"lat"})
              .sel(time=TIME_RANGE, lon=LON_SLICE, lat=LAT_SLICE_ERA5)
        )
        ds_list.append(ds[vars_])
    ds_era5 = xr.merge(ds_list)
    print(f"   loaded ERA5 6-hourly ds time={ds_era5.time.size}")

    print("→ aggregating to daily means…")
    ds_era5 = ds_era5.resample(time="1D").mean()
    print(f"   now daily ds time={ds_era5.time.size}")

    print("→ grouping into 8-day composites (lazy)…")
    ds8 = make_8day(ds_era5)
    print(f"   grouped into {ds8.time.size} blocks  ← should be ≃253")

    print("→ re-encoding time as days since 2016-01-01…")
    ds8 = encode_time_as_days(ds8, REF_DATE)

    out_path = OUT_DIR/"era5_8day.nc"
    print(f"→ writing {out_path}…")
    with ProgressBar():
        ds8.to_netcdf(
            out_path,
            engine="netcdf4",
            encoding={v: {"zlib":True,"complevel":4} for v in ds8.data_vars},
        )
    print("✅ Wrote ERA5 8-day composites")


# ── CMEMS ───────────────────────────────────────────────────────────────────────
def process_cmems():
    print("→ opening CMEMS files lazily with Dask…")
    cmems_files = {
        "cmems_mod_glo_phy_my_0.083deg_P1D-m_1742773956384.nc": ["uo","vo","zos"],
        "cmems_mod_glo_phy_my_0.083deg_P1D-m_1742774147747.nc": ["so","thetao"],
    }
    ds_list = []
    for fn, vars_ in cmems_files.items():
        ds = (
            xr.open_dataset(CMEMS_DIR/fn, decode_times=True, chunks={"time":200})
              .squeeze("depth", drop=True)
              .rename({"longitude":"lon","latitude":"lat"})
              .sel(time=TIME_RANGE, lon=LON_SLICE, lat=LAT_SLICE_CMEMS)
        )
        ds_list.append(ds[vars_])
    ds_cmems = xr.merge(ds_list)
    print(f"   loaded CMEMS ds time={ds_cmems.time.size}, lat={ds_cmems.lat.size}")

    print("→ grouping into 8-day composites (lazy)…")
    ds8 = make_8day(ds_cmems)
    print(f"   grouped into {ds8.time.size} blocks")

    print("→ re-encoding time as days since 2016-01-01…")
    ds8 = encode_time_as_days(ds8, REF_DATE)

    out_path = OUT_DIR/"cmems_8day.nc"
    print(f"→ writing {out_path}…")
    with ProgressBar():
        ds8.to_netcdf(
            out_path,
            engine="netcdf4",
            encoding={v: {"zlib":True,"complevel":4} for v in ds8.data_vars},
        )
    print("✅ Wrote CMEMS 8-day composites")

if __name__ == "__main__":
    process_era5()
    process_cmems()
