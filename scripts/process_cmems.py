#!/usr/bin/env python3
"""
Resample daily CMEMS Global Ocean Physics fields to 8-day means and
re-grid to the MODIS 4-km grid (2016-01-01 → 2021-06-30).

Outputs  →  processed/cmems_<var>_8day_4km.nc
"""

from align_utils import to_datetime, resample_8day, regrid_to_modis
import xarray as xr, os, pathlib

# ----------------------------------------------------------------------
CMEMS_DIR = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/copernicus")
OUT_DIR   = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "uo"    : "cmems_mod_glo_phy_my_0.083deg_P1D-m_1742773956384.nc",
    "vo"    : "cmems_mod_glo_phy_my_0.083deg_P1D-m_1742773956384.nc",
    "zos"   : "cmems_mod_glo_phy_my_0.083deg_P1D-m_1742773956384.nc",
    "so"    : "cmems_mod_glo_phy_my_0.083deg_P1D-m_1742774147747.nc",
    "thetao": "cmems_mod_glo_phy_my_0.083deg_P1D-m_1742774147747.nc",
}
T0, T1 = "2016-01-01", "2021-06-30"

# ----------------------------------------------------------------------
for var, fname in FILES.items():
    out = OUT_DIR / f"cmems_{var}_8day_4km.nc"
    if out.exists():
        print(f"✔︎ {out.name} exists — skip")
        continue

    src = CMEMS_DIR / fname
    if not src.is_file():
        raise FileNotFoundError(src)

    print(f"⏳ processing {var} from {fname}")
    ds = xr.open_dataset(src, engine="netcdf4", decode_times=False)

    # ---- rename dims to lat/lon for interp ----
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = to_datetime(ds, "time").sel(time=slice(T0, T1))

    # ---- drop depth dimension if present ----
    if "depth" in ds[var].dims:
        ds[var] = ds[var].isel(depth=0, drop=True)

    # ---- 8-day mean and regrid ----
    da8  = resample_8day(ds[var])
    da4k = regrid_to_modis(da8)

    da4k.to_netcdf(out)
    print(f"✅ wrote {out.name}")
