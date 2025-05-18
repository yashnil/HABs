import xarray as xr, numpy as np, pandas as pd, functools

REGRIDDER_CACHE = {}
GRID = np.load("modis_4km_grid.npz")
LAT_TGT = xr.DataArray(GRID["lats"], dims=("y","x"))
LON_TGT = xr.DataArray(GRID["lons"], dims=("y","x"))

def to_datetime(ds, time_dim):
    """Epoch-seconds → datetime64, keep original dim name."""
    ds[time_dim] = pd.to_datetime(ds[time_dim].values, unit="s")
    return ds

@functools.lru_cache(maxsize=None)
def _target_grid():
    return xr.Dataset({"lat": LAT_TGT, "lon": LON_TGT})

def regrid_to_modis(da):
    """Bilinear interp via xarray; no ESMF required."""
    ds_tgt = _target_grid()
    da_i = (da.interp(lat=ds_tgt.lat[:,0], lon=ds_tgt.lon[0],
                      method="linear")
              .rename({"lat":"y", "lon":"x"}))
    # broadcast to 2-D grid
    return da_i.broadcast_like(ds_tgt.lat)

def resample_8day(da):
    return (da
            .resample(time="8D", label="left", closed="left")
            .mean())
