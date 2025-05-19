
#!/usr/bin/env python3
"""Save the 279×502 ERA-5 / CMEMS lat-lon grid to scripts/modis_4km_grid.npz"""

import xarray as xr, numpy as np, pathlib

ERA_EXAMPLE = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/era5_avg_sdswrf_8day_4km.nc"
)
ds = xr.open_dataset(ERA_EXAMPLE)
lons2d, lats2d = ds.x.values[np.newaxis, :], ds.y.values[:, np.newaxis]   # 1-D → 2-D

out = pathlib.Path(__file__).resolve().parents[1] / "scripts/modis_4km_grid.npz"
np.savez(out, lons=np.tile(lons2d, (lats2d.size, 1)),  # 2-D mesh
               lats=np.tile(lats2d, (1, lons2d.size)))
print("✅ wrote", out)
