
# grid_modis.py  – run once, reuse the npz file

import numpy as np

# MODIS bounds
LON_W, LON_E = -125.0, -115.0
LAT_S, LAT_N =  32.0,   50.0

# 4 km ≈ 0.0416667° at equator; MODIS uses constant lon/lat step for L3m
STEP = 4 / 111.195  # 4 km / 1° lat ≈ 0.036°  (use 0.0416667 if you prefer)

lats = np.arange(LAT_S, LAT_N + STEP, STEP)  # south→north
lons = np.arange(LON_W, LON_E + STEP, STEP)  # west→east

lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
np.savez("modis_4km_grid.npz", lons=lon2d, lats=lat2d)

print("✅ saved modis_4km_grid.npz  with shape", lat2d.shape)
