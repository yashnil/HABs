import xarray as xr, numpy as np, pathlib, pandas as pd

root   = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
lab    = xr.open_dataarray(root/"labels.zarr")          # (time,lat,lon)
strip  = xr.open_dataarray(root/"coastal_strip.zarr")   # (lat,lon)

# 1️⃣  Are lats *monotonically* ordered the same way?
print("labels lat ascending?", np.all(np.diff(lab.lat)   > 0))
print("strip  lat ascending?", np.all(np.diff(strip.lat) > 0))

# 2️⃣  Compare the first/last few values
print("lab lat   :", lab.lat[:5].values, "...", lab.lat[-5:].values)
print("strip lat :", strip.lat[:5].values, "...", strip.lat[-5:].values)

# 3️⃣  How many *exact* matches?
lat_matches = np.isclose(lab.lat.values[:,None], strip.lat.values).any(1).sum()
lon_matches = np.isclose(lab.lon.values[:,None], strip.lon.values).any(1).sum()
print(f"lat exact matches : {lat_matches}/{lab.sizes['lat']}")
print(f"lon exact matches : {lon_matches}/{lab.sizes['lon']}")
