import xarray as xr, numpy as np, pathlib

ROOT = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
labels_da = xr.open_zarr(ROOT/"labels.zarr")["labels"]
strip_da  = xr.open_dataarray(ROOT/"coastal_strip.zarr")

print("coords identical?",
      np.array_equal(labels_da.lat, strip_da.lat),
      np.array_equal(labels_da.lon, strip_da.lon))

print("strip unique values :", np.unique(strip_da))
print("labels positives    :", int((labels_da==1).sum()))
print("overlap positives   :", int(((labels_da==1) & (strip_da==1)).sum()))