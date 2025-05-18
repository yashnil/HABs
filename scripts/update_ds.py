
import xarray as xr, pathlib
ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_dataset_8day_4km_common.nc")
ds = ds.drop_vars(["lat","lon"])                # keep y,x coords only (optional)
ds.hab_occurrence.attrs = {"long_name":"HAB occurrence mask"}   # cosmetics
ds.to_netcdf("/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_dataset_8day_4km_clean.nc")
