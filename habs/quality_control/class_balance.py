#!/usr/bin/env python3
"""
class_balance.py  –  quick HAB / non-HAB pixel counts
=====================================================

Walks through *labels.zarr* until it finds the **first** (time, lat, lon)
array (3-D) – no matter how deeply nested – and then prints

• total water pixels  
• HAB-positive pixels  
• HAB-negative pixels  
• positive-class share  
• recommended `pos_weight` for `torch.nn.BCEWithLogitsLoss`

Run
----
micromamba activate habs_env
python -m habs.quality_control.class_balance
"""
from pathlib import Path
import numpy as np
import xarray as xr
import zarr

# ------------------------------------------------------------------ paths ----
ROOT  = Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed")
STORE = ROOT / "labels.zarr"        # written by build_labels.py

# ------------------------------------------------------------------ helpers ---
def first_3d_array(znode: zarr.Group):
    """
    Depth-first search through a zarr Group.
    Returns (zarr_array, logical_key) or (None, None)
    """
    # arrays in *this* group
    for key in znode.array_keys():
        arr = znode[key]
        if arr.ndim == 3:
            return arr, f"{znode.basename}/{key}".lstrip("/")
    # recurse into sub-groups
    for subkey in znode.group_keys():
        arr, name = first_3d_array(znode[subkey])
        if arr is not None:
            return arr, name
    return None, None

# ------------------------------------------------------------------ open ------
lbl_da = None

# 1) try Xarray *dataarray* opener
try:
    lbl_da = xr.open_dataarray(STORE, consolidated=False)
except Exception:
    pass

# 2) try Xarray *dataset* opener
if lbl_da is None:
    try:
        ds = xr.open_zarr(STORE, consolidated=False)
        if ds.data_vars:
            lbl_da = ds[list(ds.data_vars)[0]]
    except Exception:
        pass

# 3) raw Zarr walk (last resort)
if lbl_da is None:
    root = zarr.open(str(STORE), mode="r")
    arr, key = first_3d_array(root)
    if arr is None:
        raise RuntimeError("❌  No 3-D array found anywhere inside labels.zarr")
    print(f"(found 3-D array at '{key}')")
    lbl_da = xr.DataArray(np.asarray(arr),
                          dims=["time", "lat", "lon"],
                          name=key.split("/")[-1])

# ------------------------------------------------------------------ stats -----
lbl_da = lbl_da.astype("int8")        # (time, lat, lon)

pos = int((lbl_da == 1).sum())
neg = int((lbl_da == 0).sum())
tot = pos + neg

ratio       = pos / tot if tot else 0.0
pos_weight  = neg / pos if pos else np.inf

print(f"Total water pixels  : {tot:,}")
print(f"‣ HAB-positive (1)  : {pos:,}")
print(f"‣ HAB-negative (0)  : {neg:,}")
print(f"Class ratio         : positives = {ratio:.4%}")
print(f"Suggested pos_weight: {pos_weight:.2f}")

'''

To run: 

micromamba activate habs_env
python -m habs.quality_control.class_balance
'''