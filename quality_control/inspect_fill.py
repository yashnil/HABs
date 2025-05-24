
#!/usr/bin/env python3
"""
01 – Inspect-and-(optionally) fill NaNs in the MODIS fields
-----------------------------------------------------------
* Loads root_dataset.nc
* Prints global min / max / NaN counts for every variable
* Draws a quick NaN-fraction map for each MODIS layer
* Optionally fills small NaN holes (< N contiguous pixels) by
  spatial nearest-neighbour, then writes a cleaned copy
  (root_dataset_filled.nc) **only if you set FILL=True**.

Run:
    python preprocess/01_inspect_fill.py          # just stats + plots
    python preprocess/01_inspect_fill.py --fill   # fill & write new file
"""

import argparse, itertools, pathlib
import xarray as xr, numpy as np, matplotlib.pyplot as plt
import matplotlib.ticker as mt

# ---------------- user paths ---------------------------------------------------
DATA = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed/root_dataset.nc")
OUT  = DATA.with_name("root_dataset_filled.nc")

MODIS_VARS = ["chlor_a", "Kd_490", "nflh", "sst"]   # vars to inspect / fill
FILL       = False                                  # overridden by CLI
HOLE_SIZE  = 9                                      # max (#pixels) contiguous hole to fill
# --------------------------------------------------------------------------------

# ── CLI flag -------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Inspect/fill NaNs in MODIS layers")
parser.add_argument("--fill", action="store_true", help="fill NaNs and write *_filled.nc")
args = parser.parse_args()
FILL = args.fill

# ── load -----------------------------------------------------------------------
ds = xr.open_dataset(DATA)
print(f"Loaded {DATA.name}   dims = {dict(ds.sizes)}")

# ── quick global stats ---------------------------------------------------------
def stats(da):
    good = int(np.isfinite(da).sum())
    bad  = int(np.isnan(da).sum())
    return good, bad, da.min().values, da.max().values

tbl = []
for v, da in ds.data_vars.items():
    good, bad, vmin, vmax = stats(da)
    tbl.append((v, good, bad, vmin, vmax))

hdr = f"{'var':12s} {'good':>9s} {'NaN':>9s} {'min':>11s} {'max':>11s}"
print(hdr)
print("-"*len(hdr))
for v, g, n, lo, hi in tbl:
    print(f"{v:12s} {g:9d} {n:9d} {lo:11.3g} {hi:11.3g}")

# ── NaN-fraction maps for MODIS layers -----------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for ax, var in zip(axs.flat, MODIS_VARS):
    frac = ds[var].isnull().mean(dim="time")        # 0 … 1
    im   = frac.plot(ax=ax, vmin=0, vmax=1, cmap="magma_r",
                     cbar_kwargs={"shrink":0.7})
    ax.set_title(f"{var} – NaN fraction")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.xaxis.set_major_formatter(mt.FormatStrFormatter('%.1f'))
plt.suptitle("NaN fraction per pixel (MODIS 2016-01-09 … 2021-06-23)")
plt.show()

# ── optional in-place filling ---------------------------------------------------
if FILL:
    import scipy.ndimage as ndi

    def fill_small_holes(da, n_pix=HOLE_SIZE):
        """Fill NaN clusters ≤ n_pix using nearest spatial neighbour (per-time)."""
        filled = []
        for t in da.time:
            slice_ = da.sel(time=t)
            mask   = np.isnan(slice_)
            # label contiguous NaN blobs
            lbl, nblob = ndi.label(mask)
            out = slice_.copy()
            for lab in range(1, nblob+1):
                idx = np.where(lbl == lab)
                if idx[0].size <= n_pix:          # small hole
                    # nearest non-NaN value
                    yy, xx = idx[0][0], idx[1][0]
                    value  = slice_.values
                    # brute nearest (could use KD-tree for speed)
                    dist   = np.sqrt((~mask)*1e9 + (np.indices(mask.shape)[0]-yy)**2 +
                                     (np.indices(mask.shape)[1]-xx)**2)
                    jj, ii = np.unravel_index(dist.argmin(), dist.shape)
                    out.values[idx] = value[jj, ii]
            filled.append(out)
        return xr.concat(filled, dim="time")

    print("\n→ Filling small NaN holes in MODIS layers …")
    for v in MODIS_VARS:
        before = int(ds[v].isnull().sum())
        ds[v]  = fill_small_holes(ds[v])
        after  = int(ds[v].isnull().sum())
        print(f"   {v:8s}: NaNs {before:,} → {after:,}")

    print(f"→ writing cleaned cube → {OUT}")
    ds.to_netcdf(OUT,
        encoding={var: {"zlib": True, "complevel": 4} for var in ds.data_vars})
    print("✅ wrote", OUT)
else:
    print("\n(run again with  --fill  if you want to write a cleaned file)")
