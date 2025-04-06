#!/usr/bin/env python3
"""
Generate 8‑day PNG maps (2016‑01‑01→2025‑01‑08) for MODIS‑Aqua
chlor_a, Kd_490, nflh and sst, saving each variable in its own folder:

    ~/Desktop/plots/chlorophyll/
    ~/Desktop/plots/kd490/
    ~/Desktop/plots/nflh/
    ~/Desktop/plots/sst/

Run with:  python3 make_modis_pngs.py
"""

import subprocess, os, datetime, pathlib

# ----------------------------------------------------------------------
# 0.  Point Python at your OCSSW install
# ----------------------------------------------------------------------
OCSSWROOT = "/Users/yashnilmohanty/SeaDAS/ocssw"
os.environ["OCSSWROOT"] = OCSSWROOT
os.environ["PATH"] = f"{OCSSWROOT}/bin:" + os.environ["PATH"]   # puts l3mapgen on PATH

# ----------------------------------------------------------------------
# 1.  Define file templates and plotting parameters
# ----------------------------------------------------------------------
FEATURES = {
    "chlorophyll": {
        "product": "chlor_a",
        "template": "/Users/yashnilmohanty/Desktop/HABs_Research/Data/chlorophyll/"
                    "AQUA_MODIS.{start}_{end}.L3b.8D.CHL.x.nc",
        "datamin": "0.05", "datamax": "20", "scale": "log"
    },
    "kd490": {
        "product": "Kd_490",          # capital K
        "template": "/Users/yashnilmohanty/Desktop/HABs_Research/Data/kd490/"
                    "AQUA_MODIS.{start}_{end}.L3b.8D.KD.x.nc",
        "datamin": "0.01", "datamax": "0.5", "scale": "log"
    },
    "nflh": {
        "product": "nflh",
        "template": "/Users/yashnilmohanty/Desktop/HABs_Research/Data/nFLH/"
                    "AQUA_MODIS.{start}_{end}.L3b.8D.FLH.x.nc",
        "datamin": "0", "datamax": "3", "scale": "log"
    },
    "sst": {
        "product": "sst",
        "template": "/Users/yashnilmohanty/Desktop/HABs_Research/Data/seaSurfaceTemperature/"
                    "AQUA_MODIS.{start}_{end}.L3b.8D.NSST.x.nc",
        "datamin": "-2", "datamax": "35", "scale": "linear"
    },
}

# geographic box & grid
BOUNDS = dict(west=-125, east=-115, south=32, north=50, resolution="4km")

# ----------------------------------------------------------------------
# 2.  Walk through every 8‑day composite, restarting 1 Jan each year
# ----------------------------------------------------------------------
STEP = datetime.timedelta(days=8)
FIRST_YEAR, LAST_YEAR = 2016, 2025
GLOBAL_END = datetime.date(2025, 1, 1)        # last start date

for year in range(FIRST_YEAR, LAST_YEAR + 1):
    current = datetime.date(year, 1, 1)
    while current <= GLOBAL_END and current.year == year:
        period_end = current + STEP - datetime.timedelta(days=1)
        s = current.strftime("%Y%m%d")
        e = period_end.strftime("%Y%m%d")

        for name, cfg in FEATURES.items():

            # create ~/Desktop/plots/<variable>/ if it doesn’t exist
            outdir = pathlib.Path(f"/Users/yashnilmohanty/Desktop/plots/{name}")
            outdir.mkdir(parents=True, exist_ok=True)

            infile  = cfg["template"].format(start=s, end=e)
            outfile = outdir / f"{name}_{s}_{e}.png"
            if outfile.exists():
                continue                    # already done
            if not os.path.isfile(infile):
                print(f"⚠️  Missing file, skipping: {infile}")
                continue

            cmd = [
                "l3mapgen",
                f"ifile={infile}",
                f"ofile={outfile}",
                f"product={cfg['product']}",
                "projection=platecarree",
                *(f"{k}={v}" for k, v in BOUNDS.items()),
                "oformat=png",
                "apply_pal=yes",
                f"scale_type={cfg['scale']}",
                f"datamin={cfg['datamin']}",
                f"datamax={cfg['datamax']}",
            ]
            print("→", outfile.relative_to(outdir.parent))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌  {name} failed for {s}: {e}. Continuing…")

        # next 8‑day block
        current += STEP
