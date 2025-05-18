#!/usr/bin/env bash
# batch_modis_l3m.sh  –  map all MODIS-Aqua L3b files (2016-present)
#                        to 4-km Plate-Carrée L3m NetCDFs
#                        for chlor_a, kd_490, nflh, sst

set -euo pipefail

export OCSSWROOT=/Users/yashnilmohanty/SeaDAS/ocssw
source "$OCSSWROOT/OCSSW_bash.env"

DATAROOT=/Users/yashnilmohanty/Desktop/HABs_Research/Data
OUTBASE=/Users/yashnilmohanty/Desktop/HABs_Research/Processed/modis_l3m

# product   code   sub-directory name under $DATAROOT
declare -a CONFIGS=(
  "chlor_a CHL chlorophyll"
  "Kd_490  KD  kd490"
  "nflh    FLH nFLH"
  "sst     NSST seaSurfaceTemperature"
)

mkdir -p "$OUTBASE"

for cfg in "${CONFIGS[@]}"; do
  read -r PRODUCT CODE SUBDIR <<<"$cfg"
  INDIR="$DATAROOT/$SUBDIR"
  OUTDIR="$OUTBASE/$SUBDIR"
  mkdir -p "$OUTDIR"

  echo "▶︎ processing $PRODUCT  from  $INDIR"
  shopt -s nullglob                 # avoid literal pattern if no match
  for f in "$INDIR"/AQUA_MODIS.*.L3b.8D."$CODE".x.nc; do
      base=$(basename "$f" .L3b.8D."$CODE".x.nc)
      year=${base:11:4}             # YYYY of first composite day
      [[ $year -lt 2016 ]] && continue

      of="$OUTDIR/${base}_4km_L3m.nc"
      [[ -f "$of" ]] && { echo "✔︎ exists $of"; continue; }

      echo "→ mapping $base  →  $(basename "$of")"
      l3mapgen \
          ifile="$f" ofile="$of" \
          product="$PRODUCT" projection=platecarree \
          west=-125 east=-115 south=32 north=50 \
          resolution=4km oformat=netcdf4
  done
done

echo "✅ all MODIS products mapped (2016-present)"
