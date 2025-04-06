#!/bin/bash

# Ensure environment for l3mapgen
export OCSSWROOT="/Users/yashnilmohanty/ocssw"
export OCDATAROOT="$OCSSWROOT/share"
export PATH="$OCSSWROOT/bin:$PATH"

CHL_DIR="/Users/yashnilmohanty/Desktop/HABs_Research/Data/chlorophyll"

for infile in "$CHL_DIR"/*.L3b.8D.CHL.x.nc
do
  outfile="${infile/.L3b./.L3m.}"
  outfile="${outfile%.nc}.map.nc"

  echo "Mapping '$infile' -> '$outfile'"

  l3mapgen \
    ifile="$infile" \
    ofile="$outfile" \
    product=chlor_a \
    resolution=4km \
    west=-125 east=-115 south=32 north=50 \
    projection=platecarree
done

echo "Done mapping all files."
