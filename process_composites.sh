#!/bin/bash

# used for processing puff and grid composites for the vis_grid.py code

DIR_PROCESS=$1
PUFF_OUTPUT=$2
GRID_OUTPUT=$3
ECC=$4  # Optional flag for eccentricity

# Temporary files
PUFF_TEMP="${DIR_PROCESS}/tmp_puff.dat"
GRID_TEMP="${DIR_PROCESS}/tmp_grid.dat"

# Join puff files
echo "Joining puff files..."
find "${DIR_PROCESS}" -name '*puff*.dat' ! -name 'tmp_puff.dat' -exec cat {} \; > "${PUFF_TEMP}"

# Join grid files
echo "Joining grid files..."
find "${DIR_PROCESS}" -name 'CME*.dat' ! -name '*puff*.dat' ! -name 'tmp_grid.dat' -exec cat {} \; > "${GRID_TEMP}"

# Process and clean puff files
echo "Processing puff files..."
if [ "${ECC}" == '--eccentricity' ]; then
    util_CleanILE.py "${PUFF_TEMP}" "${ECC}" | sort -rg -k11 > "${PUFF_OUTPUT}"
else
    util_CleanILE.py "${PUFF_TEMP}" "${ECC}" | sort -rg -k10 > "${PUFF_OUTPUT}"
fi

# Process and clean grid files
echo "Processing grid files..."
if [ "${ECC}" == '--eccentricity' ]; then
    util_CleanILE.py "${GRID_TEMP}" "${ECC}" | sort -rg -k11 > "${GRID_OUTPUT}"
else
    util_CleanILE.py "${GRID_TEMP}" "${ECC}" | sort -rg -k10 > "${GRID_OUTPUT}"
fi

# Clean up temporary files
rm -f "${PUFF_TEMP}" "${GRID_TEMP}"

echo "Processing complete."
