#!/bin/bash
#eval echo ${PRELOAD_FLAG}
#eval echo ${BIN_DIR}
dp=/sciclone/pscr/lyang11/NVBitFI/nvbit_release/tools/nvbitfi/nvbit_data/hotspot
eval ${PRELOAD_FLAG} ${BIN_DIR}/hotspot 256 2 2 ${dp}/temp_256 ${dp}/power_256 output.out > stdout.txt 2> stderr.txt
#./hotspot 64 2 1 ../../data/hotspot/temp_64 ../../data/hotspot/power_64 output.out >& log_run_64.txt
#./hotspot 256 2 2 ./temp_256 ./power_256 golden.txt
