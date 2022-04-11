#!/bin/bash
#eval echo ${PRELOAD_FLAG}
#eval echo ${BIN_DIR}
dp=/sciclone/pscr/lyang11/robust_proj/nvbit_release/tools/nvbitfi/nvbit_data/gaussian
eval ${PRELOAD_FLAG} ${BIN_DIR}/gaussian -f ${dp}/matrix80.txt > stdout.txt 2> stderr.txt

