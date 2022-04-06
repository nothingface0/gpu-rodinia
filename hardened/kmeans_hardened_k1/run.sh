#!/bin/bash
#eval echo ${PRELOAD_FLAG}
#eval echo ${BIN_DIR}
dp=/sciclone/pscr/lyang11/robust_proj/nvbit_release/tools/nvbitfi/nvbit_data/kmeans
eval ${PRELOAD_FLAG} ${BIN_DIR}/kmeans -o -i ${dp}/800_34.txt > stdout.txt 2> stderr.txt
