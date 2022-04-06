#!/bin/bash
#eval echo ${PRELOAD_FLAG}
#eval echo ${BIN_DIR}
eval ${PRELOAD_FLAG} ${BIN_DIR}/lud_cuda -s 128 -v  > stdout.txt 2> stderr.txt

