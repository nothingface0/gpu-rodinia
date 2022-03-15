#!/bin/bash
#eval echo ${PRELOAD_FLAG}
#eval echo ${BIN_DIR}
eval ${PRELOAD_FLAG} ${BIN_DIR}/srad 2 0.5 128 128 > stdout.txt 2> stderr.txt
