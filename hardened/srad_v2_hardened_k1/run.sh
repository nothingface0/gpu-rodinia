#!/bin/bash
#eval echo ${PRELOAD_FLAG}
#eval echo ${BIN_DIR}
eval ${PRELOAD_FLAG} ${BIN_DIR}/srad 256 256 0 127 0 127 0.5 2 > stdout.txt 2> stderr.txt
