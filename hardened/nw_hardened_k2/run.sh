#!/bin/bash
#eval echo ${PRELOAD_FLAG}
#eval echo ${BIN_DIR}
eval ${PRELOAD_FLAG} ${BIN_DIR}/needle 288 10 > stdout.txt 2> stderr.txt

