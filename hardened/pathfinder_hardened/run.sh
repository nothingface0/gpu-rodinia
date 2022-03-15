#!/bin/bash
#eval echo ${PRELOAD_FLAG}
#eval echo ${BIN_DIR}
eval ${PRELOAD_FLAG} ${BIN_DIR}/pathfinder 10000 100 20 > stdout.txt 2> stderr.txt

