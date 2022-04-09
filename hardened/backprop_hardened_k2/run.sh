#!/bin/bash
#echo ${PRELOAD_FLAG}
eval ${PRELOAD_FLAG} ${BIN_DIR}/backprop 8192 > stdout.txt 2> stderr.txt
