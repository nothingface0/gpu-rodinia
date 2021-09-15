#!/bin/bash
eval ${PRELOAD_FLAG} ${BIN_DIR}/lud_cuda -s 128 -v > stdout.txt 2> stderr.txt
