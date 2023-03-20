#!/bin/sh

IMPUTE4_PATH=$1
DATA_DIR=$2
GENETIC_MAP=$3
LENGTH=$4

cd ${DATA_DIR}

${IMPUTE4_PATH} \
    -h ref.hap \
    -l ref.legend \
    -m ${GENETIC_MAP} \
    -g in.haps \
    -no_maf_align \
    -int 0 ${LENGTH} \
    -o out \
    -o_gz
