#!/bin/sh
RELATE_BIN_DIR="${1}" # should contain Relate and RelateFileFormats binaries
RELATE_TMP_DIR="${2}" # should contain .haps / .sample
MAP_FILE="${3}"
COAL_FILE="${4}"
IN_PREFIX="in"
MID_PREFIX="data"
OUT_PREFIX="out"

cd ${RELATE_TMP_DIR}
pwd

echo "Expecting ${IN_PREFIX}.haps, ${IN_PREFIX}.sample"
echo "Will write to ${OUT_PREFIX}.trees, will delete ${IN_PREFIX}*, ${MID_PREFIX}*"
echo "Map file = ${MAP_FILE}"
echo "Coal file = ${COAL_FILE}"

${RELATE_BIN_DIR}/Relate \
  --mode All \
  -m 1.65e-8 \
  --coal ${COAL_FILE} \
  --haps ${IN_PREFIX}.haps \
  --sample ${IN_PREFIX}.sample \
  --map ${MAP_FILE} \
  --seed 1 \
  -o ${MID_PREFIX}

${RELATE_BIN_DIR}/RelateFileFormats \
  --mode ConvertToTreeSequence \
  -i ${MID_PREFIX} \
  -o ${OUT_PREFIX}

pwd
echo "Removing the following files:"
ls ${IN_PREFIX}* ${MID_PREFIX}*
rm ${IN_PREFIX}* ${MID_PREFIX}*
