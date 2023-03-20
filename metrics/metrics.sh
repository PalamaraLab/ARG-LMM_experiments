#!/bin/bash
# Example script for running metrics.py

NUM_SNP_SAMPLES=300
NUM_SEQUENCE_SAMPLES=0
SIM_LENGTH=5e6
# NUM_SNP_SAMPLES=0
# NUM_SEQUENCE_SAMPLES=300
# SIM_LENGTH=1e6

# thread is ARG-Needle, upgma is ASMC-clust
ALGORITHMS=("thread" "upgma" "relate" "tsinfer" "tsinfer_sparse")
ALGORITHM_INDEX=0 # can change this to use other inference modes
ALGORITHM=${ALGORITHMS[$(($ALGORITHM_INDEX))]}

# CONDITION=0 DECODING_INDEX=0  # CEU
# CONDITION=1 DECODING_INDEX=1  # const15k
# CONDITION=2 DECODING_INDEX=2  # const10k
# CONDITION=5 DECODING_INDEX=0  # CEU, rho x 2
# CONDITION=6 DECODING_INDEX=0  # CEU, rho / 2
CONDITION=0
DECODING_INDEX=0

HASH_TOPK=64
SEQUENCE_HASH_CM=0.1
SNP_HASH_CM=0.5
HASH_WORD_SIZE=16
HASH_TOLERANCE=1
ASMC_PAD_CM=2
ASMC_CLUST_CHUNK_SITES=300

HASH_VERSION=4 # only used for file naming
DECODER_MAP_MEAN_CHOICE=3 # only used for file naming

ASMC_DECODING_PREFIX="../common/decoding_quantities/" # relative to the metrics/ directory
DECODING_STRINGS=("30-100-2000" "30-100-2000_const15k" "30-100-2000_const10k")
ASMC_DECODING_STRING="${DECODING_STRINGS[$DECODING_INDEX]}"
ASMC_DECODING_FILE="${ASMC_DECODING_PREFIX}${ASMC_DECODING_STRING}.decodingQuantities.gz"

START_SEED=1 # Can edit this
NUM_SEEDS=1
echo "Start seed: ${START_SEED}"

KC_MERGING=0 # 1 is comprehensive, but slower
SAVE_TREES=1
STAB_EVAL_ONLY=1
TSINFER_RESOLVE=0 # 1 is comprehensive, but slower
PERMUTATION_SEED=0
GENOTYPING_ERROR=0

CREDIBLE=0 # only used for file naming
ASMC_DECODING_STRING="_c${CREDIBLE}"

if [[ ${DECODING_INDEX} == "1" ]]; then ASMC_DECODING_STRING="${ASMC_DECODING_STRING}_q${DECODING_STRINGS[$DECODING_INDEX]}"; fi
if [[ ${DECODING_INDEX} == "2" ]]; then ASMC_DECODING_STRING="${ASMC_DECODING_STRING}_q${DECODING_STRINGS[$DECODING_INDEX]}"; fi

ASMC_CLUST=0
if [[ $ALGORITHM = "upgma" ]]; then ASMC_CLUST=1; fi

# Form unique strings for saving experiment results
if [[ $ALGORITHM = "thread" ]]; then ALGORITHM_STRING="thread${DECODER_MAP_MEAN_CHOICE}${ASMC_DECODING_STRING}_pad${ASMC_PAD_CM}_hk${HASH_TOPK}_hv${HASH_VERSION}${HASH_TOLERANCE}_hw${HASH_WORD_SIZE}"; fi
if [[ $ALGORITHM = "upgma" ]]; then ALGORITHM_STRING="upgma${ASMC_DECODING_STRING}_pad${ASMC_PAD_CM}"; fi
if [[ $ALGORITHM = "relate" ]]; then ALGORITHM_STRING="${ALGORITHM}"; fi
if [[ $ALGORITHM = "tsinfer" ]]; then ALGORITHM_STRING="${ALGORITHM}"; fi
if [[ $ALGORITHM = "tsinfer_sparse" ]]; then ALGORITHM_STRING="${ALGORITHM}"; fi
if [[ $PERMUTATION_SEED != 0 ]]; then ALGORITHM_STRING="${ALGORITHM_STRING}_p${PERMUTATION_SEED}"; fi
if [[ $GENOTYPING_ERROR != 0 ]]; then ALGORITHM_STRING="${ALGORITHM_STRING}_g${GENOTYPING_ERROR}"; fi
ALGORITHM_STRING="${ALGORITHM_STRING}_ns${NUM_SEQUENCE_SAMPLES}_na${NUM_SNP_SAMPLES}_l${SIM_LENGTH}_s${START_SEED}-${NUM_SEEDS}"

# snarg is shorthand for "SNP-inferred ARG"
BASE_PATH="../data/snarg_cond${CONDITION}/regular/"
mkdir -p "${BASE_PATH}"
BASE_PATH="${BASE_PATH}${ALGORITHM_STRING}"

echo "Base path: ${BASE_PATH}"
PKL_PATH="${BASE_PATH}.pkl"

if [ -f ${BASE_PATH}.log ]; then mv ${BASE_PATH}.log ${BASE_PATH}.log.bk; fi
if [ -f ${BASE_PATH}.pkl ]; then mv ${BASE_PATH}.pkl ${BASE_PATH}.pkl.bk; fi
echo () { builtin echo "$@" | tee -a ${BASE_PATH}.log; }

echo "*********************************"
echo "Run on host: `hostname`"
echo "Operating system: `uname -s`"
echo "Username: `whoami`"
echo "Started at: `date`"
echo "*********************************"

export OMP_NUM_THREADS=${N_SLOTS:-1}
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# https://stackoverflow.com/a/12797512
python3 -u metrics.py \
    --condition ${CONDITION} \
    --genotyping_error ${GENOTYPING_ERROR} \
    --kc_merging ${KC_MERGING} \
    --num_seeds ${NUM_SEEDS} \
    --permutation_seed ${PERMUTATION_SEED} \
    --pkl_path ${PKL_PATH} \
    --save_trees ${SAVE_TREES} \
    --sim_length ${SIM_LENGTH} \
    --stab_eval_only ${STAB_EVAL_ONLY} \
    --start_seed ${START_SEED} \
    --tsinfer_resolve ${TSINFER_RESOLVE} \
    `# number of samples` \
    --num_sequence_samples ${NUM_SEQUENCE_SAMPLES} \
    --num_snp_samples ${NUM_SNP_SAMPLES} \
    `# ARG inference parameters` \
    --algorithm ${ALGORITHM} \
    --asmc_clust ${ASMC_CLUST} \
    --asmc_clust_chunk_sites ${ASMC_CLUST_CHUNK_SITES} \
    --asmc_decoding_file ${ASMC_DECODING_FILE} \
    --asmc_pad_cm ${ASMC_PAD_CM} \
    --asmc_tmp_string ${ALGORITHM_STRING} \
    --hash_tolerance ${HASH_TOLERANCE} \
    --hash_topk ${HASH_TOPK} \
    --hash_word_size ${HASH_WORD_SIZE} \
    --sequence_hash_cm ${SEQUENCE_HASH_CM} \
    --snp_hash_cm ${SNP_HASH_CM} \
    2>&1 | tee -a ${BASE_PATH}.log

echo "********************************************"
echo "Finished at: `date`"
echo "********************************************"
exit 0
