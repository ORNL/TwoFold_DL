#!/usr/bin/env bash

export NENSEMBLE=1

module load gcc/9.3.0
module load open-ce/1.5.0-py39-0
conda activate /autofs/nccs-svm1_proj/bif136/summit-env

export HF_DATASETS_CACHE=/tmp
export TRANSFORMERS_CACHE=/tmp
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export NNODES=`echo $LSB_MCPU_HOSTS | awk '{for (j=3; j <= NF; j+=2) { print $j }}' | wc -l`
export NNODES=$((${NNODES}/${NENSEMBLE}))
export NWORKERS=$((${NNODES}*6))

export NCROSS=3
export LR=3e-5
export PER_DEVICE_BATCH_SIZE=1
export CLUSTER=summit
export BATCH_SIZE=$((${PER_DEVICE_BATCH_SIZE}*${NWORKERS}))

export TORCH_DISTRIBUTED_DEBUG=INFO

export ENSEMBLE_ID=1

export GLOBAL_ID=$(((${LSB_JOBINDEX}-1)*${NENSEMBLE}+${ENSEMBLE_ID}))
export ID_STR=${CLUSTER}_bs${BATCH_SIZE}_lr${LR}_ncross_${NCROSS}_${GLOBAL_ID}

export LD_PRELOAD="${OLCF_GCC_ROOT}/lib64/libstdc++.so.6"

