#!/usr/bin/env bash
#BSUB -P ABC123
#BSUB -W 2:00
#BSUB -nnodes 16
#BSUB -q batch
#BSUB -J "structure[1]"
#BSUB -o structure.o%J
#BSUB -e structure.e%J

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

export TRAIN_DATASET='/path/to/train/set'
export TEST_DATASET='/path/to/test/set'

export NCROSS=3
export LR=1e-5
export PER_DEVICE_BATCH_SIZE=1
export CLUSTER=summit
export BATCH_SIZE=$((${PER_DEVICE_BATCH_SIZE}*${NWORKERS}))

export TORCH_DISTRIBUTED_DEBUG=INFO

for ENSEMBLE_ID in `seq 1 ${NENSEMBLE}`; do
    export GLOBAL_ID=$(((${LSB_JOBINDEX}-1)*${NENSEMBLE}+${ENSEMBLE_ID}))
    export ID_STR=${CLUSTER}_bs${BATCH_SIZE}_lr${LR}_ncross_${NCROSS}_${GLOBAL_ID}

    final_checkpoint=`ls -t ./results_${ID_STR}  | grep "checkpoint" | head -1 | awk 'BEGIN {FS="-"} {printf $2}'`
    echo "Removing ./results_${ID_STR}/checkpoint-$final_checkpoint to ensure we progress in training" 
    rm -r ./results_${ID_STR}/checkpoint-$final_checkpoint

    export LD_PRELOAD="${OLCF_GCC_ROOT}/lib64/libstdc++.so.6"
    jsrun -n ${NNODES} -g 6 -a 6 -c 42 python ../train.py \
    --smiles_tokenizer_dir='/gpfs/alpine/world-shared/med106/blnchrd/models/bert_large_plus_clean_regex/tokenizer'\
    --smiles_model_dir='/gpfs/alpine/world-shared/med106/blnchrd/automatedmutations/pretraining/run/job_86neeM/output'\
    --model_type='regex' \
    --seq_model_name='Rostlab/prot_bert_bfd'\
    --train_dataset=${TRAIN_DATASET}\
    --test_dataset=${TEST_DATASET}\
    --train_size=13753\
    --n_cross_attn=${NCROSS}\
    --output_dir=./results_${ID_STR}\
    --max_steps=150000\
    --per_device_train_batch_size=${PER_DEVICE_BATCH_SIZE}\
    --per_device_eval_batch_size=${PER_DEVICE_BATCH_SIZE}\
    --learning_rate=${LR}\
    --weight_decay=0.01\
    --logging_dir=./logs_${ID_STR}\
    --logging_steps=1\
    --lr_scheduler_type=constant_with_warmup\
    --evaluation_strategy="steps"\
    --eval_steps=100\
    --gradient_accumulation_steps=1\
    --fp16=False\
    --save_strategy="steps"\
    --save_steps=100\
    --warmup_steps=10\
    --optim=adafactor\
    --ignore_data_skip\
    --gradient_checkpointing\
    --seed=$((42+${GLOBAL_ID})) &
done
wait
