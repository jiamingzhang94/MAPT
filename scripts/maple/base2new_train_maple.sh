#!/bin/bash

#cd ../..
export CUDA_VISIBLE_DEVICES=0
# custom config
DATA="/home/dycpu6_8tssd1/jmzhang/datasets/"
TRAINER=MaPLe

DATASET=oxford_flowers
SEED=1

CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--adv-train True \


#if [ -d "$DIR" ]; then
#    echo "Results are available in ${DIR}. Resuming..."
#    python train.py \
#    --root ${DATA} \
#    --seed ${SEED} \
#    --trainer ${TRAINER} \
#    --dataset-config-file configs/datasets/${DATASET}.yaml \
#    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#    --output-dir ${DIR} \
#    DATASET.NUM_SHOTS ${SHOTS} \
#    DATASET.SUBSAMPLE_CLASSES base
#else
#    echo "Run this job and save the output to ${DIR}"
#    python train.py \
#    --root ${DATA} \
#    --seed ${SEED} \
#    --trainer ${TRAINER} \
#    --dataset-config-file configs/datasets/${DATASET}.yaml \
#    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#    --output-dir ${DIR} \
#    DATASET.NUM_SHOTS ${SHOTS} \
#    DATASET.SUBSAMPLE_CLASSES base
#fi