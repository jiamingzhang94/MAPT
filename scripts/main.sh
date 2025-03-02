#!/bin/bash

#cd ../..
#export CUDA_VISIBLE_DEVICES=2
# custom config
DATA="/home/dycpu6_8tssd1/jmzhang/datasets/"
TRAINER=MaPLe

#DATASETS=(
#    "oxford_pets"
#    "oxford_flowers"
#    "fgvc_aircraft"
#    "dtd"
#    "eurosat"
#    "stanford_cars"
#    "food101"
#    "sun397"
#    "caltech101"
#    "ucf101"
#    "imagenet"
#    # "imagenet_sketch"
#    # "imagenetv2"
#    # "imagenet_a"
#    # "imagenet_r"
#)
DATASET=oxford_pets
SEED=1
CFG=vit_b16_c2_ep5_batch4_2ctx

DIR=checkpoints/${DATASET}/${TRAINER}/${CFG}/feature
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--adv-train True \
--feature True


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