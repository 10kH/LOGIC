#!/bin/bash

CUDA_DEVICE=0
BATCH_SIZE=8
ACCUMULATE_STEP=4
LR=5e-6
EPOCHS=30
ENC_MAX_LENGTH=350
DEC_MAX_LENGTH=850
SEED=428
REASONING_GENERATION_ALPHA=3
N_SAMPLE=13477

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

for CUDA in 0
do
    DIR_SUFFIX=ALL_${N_SAMPLE}_b_${BATCH_SIZE}_accumualate_step_${ACCUMULATE_STEP}_for_generation
    LOG_DIR=./logs/$DIR_SUFFIX
    mkdir $LOG_DIR
    for SEED in 0
    do
        OUTPUT_DIR=../save_dir/ALL_${N_SAMPLE}_SEED_${SEED}_LR_${LR}_for_generation
        nohup accelerate launch main.py \
          --batch_size $BATCH_SIZE \
          --accumulate_step $ACCUMULATE_STEP \
          --output_dir $OUTPUT_DIR \
          --enc_max_length $ENC_MAX_LENGTH \
          --dec_max_length $DEC_MAX_LENGTH \
          --reasoning_generation_alpha $REASONING_GENERATION_ALPHA \
          --is_short \
          --is_long \
          --with_new_topic \
          --with_unlikelihood_training \
          --with_topic_prediction \
          --seed $SEED \
          --n_sample $N_SAMPLE \
          --lr $LR \
          --epoch $EPOCHS > ${LOG_DIR}/ALL_${N_SAMPLE}_SEED_${SEED}_LR_${LR}_for_generation.log
    done
done


<<'END'

for N_SAMPLE in 1500 3000 4500 6000 7500
do
    DIR_SUFFIX=ALL_${N_SAMPLE}_b_${BATCH_SIZE}_accumualate_step_${ACCUMULATE_STEP}_REASOING_ALPHA_${REASONING_GENERATION_ALPHA}_wiki_without_shuffling
    LOG_DIR=./logs/$DIR_SUFFIX
    mkdir $LOG_DIR
    for SEED in 0 1 2 3 4
    do
        OUTPUT_DIR=../save_dir/ALL_${N_SAMPLE}_SEED_${SEED}_LR_${LR}_REASOING_ALPHA_${REASONING_GENERATION_ALPHA}_wiki_without_shuffling
        nohup accelerate launch main.py \
          --batch_size $BATCH_SIZE \
          --accumulate_step $ACCUMULATE_STEP \
          --output_dir $OUTPUT_DIR \
          --enc_max_length $ENC_MAX_LENGTH \
          --dec_max_length $DEC_MAX_LENGTH \
          --reasoning_generation_alpha $REASONING_GENERATION_ALPHA \
          --is_short \
          --is_long \
          --with_wiki \
          --with_unlikelihood_training \
          --with_topic_prediction \
          --seed $SEED \
          --n_sample $N_SAMPLE \
          --lr $LR \
          --epoch $EPOCHS > ${LOG_DIR}/ALL_${N_SAMPLE}_SEED_${SEED}_LR_${LR}_REASOING_ALPHA_${REASONING_GENERATION_ALPHA}_wiki_without_shuffling.log
    done
done


for N_SAMPLE in $CUDA_DEVICE
do
    for SEED in 0 1 2 3 4
    do
        OUTPUT_DIR=../save_dir/ALL_${N_SAMPLE}_SEED_${SEED}_LR_${LR}_REASOING_ALPHA_${REASONING_GENERATION_ALPHA}_new_topic_chagpt
        nohup accelerate launch main.py \
          --batch_size $BATCH_SIZE \
          --accumulate_step $ACCUMULATE_STEP \
          --output_dir $OUTPUT_DIR \
          --enc_max_length $ENC_MAX_LENGTH \
          --dec_max_length $DEC_MAX_LENGTH \
          --reasoning_generation_alpha $REASONING_GENERATION_ALPHA \
          --is_short \
          --is_long \
          --with_new_topic \
          --with_shuffling \
          --with_unlikelihood_training \
          --with_topic_prediction \
          --seed $SEED \
          --n_sample $N_SAMPLE \
          --lr $LR \
          --epoch $EPOCHS > ${LOG_DIR}/ALL_${N_SAMPLE}_SEED_${SEED}_LR_${LR}_REASOING_ALPHA_${REASONING_GENERATION_ALPHA}_new_topic_chagpt.log
    done
done
END