#!/bin/bash

CUDA_DEVICE=0
BATCH_SIZE=8
ACCUMULATE_STEP=4
LR=5e-5
EPOCHS=20
ENC_MAX_LENGTH=350
DEC_MAX_LENGTH=850
SEED=428
REASONING_GENERATION_ALPHA=3

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

for CUDA in 0
do
    TEST_TARGET="climate change is a real concern"
    DIR_SUFFIX=CC_LLM_ARG
    LOG_DIR=./logs/$DIR_SUFFIX
    mkdir $LOG_DIR
    for SEED in 1 2 3 4 5
    do
        OUTPUT_DIR=../save_dir/SEED_${SEED}_CC_LLM_ARG
        nohup accelerate launch main_sem16.py \
          --batch_size $BATCH_SIZE \
          --accumulate_step $ACCUMULATE_STEP \
          --output_dir $OUTPUT_DIR \
          --enc_max_length $ENC_MAX_LENGTH \
          --dec_max_length $DEC_MAX_LENGTH \
          --reasoning_generation_alpha $REASONING_GENERATION_ALPHA \
          --is_short \
          --is_long \
          --target_knowledge_made_by_llm \
          --with_unlikelihood_training \
          --with_topic_prediction \
          --seed $SEED \
          --lr $LR \
          --test_target "$TEST_TARGET" \
          --epoch $EPOCHS > ${LOG_DIR}/SEED_${SEED}.log
    done
done
