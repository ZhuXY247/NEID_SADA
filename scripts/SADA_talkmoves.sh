#!/usr/bin/env bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
mkdir -p logs save_models/checkpoints

for d in 'talkmoves' 'mathdial'
do
  for l in 0.1 0.25  # Labeled Ratio
  do
    for k in 0.5 0.75 # Known Class Ratio
    do
      for i in "CONTEXT" "ORIGINAL" # Input Strategy
      do
        for w in 0 1 # With Speaker
        do
          for s in 42 43 44 45 46 47 48 49 # Seed
          do
            r=0.35 
            strat="SADA"
            
            SAVE_DIR="./save_models/${d}_l${l}_k${k}_i${i}_w${w}_s${s}"
            LOG_FILE="logs/${d}_l${l}_k${k}_i${i}_w${w}_s${s}.log"
            S2_CKPT="./save_models/checkpoints/${d}_l${l}_k${k}_s${s}_in${i}_spk${w}_step2.pth"
            S3_CKPT="./save_models/checkpoints/${d}_l${l}_k${k}_s${s}_in${i}_spk${w}_r${r}_step3_student.pth"

            mkdir -p "$SAVE_DIR"

            python main.py \
              --internal_dataset $d \
              --internal_max_seq_length 85 \
              --external_dataset clinc \
              --external_max_seq_length 30 \
              --known_cls_ratio $k \
              --labeled_ratio $l \
              --seed $s \
              --num_pretrain_epochs 100 \
              --num_distillate_epochs 100 \
              --num_train_epochs 10 \
              --train_batch_size 64 \
              --view_strategy $strat \
              --lr '1e-5' \
              --bert_model "./model/bert-base-uncased" \
              --tokenizer "./model/bert-base-uncased" \
              --input_strategy $i \
              --with_speaker $w \
              --ratio $r \
              --mask_threshold $r \
              --save_model_path "${SAVE_DIR}/final" \
              --step2_ckpt "$S2_CKPT" \
              --step3_ckpt "$S3_CKPT" \
              --with_mask --with_pos \
              > "$LOG_FILE" 2>&1
          done
        done
      done
    done
  done
done