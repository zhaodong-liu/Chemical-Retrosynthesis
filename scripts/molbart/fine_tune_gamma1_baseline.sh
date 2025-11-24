#!/bin/bash

python -m molbart.fine_tune \
  --dataset UsptoTXT_gamma \
  --data_path /gpfsnyu/scratch/hh3043/chem_new/NLP_project/retro_111.txt \
  --model_path /gpfsnyu/scratch/hh3043/Chemformer/models/pre-trained/combined/step_1000000.ckpt \
  --task backward_prediction \
  --epochs 10 \
  --lr 0.001 \
  --schedule cycle \
  --batch_size 128 \
  --acc_batches 4 \
  --augment all \
  --aug_prob 0.5 \
  --gpus 1 \
  --gamma 1 \
  # --fix_encoder \
  # --encoder_lora \
  # --fix_encoder \
  # --fix_encoder \
  # --fix_decoder \
  # --add_end_layer \
  # --insert_mid_layer \

