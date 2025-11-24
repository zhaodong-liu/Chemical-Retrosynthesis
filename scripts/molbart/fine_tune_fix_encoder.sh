#!/bin/bash

python -m molbart.fine_tune \
  --dataset uspto_50 \
  --data_path /scratch/zl4789/NLP/project/Chemformer-release-1.0/data/seq-to-seq_datasets/uspto_50.pickle \
  --model_path /scratch/zl4789/NLP/project/Chemformer-release-1.0/models/pre-trained/combined/step=1000000.ckpt \
  --task backward_prediction \
  --epochs 100 \
  --lr 0.001 \
  --schedule cycle \
  --batch_size 128 \
  --acc_batches 4 \
  --augment all \
  --aug_prob 0.5 \
  --fix_encoder \
  --gpus 1 \

