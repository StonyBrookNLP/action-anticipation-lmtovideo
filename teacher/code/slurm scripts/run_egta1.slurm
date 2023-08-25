#!/bin/bash
#SBATCH -p gpu                   # Asking to assign "gpu" partition(queue)
#SBATCH --gres=gpu:1         # Asking to assign one GPU. You can ask two or more GPUs, but unless your project is coded to run parallel, it will only waste resources.
#SBATCH —time=36:00:00         # Asking time. This example sets to run unto 12 hours. "gpu" currently let you run an experiment upto 36 hours.
#SBATCH --output sample.out    # Indicates a file to redirect STDOUT to. "test.out.%j" would append job ID to the filename.
#SBATCH --error sample.err       # Indicates a file to redirect STDERR to. "test.err.%j" would append job ID to the filename.

# CUDA_VISIBLE_DEVICES=0 python ./code/egtea_agg.py

# BERT based experiments

CUDA_VISIBLE_DEVICES=0 python ./code/egtea_finetuning.py \
-model_type roberta \
-batch_size 14 \
-num_epochs 4 \
-max_len 512 \
-checkpoint_path ./out/roberta_pretrained/checkpoint-200000 \
-weigh_classes True \
-hist_len 15 \
-gappy_hist True \
-multi_task True \
-sort_seg True \

CUDA_VISIBLE_DEVICES=0 python ./code/egtea_finetuning.py \
-model_type bert \
-batch_size 14 \
-num_epochs 4 \
-max_len 512 \
-checkpoint_path ./out/bert_pretrained/checkpoint-200000 \
-weigh_classes True \
-hist_len 15 \
-gappy_hist True \
-multi_task True \
-sort_seg True \