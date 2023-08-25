#!/bin/bash
#SBATCH -p gpu                   # Asking to assign "gpu" partition(queue)
#SBATCH --gres=gpu:1         # Asking to assign one GPU. You can ask two or more GPUs, but unless your project is coded to run parallel, it will only waste resources.
#SBATCH —time=36:00:00         # Asking time. This example sets to run unto 12 hours. "gpu" currently let you run an experiment upto 36 hours.
#SBATCH --output sample.out    # Indicates a file to redirect STDOUT to. "test.out.%j" would append job ID to the filename.
#SBATCH --error sample.err       # Indicates a file to redirect STDERR to. "test.err.%j" would append job ID to the filename.

CUDA_VISIBLE_DEVICES=2 python ./code/epic55_finetuning.py \
-model_type distillbert \
-batch_size  16 \
-num_epochs 10 \
-max_len 512 \
-hist_len 2 \
-checkpoint_path ./out/distilbert_pretrained/checkpoint-200000 \
-weigh_classes False \
-multi_task True

CUDA_VISIBLE_DEVICES=2 python ./code/epic55_finetuning.py \
-model_type distillbert \
-batch_size  16 \
-num_epochs 10 \
-max_len 512 \
-hist_len 3 \
-checkpoint_path ./out/distilbert_pretrained/checkpoint-200000 \
-weigh_classes False \
-multi_task True

CUDA_VISIBLE_DEVICES=2 python ./code/epic55_finetuning.py \
-model_type distillbert \
-batch_size  16 \
-num_epochs 10 \
-max_len 512 \
-hist_len 4 \
-checkpoint_path ./out/distilbert_pretrained/checkpoint-200000 \
-weigh_classes False \
-multi_task True