#!/usr/bin/env bash

#SBATCH -J "AudioCap"
#SBATCH -o outputs/outputs_run/out_%A_%a.txt
#SBATCH -e outputs/errs_run/err_%A_%a.txt
#SBATCH --mem-per-cpu=70000
#SBATCH --nodes 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH -t 6-23:59:00

export PYTHONPATH=$PYTHONPATH:.
source activate audio-captioning

python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling1/sub1_same_training_params_baseline_epoch_1000_attn_2_512 -v

echo Done!
