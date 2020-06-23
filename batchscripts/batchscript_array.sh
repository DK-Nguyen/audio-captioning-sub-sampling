#!/usr/bin/env bash

#SBATCH -J "AudioCap"
#SBATCH -o outputs/outputs_run/out_%A_%a.txt
#SBATCH -e outputs/errs_run/err_%A_%a.txt
#SBATCH --mem-per-cpu=70000
#SBATCH --nodes 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --array=1-2
#SBATCH -t 5-23:59:00

export PYTHONPATH=$PYTHONPATH:.
source activate audio-captioning

case $SLURM_ARRAY_TASK_ID in
  1) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling1/sub1_no_attn_lr_1e-4_loss_thr_1e-4 -v;;
  2) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling1/sub1_no_attn_lr_1e-4_loss_thr_1e-4_clamp_0.5 -v;;
esac

echo Done!
