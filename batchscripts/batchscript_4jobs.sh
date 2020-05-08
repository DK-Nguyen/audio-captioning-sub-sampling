#!/usr/bin/env bash

#SBATCH -J "AudioCap"
#SBATCH -o outputs/outputs_run/out_%A_%a.txt
#SBATCH -e outputs/errs_run/err_%A_%a.txt
#SBATCH --mem-per-cpu=70000
#SBATCH --nodes 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --array=1-4
#SBATCH -t 5-23:59:00

export PYTHONPATH=$PYTHONPATH:.
source activate audio-captioning

case $SLURM_ARRAY_TASK_ID in
  1) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_1/lr_1e-4_grad_0.5_loss_thr_1e-3 -v;;
  2) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_1/lr_1e-4_grad_0.5_loss_thr_1e-4 -v;;
  3) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_1/lr_5e-5_grad_1.0_loss_thr_1e-3 -v;;
  4) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_1/lr_5e-5_grad_1.0_loss_thr_1e-4 -v;;

esac

echo Done!
