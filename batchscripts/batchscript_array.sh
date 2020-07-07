#!/usr/bin/env bash

#SBATCH -J "AudioCap"
#SBATCH -o outputs/outputs_run/out_%A_%a.txt
#SBATCH -e outputs/errs_run/err_%A_%a.txt
#SBATCH --mem-per-cpu=70000
#SBATCH --nodes 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --array=1-5
#SBATCH -t 5-23:59:00

export PYTHONPATH=$PYTHONPATH:.
source activate audio-captioning

case $SLURM_ARRAY_TASK_ID in
  1) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling1/sub1_same_training_params_baseline_epoch_1000_attn_1 -v;;
  2) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling2/sub2_same_training_params_baseline_epoch_1000_attn_1 -v;;
  3) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling4/sub4_same_training_params_baseline_epoch_1000_attn_1 -v;;
  4) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling8/sub8_same_training_params_baseline_epoch_1000_attn_1 -v;;
  5) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling16/sub16_same_training_params_baseline_epoch_1000_attn_1 -v;;
esac

echo Done!
