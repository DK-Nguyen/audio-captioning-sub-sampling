#!/usr/bin/env bash

#SBATCH -J "AudioCap"
#SBATCH -o outputs/outputs_run/out_%A_%a.txt
#SBATCH -e outputs/errs_run/err_%A_%a.txt
#SBATCH --mem-per-cpu=70000
#SBATCH --nodes 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --array=1-12
#SBATCH -t 5-23:59:00

export PYTHONPATH=$PYTHONPATH:.
source activate audio-captioning

case $SLURM_ARRAY_TASK_ID in
  1) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_2_128_class_weight_0.01 -v;;
  2) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_2_128_class_weight_0.2 -v;;
  3) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_2_256_class_weight_0.01 -v;;
  4) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_2_256_class_weight_0.2 -v;;
  5) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_2_512_class_weight_0.01 -v;;
  6) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_2_512_class_weight_0.2 -v;;
  7) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_3_128_class_weight_0.01 -v;;
  8) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_3_128_class_weight_0.2 -v;;
  9) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_3_256_class_weight_0.01 -v;;
  10) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_3_256_class_weight_0.2 -v;;
  11) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_3_512_class_weight_0.01 -v;;
  12) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/settings_downsampling_4/down_4_attn_3_512_class_weight_0.2 -v;;
esac


echo Done!
