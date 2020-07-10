#!/usr/bin/env bash

#SBATCH -J "AudioCap"
#SBATCH -o outputs/outputs_run/out_%A_%a.txt
#SBATCH -e outputs/errs_run/err_%A_%a.txt
#SBATCH --mem-per-cpu=70000
#SBATCH --nodes 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --array=1-10
#SBATCH -t 5-23:59:00

export PYTHONPATH=$PYTHONPATH:.
source activate audio-captioning

case $SLURM_ARRAY_TASK_ID in
  1) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling1/sub1linear_same_training_params_baseline_epoch_1000_no_attention -v;;
  2) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling2/sub2linear_same_training_params_baseline_epoch_1000_no_attention -v;;
  3) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling4/sub4linear_same_training_params_baseline_epoch_1000_no_attention -v;;
  4) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling8/sub8linear_same_training_params_baseline_epoch_1000_no_attention -v;;
  5) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling16/sub16linear_same_training_params_baseline_epoch_1000_no_attention -v;;
  6) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling1/sub1maxoutlinear_same_training_params_baseline_epoch_1000_no_attention -v;;
  7) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling2/sub2maxoutlinear_same_training_params_baseline_epoch_1000_no_attention -v;;
  8) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling4/sub4maxoutlinear_same_training_params_baseline_epoch_1000_no_attention -v;;
  9) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling8/sub8maxoutlinear_same_training_params_baseline_epoch_1000_no_attention -v;;
  10) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling16/sub16maxoutlinear_same_training_params_baseline_epoch_1000_no_attention -v;;
esac

echo Done!
