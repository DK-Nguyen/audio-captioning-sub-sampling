#!/usr/bin/env bash

#SBATCH -J "AudioCap"
#SBATCH -o outputs/outputs_run/out_%A_%a.txt
#SBATCH -e outputs/errs_run/err_%A_%a.txt
#SBATCH --mem-per-cpu=70000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --nodes 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH -t 5-23:59:00

export PYTHONPATH=$PYTHONPATH:.
source activate audio-captioning

python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling8/sub8_params_same_baseline_hasdecdropout_epoch_1000 -v

echo Done!
