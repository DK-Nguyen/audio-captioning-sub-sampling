#!/usr/bin/env bash

#SBATCH -J "AudioCap"
#SBATCH -o outputs/outputs_run/out_%A_%a.txt
#SBATCH -e outputs/errs_run/err_%A_%a.txt
#SBATCH --mem-per-cpu=70000
#SBATCH --nodes 1
#SBATCH -p gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --array=1-20
#SBATCH -t 5-23:59:00

export PYTHONPATH=$PYTHONPATH:.
source activate audio-captioning

case $SLURM_ARRAY_TASK_ID in
  1) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling1/no_attention/sub1Dropping_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  2) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling2/no_attention/sub2Dropping_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  3) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling4/no_attention/sub4Dropping_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  4) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling8/no_attention/sub8Dropping_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  5) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling16/no_attention/sub16Dropping_sameTrainingParamsBaseline_1000epochs_noAttention -v;;

  6) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling1/no_attention/sub1Linear_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  7) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling2/no_attention/sub2Linear_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  8) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling4/no_attention/sub4Linear_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  9) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling8/no_attention/sub8Linear_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  10) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling16/no_attention/sub16Linear_sameTrainingParamsBaseline_1000epochs_noAttention -v;;

  11) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling1/no_attention/sub1Maxout_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  12) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling2/no_attention/sub2Maxout_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  13) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling4/no_attention/sub4Maxout_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  14) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling8/no_attention/sub8Maxout_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  15) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling16/no_attention/sub16Maxout_sameTrainingParamsBaseline_1000epochs_noAttention -v;;

  16) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling1/no_attention/sub1Rnn_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  17) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling2/no_attention/sub2Rnn_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  18) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling4/no_attention/sub4Rnn_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  19) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling8/no_attention/sub8Rnn_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
  20) python main.py -c main_settings -j ${SLURM_JOBID} -d settings/subsampling16/no_attention/sub16Rnn_sameTrainingParamsBaseline_1000epochs_noAttention -v;;
esac

echo Done!
