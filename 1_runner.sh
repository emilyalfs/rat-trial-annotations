#!/bin/bash -l
#SBATCH --gres=gpu:rtx_a4000:1
#SBATCH --mem=2GB
#SBATCH --time=0-02:20:00
#SBATCH --constraint=avx
#SBATCH --partition=ksu-gen-gpu.q

module load protobuf-python/4.24.0-GCCcore-12.3.0
source /homes/emilyalfs/yolo_rats/up_yolo/bin/activate
export PYTHONDONTWRITEBYTECODE=1

python keypoint.py $1 cuda $2
