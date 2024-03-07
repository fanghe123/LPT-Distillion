#!/bin/bash
#SBATCH -p hebhdnormal
#SBATCH -N 2
#SBATCH -n 64
#SBATCH --gres=dcu:4
#SBATCH -J zhengliu
source /public/home/acp15ony6v/miniconda3/etc/profile.d/conda.sh
conda activate paddle
module rm compiler/rocm/2.9
module load compiler/dtk/23.10
module list
cd /public/home/acp15ony6v/123/tools
python train.py -c /public/home/acp15ony6v/123/configs/rtdetr/rtdetr_r50vd_6x_coco.yml --slim_config /public/home/acp15ony6v/123/configs/slim/distill/rtdetr_distill.yml
