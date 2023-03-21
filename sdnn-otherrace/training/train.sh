#!/bin/bash

#SBATCH --nodes=1    # Each node has 16 or 20 CPU cores.
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Elaheh.Akbarifathkouhi@psychol.uni-giessen.de

#SBATCH --partition=single
##SBATCH --partition=normal

#SBATCH --job-name=vgg_DualRace

#SBATCH --time=4-23:00:00

#SBATCH --gres=gpu:1

#SBATCH --mem=12G

#SBATCH --output='/home/elaheh_akbari/new/sdnn-otherrace/output/train/%A_%a.out'

CONFIG_FILE='../configs/vgg/face_dual_whitasia.yaml'
OUTPUT_FILE=./%A_%a.out

# SCRIPT=/home/elaheh_akbari/new/sdnn-otherrace/training/train.py
SCRIPT=./train.py

hostname
date

date
echo "Running python script..."
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT --config_file $CONFIG_FILE --num_epochs 200 --read_seed 1 --maxout True --save_freq 10 --valid_freq 1 --use_scheduler "True" --pretrained "True" # --custom_learning_rate 0.0001

date
echo "Job completed"
