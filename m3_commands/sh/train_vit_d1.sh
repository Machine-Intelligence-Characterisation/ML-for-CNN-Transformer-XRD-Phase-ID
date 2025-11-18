#!/bin/bash
#SBATCH --job-name=vit_d1_train
#SBATCH --output=../slurm_outputs/vit_d1_train_%j.out
#SBATCH --error=../slurm_outputs/vit_d1_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Load modules
module load python/3.9
module load cuda/11.8

# Activate environment (if using conda)
# conda activate xrd_env

# Navigate to project directory
cd /home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase

# Install required packages if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm pandas

# Run VIT training
python src/training/train_vit_d1.py \
    --epochs 150 \
    --warmup-epochs 15 \
    --lr 8e-5 \
    --batch_size 64 \
    --embed_dim 128 \
    --num_classes 23073 \
    --data_dir_train training_data/D1_full/train/ \
    --data_dir_val training_data/D1_full/val/ \
    --anno_train training_data/D1_full/anno_train.csv \
    --anno_val training_data/D1_full/anno_val.csv \
    --output_dir trained_models/ \
    --model_name VIT_D1_full \
    --device cuda:0

echo "VIT training completed!"
