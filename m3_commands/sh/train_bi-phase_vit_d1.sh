#!/bin/bash
#SBATCH --job-name=bi-phase_vit_d1
#SBATCH --account=sy86
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=../slurm_outputs/bi-phase_vit_d1_%j.out
#SBATCH --error=../slurm_outputs/bi-phase_vit_d1_%j.err

# Load required modules
module load python/3.9.0
module load cuda/11.8.0

# Activate virtual environment
source /home/ankitag/sy86/ankitag/venv/bin/activate

# Navigate to project directory
cd /home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run bi-phase VIT training
python src/training/train_bi-phase_flexible.py \
    --model_type vit \
    --embed_dim 128 \
    --num_classes 23073 \
    --epochs 150 \
    --warmup-epochs 15 \
    --lr 8e-4 \
    --batch_size 64 \
    --load_path pretrained/VITsingle_phase_D1.pth \
    --data_dir_train training_data/D1_full/train \
    --data_dir_val training_data/D1_full/val \
    --anno_struc training_data/D1_full/anno_struc.csv \
    --progress_bar True

echo "Bi-phase VIT training completed!"
