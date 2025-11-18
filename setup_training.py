#!/usr/bin/env python3
"""
Training script for CPICANN model on D1_full data
Replicates the paper: "Crystal phase identification with extreme learning machine and convolutional neural network"
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def create_training_config():
    """Create configuration for training with D1_full data"""
    
    config = {
        # Data paths
        'data_dir_train': '/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_data/D1_full/train',
        'data_dir_val': '/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_data/D1_full/val',
        'anno_train': '/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_data/D1_full/anno_train.csv',
        'anno_val': '/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_data/D1_full/anno_val.csv',
        
        # Model parameters
        'num_classes': 23073,
        'embed_dim': 128,
        
        # Training parameters (from paper)
        'epochs': 200,
        'warmup_epochs': 20,
        'lr': 8e-5,
        'batch_size': 128,
        'num_workers': 16,
        
        # Output directory
        'output_dir': '/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_outputs'
    }
    
    return config

def check_data_availability(config):
    """Check if all required data is available"""
    print("🔍 Checking data availability...")
    
    required_paths = [
        config['data_dir_train'],
        config['data_dir_val'],
        config['anno_train'],
        config['anno_val']
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("❌ Missing data paths:")
        for path in missing_paths:
            print(f"   {path}")
        return False
    
    # Check data counts
    try:
        import pandas as pd
        train_anno = pd.read_csv(config['anno_train'])
        val_anno = pd.read_csv(config['anno_val'])
        
        print(f"✅ Training annotations: {len(train_anno)} samples")
        print(f"✅ Validation annotations: {len(val_anno)} samples")
        
        # Check if data files exist
        train_files = os.listdir(config['data_dir_train'])
        val_files = os.listdir(config['data_dir_val'])
        
        print(f"✅ Training data files: {len(train_files)} files")
        print(f"✅ Validation data files: {len(val_files)} files")
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False
    
    return True

def create_training_script(config):
    """Create a modified training script for D1_full data"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Modified training script for D1_full data
Based on train_single-phase.py but adapted for your data paths
"""

import argparse
import math
import os
import sys

# Add src to path
sys.path.append('/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/src')

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loading.dataset import XrdDataset
from models.CPICANN import CPICANN
from utils.focal_loss import FocalLoss
from utils.logger import Logger


def get_acc(cls, label):
    cls_acc = sum(cls.argmax(1) == label.int()) / cls.shape[0]
    return cls_acc


def run_one_epoch(model, dataloader, criterion, optimizer, epoch, mode):
    if mode == 'Train':
        model.train()
        criterion.train()
        desc = 'Training... '
    else:
        model.eval()
        criterion.eval()
        desc = 'Evaluating... '

    epoch_loss, cls_acc = 0, 0
    if args.progress_bar:
        pbar = tqdm(total=len(dataloader.dataset), desc=desc, unit='data')
    iters = len(dataloader)
    
    for i, batch in enumerate(dataloader):
        data = batch[0].to(device)
        label_cls = batch[1].to(device)

        if mode == 'Train':
            adjust_learning_rate_withWarmup(optimizer, epoch + i / iters, args)

            logits = model(data)
            loss = criterion(logits, label_cls.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(data)
                loss = criterion(logits, label_cls.long())

        epoch_loss += loss.item()
        if args.progress_bar:
            pbar.update(len(data))
            pbar.set_postfix(**{{'loss': loss.item()}})

        _cls_acc = get_acc(logits, label_cls)
        cls_acc += _cls_acc.item()

    return epoch_loss / iters, cls_acc * 100 / iters


def print_log(epoch, loss_train, loss_val, acc_train, acc_val, lr):
    log.printlog('---------------- Epoch {{}} ----------------'.format(epoch))

    log.printlog('loss_train : {{}}'.format(round(loss_train, 4)))
    log.printlog('loss_val   : {{}}'.format(round(loss_val, 4)))

    log.printlog('acc_train  : {{}}%'.format(round(acc_train, 4)))
    log.printlog('acc_val    : {{}}%'.format(round(acc_val, 4)))

    log.train_writer.add_scalar('loss', loss_train, epoch)
    log.val_writer.add_scalar('loss', loss_val, epoch)

    log.train_writer.add_scalar('acc', acc_train, epoch)
    log.val_writer.add_scalar('acc', acc_val, epoch)

    log.train_writer.add_scalar('lr', lr, epoch)


def save_checkpoint(state, is_best, filepath, filename):
    if (state['epoch']) % 10 == 0 or state['epoch'] == 1:
        os.makedirs(filepath, exist_ok=True)
        torch.save(state, filepath + filename)
        log.printlog('checkpoint saved!')
        if is_best:
            torch.save(state, '{{}}/model_best.pth'.format(filepath))
            log.printlog('best model saved!')


def adjust_learning_rate_withWarmup(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    print('>>>>  Running on {{}}  <<<<'.format(device))

    model = CPICANN(embed_dim={config['embed_dim']}, num_classes={config['num_classes']})
    model.to(device)
    if rank == 0:
        log.printlog(model)

    trainset = XrdDataset(args.data_dir_train, args.anno_train)
    valset = XrdDataset(args.data_dir_val, args.anno_val)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=True)

        train_loader = DataLoader(trainset, batch_size={config['batch_size']}, num_workers={config['num_workers']}, pin_memory=True, drop_last=True, sampler=train_sampler)
        val_loader = DataLoader(valset, batch_size={config['batch_size']}, num_workers={config['num_workers']}, pin_memory=True, drop_last=True, sampler=val_sampler)

        model = DDP(model, device_ids=[device], output_device=local_rank, find_unused_parameters=False)
    else:
        train_loader = DataLoader(trainset, batch_size={config['batch_size']}, num_workers={config['num_workers']}, pin_memory=True, shuffle=True)
        val_loader = DataLoader(valset, batch_size={config['batch_size']}, num_workers={config['num_workers']}, pin_memory=True, shuffle=True)

    criterion = FocalLoss(class_num={config['num_classes']}, device=device)

    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    start_epoch = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        loss_train, acc_train = run_one_epoch(model, train_loader, criterion, optimizer, epoch, mode='Train')

        loss_val, acc_val = run_one_epoch(model, val_loader, criterion, optimizer, epoch, mode='Eval')

        if rank == 0:
            print_log(epoch,  loss_train, loss_val, acc_train, acc_val, optimizer.param_groups[0]['lr'])
            save_checkpoint({{'epoch': epoch,
                             'model': model.module.state_dict() if distributed else model.state_dict(),
                             'optimizer': optimizer}}, is_best=False,
                            filepath='{{}}/checkpoints/'.format(log.get_path()),
                            filename='checkpoint_{{:04d}}.pth'.format(epoch))


if __name__ == '__main__':
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
        print(f"[init] == local rank: {{local_rank}}, global rank: {{rank}} ==")
        distributed = True
    else:
        rank = 0
        device = 'cuda:0'
        distributed = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--progress_bar", type=bool, default=True)

    parser.add_argument('--epochs', default={config['epochs']}, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup-epochs', default={config['warmup_epochs']}, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('--lr', '--learning-rate', default={config['lr']}, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')

    parser.add_argument('--data_dir_train', default='{config['data_dir_train']}', type=str)
    parser.add_argument('--data_dir_val', default='{config['data_dir_val']}', type=str)
    parser.add_argument('--anno_train', default='{config['anno_train']}', type=str,
                        help='path to annotation file for training data')
    parser.add_argument('--anno_val', default='{config['anno_val']}', type=str,
                        help='path to annotation file for validation data')
    parser.add_argument('--num_classes', default={config['num_classes']}, type=int, metavar='N')

    args = parser.parse_args()

    if rank == 0:
        log = Logger(val=True)

    main()
    print('THE END')
'''
    
    script_path = '/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/train_d1_full.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path

def create_slurm_job_script(config):
    """Create SLURM job script for HPC training"""
    
    slurm_script = f'''#!/bin/bash
#SBATCH --job-name=CPICANN_D1_Training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=sy86

#SBATCH --output={config['output_dir']}/slurm_outputs/SLURM%j.out
#SBATCH --error={config['output_dir']}/slurm_outputs/SLURM%j.err

# Load necessary modules
module load cuda

# Display GPU information
nvidia-smi

# Activate virtual environment
source /home/ankitag/sy86/ankitag/xrd_env/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH=/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/src:$PYTHONPATH

# Create output directory
mkdir -p {config['output_dir']}/slurm_outputs

# Run the training script
cd /home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase
python train_d1_full.py

echo "Training completed!"
'''
    
    script_path = '/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/submit_training_job.sh'
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    print("🚀 CPICANN Training Setup for D1_full Data")
    print("=" * 50)
    
    # Create configuration
    config = create_training_config()
    
    # Check data availability
    if not check_data_availability(config):
        print("\n❌ Data not ready. Please wait for rsync transfer to complete.")
        print("Run: ./check_progress.sh")
        return
    
    print("\n✅ Data is ready!")
    
    # Create training script
    print("\n📝 Creating training script...")
    train_script = create_training_script(config)
    print(f"✅ Created: {train_script}")
    
    # Create SLURM job script
    print("\n📝 Creating SLURM job script...")
    slurm_script = create_slurm_job_script(config)
    print(f"✅ Created: {slurm_script}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(f"{config['output_dir']}/slurm_outputs", exist_ok=True)
    
    print("\n🎯 Training Options:")
    print("=" * 30)
    print("1. Local training (if you have GPU):")
    print(f"   python {train_script}")
    print()
    print("2. HPC training (recommended):")
    print(f"   sbatch {slurm_script}")
    print()
    print("3. Monitor job:")
    print("   squeue -u $USER")
    print("   tail -f /home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_outputs/slurm_outputs/SLURM<JOB_ID>.out")
    
    print("\n📊 Training Configuration:")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Learning Rate: {config['lr']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Warmup Epochs: {config['warmup_epochs']}")
    print(f"   Model: CPICANN (embed_dim={config['embed_dim']})")
    print(f"   Classes: {config['num_classes']}")
    
    print("\n✅ Setup complete! Ready to train.")

if __name__ == "__main__":
    main()
