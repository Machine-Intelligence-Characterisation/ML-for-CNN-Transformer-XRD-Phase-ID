#!/usr/bin/env python3
"""
Training script for VIT (Attention-only) model on D1_full dataset
Based on train_single-phase.py but modified for VIT architecture
"""

import argparse
import math
import os
import sys

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from data_loading.dataset import XrdDataset
from models.ATTENTIONonly import VIT
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
        cls_acc += get_acc(logits, label_cls)

        if args.progress_bar:
            pbar.update(data.shape[0])
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{get_acc(logits, label_cls):.4f}'})

    if args.progress_bar:
        pbar.close()

    return epoch_loss / len(dataloader), cls_acc / len(dataloader)


def adjust_learning_rate_withWarmup(optimizer, epoch, args):
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr * epoch / args.warmup_epochs
    else:
        lr = lr * 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VIT Training on D1_full Dataset')
    
    # Model parameters
    parser.add_argument('--embed_dim', default=128, type=int, help='embedding dimension')
    parser.add_argument('--num_classes', default=23073, type=int, help='number of classes')
    
    # Training parameters
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--warmup-epochs', default=20, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', '--learning-rate', default=8e-5, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    
    # Data parameters
    parser.add_argument('--data_dir_train', default='training_data/D1_full/train/', type=str)
    parser.add_argument('--data_dir_val', default='training_data/D1_full/val/', type=str)
    parser.add_argument('--anno_train', default='training_data/D1_full/anno_train.csv', type=str)
    parser.add_argument('--anno_val', default='training_data/D1_full/anno_val.csv', type=str)
    
    # Output parameters
    parser.add_argument('--output_dir', default='trained_models/', type=str)
    parser.add_argument('--model_name', default='VIT_D1_full', type=str)
    
    # Other parameters
    parser.add_argument('--progress_bar', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', type=str)
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model = VIT(embed_dim=args.embed_dim, num_classes=args.num_classes)
    model.to(device)
    
    # Initialize datasets
    trainset = XrdDataset(args.data_dir_train, args.anno_train)
    valset = XrdDataset(args.data_dir_val, args.anno_val)
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")
    
    # Initialize optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = FocalLoss(alpha=1, gamma=2, num_classes=args.num_classes)
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training
        train_loss, train_acc = run_one_epoch(model, trainloader, criterion, optimizer, epoch, 'Train')
        
        # Validation
        val_loss, val_acc = run_one_epoch(model, valloader, criterion, optimizer, epoch, 'Val')
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'args': args
            }, f"{args.output_dir}/{args.model_name}_best.pth")
            print(f"New best model saved with accuracy: {val_acc:.4f}")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'args': args
            }, f"{args.output_dir}/{args.model_name}_epoch_{epoch+1:03d}.pth")
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.4f}")
