#!/usr/bin/env python3
"""
Custom Dataset Training Setup for SimXRD-4M Data
Adapts CPICANN training for your own XRD dataset
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

# Add src to path
sys.path.append('/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/src')

from models.CPICANN import CPICANN
from utils.focal_loss import FocalLoss
from utils.logger import Logger


class SimXRDDataset(Dataset):
    """
    Custom dataset class for SimXRD-4M data
    Adapts to your specific data format and structure
    """
    
    def __init__(self, data_dir, annotations_file, transform=None):
        """
        Initialize SimXRD dataset
        
        Args:
            data_dir: Directory containing XRD pattern files
            annotations_file: CSV file with sample IDs and labels
            transform: Optional data transformations
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load annotations
        self.annotations = pd.read_csv(annotations_file)
        
        # Extract unique classes for mapping
        self.unique_classes = sorted(self.annotations['label'].unique())
        self.num_classes = len(self.unique_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}
        
        print(f"Dataset loaded: {len(self.annotations)} samples, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Get sample info
        sample_info = self.annotations.iloc[idx]
        sample_id = sample_info['sample_id']  # Adjust column name as needed
        label = sample_info['label']  # Adjust column name as needed
        
        # Load XRD pattern data
        data_path = os.path.join(self.data_dir, f"{sample_id}.csv")  # Adjust file extension as needed
        
        try:
            # Load data (adjust format as needed)
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path, header=None).values.astype(np.float32)
            elif data_path.endswith('.npy'):
                data = np.load(data_path).astype(np.float32)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            # Ensure data is in correct format (2D: [channels, features])
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim == 2 and data.shape[0] > data.shape[1]:
                data = data.T  # Transpose if needed
            
            # Apply transformations
            if self.transform:
                data = self.transform(data)
            
            # Convert label to index
            label_idx = self.class_to_idx[label]
            
            return torch.tensor(data, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)
            
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            # Return dummy data to avoid breaking the training loop
            dummy_data = torch.zeros(1, 1000, dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_data, dummy_label


def create_simxrd_annotations(data_dir, output_file, train_split=0.8):
    """
    Create annotation file from SimXRD-4M data directory
    
    Args:
        data_dir: Directory containing XRD files
        output_file: Path to save annotation CSV
        train_split: Fraction of data to use for training
    """
    print(f"Creating annotations from {data_dir}...")
    
    # Get all data files
    data_files = []
    for ext in ['*.csv', '*.npy', '*.txt']:
        data_files.extend(Path(data_dir).glob(ext))
    
    print(f"Found {len(data_files)} data files")
    
    # Extract labels from filenames or create dummy labels
    # Adjust this based on your SimXRD-4M data structure
    annotations = []
    
    for i, file_path in enumerate(data_files):
        sample_id = file_path.stem  # filename without extension
        
        # Option 1: Extract label from filename (adjust pattern as needed)
        # label = extract_label_from_filename(sample_id)
        
        # Option 2: Use dummy labels (replace with your actual labeling logic)
        label = f"class_{i % 100}"  # Create 100 dummy classes
        
        # Option 3: Load label from metadata file
        # label = load_label_from_metadata(sample_id)
        
        annotations.append({
            'sample_id': sample_id,
            'label': label,
            'file_path': str(file_path)
        })
    
    # Create DataFrame
    df = pd.DataFrame(annotations)
    
    # Split into train/val
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    split_idx = int(len(df) * train_split)
    
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    # Save annotations
    train_file = output_file.replace('.csv', '_train.csv')
    val_file = output_file.replace('.csv', '_val.csv')
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    
    print(f"Created train annotations: {train_file} ({len(train_df)} samples)")
    print(f"Created val annotations: {val_file} ({len(val_df)} samples)")
    
    return train_file, val_file


def train_simxrd_model(config):
    """Train CPICANN model on SimXRD-4M data"""
    
    print("🚀 Starting SimXRD-4M Training")
    print("=" * 40)
    
    # Create datasets
    print("📥 Loading datasets...")
    train_dataset = SimXRDDataset(
        data_dir=config['data_dir'],
        annotations_file=config['train_annotations']
    )
    
    val_dataset = SimXRDDataset(
        data_dir=config['data_dir'],
        annotations_file=config['val_annotations']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    print("🏗️ Initializing model...")
    model = CPICANN(
        embed_dim=config['embed_dim'],
        num_classes=train_dataset.num_classes
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Initialize loss and optimizer
    criterion = FocalLoss(class_num=train_dataset.num_classes, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    
    # Initialize logger
    logger = Logger(val=True)
    
    print(f"✅ Model initialized: {train_dataset.num_classes} classes")
    print(f"✅ Device: {device}")
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Log results
        print(f"Epoch {epoch+1}/{config['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': train_dataset.num_classes
            }, f"{config['output_dir']}/best_model.pth")
            print(f"  💾 Best model saved! Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, f"{config['output_dir']}/checkpoint_epoch_{epoch+1}.pth")
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train CPICANN on SimXRD-4M data')
    
    # Data arguments
    parser.add_argument('--data_dir', required=True, help='Directory containing SimXRD-4M data files')
    parser.add_argument('--output_dir', default='./simxrd_training_outputs', help='Output directory for models and logs')
    parser.add_argument('--create_annotations', action='store_true', help='Create annotation files from data directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=128, help='Model embedding dimension')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Data split
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of data for training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'embed_dim': args.embed_dim,
        'num_workers': args.num_workers,
        'train_split': args.train_split
    }
    
    # Create annotations if requested
    if args.create_annotations:
        print("📝 Creating annotation files...")
        train_annotations, val_annotations = create_simxrd_annotations(
            args.data_dir, 
            os.path.join(args.output_dir, 'annotations.csv'),
            args.train_split
        )
        config['train_annotations'] = train_annotations
        config['val_annotations'] = val_annotations
    else:
        # Use existing annotation files
        config['train_annotations'] = os.path.join(args.output_dir, 'annotations_train.csv')
        config['val_annotations'] = os.path.join(args.output_dir, 'annotations_val.csv')
        
        if not os.path.exists(config['train_annotations']):
            print("❌ Training annotations not found. Use --create_annotations to generate them.")
            return
    
    # Start training
    train_simxrd_model(config)


if __name__ == "__main__":
    main()

