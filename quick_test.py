#!/usr/bin/env python3
"""
Quick test script for pretrained models
Tests one model on a small subset of data
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append('src')

from data_loading.dataset import XrdDataset
from models.CPICANN import CPICANN
from models.CNNonly import CNN
from models.ATTENTIONonly import VIT

def quick_test():
    """Quick test of a pretrained model"""
    print("🚀 Quick Pretrained Model Test")
    print("=" * 40)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = XrdDataset(
        data_dir="training_data/D1_full/val",
        annotations_file="training_data/D1_full/anno_val.csv"
    )
    print(f"✓ Dataset created: {len(dataset)} samples")
    
    # Use only first 100 samples for quick test
    subset = Subset(dataset, range(min(100000, len(dataset))))
    data_loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    # Test CPICANN model (using single-phase instead of bi-phase due to corruption)
    model_path = "/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/trained_models/ATTENTIONonlyD1.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    
    try:
        # Load model
        model = VIT(embed_dim=128, num_classes=23073)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
        
        # Test model
        print("Testing model...")
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc="Testing")):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                pred = output.argmax(dim=1)
                
                # Calculate accuracy
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Print sample details for first batch
                if batch_idx == 0:
                    print(f"\n📊 Sample Analysis (Batch {batch_idx + 1}):")
                    print(f"  Input shape: {data.shape}")
                    print(f"  Sample input (first 10 values): {data[0, :10].cpu().numpy()}")
                    print(f"  True labels: {target.cpu().numpy()}")
                    print(f"  Model output logits (first 5): {output[0, :5].cpu().numpy()}")
                    print(f"  Predicted classes: {pred.cpu().numpy()}")
        
        accuracy = 100. * correct / total
        print(f"\n🎯 Results:")
        print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"  Samples tested: {total}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

if __name__ == "__main__":
    quick_test()

