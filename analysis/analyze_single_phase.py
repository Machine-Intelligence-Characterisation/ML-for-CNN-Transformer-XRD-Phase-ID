#!/usr/bin/env python3
"""
Single-phase CPICANN model analysis script
Analyzes confident predictions and finds examples of high-confidence single-phase identification
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pandas as pd

# Add src to path
sys.path.append('src')

from data_loading.dataset import XrdDataset
from models.CPICANN import CPICANN

def analyze_single_phase_predictions():
    """Analyze single-phase model predictions in detail"""
    print("🔬 Single-Phase CPICANN Model Analysis")
    print("=" * 50)
    
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
    
    # Load single-phase model
    model_path = "/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/trained_models/CPICANNsingle_phase_D1.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Loading single-phase model: {model_path}")
    
    try:
        # Load model
        model = CPICANN(embed_dim=128, num_classes=23073)
        
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
        print("✓ Single-phase model loaded successfully")
        
        # Test on 1000 samples for detailed analysis
        subset = Subset(dataset, range(1000))
        data_loader = DataLoader(subset, batch_size=32, shuffle=False)
        
        print("\nAnalyzing single-phase predictions...")
        
        all_outputs = []
        all_targets = []
        all_data_samples = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc="Analyzing")):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Store results
                all_outputs.append(output.cpu())
                all_targets.append(target.cpu())
                all_data_samples.append(data.cpu())
                
                # Print sample details for first batch
                if batch_idx == 0:
                    print(f"\n📊 Sample Analysis (First Batch):")
                    print(f"  Input shape: {data.shape}")
                    print(f"  True labels: {target.cpu().numpy()}")
                    print(f"  Model output logits (first 5): {output[0, :5].cpu().numpy()}")
                    
                    # Show probabilities for first sample
                    probs = F.softmax(output[0], dim=0)
                    top_probs, top_indices = torch.topk(probs, 5)
                    print(f"  Top-5 probabilities: {top_probs.cpu().numpy()}")
                    print(f"  Top-5 phase IDs: {top_indices.cpu().numpy()}")
        
        # Concatenate all outputs
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_data_samples = torch.cat(all_data_samples, dim=0)
        
        # Calculate probabilities
        probs = F.softmax(all_outputs, dim=1)
        top_1_probs, top_1_indices = torch.topk(probs, 1, dim=1)
        top_2_probs, top_2_indices = torch.topk(probs, 2, dim=1)
        
        # Find confident predictions
        print(f"\n🎯 Confidence Analysis:")
        print("=" * 40)
        
        # High confidence top-1 predictions (correct)
        high_conf_correct = []
        high_conf_incorrect = []
        
        for i, true_label in enumerate(all_targets):
            true_phase = true_label.item()
            pred_phase = top_1_indices[i, 0].item()
            pred_prob = top_1_probs[i, 0].item()
            
            if pred_prob > 0.8:  # High confidence threshold
                if pred_phase == true_phase:
                    high_conf_correct.append({
                        'sample_idx': i,
                        'true_phase': true_phase,
                        'pred_phase': pred_phase,
                        'pred_prob': pred_prob,
                        'data': all_data_samples[i]
                    })
                else:
                    high_conf_incorrect.append({
                        'sample_idx': i,
                        'true_phase': true_phase,
                        'pred_phase': pred_phase,
                        'pred_prob': pred_prob,
                        'data': all_data_samples[i]
                    })
        
        print(f"High confidence correct predictions (>0.8): {len(high_conf_correct)}")
        print(f"High confidence incorrect predictions (>0.8): {len(high_conf_incorrect)}")
        
        # Show examples of high confidence correct predictions
        if high_conf_correct:
            print(f"\n✅ High Confidence CORRECT Predictions:")
            print("-" * 60)
            for i, case in enumerate(high_conf_correct[:5]):  # Show first 5
                print(f"Sample {case['sample_idx']+1}:")
                print(f"  True Phase: {case['true_phase']}")
                print(f"  Predicted Phase: {case['pred_phase']}")
                print(f"  Confidence: {case['pred_prob']:.4f}")
                
                # Show top-5 predictions for this sample
                sample_probs = probs[case['sample_idx']]
                top_5_probs, top_5_indices = torch.topk(sample_probs, 5)
                print(f"  Top-5 predictions:")
                for j in range(5):
                    phase_id = top_5_indices[j].item()
                    prob = top_5_probs[j].item()
                    is_correct = "✓" if phase_id == case['true_phase'] else "✗"
                    print(f"    {j+1}. Phase {phase_id:5d} (prob: {prob:.4f}) {is_correct}")
                print()
        
        # Show examples of high confidence incorrect predictions
        if high_conf_incorrect:
            print(f"\n❌ High Confidence INCORRECT Predictions:")
            print("-" * 60)
            for i, case in enumerate(high_conf_incorrect[:3]):  # Show first 3
                print(f"Sample {case['sample_idx']+1}:")
                print(f"  True Phase: {case['true_phase']}")
                print(f"  Predicted Phase: {case['pred_phase']}")
                print(f"  Confidence: {case['pred_prob']:.4f}")
                
                # Show top-5 predictions for this sample
                sample_probs = probs[case['sample_idx']]
                top_5_probs, top_5_indices = torch.topk(sample_probs, 5)
                print(f"  Top-5 predictions:")
                for j in range(5):
                    phase_id = top_5_indices[j].item()
                    prob = top_5_probs[j].item()
                    is_correct = "✓" if phase_id == case['true_phase'] else "✗"
                    print(f"    {j+1}. Phase {phase_id:5d} (prob: {prob:.4f}) {is_correct}")
                print()
        
        # Find confident top-2 predictions (where true phase is in top-2)
        print(f"\n🔍 Confident Top-2 Predictions:")
        print("=" * 50)
        
        confident_top2 = []
        for i, true_label in enumerate(all_targets):
            true_phase = true_label.item()
            top_2_phases = top_2_indices[i].tolist()
            top_2_probs_values = top_2_probs[i].tolist()
            
            if true_phase in top_2_phases:
                # Calculate confidence as sum of top-2 probabilities
                confidence = sum(top_2_probs_values)
                if confidence > 0.9:  # High confidence threshold
                    true_phase_position = top_2_phases.index(true_phase)
                    confident_top2.append({
                        'sample_idx': i,
                        'true_phase': true_phase,
                        'top_2_phases': top_2_phases,
                        'top_2_probs': top_2_probs_values,
                        'true_phase_position': true_phase_position,
                        'confidence': confidence,
                        'data': all_data_samples[i]
                    })
        
        print(f"Confident top-2 predictions (>0.9): {len(confident_top2)}")
        
        if confident_top2:
            print(f"\n📋 Confident Top-2 Examples:")
            print("-" * 60)
            for i, case in enumerate(confident_top2[:5]):  # Show first 5
                print(f"Sample {case['sample_idx']+1}:")
                print(f"  True Phase: {case['true_phase']}")
                print(f"  Top-2 Phases: {case['top_2_phases']}")
                print(f"  Top-2 Probs: {[f'{p:.4f}' for p in case['top_2_probs']]}")
                print(f"  True Phase Position: {case['true_phase_position']+1}")
                print(f"  Total Confidence: {case['confidence']:.4f}")
                
                # Show top-5 predictions for this sample
                sample_probs = probs[case['sample_idx']]
                top_5_probs, top_5_indices = torch.topk(sample_probs, 5)
                print(f"  Top-5 predictions:")
                for j in range(5):
                    phase_id = top_5_indices[j].item()
                    prob = top_5_probs[j].item()
                    is_correct = "✓" if phase_id == case['true_phase'] else "✗"
                    print(f"    {j+1}. Phase {phase_id:5d} (prob: {prob:.4f}) {is_correct}")
                print()
        
        # Overall statistics
        print(f"\n📊 Overall Statistics:")
        print("=" * 30)
        
        # Calculate accuracy metrics
        correct_top1 = (top_1_indices.squeeze() == all_targets).sum().item()
        correct_top2 = 0
        for i, true_label in enumerate(all_targets):
            if true_label.item() in top_2_indices[i]:
                correct_top2 += 1
        
        print(f"Top-1 accuracy: {correct_top1/len(all_targets):.2%} ({correct_top1}/{len(all_targets)})")
        print(f"Top-2 accuracy: {correct_top2/len(all_targets):.2%} ({correct_top2}/{len(all_targets)})")
        
        # Confidence distribution
        avg_top1_prob = top_1_probs.mean().item()
        print(f"Average top-1 confidence: {avg_top1_prob:.4f}")
        
        # Count by confidence levels
        very_high_conf = (top_1_probs > 0.9).sum().item()
        high_conf = ((top_1_probs > 0.7) & (top_1_probs <= 0.9)).sum().item()
        medium_conf = ((top_1_probs > 0.5) & (top_1_probs <= 0.7)).sum().item()
        low_conf = (top_1_probs <= 0.5).sum().item()
        
        print(f"\nConfidence Distribution:")
        print(f"  Very High (>0.9): {very_high_conf} samples ({very_high_conf/len(all_targets):.1%})")
        print(f"  High (0.7-0.9): {high_conf} samples ({high_conf/len(all_targets):.1%})")
        print(f"  Medium (0.5-0.7): {medium_conf} samples ({medium_conf/len(all_targets):.1%})")
        print(f"  Low (≤0.5): {low_conf} samples ({low_conf/len(all_targets):.1%})")
        
        print(f"\n✅ Analysis complete!")
        print(f"Key findings:")
        print(f"- Found {len(high_conf_correct)} high-confidence correct predictions")
        print(f"- Found {len(confident_top2)} confident top-2 predictions")
        print(f"- Model shows varying confidence levels across samples")
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_single_phase_predictions()
