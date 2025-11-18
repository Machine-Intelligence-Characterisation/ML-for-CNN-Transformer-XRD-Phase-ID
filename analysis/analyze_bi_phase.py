#!/usr/bin/env python3
"""
Bi-phase model analysis script
Analyzes how many phases the bi-phase model recommends and their probabilities
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append('src')

from data_loading.dataset import XrdDataset
from models.CPICANN import CPICANN

def analyze_bi_phase_recommendations():
    """Analyze bi-phase model recommendations in detail"""
    print("🔬 Bi-Phase Model Recommendation Analysis")
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
    
    # Load bi-phase model
    model_path = "/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/trained_models/CPICANNbi_phase_D1.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Loading bi-phase model: {model_path}")
    
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
        print("✓ Bi-phase model loaded successfully")
        
        # Test on 100 samples for detailed analysis
        subset = Subset(dataset, range(100))
        data_loader = DataLoader(subset, batch_size=32, shuffle=False)
        
        print("\nAnalyzing phase recommendations...")
        
        all_recommendations = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc="Analyzing")):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Get probabilities
                probs = F.softmax(output, dim=1)
                
                # Store results
                all_recommendations.append(probs.cpu())
                all_targets.append(target.cpu())
                
                # Detailed analysis for first batch
                if batch_idx == 0:
                    print(f"\n📊 Detailed Analysis (First Batch):")
                    print(f"  Input shape: {data.shape}")
                    print(f"  True labels: {target.cpu().numpy()}")
                    
                    # Analyze each sample in first batch
                    for i in range(min(5, len(target))):
                        true_label = target[i].item()
                        sample_probs = probs[i]
                        
                        # Get top-10 recommendations
                        top_probs, top_indices = torch.topk(sample_probs, 10)
                        
                        print(f"\n  Sample {i+1} (True Phase: {true_label}):")
                        print(f"    Top-10 Phase Recommendations:")
                        
                        for j in range(10):
                            phase_id = top_indices[j].item()
                            probability = top_probs[j].item()
                            is_correct = "✓ CORRECT" if phase_id == true_label else ""
                            print(f"      {j+1:2d}. Phase {phase_id:5d} (prob: {probability:.4f}) {is_correct}")
        
        # Concatenate all results
        all_recommendations = torch.cat(all_recommendations, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Analyze recommendation patterns
        print(f"\n📈 Recommendation Pattern Analysis:")
        print("=" * 50)
        
        # Calculate top-k accuracy
        k_values = [1, 2, 3, 5, 10]
        print(f"Top-K Accuracy Results:")
        
        for k in k_values:
            top_k_probs, top_k_indices = torch.topk(all_recommendations, k, dim=1)
            
            correct = 0
            for i, true_label in enumerate(all_targets):
                if true_label.item() in top_k_indices[i]:
                    correct += 1
            
            accuracy = correct / len(all_targets)
            print(f"  Top-{k:2d} accuracy: {accuracy:.2%} ({correct}/{len(all_targets)})")
        
        # Analyze probability distributions
        print(f"\n📊 Probability Distribution Analysis:")
        print("=" * 50)
        
        # Get top-3 recommendations for all samples
        top_3_probs, top_3_indices = torch.topk(all_recommendations, 3, dim=1)
        
        # Calculate average probabilities for top-3
        avg_top1_prob = top_3_probs[:, 0].mean().item()
        avg_top2_prob = top_3_probs[:, 1].mean().item()
        avg_top3_prob = top_3_probs[:, 2].mean().item()
        
        print(f"Average probabilities in top-3 recommendations:")
        print(f"  Top-1: {avg_top1_prob:.4f}")
        print(f"  Top-2: {avg_top2_prob:.4f}")
        print(f"  Top-3: {avg_top3_prob:.4f}")
        
        # Analyze confidence levels
        print(f"\n🎯 Confidence Analysis:")
        print("=" * 30)
        
        # Count samples by confidence level
        high_confidence = (top_3_probs[:, 0] > 0.5).sum().item()
        medium_confidence = ((top_3_probs[:, 0] > 0.2) & (top_3_probs[:, 0] <= 0.5)).sum().item()
        low_confidence = (top_3_probs[:, 0] <= 0.2).sum().item()
        
        print(f"Confidence levels (based on top-1 probability):")
        print(f"  High confidence (>0.5): {high_confidence} samples ({high_confidence/len(all_targets):.1%})")
        print(f"  Medium confidence (0.2-0.5): {medium_confidence} samples ({medium_confidence/len(all_targets):.1%})")
        print(f"  Low confidence (≤0.2): {low_confidence} samples ({low_confidence/len(all_targets):.1%})")
        
        # Analyze phase diversity in recommendations
        print(f"\n🔄 Phase Diversity Analysis:")
        print("=" * 40)
        
        # Count unique phases in top-3 recommendations
        unique_phases_per_sample = []
        for i in range(len(all_targets)):
            unique_phases = len(set(top_3_indices[i].tolist()))
            unique_phases_per_sample.append(unique_phases)
        
        avg_unique_phases = np.mean(unique_phases_per_sample)
        print(f"Average unique phases in top-3 recommendations: {avg_unique_phases:.2f}")
        
        # Count how many samples have the true phase in top-2 (bi-phase scenario)
        bi_phase_correct = 0
        for i, true_label in enumerate(all_targets):
            if true_label.item() in top_3_indices[i, :2]:  # Top-2
                bi_phase_correct += 1
        
        bi_phase_accuracy = bi_phase_correct / len(all_targets)
        print(f"Bi-phase accuracy (true phase in top-2): {bi_phase_accuracy:.2%}")
        
        # NEW: Analyze bi-phase scenarios where 2 out of top-3 are correct
        print(f"\n🔬 Bi-Phase Scenario Analysis:")
        print("=" * 50)
        
        # For this analysis, we need to understand what constitutes "correct phases"
        # Since we're testing on single-phase patterns, we'll look for cases where
        # the true phase appears multiple times in top-3 (which shouldn't happen)
        # OR cases where the model is very confident about the top recommendation
        
        bi_phase_scenarios = []
        high_confidence_cases = []
        
        for i, true_label in enumerate(all_targets):
            true_phase = true_label.item()
            top_3_phases = top_3_indices[i].tolist()
            top_3_probs_values = top_3_probs[i].tolist()
            
            # Check if true phase appears in top-3
            if true_phase in top_3_phases:
                true_phase_position = top_3_phases.index(true_phase)
                true_phase_prob = top_3_probs_values[true_phase_position]
                
                # Case 1: True phase is in top-2 with high confidence
                if true_phase_position < 2 and true_phase_prob > 0.3:
                    bi_phase_scenarios.append({
                        'sample_idx': i,
                        'true_phase': true_phase,
                        'top_3_phases': top_3_phases,
                        'top_3_probs': top_3_probs_values,
                        'true_phase_position': true_phase_position,
                        'true_phase_prob': true_phase_prob,
                        'scenario_type': 'high_confidence_top2'
                    })
                
                # Case 2: Very high confidence in top-1 (potential bi-phase case)
                elif top_3_probs_values[0] > 0.6:
                    high_confidence_cases.append({
                        'sample_idx': i,
                        'true_phase': true_phase,
                        'top_3_phases': top_3_phases,
                        'top_3_probs': top_3_probs_values,
                        'true_phase_position': true_phase_position,
                        'true_phase_prob': true_phase_prob,
                        'scenario_type': 'high_confidence_top1'
                    })
        
        print(f"Bi-phase scenario cases found: {len(bi_phase_scenarios)}")
        print(f"High confidence cases: {len(high_confidence_cases)}")
        
        # Show detailed examples
        if bi_phase_scenarios:
            print(f"\n📋 Bi-Phase Scenario Examples:")
            print("-" * 60)
            for i, scenario in enumerate(bi_phase_scenarios[:5]):  # Show first 5
                print(f"Sample {scenario['sample_idx']+1}:")
                print(f"  True Phase: {scenario['true_phase']}")
                print(f"  Top-3 Phases: {scenario['top_3_phases']}")
                print(f"  Top-3 Probs: {[f'{p:.3f}' for p in scenario['top_3_probs']]}")
                print(f"  True Phase Position: {scenario['true_phase_position']+1}")
                print(f"  True Phase Prob: {scenario['true_phase_prob']:.3f}")
                print()
        
        if high_confidence_cases:
            print(f"\n📋 High Confidence Cases:")
            print("-" * 60)
            for i, case in enumerate(high_confidence_cases[:5]):  # Show first 5
                print(f"Sample {case['sample_idx']+1}:")
                print(f"  True Phase: {case['true_phase']}")
                print(f"  Top-3 Phases: {case['top_3_phases']}")
                print(f"  Top-3 Probs: {[f'{p:.3f}' for p in case['top_3_probs']]}")
                print(f"  True Phase Position: {case['true_phase_position']+1}")
                print(f"  True Phase Prob: {case['true_phase_prob']:.3f}")
                print()
        
        # Additional analysis: Check for cases where model is uncertain (low confidence)
        uncertain_cases = []
        for i, true_label in enumerate(all_targets):
            true_phase = true_label.item()
            top_3_phases = top_3_indices[i].tolist()
            top_3_probs_values = top_3_probs[i].tolist()
            
            # Low confidence in top-1 (potential bi-phase scenario)
            if top_3_probs_values[0] < 0.3 and true_phase in top_3_phases:
                true_phase_position = top_3_phases.index(true_phase)
                uncertain_cases.append({
                    'sample_idx': i,
                    'true_phase': true_phase,
                    'top_3_phases': top_3_phases,
                    'top_3_probs': top_3_probs_values,
                    'true_phase_position': true_phase_position,
                    'scenario_type': 'uncertain'
                })
        
        print(f"\n🤔 Uncertain Cases (Low Confidence): {len(uncertain_cases)}")
        if uncertain_cases:
            print("These cases might benefit from bi-phase analysis:")
            for i, case in enumerate(uncertain_cases[:3]):  # Show first 3
                print(f"Sample {case['sample_idx']+1}: True Phase {case['true_phase']} at position {case['true_phase_position']+1}")
                print(f"  Top-3 Probs: {[f'{p:.3f}' for p in case['top_3_probs']]}")
        
        # Show most frequently recommended phases
        print(f"\n📋 Most Frequently Recommended Phases:")
        print("=" * 50)
        
        phase_counts = {}
        for batch_indices in top_3_indices:
            for phase_id in batch_indices:
                phase_id = phase_id.item()
                phase_counts[phase_id] = phase_counts.get(phase_id, 0) + 1
        
        sorted_phases = sorted(phase_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Top-10 most frequently recommended phases:")
        for i, (phase_id, count) in enumerate(sorted_phases[:10]):
            percentage = count / (len(all_targets) * 3) * 100
            print(f"  {i+1:2d}. Phase {phase_id:5d}: {count:3d} times ({percentage:5.1f}%)")
        
        print(f"\n✅ Analysis complete!")
        print(f"Key findings:")
        print(f"- Bi-phase model recommends top-3 phases with varying confidence")
        print(f"- Top-2 accuracy shows bi-phase recommendation effectiveness")
        print(f"- Model shows different confidence levels across samples")
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_bi_phase_recommendations()
