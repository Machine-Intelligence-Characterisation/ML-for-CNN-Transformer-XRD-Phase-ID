#!/usr/bin/env python3
"""
Test script for evaluating pretrained models on D1_full data
Tests all available pretrained models and compares their performance
"""

import torch
import torch.nn.functional as F
import os
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your models and dataset
from src.models.CPICANN import CPICANN
from src.models.CNNonly import CNN
from src.models.ATTENTIONonly import VIT
from src.data_loading.dataset import XrdDataset


class ModelTester:
    def __init__(self, data_dir, annotations_file, batch_size=64):
        """
        Initialize the model tester
        
        Args:
            data_dir: Path to the data directory (train or val)
            annotations_file: Path to the annotations CSV file
            batch_size: Batch size for evaluation
        """
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Available pretrained models
        self.pretrained_models = {
            'CPICANNsingle_phase_D1': {
                'model_class': CPICANN,
                'file': 'CPICANNsingle_phase_D1.pth',
                'num_classes': 23073
            },
            # 'CPICANNbi_phase_D1': {  # SKIPPED: File corrupted (86MB vs expected 165MB)
            #     'model_class': CPICANN,
            #     'file': 'CPICANNbi_phase_D1.pth',
            #     'num_classes': 23073
            # },
            'CNNonlyD1': {
                'model_class': CNN,
                'file': 'CNNonlyD1.pth',
                'num_classes': 23073
            },
            'ATTENTIONonlyD1': {
                'model_class': VIT,
                'file': 'ATTENTIONonlyD1.pth',
                'num_classes': 23073
            }
        }
        
        # Check which models are actually available
        self.available_models = {}
        trained_models_dir = '/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/trained_models'
        
        for model_name, config in self.pretrained_models.items():
            model_path = os.path.join(trained_models_dir, config['file'])
            if os.path.exists(model_path):
                self.available_models[model_name] = config
                print(f"✓ Found: {model_name}")
            else:
                print(f"✗ Missing: {model_name}")
        
        print(f"\nFound {len(self.available_models)} available models")
    
    def create_dataset(self):
        """Create the dataset and data loader"""
        print(f"\nCreating dataset from:")
        print(f"  Data dir: {self.data_dir}")
        print(f"  Annotations: {self.annotations_file}")
        
        try:
            dataset = XrdDataset(
                data_dir=self.data_dir,
                annotations_file=self.annotations_file
            )
            
            data_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            print(f"✓ Dataset created successfully")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Batches: {len(data_loader)}")
            
            # Show sample data from dataset
            print(f"\n--- Sample Dataset Information ---")
            sample_data, sample_label = dataset[0]
            print(f"Sample data shape: {sample_data.shape}")
            print(f"Sample data type: {sample_data.dtype}")
            print(f"Sample data range: [{sample_data.min():.4f}, {sample_data.max():.4f}]")
            print(f"Sample label: {sample_label}")
            print(f"Sample data (first 10 values): {sample_data[0, :10]}")
            print("--- End Sample Information ---\n")
            
            return data_loader
            
        except Exception as e:
            print(f"✗ Error creating dataset: {e}")
            return None
    
    def load_model(self, model_name, model_config):
        """Load a pretrained model"""
        try:
            print(f"\nLoading {model_name}...")
            
            # Initialize model with correct embed_dim
            if model_config['model_class'] == CPICANN:
                model = model_config['model_class'](embed_dim=128, num_classes=model_config['num_classes'])
            else:
                model = model_config['model_class'](num_classes=model_config['num_classes'])
            
            # Load pretrained weights
            model_path = f"/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/trained_models/{model_config['file']}"
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            print(f"✓ {model_name} loaded successfully")
            return model
            
        except Exception as e:
            print(f"✗ Error loading {model_name}: {e}")
            return None
    
    def evaluate_model(self, model, data_loader, model_name):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")
        
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        # Progress bar
        pbar = tqdm(data_loader, desc=f"Evaluating {model_name}")
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(pbar):
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                try:
                    # Forward pass
                    outputs = model(data)
                    
                    # Print sample inputs and outputs for first few batches
                    if batch_idx < 3:  # Print details for first 3 batches
                        print(f"\n--- Batch {batch_idx + 1} Details ---")
                        print(f"Input shape: {data.shape}")
                        print(f"Input data range: [{data.min().item():.4f}, {data.max().item():.4f}]")
                        print(f"Input sample (first sample, first channel): {data[0, 0, :10].cpu().numpy()}")
                        print(f"True labels: {labels[:5].cpu().numpy()}")
                        print(f"Model outputs shape: {outputs.shape}")
                        print(f"Output logits (first 5 samples, top 5 classes):")
                        top5_indices = torch.topk(outputs[:5], 5, dim=1).indices
                        top5_values = torch.topk(outputs[:5], 5, dim=1).values
                        for i in range(5):
                            print(f"  Sample {i}: Classes {top5_indices[i].cpu().numpy()} with scores {top5_values[i].cpu().numpy()}")
                        print(f"Predicted classes: {torch.max(outputs, 1)[1][:5].cpu().numpy()}")
                        print(f"Correct predictions: {(torch.max(outputs, 1)[1] == labels)[:5].cpu().numpy()}")
                        print("--- End Batch Details ---\n")
                    
                    # Calculate loss
                    loss = F.cross_entropy(outputs, labels)
                    total_loss += loss.item()
                    
                    # Get predictions
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    accuracy = 100 * correct / total
                    pbar.set_postfix({
                        'Acc': f'{accuracy:.2f}%',
                        'Loss': f'{loss.item():.4f}'
                    })
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Calculate final metrics
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(data_loader)
        
        print(f"✓ {model_name} evaluation complete")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Correct: {correct}/{total}")
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def test_all_models(self, max_samples=None):
        """Test all available pretrained models"""
        print("=" * 60)
        print("PRETRAINED MODEL EVALUATION")
        print("=" * 60)
        
        # Create dataset
        data_loader = self.create_dataset()
        if data_loader is None:
            print("Failed to create dataset. Exiting.")
            return
        
        # Limit samples if specified
        if max_samples:
            print(f"\nLimiting evaluation to {max_samples} samples")
            limited_data = []
            count = 0
            for batch in data_loader:
                limited_data.append(batch)
                count += batch[0].size(0)
                if count >= max_samples:
                    break
            
            # Create limited data loader
            from torch.utils.data import TensorDataset
            all_data = torch.cat([batch[0] for batch in limited_data])
            all_labels = torch.cat([batch[1] for batch in limited_data])
            limited_dataset = TensorDataset(all_data, all_labels)
            data_loader = DataLoader(limited_dataset, batch_size=self.batch_size, shuffle=False)
            print(f"Limited dataset: {len(limited_dataset)} samples")
        
        # Results storage
        results = []
        
        # Test each model
        for model_name, model_config in self.available_models.items():
            print(f"\n{'='*40}")
            print(f"Testing: {model_name}")
            print(f"{'='*40}")
            
            # Load model
            model = self.load_model(model_name, model_config)
            if model is None:
                print(f"Skipping {model_name} due to loading error")
                continue
            
            # Evaluate model
            result = self.evaluate_model(model, data_loader, model_name)
            results.append(result)
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """Print a summary of all results"""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        # Sort by accuracy
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"{'Model Name':<25} {'Accuracy':<10} {'Loss':<10} {'Correct/Total':<15}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['model_name']:<25} {result['accuracy']:<10.2f} {result['loss']:<10.4f} {result['correct']}/{result['total']}")
        
        # Best model
        if results:
            best = results[0]
            print(f"\n🏆 Best Model: {best['model_name']}")
            print(f"   Accuracy: {best['accuracy']:.2f}%")
            print(f"   Loss: {best['loss']:.4f}")


def main():
    """Main function to run the evaluation"""
    
    # Configuration
    DATA_DIR = "/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_data/D1_full/val"
    ANNOTATIONS_FILE = "/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_data/D1_full/anno_val.csv"
    BATCH_SIZE = 64
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("Please ensure the rsync transfer is complete.")
        return
    
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"❌ Annotations file not found: {ANNOTATIONS_FILE}")
        return
    
    # Initialize tester
    tester = ModelTester(DATA_DIR, ANNOTATIONS_FILE, BATCH_SIZE)
    
    # Test all models
    print(f"\nStarting evaluation...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Annotations: {ANNOTATIONS_FILE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # You can limit samples for quick testing
    # results = tester.test_all_models(max_samples=1000)  # Quick test
    results = tester.test_all_models()  # Full evaluation
    
    if results and len(results) > 0:
        print(f"\n✅ Evaluation complete! Tested {len(results)} models.")
    else:
        print(f"\n❌ Evaluation failed.")


if __name__ == "__main__":
    main()
