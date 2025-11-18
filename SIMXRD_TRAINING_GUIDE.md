# Training CPICANN on SimXRD-4M Dataset

This guide explains how to train the CPICANN model on your own SimXRD-4M dataset instead of the paper's D1_full data.

## 📋 Overview

The SimXRD-4M dataset is a large collection of simulated XRD patterns. This guide helps you:
1. **Prepare your data** in the correct format
2. **Train CPICANN** on your custom dataset
3. **Validate and evaluate** the trained model

## 🗂️ Data Structure Requirements

### Expected Input Format
Your SimXRD-4M data should be organized as:

```
simxrd_data/
├── patterns/                    # XRD pattern files
│   ├── sample_001.csv          # Individual XRD patterns
│   ├── sample_002.csv
│   └── ...
├── metadata.csv                # Optional: sample labels
└── README.txt                  # Optional: data description
```

### Supported File Formats
- **CSV files**: Each file contains 1D XRD pattern data
- **NPY files**: NumPy arrays with XRD patterns
- **TXT files**: Text files with XRD data

### Label Sources
Labels can come from:
1. **Metadata file**: CSV with sample_id and label columns
2. **Filename structure**: `sample_class1_001.csv` → label = "class1"
3. **Directory structure**: `data/class1/sample001.csv` → label = "class1"

## 🚀 Step-by-Step Process

### Step 1: Data Preparation
```bash
# Prepare your SimXRD-4M data
python prepare_simxrd_data.py \
    --input_dir /path/to/your/simxrd_data \
    --output_dir /path/to/prepared_data
```

This will:
- ✅ Analyze your data structure
- ✅ Extract labels from metadata/filenames
- ✅ Convert data to CPICANN format
- ✅ Split into train/validation sets
- ✅ Create annotation files

### Step 2: Training
```bash
# Train CPICANN on your data
python train_simxrd.py \
    --data_dir /path/to/prepared_data \
    --output_dir ./simxrd_training_outputs \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4
```

### Step 3: Validation
```bash
# Validate the trained model
python validate_model.py \
    --model_path ./simxrd_training_outputs/best_model.pth \
    --data_dir /path/to/prepared_data/val \
    --anno_val /path/to/prepared_data/annotations_val.csv
```

## 🔧 Customization Options

### Model Architecture
```python
# Modify model parameters in train_simxrd.py
model = CPICANN(
    embed_dim=256,        # Increase for larger models
    num_classes=your_num_classes,
    nhead=16,             # More attention heads
    num_encoder_layers=8  # Deeper transformer
)
```

### Training Parameters
```bash
# Adjust training parameters
python train_simxrd.py \
    --epochs 200 \           # More epochs
    --batch_size 128 \       # Larger batch size
    --lr 5e-5 \             # Lower learning rate
    --embed_dim 256         # Larger model
```

### Data Augmentation
Add data augmentation to `SimXRDDataset`:
```python
class SimXRDDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None, augment=False):
        # ... existing code ...
        self.augment = augment
    
    def __getitem__(self, idx):
        # ... load data ...
        
        if self.augment:
            # Add noise
            data += np.random.normal(0, 0.01, data.shape)
            # Scale intensity
            data *= np.random.uniform(0.9, 1.1)
        
        return torch.tensor(data), torch.tensor(label_idx)
```

## 📊 Expected Results

### Performance Metrics
- **Training Accuracy**: 85-95% (depending on data quality)
- **Validation Accuracy**: 80-90%
- **Training Time**: 2-8 hours (depending on dataset size)
- **Convergence**: Usually within 50-100 epochs

### Model Size
- **Parameters**: ~1-5M (depending on embed_dim)
- **Model File**: 10-50MB
- **Memory Usage**: 2-8GB GPU memory

## 🎯 SimXRD-4M Specific Considerations

### Data Characteristics
- **Pattern Length**: Typically 1000-2000 data points
- **Intensity Range**: 0-1 (normalized)
- **Noise Level**: Simulated noise may be different from real data
- **Class Distribution**: May be imbalanced

### Recommended Settings
```bash
# For SimXRD-4M dataset
python train_simxrd.py \
    --epochs 150 \           # SimXRD data may need more epochs
    --batch_size 128 \       # Larger batch for stability
    --lr 1e-4 \             # Conservative learning rate
    --embed_dim 128         # Standard size
```

### Data Quality Checks
```python
# Add to your training script
def check_data_quality(dataset):
    """Check data quality before training"""
    print("🔍 Checking data quality...")
    
    # Check data shapes
    sample_data, _ = dataset[0]
    print(f"Data shape: {sample_data.shape}")
    
    # Check for NaN values
    for i in range(min(100, len(dataset))):
        data, _ = dataset[i]
        if torch.isnan(data).any():
            print(f"⚠️ NaN values found in sample {i}")
    
    # Check data ranges
    all_data = torch.stack([dataset[i][0] for i in range(min(100, len(dataset)))])
    print(f"Data range: {all_data.min():.4f} to {all_data.max():.4f}")
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   python train_simxrd.py --batch_size 32
   ```

2. **Poor Convergence**
   ```bash
   # Adjust learning rate
   python train_simxrd.py --lr 5e-5
   ```

3. **Data Loading Errors**
   ```bash
   # Check data format
   python prepare_simxrd_data.py --input_dir your_data --output_dir test_output
   ```

4. **Class Imbalance**
   ```python
   # Use weighted loss
   from torch.nn import CrossEntropyLoss
   criterion = CrossEntropyLoss(weight=class_weights)
   ```

### Debug Mode
```bash
# Test with small dataset
python train_simxrd.py \
    --data_dir your_data \
    --epochs 5 \
    --batch_size 16
```

## 📈 Monitoring Training

### TensorBoard Logs
```bash
# Start TensorBoard
tensorboard --logdir simxrd_training_outputs/

# Access at: http://localhost:6006
```

### Key Metrics to Watch
- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should decrease and stabilize
- **Training Accuracy**: Should increase to 90%+
- **Validation Accuracy**: Should reach 80-90%
- **Learning Rate**: Follows cosine schedule

## 🎉 Success Criteria

Training is successful when:
- ✅ Validation accuracy > 80%
- ✅ Training converges smoothly
- ✅ No overfitting (val acc ≈ train acc)
- ✅ Model saves correctly
- ✅ Validation script runs without errors

## 📚 Comparison with Paper Results

### Original Paper (D1_full data)
- **Accuracy**: ~85-90%
- **Classes**: 23,073 crystal structures
- **Data**: Real XRD patterns

### Your SimXRD-4M Results
- **Expected Accuracy**: 80-90% (may vary)
- **Classes**: Your custom classes
- **Data**: Simulated XRD patterns

### Key Differences
1. **Data Source**: Simulated vs. real XRD patterns
2. **Class Distribution**: May be different from paper
3. **Noise Characteristics**: Simulated vs. experimental noise
4. **Pattern Quality**: May be cleaner than real data

## 🔄 Next Steps

After successful training:
1. **Evaluate on test set**
2. **Compare with baseline models**
3. **Analyze misclassified samples**
4. **Fine-tune hyperparameters**
5. **Deploy model for inference**

## 📞 Support

If you encounter issues:
1. Check data format with `prepare_simxrd_data.py`
2. Test with small dataset first
3. Verify GPU memory usage
4. Check data quality and labels
5. Review training logs for errors

