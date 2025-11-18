#!/usr/bin/env python3
"""
Data Preparation Script for SimXRD-4M Dataset
Converts your SimXRD-4M data to the format expected by CPICANN training
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import json


class SimXRDDataPreparator:
    """Prepare SimXRD-4M data for CPICANN training"""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_data_structure(self) -> Dict:
        """Analyze the structure of your SimXRD-4M data"""
        print("🔍 Analyzing SimXRD-4M data structure...")
        
        analysis = {
            'file_count': 0,
            'file_extensions': set(),
            'sample_files': [],
            'metadata_files': [],
            'directory_structure': {}
        }
        
        # Scan directory structure
        for root, dirs, files in os.walk(self.input_dir):
            rel_path = os.path.relpath(root, self.input_dir)
            analysis['directory_structure'][rel_path] = len(files)
            
            for file in files:
                analysis['file_count'] += 1
                file_ext = Path(file).suffix.lower()
                analysis['file_extensions'].add(file_ext)
                
                file_path = Path(root) / file
                
                # Categorize files
                if any(keyword in file.lower() for keyword in ['pattern', 'xrd', 'diffraction']):
                    analysis['sample_files'].append(str(file_path))
                elif any(keyword in file.lower() for keyword in ['meta', 'info', 'label', 'annotation']):
                    analysis['metadata_files'].append(str(file_path))
        
        print(f"📊 Analysis Results:")
        print(f"  Total files: {analysis['file_count']}")
        print(f"  File extensions: {list(analysis['file_extensions'])}")
        print(f"  Sample files: {len(analysis['sample_files'])}")
        print(f"  Metadata files: {len(analysis['metadata_files'])}")
        
        return analysis
    
    def extract_labels_from_metadata(self, metadata_files: List[str]) -> Dict[str, str]:
        """Extract labels from metadata files"""
        print("📋 Extracting labels from metadata...")
        
        labels = {}
        
        for metadata_file in metadata_files:
            try:
                if metadata_file.endswith('.csv'):
                    df = pd.read_csv(metadata_file)
                    # Adjust column names based on your metadata format
                    if 'sample_id' in df.columns and 'label' in df.columns:
                        for _, row in df.iterrows():
                            labels[row['sample_id']] = row['label']
                    elif 'filename' in df.columns and 'class' in df.columns:
                        for _, row in df.iterrows():
                            labels[row['filename']] = row['class']
                
                elif metadata_file.endswith('.json'):
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        # Adjust based on your JSON structure
                        for item in data:
                            if 'id' in item and 'label' in item:
                                labels[item['id']] = item['label']
                
                elif metadata_file.endswith('.txt'):
                    with open(metadata_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                labels[parts[0]] = parts[1]
                
            except Exception as e:
                print(f"⚠️ Error reading {metadata_file}: {e}")
        
        print(f"✅ Extracted {len(labels)} labels")
        return labels
    
    def extract_labels_from_filenames(self, sample_files: List[str]) -> Dict[str, str]:
        """Extract labels from filenames (if structured)"""
        print("📋 Extracting labels from filenames...")
        
        labels = {}
        
        for file_path in sample_files:
            filename = Path(file_path).stem
            
            # Option 1: Label is in filename (adjust pattern as needed)
            # Example: "sample_class1_001.csv" -> label = "class1"
            if '_' in filename:
                parts = filename.split('_')
                if len(parts) >= 2:
                    label = parts[1]  # Adjust index as needed
                    labels[filename] = label
            
            # Option 2: Label is directory name
            # Example: "data/class1/sample001.csv" -> label = "class1"
            parent_dir = Path(file_path).parent.name
            if parent_dir != '.':
                labels[filename] = parent_dir
            
            # Option 3: Create dummy labels (replace with your logic)
            if filename not in labels:
                labels[filename] = f"class_{hash(filename) % 100}"
        
        print(f"✅ Extracted {len(labels)} labels from filenames")
        return labels
    
    def convert_data_format(self, sample_files: List[str], labels: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """Convert data to CPICANN format"""
        print("🔄 Converting data format...")
        
        converted_files = []
        failed_files = []
        
        # Create output directories
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # Split data (80% train, 20% val)
        np.random.seed(42)
        is_train = np.random.random(len(sample_files)) < 0.8
        
        for i, file_path in enumerate(sample_files):
            try:
                filename = Path(file_path).stem
                
                # Load data
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path, header=None).values
                elif file_path.endswith('.npy'):
                    data = np.load(file_path)
                elif file_path.endswith('.txt'):
                    data = np.loadtxt(file_path)
                else:
                    print(f"⚠️ Unsupported format: {file_path}")
                    failed_files.append(file_path)
                    continue
                
                # Ensure data is 1D (XRD pattern)
                if data.ndim > 1:
                    data = data.flatten()
                
                # Normalize data (optional)
                data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
                
                # Determine output directory
                output_dir = train_dir if is_train[i] else val_dir
                
                # Save as CSV (CPICANN format)
                output_file = output_dir / f"{filename}.csv"
                pd.DataFrame(data).to_csv(output_file, header=False, index=False)
                
                converted_files.append(str(output_file))
                
            except Exception as e:
                print(f"❌ Error converting {file_path}: {e}")
                failed_files.append(file_path)
        
        print(f"✅ Converted {len(converted_files)} files")
        if failed_files:
            print(f"❌ Failed to convert {len(failed_files)} files")
        
        return converted_files, failed_files
    
    def create_annotations(self, labels: Dict[str, str], train_files: List[str], val_files: List[str]):
        """Create annotation CSV files"""
        print("📝 Creating annotation files...")
        
        # Create train annotations
        train_annotations = []
        for file_path in train_files:
            filename = Path(file_path).stem
            if filename in labels:
                train_annotations.append({
                    'sample_id': filename,
                    'label': labels[filename],
                    'file_path': file_path
                })
        
        # Create val annotations
        val_annotations = []
        for file_path in val_files:
            filename = Path(file_path).stem
            if filename in labels:
                val_annotations.append({
                    'sample_id': filename,
                    'label': labels[filename],
                    'file_path': file_path
                })
        
        # Save annotations
        train_df = pd.DataFrame(train_annotations)
        val_df = pd.DataFrame(val_annotations)
        
        train_anno_file = self.output_dir / 'annotations_train.csv'
        val_anno_file = self.output_dir / 'annotations_val.csv'
        
        train_df.to_csv(train_anno_file, index=False)
        val_df.to_csv(val_anno_file, index=False)
        
        print(f"✅ Created train annotations: {train_anno_file} ({len(train_df)} samples)")
        print(f"✅ Created val annotations: {val_anno_file} ({len(val_df)} samples)")
        
        return str(train_anno_file), str(val_anno_file)
    
    def prepare_dataset(self) -> Dict[str, str]:
        """Main method to prepare the dataset"""
        print("🚀 Preparing SimXRD-4M dataset for CPICANN training")
        print("=" * 60)
        
        # Analyze data structure
        analysis = self.analyze_data_structure()
        
        # Extract labels
        labels = {}
        
        # Try metadata files first
        if analysis['metadata_files']:
            labels.update(self.extract_labels_from_metadata(analysis['metadata_files']))
        
        # Extract from filenames if no metadata
        if not labels and analysis['sample_files']:
            labels.update(self.extract_labels_from_filenames(analysis['sample_files']))
        
        if not labels:
            print("❌ No labels found! Please check your data structure.")
            return {}
        
        # Convert data format
        converted_files, failed_files = self.convert_data_format(analysis['sample_files'], labels)
        
        # Split into train/val
        train_files = [f for f in converted_files if '/train/' in f]
        val_files = [f for f in converted_files if '/val/' in f]
        
        # Create annotations
        train_anno, val_anno = self.create_annotations(labels, train_files, val_files)
        
        # Create summary
        summary = {
            'total_samples': len(converted_files),
            'train_samples': len(train_files),
            'val_samples': len(val_files),
            'num_classes': len(set(labels.values())),
            'train_annotations': train_anno,
            'val_annotations': val_anno,
            'output_dir': str(self.output_dir),
            'failed_files': len(failed_files)
        }
        
        # Save summary
        summary_file = self.output_dir / 'preparation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Dataset preparation complete!")
        print(f"📊 Summary:")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Train samples: {summary['train_samples']}")
        print(f"  Val samples: {summary['val_samples']}")
        print(f"  Number of classes: {summary['num_classes']}")
        print(f"  Output directory: {summary['output_dir']}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Prepare SimXRD-4M data for CPICANN training')
    
    parser.add_argument('--input_dir', required=True, help='Directory containing SimXRD-4M data')
    parser.add_argument('--output_dir', required=True, help='Output directory for prepared data')
    
    args = parser.parse_args()
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    # Prepare dataset
    preparator = SimXRDDataPreparator(args.input_dir, args.output_dir)
    summary = preparator.prepare_dataset()
    
    if summary:
        print(f"\n✅ Ready for training!")
        print(f"Run: python train_simxrd.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()

