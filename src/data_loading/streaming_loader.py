"""
Streaming data loader that can work with remote or mounted directories
No need to copy 30GB of data locally
"""

import os
import zipfile
import tempfile
import pandas as pd
from datasets import Dataset
from typing import Optional, List, Dict, Any, Iterator
import subprocess
import sys


class StreamingDatasetCPICANNLoader:
    """
    Streaming loader that can work with remote/mounted directories
    Processes data without copying everything locally
    """
    
    def __init__(self, remote_data_dir: Optional[str] = None, local_cache_dir: Optional[str] = None):
        """
        Initialize streaming loader
        
        Args:
            remote_data_dir: Path to remote/mounted directory containing D1.zip and D2.zip
            local_cache_dir: Local directory for temporary files (small cache)
        """
        self.remote_data_dir = remote_data_dir
        self.local_cache_dir = local_cache_dir or "/tmp/datasetCPICANN_cache"
        self.files_to_process = ["D1.zip", "D2.zip"]
        
        # Create local cache directory
        os.makedirs(self.local_cache_dir, exist_ok=True)
    
    def check_remote_access(self) -> bool:
        """
        Check if remote directory is accessible
        
        Returns:
            True if accessible, False otherwise
        """
        if not self.remote_data_dir:
            return False
        
        try:
            # Check if directory exists and is readable
            result = subprocess.run(['ls', self.remote_data_dir], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def list_remote_files(self) -> Dict[str, str]:
        """
        List available files in remote directory
        
        Returns:
            Dictionary mapping file names to full paths
        """
        if not self.remote_data_dir:
            return {}
        
        available_files = {}
        for filename in self.files_to_process:
            file_path = os.path.join(self.remote_data_dir, filename)
            if os.path.exists(file_path):
                available_files[filename] = file_path
                print(f"Found remote file: {filename}")
            else:
                print(f"Remote file not found: {filename}")
        
        return available_files
    
    def stream_zip_contents(self, zip_path: str) -> Iterator[Dict[str, Any]]:
        """
        Stream contents of a zip file without extracting everything
        
        Args:
            zip_path: Path to zip file
            
        Yields:
            Dictionary containing sample data
        """
        print(f"Streaming contents from {zip_path}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files in zip
                file_list = zip_ref.namelist()
                print(f"Found {len(file_list)} files in zip")
                
                # Process files in batches to avoid memory issues
                batch_size = 1000
                for i in range(0, len(file_list), batch_size):
                    batch_files = file_list[i:i + batch_size]
                    
                    for file_name in batch_files:
                        if file_name.endswith('.pkl') or file_name.endswith('.pickle'):
                            try:
                                # Extract and process individual file
                                with zip_ref.open(file_name) as f:
                                    import pickle
                                    data = pickle.load(f)
                                    
                                    # Convert to standard format
                                    if isinstance(data, dict):
                                        yield data
                                    elif isinstance(data, list):
                                        for item in data:
                                            yield item
                                    else:
                                        yield {'data': data}
                                        
                            except Exception as e:
                                print(f"Error processing {file_name}: {e}")
                                continue
                                
        except Exception as e:
            print(f"Error streaming zip file {zip_path}: {e}")
            raise e
    
    def stream_csv_data(self, csv_path: str) -> Iterator[Dict[str, Any]]:
        """
        Stream CSV data in chunks
        
        Args:
            csv_path: Path to CSV file
            
        Yields:
            Dictionary containing sample data
        """
        print(f"Streaming CSV data from {csv_path}...")
        
        try:
            # Read CSV in chunks to avoid memory issues
            chunk_size = 1000
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                for _, row in chunk.iterrows():
                    yield row.to_dict()
                    
        except Exception as e:
            print(f"Error streaming CSV {csv_path}: {e}")
            raise e
    
    def create_streaming_dataset(self, data_generator: Iterator[Dict[str, Any]], 
                                dataset_name: str, max_samples: Optional[int] = None) -> Dataset:
        """
        Create a Hugging Face Dataset from a streaming generator
        
        Args:
            data_generator: Iterator yielding data samples
            dataset_name: Name of the dataset
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            Hugging Face Dataset object
        """
        print(f"Creating streaming dataset: {dataset_name}")
        
        # Collect samples in batches
        batch_size = 1000
        all_samples = []
        sample_count = 0
        
        for sample in data_generator:
            all_samples.append(sample)
            sample_count += 1
            
            # Process in batches to avoid memory issues
            if len(all_samples) >= batch_size:
                print(f"Processed {sample_count} samples...")
                
                # Create temporary dataset from batch
                if sample_count == batch_size:  # First batch
                    dataset = Dataset.from_list(all_samples)
                else:
                    # Append to existing dataset
                    batch_dataset = Dataset.from_list(all_samples)
                    dataset = dataset.add_item(batch_dataset)
                
                all_samples = []  # Clear batch
            
            # Check if we've reached the limit
            if max_samples and sample_count >= max_samples:
                break
        
        # Add remaining samples
        if all_samples:
            if sample_count <= batch_size:
                dataset = Dataset.from_list(all_samples)
            else:
                batch_dataset = Dataset.from_list(all_samples)
                dataset = dataset.add_item(batch_dataset)
        
        print(f"Created dataset {dataset_name} with {sample_count} samples")
        return dataset
    
    def load_dataset_streaming(self, max_samples_per_file: Optional[int] = None) -> Dict[str, Dataset]:
        """
        Load datasets using streaming approach
        
        Args:
            max_samples_per_file: Maximum samples to process per file (None for all)
            
        Returns:
            Dictionary with 'D1' and 'D2' datasets
        """
        if not self.check_remote_access():
            raise ValueError(f"Cannot access remote directory: {self.remote_data_dir}")
        
        available_files = self.list_remote_files()
        
        if not available_files:
            raise ValueError("No D1.zip or D2.zip files found in remote directory")
        
        datasets_dict = {}
        
        for filename, file_path in available_files.items():
            dataset_name = filename.replace('.zip', '')  # 'D1.zip' -> 'D1'
            print(f"Loading {dataset_name} dataset using streaming...")
            
            try:
                # Create streaming generator
                data_generator = self.stream_zip_contents(file_path)
                
                # Create dataset from generator
                dataset = self.create_streaming_dataset(
                    data_generator, 
                    dataset_name, 
                    max_samples_per_file
                )
                
                datasets_dict[dataset_name] = dataset
                print(f"Successfully loaded {dataset_name} with {len(dataset)} samples")
                
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                continue
        
        return datasets_dict
    
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get information about a dataset
        
        Args:
            dataset: Hugging Face Dataset object
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'num_samples': len(dataset),
            'features': list(dataset.features.keys()) if hasattr(dataset, 'features') else None,
            'column_names': dataset.column_names if hasattr(dataset, 'column_names') else None
        }
        
        # Get sample of first few rows
        if len(dataset) > 0:
            info['sample_data'] = dataset[0] if len(dataset) > 0 else None
        
        return info


def setup_remote_access():
    """
    Helper function to set up remote access
    """
    print("Setting up remote access to your Mac...")
    print("=" * 40)
    
    print("Option 1: SSHFS Mount (Recommended)")
    print("Run this on M3:")
    print("  sshfs username@your-macbook-ip:/Users/username/Downloads /home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_data/macbook_data")
    
    print("\nOption 2: rsync Transfer")
    print("Run this on M3:")
    print("  rsync -avz --progress username@your-macbook-ip:/Users/username/Downloads/D1.zip /home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_data/")
    print("  rsync -avz --progress username@your-macbook-ip:/Users/username/Downloads/D2.zip /home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_data/")
    
    print("\nOption 3: Cloud Storage")
    print("Upload to Google Drive/Dropbox from Mac, then download on M3")
    
    print("\nOption 4: Direct Network Access")
    print("Enable file sharing on Mac, then access via network path")


def main():
    """
    Example usage of streaming loader
    """
    print("Streaming DatasetCPICANN Loader")
    print("=" * 30)
    
    # Set up remote access first
    setup_remote_access()
    
    # Example usage (after setting up remote access)
    remote_dir = "/home/ankitag/sy86/ankitag/Ankita_CPICANN_Phase/training_data/macbook_data"
    
    if os.path.exists(remote_dir):
        print(f"\nUsing remote directory: {remote_dir}")
        
        loader = StreamingDatasetCPICANNLoader(remote_data_dir=remote_dir)
        
        try:
            # Load datasets with streaming (process first 1000 samples for testing)
            datasets = loader.load_dataset_streaming(max_samples_per_file=1000)
            
            # Print information about loaded datasets
            for name, dataset in datasets.items():
                print(f"\n{name} Dataset Info:")
                info = loader.get_dataset_info(dataset)
                for key, value in info.items():
                    print(f"  {key}: {value}")
                    
        except Exception as e:
            print(f"Error: {e}")
            print("Please set up remote access first")
    else:
        print(f"\nRemote directory not found: {remote_dir}")
        print("Please set up remote access first")


if __name__ == "__main__":
    main()
