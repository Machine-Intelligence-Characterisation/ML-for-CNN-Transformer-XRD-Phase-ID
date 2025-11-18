"""
Data loader for datasetCPICANN using Hugging Face Datasets Library
Specifically designed to work with D1 and D2 files only
"""

import os
import zipfile
import tempfile
import shutil
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
import numpy as np
from typing import Optional, List, Dict, Any
import pickle


class DatasetCPICANNLoader:
    """
    Loader for datasetCPICANN that efficiently handles D1 and D2 files
    Supports both Hugging Face download and local file usage
    """
    
    def __init__(self, cache_dir: Optional[str] = None, local_data_dir: Optional[str] = None):
        """
        Initialize the data loader
        
        Args:
            cache_dir: Directory to cache downloaded files. If None, uses default cache.
            local_data_dir: Directory containing local D1.zip and D2.zip files. If provided, 
                           will use local files instead of downloading from Hugging Face.
        """
        self.repo_id = "caobin/datasetCPICANN"
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/datasets")
        self.local_data_dir = local_data_dir
        self.files_to_download = ["D1.zip", "D2.zip"]
    
    def get_local_files(self) -> Dict[str, str]:
        """
        Get local D1.zip and D2.zip files if they exist
        
        Returns:
            Dictionary mapping file names to local paths, or empty dict if not found
        """
        if not self.local_data_dir or not os.path.exists(self.local_data_dir):
            return {}
        
        local_files = {}
        for filename in self.files_to_download:
            file_path = os.path.join(self.local_data_dir, filename)
            if os.path.exists(file_path):
                local_files[filename] = file_path
                print(f"Found local file: {filename} at {file_path}")
            else:
                print(f"Local file not found: {filename} at {file_path}")
        
        return local_files
        
    def download_files(self, local_dir: Optional[str] = None, token: Optional[str] = None) -> Dict[str, str]:
        """
        Download D1.zip and D2.zip files from Hugging Face
        
        Args:
            local_dir: Local directory to save files. If None, uses cache directory.
            token: Hugging Face token for authentication (if required)
            
        Returns:
            Dictionary mapping file names to local paths
        """
        if local_dir is None:
            local_dir = os.path.join(self.cache_dir, "datasetCPICANN")
        
        os.makedirs(local_dir, exist_ok=True)
        
        downloaded_files = {}
        
        for filename in self.files_to_download:
            print(f"Downloading {filename}...")
            try:
                local_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    local_dir=local_dir,
                    token=token,  # Pass token for authentication
                    resume_download=True  # Resume interrupted downloads
                )
                downloaded_files[filename] = local_path
                print(f"Downloaded {filename} to {local_path}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if "401" in str(e) or "Unauthorized" in str(e):
                    print("\nThis appears to be an authentication issue.")
                    print("Please try one of the following:")
                    print("1. Run: huggingface-cli login")
                    print("2. Set environment variable: export HUGGINGFACE_HUB_TOKEN='your_token'")
                    print("3. Pass token to load_dataset(token='your_token')")
                    print("4. Request access to the dataset if it's private")
                raise e
            
        return downloaded_files
    
    def extract_and_load_data(self, zip_path: str) -> Dataset:
        """
        Extract zip file and load the data into a Hugging Face Dataset
        
        Args:
            zip_path: Path to the zip file
            
        Returns:
            Hugging Face Dataset object
        """
        print(f"Extracting {zip_path}...")
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find pickle files in extracted directory
            pickle_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.pkl') or file.endswith('.pickle'):
                        pickle_files.append(os.path.join(root, file))
            
            if not pickle_files:
                raise ValueError(f"No pickle files found in {zip_path}")
            
            print(f"Found {len(pickle_files)} pickle files")
            
            # Load data from pickle files
            all_data = []
            for pickle_file in pickle_files:
                print(f"Loading {pickle_file}...")
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    
                    # Handle different data structures
                    if isinstance(data, dict):
                        # If it's a dictionary, convert to list of records
                        if 'data' in data:
                            all_data.extend(data['data'])
                        elif 'samples' in data:
                            all_data.extend(data['samples'])
                        else:
                            # Assume each key-value pair is a sample
                            for key, value in data.items():
                                all_data.append({'id': key, 'data': value})
                    elif isinstance(data, list):
                        all_data.extend(data)
                    else:
                        # Single sample
                        all_data.append({'data': data})
            
            print(f"Loaded {len(all_data)} samples")
            
            # Create Hugging Face Dataset
            dataset = Dataset.from_list(all_data)
            return dataset
    
    def load_dataset(self, local_dir: Optional[str] = None, streaming: bool = False, token: Optional[str] = None) -> Dict[str, Dataset]:
        """
        Load D1 and D2 datasets
        
        Args:
            local_dir: Local directory to save files
            streaming: Whether to use streaming mode (not applicable for zip files)
            token: Hugging Face token for authentication (if required)
            
        Returns:
            Dictionary with 'D1' and 'D2' datasets
        """
        # First try to use local files if available
        local_files = self.get_local_files()
        
        if local_files:
            print("Using local files instead of downloading from Hugging Face")
            file_paths = local_files
        else:
            print("No local files found, downloading from Hugging Face...")
            # Download files from Hugging Face
            file_paths = self.download_files(local_dir, token)
        
        datasets_dict = {}
        
        for filename, file_path in file_paths.items():
            dataset_name = filename.replace('.zip', '')  # 'D1.zip' -> 'D1'
            print(f"Loading {dataset_name} dataset...")
            
            try:
                dataset = self.extract_and_load_data(file_path)
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
    
    def get_cache_size(self) -> str:
        """
        Get the size of cached data
        
        Returns:
            Human-readable string of cache size
        """
        cache_path = os.path.join(self.cache_dir, "datasetCPICANN")
        if os.path.exists(cache_path):
            try:
                result = os.popen(f"du -sh {cache_path}").read().strip()
                return result.split()[0] if result else "Unknown"
            except:
                return "Unknown"
        return "0B"
    
    def clear_cache(self) -> bool:
        """
        Clear cached data to free up disk space
        
        Returns:
            True if cache was cleared successfully
        """
        cache_path = os.path.join(self.cache_dir, "datasetCPICANN")
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
                print(f"Cache cleared: {cache_path}")
                return True
            except Exception as e:
                print(f"Error clearing cache: {e}")
                return False
        else:
            print("No cache to clear")
            return True
    
    def cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data
        
        Returns:
            Dictionary with cache information
        """
        cache_path = os.path.join(self.cache_dir, "datasetCPICANN")
        
        info = {
            'cache_directory': cache_path,
            'exists': os.path.exists(cache_path),
            'size': self.get_cache_size()
        }
        
        if os.path.exists(cache_path):
            # List cached files
            cached_files = []
            for file in os.listdir(cache_path):
                file_path = os.path.join(cache_path, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    cached_files.append({
                        'name': file,
                        'size_bytes': file_size,
                        'size_human': self._format_bytes(file_size)
                    })
            info['cached_files'] = cached_files
        
        return info
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}PB"


def main():
    """
    Example usage of the DatasetCPICANNLoader
    """
    # Initialize loader
    loader = DatasetCPICANNLoader()
    
    # Load datasets
    print("Loading datasetCPICANN (D1 and D2)...")
    datasets = loader.load_dataset()
    
    # Print information about loaded datasets
    for name, dataset in datasets.items():
        print(f"\n{name} Dataset Info:")
        info = loader.get_dataset_info(dataset)
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    return datasets


if __name__ == "__main__":
    datasets = main()
