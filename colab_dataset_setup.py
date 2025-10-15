"""
Google Colab Dataset Setup for Cross-Attention Plant Disease Classification
This module provides utilities to download and set up the PlantWild dataset on Google Colab
"""

import os
import zipfile
import urllib.request
import gdown
from typing import Optional, Tuple
import json
import shutil

class ColabDatasetManager:
    """Manages dataset download and setup for Google Colab"""
    
    def __init__(self, base_path: str = "/content/plant_disease_data", dataset_name: Optional[str] = None):
        """
        Args:
            base_path: Base directory to store dataset.
            dataset_name: Optional name of the dataset subdirectory (e.g., 'plantwild').
                          If None, it will try to auto-detect.
        """
        self.base_path = base_path
        
        if dataset_name:
            self.dataset_path = os.path.join(base_path, dataset_name)
        else:
            # Auto-detect dataset path
            self.dataset_path = self._find_dataset_path()

        # Define images path based on common structures
        if os.path.exists(os.path.join(self.dataset_path, "images")):
            self.images_path = os.path.join(self.dataset_path, "images")
        else:
            self.images_path = self.dataset_path
        
    def _find_dataset_path(self) -> str:
        """Automatically find the main dataset directory."""
        if not os.path.exists(self.base_path):
            return self.base_path

        items = os.listdir(self.base_path)
        # Exclude common non-dataset directories
        potential_datasets = [d for d in items if os.path.isdir(os.path.join(self.base_path, d)) and not d.startswith('.') and d not in ['__pycache__']]
        
        # If there's a single directory, assume it's the dataset directory
        if len(potential_datasets) == 1:
            print(f"Auto-detected dataset directory: {potential_datasets[0]}")
            return os.path.join(self.base_path, potential_datasets[0])
        
        # Default to base path if auto-detection is unclear
        return self.base_path
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)
        print(f"Created directories at {self.base_path}")
    
    def download_from_google_drive(self, file_id: str, output_path: str, extract: bool = True):
        """
        Download dataset from Google Drive
        
        Args:
            file_id: Google Drive file ID
            output_path: Path to save the downloaded file
            extract: Whether to extract if it's a zip file
        """
        print(f"Downloading from Google Drive...")
        
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
            
            if extract and output_path.endswith('.zip'):
                print("Extracting dataset...")
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(self.base_path)
                
                # Remove the zip file after extraction
                os.remove(output_path)
                print("Extraction completed and zip file removed")
            
            return True
        except Exception as e:
            print(f"Error downloading from Google Drive: {e}")
            return False
    
    def download_from_url(self, url: str, output_path: str, extract: bool = True):
        """
        Download dataset from direct URL
        
        Args:
            url: Direct URL to download from
            output_path: Path to save the downloaded file
            extract: Whether to extract if it's a zip file
        """
        print(f"Downloading from {url}...")
        
        try:
            urllib.request.urlretrieve(url, output_path)
            
            if extract and output_path.endswith('.zip'):
                print("Extracting dataset...")
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(self.base_path)
                
                os.remove(output_path)
                print("Extraction completed")
            
            return True
        except Exception as e:
            print(f"Error downloading from URL: {e}")
            return False
    
    def upload_from_local(self, source_path: str):
        """
        Upload dataset from local files (if running locally)
        
        Args:
            source_path: Path to local dataset
        """
        if os.path.exists(source_path):
            shutil.copytree(source_path, self.dataset_path, dirs_exist_ok=True)
            print(f"Copied dataset from {source_path} to {self.dataset_path}")
            return True
        else:
            print(f"Source path {source_path} does not exist")
            return False
    
    def create_sample_dataset(self, num_classes: int = 5, samples_per_class: int = 10):
        """
        Create a small sample dataset for testing (downloads sample images)
        
        Args:
            num_classes: Number of classes to create
            samples_per_class: Number of sample images per class
        """
        print(f"Creating sample dataset with {num_classes} classes and {samples_per_class} samples each...")
        
        # Sample class names and create directories
        sample_classes = [
            "apple_black_rot", "apple_scab", "tomato_leaf_mold", 
            "grape_leaf_blight", "corn_rust"
        ][:num_classes]
        
        # Create sample prompts
        sample_prompts = {}
        for class_name in sample_classes:
            class_dir = os.path.join(self.images_path, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create sample text prompts
            sample_prompts[class_name] = [
                f"{class_name.replace('_', ' ')} shows typical disease symptoms on plant leaves",
                f"Plant affected by {class_name.replace('_', ' ')} displaying characteristic lesions",
                f"{class_name.replace('_', ' ')} infection visible as discoloration on plant tissue",
                f"Symptoms of {class_name.replace('_', ' ')} appearing as spots or marks",
                f"{class_name.replace('_', ' ')} disease manifestation on plant surface"
            ]
            
            # Create placeholder images (you'll need to replace with actual images)
            for i in range(samples_per_class):
                # Create a simple placeholder image file
                placeholder_path = os.path.join(class_dir, f"sample_{i:03d}.jpg")
                # Note: In actual use, you'd put real images here
                with open(placeholder_path, 'w') as f:
                    f.write("# Placeholder - replace with actual image")
        
        # Save sample prompts
        prompts_file = os.path.join(self.base_path, "plantwild_prompts.json")
        with open(prompts_file, 'w') as f:
            json.dump(sample_prompts, f, indent=2)
        
        print(f"Sample dataset created at {self.dataset_path}")
        print(f"Sample prompts saved to {prompts_file}")
        
        return True
    
    def verify_dataset(self) -> Tuple[bool, dict]:
        """
        Verify that the dataset is properly set up
        
        Returns:
            Tuple of (is_valid, info_dict)
        """
        info = {
            'dataset_path_exists': os.path.exists(self.dataset_path),
            'images_path_exists': os.path.exists(self.images_path),
            'prompts_file_exists': False,
            'num_classes': 0,
            'total_images': 0,
            'class_distribution': {}
        }
        
        # Check for prompts file in different locations
        possible_prompts_files = [
            os.path.join(self.base_path, "plantwild_prompts.json"),
            os.path.join(self.dataset_path, "plantwild_prompts.json"),
            os.path.join(self.images_path, "plantwild_prompts.json"),
            "plantwild_prompts.json"  # In current directory
        ]
        
        prompts_file = None
        for pf in possible_prompts_files:
            if os.path.exists(pf):
                prompts_file = pf
                info['prompts_file_exists'] = True
                break
        
        if info['prompts_file_exists']:
            try:
                with open(prompts_file, 'r') as f:
                    prompts_data = json.load(f)
                info['num_classes'] = len(prompts_data)
            except:
                pass
        
        # Check for class folders in the images path
        images_found = False
        if os.path.exists(self.images_path):
            all_items = os.listdir(self.images_path)
            directories = [d for d in all_items if os.path.isdir(os.path.join(self.images_path, d))]
            
            # Filter out system directories
            excluded_dirs = {'__pycache__', '.git', 'checkpoints', 'logs', 'results', '.ipynb_checkpoints'}
            class_dirs = [d for d in directories if d not in excluded_dirs]
            
            images_found = len(class_dirs) > 0
            
            # Count images in each class directory
            for class_dir in class_dirs:
                class_path = os.path.join(self.images_path, class_dir)
                try:
                    files = os.listdir(class_path)
                    image_files = [f for f in files 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                    if image_files:  # Only count directories with actual images
                        info['class_distribution'][class_dir] = len(image_files)
                        info['total_images'] += len(image_files)
                except (PermissionError, FileNotFoundError):
                    continue
        
        info['images_path_exists'] = images_found
        
        # Update num_classes based on image folders if prompts are missing
        if not info['prompts_file_exists'] and images_found:
            info['num_classes'] = len(info['class_distribution'])

        # Check validity - we need images to proceed
        is_valid = (info['dataset_path_exists'] and 
                   info['images_path_exists'] and 
                   info['total_images'] > 0)
        
        return is_valid, info
    
    def print_dataset_info(self):
        """Print detailed information about the dataset"""
        is_valid, info = self.verify_dataset()
        
        print("=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
        print(f"Dataset Valid: {'✅ YES' if is_valid else '❌ NO'}")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Images Path: {self.images_path}")
        print(f"Prompts File Exists: {'✅' if info['prompts_file_exists'] else '❌'}")
        print(f"Number of Classes: {info['num_classes']}")
        print(f"Total Images: {info['total_images']}")
        
        if info['class_distribution']:
            print("\nClass Distribution:")
            print("-" * 40)
            for class_name, count in sorted(info['class_distribution'].items()):
                print(f"  {class_name:25s}: {count:4d} images")
        
        print("=" * 60)
        
        return is_valid, info

def setup_colab_environment():
    """Set up the complete environment for Google Colab"""
    
    print("Setting up Cross-Attention Plant Disease Classification on Google Colab")
    print("=" * 70)
    
    # Install required packages
    print("📦 Installing required packages...")
    os.system("pip install -q transformers torch torchvision torchaudio")
    os.system("pip install -q gdown pyyaml matplotlib seaborn plotly")
    os.system("pip install -q scikit-learn opencv-python albumentations")
    os.system("pip install -q dataclasses-json tqdm")
    
    print("✅ Packages installed successfully!")
    
    # Create dataset manager
    dataset_manager = ColabDatasetManager()
    dataset_manager.setup_directories()
    
    print("\n📁 Dataset directories created")
    print(f"   Base path: {dataset_manager.base_path}")
    print(f"   Dataset path: {dataset_manager.dataset_path}")
    print(f"   Images path: {dataset_manager.images_path}")
    
    return dataset_manager

# Example usage functions for different data sources
def load_from_google_drive(file_id: str):
    """
    Load dataset from Google Drive
    
    Usage:
        # Replace 'your_file_id' with actual Google Drive file ID
        load_from_google_drive('your_file_id')
    """
    dataset_manager = ColabDatasetManager()
    dataset_manager.setup_directories()
    
    zip_path = os.path.join(dataset_manager.base_path, "dataset.zip")
    success = dataset_manager.download_from_google_drive(file_id, zip_path)
    
    if success:
        dataset_manager.print_dataset_info()
    
    return dataset_manager

def load_from_url(url: str):
    """
    Load dataset from direct URL
    
    Usage:
        load_from_url('https://example.com/dataset.zip')
    """
    dataset_manager = ColabDatasetManager()
    dataset_manager.setup_directories()
    
    filename = url.split('/')[-1]
    zip_path = os.path.join(dataset_manager.base_path, filename)
    success = dataset_manager.download_from_url(url, zip_path)
    
    if success:
        dataset_manager.print_dataset_info()
    
    return dataset_manager

def create_sample_for_testing():
    """
    Create a small sample dataset for testing the code
    
    Usage:
        dataset_manager = create_sample_for_testing()
    """
    dataset_manager = ColabDatasetManager()
    dataset_manager.setup_directories()
    dataset_manager.create_sample_dataset(num_classes=3, samples_per_class=5)
    dataset_manager.print_dataset_info()
    
    return dataset_manager

if __name__ == "__main__":
    # Example setup
    dataset_manager = setup_colab_environment()
    print("\n🚀 Environment setup complete!")
    print("\nNext steps:")
    print("1. Load your dataset using one of the methods above")
    print("2. Verify the dataset with dataset_manager.print_dataset_info()")
    print("3. Run the training script")