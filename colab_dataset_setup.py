"""
Simple PlantWild v2 Dataset Loader for Google Colab
Downloads and extracts plantwild_v2.zip from Google Drive
"""

import os
import zipfile
import gdown
from typing import Tuple
import json

class PlantWildV2Loader:
    """Simple loader for PlantWild v2 dataset from Google Drive"""
    
    def __init__(self, base_path: str = "/content/plant_disease_data"):
        """
        Args:
            base_path: Where to extract the dataset
        """
        self.base_path = base_path
        self.dataset_path = None
        self.images_path = None
        
    def download_and_extract(self, drive_file_id: str):
        """
        Download plantwild_v2.zip from Google Drive and extract it
        
        Args:
            drive_file_id: Google Drive file ID for plantwild_v2.zip
            
        Returns:
            bool: Success status
        """
        print("🚀 Starting PlantWild v2 Download and Setup")
        print("=" * 50)
        
        # Create base directory
        os.makedirs(self.base_path, exist_ok=True)
        
        # Download zip file
        zip_path = os.path.join(self.base_path, "plantwild_v2.zip")
        print(f"📥 Downloading plantwild_v2.zip...")
        
        try:
            gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", zip_path, quiet=False)
            print(f"✅ Download completed: {zip_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
        
        # Extract zip file
        print(f"📦 Extracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_path)
            
            # Remove zip file after extraction
            os.remove(zip_path)
            print(f"✅ Extraction completed and zip file removed")
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
        
        # Auto-detect the extracted folder structure
        self._detect_paths()
        
        return True
    
    def _detect_paths(self):
        """Auto-detect where the dataset was extracted"""
        print("🔍 Detecting dataset structure...")
        
        # Look for common folder patterns
        possible_paths = [
            os.path.join(self.base_path, "plantwild_v2"),
            os.path.join(self.base_path, "plantwild"),
            os.path.join(self.base_path, "PlantWild"),
            os.path.join(self.base_path, "dataset"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.dataset_path = path
                print(f"📁 Found dataset folder: {os.path.basename(path)}")
                break
        
        # If no specific folder found, use base path
        if self.dataset_path is None:
            self.dataset_path = self.base_path
            print(f"📁 Using base path as dataset folder")
        
        # Look for images folder
        images_candidates = [
            os.path.join(self.dataset_path, "images"),
            os.path.join(self.dataset_path, "Images"),
            self.dataset_path  # Class folders might be directly in dataset folder
        ]
        
        for img_path in images_candidates:
            if os.path.exists(img_path):
                # Check if this path contains class folders
                try:
                    items = os.listdir(img_path)
                    class_folders = [item for item in items if os.path.isdir(os.path.join(img_path, item))]
                    if class_folders:
                        self.images_path = img_path
                        print(f"📸 Found images folder: {os.path.relpath(img_path, self.base_path)}")
                        break
                except:
                    continue
        
        if self.images_path is None:
            self.images_path = self.dataset_path
            print(f"📸 Using dataset folder for images")
    
    def verify_dataset(self) -> Tuple[bool, dict]:
        """
        Verify the dataset is properly loaded
        
        Returns:
            Tuple of (is_valid, info_dict)
        """
        if self.dataset_path is None:
            return False, {"error": "Dataset not loaded yet"}
        
        info = {
            'dataset_path': self.dataset_path,
            'images_path': self.images_path,
            'prompts_file_exists': False,
            'classes_file_exists': False,
            'num_classes': 0,
            'total_images': 0,
            'class_distribution': {}
        }
        
        # Check for prompts file
        prompts_files = [
            os.path.join(self.dataset_path, "plantwild_prompts.json"),
            os.path.join(self.images_path, "plantwild_prompts.json"),
            os.path.join(self.base_path, "plantwild_prompts.json")
        ]
        
        for pf in prompts_files:
            if os.path.exists(pf):
                info['prompts_file_exists'] = True
                try:
                    with open(pf, 'r') as f:
                        prompts_data = json.load(f)
                    info['num_classes'] = len(prompts_data)
                    print(f"📝 Found prompts file: {os.path.basename(pf)}")
                except:
                    pass
                break
        
        # Check for classes file
        classes_files = [
            os.path.join(self.dataset_path, "classes.txt"),
            os.path.join(self.images_path, "classes.txt"),
            os.path.join(self.base_path, "classes.txt")
        ]
        
        for cf in classes_files:
            if os.path.exists(cf):
                info['classes_file_exists'] = True
                print(f"📋 Found classes file: {os.path.basename(cf)}")
                break
        
        # Count images in class folders
        if os.path.exists(self.images_path):
            try:
                items = os.listdir(self.images_path)
                class_folders = [item for item in items if os.path.isdir(os.path.join(self.images_path, item))]
                
                for class_folder in class_folders:
                    class_path = os.path.join(self.images_path, class_folder)
                    try:
                        files = os.listdir(class_path)
                        image_files = [f for f in files 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                        if image_files:
                            info['class_distribution'][class_folder] = len(image_files)
                            info['total_images'] += len(image_files)
                    except:
                        continue
                
                if not info['num_classes']:  # If prompts didn't give us class count
                    info['num_classes'] = len(info['class_distribution'])
                    
            except:
                pass
        
        # Dataset is valid if we have images
        is_valid = info['total_images'] > 0
        
        return is_valid, info
    
    def print_info(self):
        """Print dataset information"""
        is_valid, info = self.verify_dataset()
        
        print("\n" + "=" * 60)
        print("PLANTWILD V2 DATASET INFORMATION")
        print("=" * 60)
        print(f"Dataset Valid: {'✅ YES' if is_valid else '❌ NO'}")
        print(f"Dataset Path: {info.get('dataset_path', 'Not found')}")
        print(f"Images Path: {info.get('images_path', 'Not found')}")
        print(f"Prompts File: {'✅' if info.get('prompts_file_exists') else '❌'}")
        print(f"Classes File: {'✅' if info.get('classes_file_exists') else '❌'}")
        print(f"Number of Classes: {info.get('num_classes', 0)}")
        print(f"Total Images: {info.get('total_images', 0)}")
        
        if info.get('class_distribution'):
            print("\nClass Distribution:")
            print("-" * 40)
            for class_name, count in sorted(info['class_distribution'].items()):
                print(f"  {class_name:30s}: {count:4d} images")
        
        print("=" * 60)
        
        return is_valid, info

def setup_plantwild_v2(drive_file_id: str, base_path: str = "/content/plant_disease_data"):
    """
    Complete setup for PlantWild v2 dataset
    
    Args:
        drive_file_id: Google Drive file ID for plantwild_v2.zip
        base_path: Where to extract the dataset
        
    Returns:
        PlantWildV2Loader: Configured loader instance
    """
    # Install gdown if needed
    try:
        import gdown
    except ImportError:
        print("📦 Installing gdown...")
        os.system("pip install -q gdown")
        import gdown
    
    # Create loader and download dataset
    loader = PlantWildV2Loader(base_path)
    
    if loader.download_and_extract(drive_file_id):
        loader.print_info()
        return loader
    else:
        print("❌ Failed to setup dataset")
        return None

def quick_setup(drive_file_id: str):
    """
    One-liner setup for PlantWild v2
    
    Usage:
        loader = quick_setup('your_google_drive_file_id')
    """
    return setup_plantwild_v2(drive_file_id)

if __name__ == "__main__":
    print("🚀 PlantWild v2 Dataset Loader")
    print("Usage:")
    print("  loader = quick_setup('your_google_drive_file_id')")
    print("  # Replace 'your_google_drive_file_id' with actual file ID")