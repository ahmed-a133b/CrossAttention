"""
Script to fix PlantWild dataset structure after extraction in Google Colab
This handles the case where the dataset is extracted directly in the base folder
"""

import os
import shutil
import json
from typing import Dict, List

def fix_plantwild_structure(base_path: str = "/content/plant_disease_data"):
    """
    Fix the PlantWild dataset structure to match expected format
    
    Args:
        base_path: Base path where dataset was extracted
    """
    print("🔧 Fixing PlantWild dataset structure...")
    
    # Expected paths
    expected_plantwild_path = os.path.join(base_path, "plantwild")
    expected_images_path = os.path.join(expected_plantwild_path, "images")
    
    # Look for class directories in various locations
    possible_locations = [
        base_path,  # Directly in base path
        os.path.join(base_path, "plantwild_vz"),  # Common extracted folder name
        os.path.join(base_path, "PlantWild"),     # Alternative name
        expected_plantwild_path,                  # Already in correct location
    ]
    
    class_dirs_found = []
    source_location = None
    
    # Find where the class directories are located
    for location in possible_locations:
        if os.path.exists(location):
            items = os.listdir(location)
            # Look for directories that look like plant disease classes
            potential_classes = [item for item in items if os.path.isdir(os.path.join(location, item))]
            
            # Filter out common non-class directories
            excluded = {'images', 'plantwild', 'prompts', '__pycache__', '.git'}
            potential_classes = [item for item in potential_classes if item not in excluded]
            
            if len(potential_classes) > 5:  # Likely found the class directories
                class_dirs_found = potential_classes
                source_location = location
                print(f"📁 Found {len(potential_classes)} class directories in {location}")
                break
    
    if not class_dirs_found:
        print("❌ No class directories found. Please check the dataset extraction.")
        return False
    
    # Create the expected directory structure
    os.makedirs(expected_images_path, exist_ok=True)
    
    # Move class directories to the correct location if needed
    if source_location != expected_images_path:
        print(f"📦 Moving class directories from {source_location} to {expected_images_path}")
        
        for class_dir in class_dirs_found:
            src_path = os.path.join(source_location, class_dir)
            dst_path = os.path.join(expected_images_path, class_dir)
            
            if os.path.exists(dst_path):
                print(f"  ⚠️  Directory {class_dir} already exists in target, skipping...")
                continue
            
            try:
                shutil.move(src_path, dst_path)
                print(f"  ✅ Moved {class_dir}")
            except Exception as e:
                print(f"  ❌ Error moving {class_dir}: {e}")
    
    # Verify the structure
    final_class_dirs = [d for d in os.listdir(expected_images_path) 
                       if os.path.isdir(os.path.join(expected_images_path, d))]
    
    print(f"\n📊 Final structure verification:")
    print(f"  Base path: {base_path}")
    print(f"  PlantWild path: {expected_plantwild_path}")
    print(f"  Images path: {expected_images_path}")
    print(f"  Number of class directories: {len(final_class_dirs)}")
    
    # Count images per class
    total_images = 0
    class_distribution = {}
    
    for class_dir in final_class_dirs[:10]:  # Show first 10 classes
        class_path = os.path.join(expected_images_path, class_dir)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        class_distribution[class_dir] = len(image_files)
        total_images += len(image_files)
    
    print(f"  Total images: {total_images}")
    print(f"  Sample class distribution:")
    for class_name, count in list(class_distribution.items())[:5]:
        print(f"    {class_name}: {count} images")
    
    if len(final_class_dirs) > 5:
        print(f"    ... and {len(final_class_dirs) - 5} more classes")
    
    return True

def check_prompts_file(base_path: str = "/content/plant_disease_data"):
    """
    Check if prompts file exists and is accessible
    
    Args:
        base_path: Base path to check for prompts file
    """
    print("\n🔍 Checking for prompts file...")
    
    possible_prompts_locations = [
        os.path.join(base_path, "plantwild_prompts.json"),
        "/content/plantwild_prompts.json",
        "plantwild_prompts.json"
    ]
    
    for prompts_file in possible_prompts_locations:
        if os.path.exists(prompts_file):
            print(f"✅ Found prompts file at: {prompts_file}")
            
            try:
                with open(prompts_file, 'r') as f:
                    prompts_data = json.load(f)
                print(f"  📝 Contains {len(prompts_data)} classes")
                
                # Show sample classes
                sample_classes = list(prompts_data.keys())[:5]
                print(f"  📋 Sample classes: {sample_classes}")
                
                return True
            except Exception as e:
                print(f"  ❌ Error reading prompts file: {e}")
                
    print("❌ Prompts file not found. You may need to upload it separately.")
    print("💡 The prompts file should contain text descriptions for each disease class.")
    return False

def create_test_config(base_path: str = "/content/plant_disease_data", num_classes: int = 5):
    """
    Create a test configuration with a subset of classes
    
    Args:
        base_path: Base path to dataset
        num_classes: Number of classes to include in test
    """
    print(f"\n⚙️  Creating test configuration with {num_classes} classes...")
    
    images_path = os.path.join(base_path, "plantwild", "images")
    
    if not os.path.exists(images_path):
        print("❌ Images directory not found. Please run fix_plantwild_structure() first.")
        return None
    
    # Get available classes
    available_classes = [d for d in os.listdir(images_path) 
                        if os.path.isdir(os.path.join(images_path, d))]
    
    # Select subset for testing
    test_classes = available_classes[:num_classes]
    
    print(f"📝 Selected test classes: {test_classes}")
    
    # Create a simple config
    test_config = {
        'dataset_root': base_path,
        'selected_classes': test_classes,
        'batch_size': 8,
        'num_text_per_class': 3,
        'text_selection': 'random'
    }
    
    return test_config

def main():
    """Main function to fix dataset and verify setup"""
    base_path = "/content/plant_disease_data"
    
    print("🚀 PlantWild Dataset Structure Fixer")
    print("=" * 50)
    
    # Step 1: Fix directory structure
    success = fix_plantwild_structure(base_path)
    
    if success:
        # Step 2: Check prompts file
        check_prompts_file(base_path)
        
        # Step 3: Create test configuration
        test_config = create_test_config(base_path, num_classes=5)
        
        print("\n🎉 Dataset structure fix completed!")
        print("\n📋 Next steps:")
        print("1. If prompts file is missing, upload 'plantwild_prompts.json'")
        print("2. Run the dataset verification:")
        print("   from colab_dataset_setup import ColabDatasetManager")
        print("   manager = ColabDatasetManager()")
        print("   manager.print_dataset_info()")
        print("3. Start training with your cross-attention model!")
        
        return test_config
    else:
        print("❌ Failed to fix dataset structure. Please check the extraction.")
        return None

if __name__ == "__main__":
    config = main()