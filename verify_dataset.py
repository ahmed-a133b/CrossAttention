#!/usr/bin/env python3
"""
Dataset Structure Verification Script
Verifies that your PlantWild dataset has the correct structure and files.
"""

import os
import json
from typing import Dict, List

def verify_dataset_structure(dataset_root: str = ".") -> Dict:
    """
    Verify the PlantWild dataset structure
    
    Args:
        dataset_root: Path to the dataset root directory
        
    Returns:
        Dict with verification results
    """
    results = {
        'dataset_root_exists': False,
        'classes_file_exists': False,
        'prompts_file_exists': False,
        'num_classes_in_file': 0,
        'num_class_folders': 0,
        'num_classes_with_prompts': 0,
        'class_folders': [],
        'classes_from_file': [],
        'missing_folders': [],
        'missing_prompts': [],
        'sample_counts': {},
        'total_images': 0,
        'is_valid': False
    }
    
    print(f"🔍 Verifying dataset structure in: {os.path.abspath(dataset_root)}")
    print("=" * 60)
    
    # Check if dataset root exists
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset root directory does not exist: {dataset_root}")
        return results
    
    results['dataset_root_exists'] = True
    print(f"✅ Dataset root directory exists: {dataset_root}")
    
    # Check for classes.txt file
    classes_file = os.path.join(dataset_root, 'classes.txt')
    if os.path.exists(classes_file):
        results['classes_file_exists'] = True
        print(f"✅ classes.txt file found")
        
        # Read classes from file
        with open(classes_file, 'r') as f:
            classes_from_file = [line.strip() for line in f if line.strip()]
        
        results['classes_from_file'] = classes_from_file
        results['num_classes_in_file'] = len(classes_from_file)
        print(f"📋 {len(classes_from_file)} classes listed in classes.txt")
    else:
        print(f"❌ classes.txt file not found")
    
    # Check for prompts file
    possible_prompts_files = [
        os.path.join(dataset_root, 'plantwild_prompts.json'),
        'plantwild_prompts.json'
    ]
    
    prompts_file = None
    for pf in possible_prompts_files:
        if os.path.exists(pf):
            prompts_file = pf
            break
    
    if prompts_file:
        results['prompts_file_exists'] = True
        print(f"✅ prompts file found: {prompts_file}")
        
        # Read prompts
        try:
            with open(prompts_file, 'r') as f:
                prompts_data = json.load(f)
            
            results['num_classes_with_prompts'] = len(prompts_data)
            print(f"📝 {len(prompts_data)} classes have text prompts")
            
            # Show sample prompts
            sample_classes = list(prompts_data.keys())[:3]
            for cls in sample_classes:
                num_prompts = len(prompts_data[cls])
                print(f"   {cls}: {num_prompts} prompts")
                
        except Exception as e:
            print(f"❌ Error reading prompts file: {e}")
    else:
        print(f"❌ plantwild_prompts.json not found")
    
    # Check for class folders
    print(f"\n📁 Scanning for class directories...")
    
    # Get all directories in dataset root
    all_items = os.listdir(dataset_root)
    directories = [item for item in all_items if os.path.isdir(os.path.join(dataset_root, item))]
    
    # Filter out system directories
    excluded_dirs = {'__pycache__', '.git', 'checkpoints', 'logs', 'results', '.ipynb_checkpoints', '.DS_Store'}
    class_folders = [d for d in directories if d not in excluded_dirs]
    
    results['class_folders'] = class_folders
    results['num_class_folders'] = len(class_folders)
    
    print(f"📂 Found {len(class_folders)} potential class directories")
    
    # Check each class folder for images
    total_images = 0
    valid_class_folders = []
    
    for folder in class_folders:
        folder_path = os.path.join(dataset_root, folder)
        
        # Count images in this folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        image_files = []
        
        try:
            files = os.listdir(folder_path)
            image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
        except PermissionError:
            print(f"   ⚠️  Permission denied accessing {folder}")
            continue
        
        image_count = len(image_files)
        total_images += image_count
        results['sample_counts'][folder] = image_count
        
        if image_count > 0:
            valid_class_folders.append(folder)
        
        # Show details for first few folders
        if len(valid_class_folders) <= 10:
            print(f"   📷 {folder}: {image_count} images")
    
    if len(valid_class_folders) > 10:
        print(f"   ... and {len(valid_class_folders) - 10} more class folders with images")
    
    results['total_images'] = total_images
    print(f"\n📊 Total images found: {total_images}")
    
    # Cross-validation between files and folders
    if results['classes_file_exists'] and class_folders:
        classes_from_file = results['classes_from_file']
        
        # Check which classes from file have corresponding folders
        missing_folders = []
        for cls in classes_from_file:
            # Check for exact match or with underscore/space variations
            folder_variants = [cls, cls.replace(' ', '_'), cls.replace('_', ' ')]
            
            if not any(variant in class_folders for variant in folder_variants):
                missing_folders.append(cls)
        
        results['missing_folders'] = missing_folders
        
        if missing_folders:
            print(f"\n⚠️  {len(missing_folders)} classes from classes.txt don't have corresponding folders:")
            for cls in missing_folders[:5]:  # Show first 5
                print(f"   - {cls}")
            if len(missing_folders) > 5:
                print(f"   ... and {len(missing_folders) - 5} more")
        else:
            print(f"✅ All classes from classes.txt have corresponding folders")
    
    # Check which folders have prompts
    if results['prompts_file_exists'] and prompts_data:
        missing_prompts = []
        for folder in valid_class_folders:
            # Check for exact match or variations
            prompt_variants = [folder, folder.replace(' ', '_'), folder.replace('_', ' ')]
            
            if not any(variant in prompts_data for variant in prompt_variants):
                missing_prompts.append(folder)
        
        results['missing_prompts'] = missing_prompts
        
        if missing_prompts:
            print(f"\n⚠️  {len(missing_prompts)} class folders don't have corresponding prompts:")
            for folder in missing_prompts[:5]:  # Show first 5
                print(f"   - {folder}")
            if len(missing_prompts) > 5:
                print(f"   ... and {len(missing_prompts) - 5} more")
        else:
            print(f"✅ All class folders have corresponding prompts")
    
    # Overall validation
    is_valid = (
        results['dataset_root_exists'] and
        results['prompts_file_exists'] and
        results['num_class_folders'] > 0 and
        results['total_images'] > 0 and
        len(results['missing_prompts']) == 0
    )
    
    results['is_valid'] = is_valid
    
    print(f"\n" + "=" * 60)
    print(f"📋 VERIFICATION SUMMARY")
    print(f"=" * 60)
    print(f"Dataset Status: {'✅ VALID' if is_valid else '❌ ISSUES FOUND'}")
    print(f"Class Folders: {results['num_class_folders']}")
    print(f"Classes in File: {results['num_classes_in_file']}")
    print(f"Classes with Prompts: {results['num_classes_with_prompts']}")
    print(f"Total Images: {results['total_images']}")
    print(f"Missing Folders: {len(results['missing_folders'])}")
    print(f"Missing Prompts: {len(results['missing_prompts'])}")
    
    if not is_valid:
        print(f"\n🔧 TO FIX ISSUES:")
        if not results['prompts_file_exists']:
            print(f"   1. Ensure plantwild_prompts.json is in the dataset directory")
        if results['total_images'] == 0:
            print(f"   2. Ensure class folders contain image files")
        if results['missing_prompts']:
            print(f"   3. Add text prompts for missing classes in plantwild_prompts.json")
    
    print(f"=" * 60)
    
    return results

def show_sample_structure(dataset_root: str = ".", num_samples: int = 3):
    """Show sample files from the dataset structure"""
    
    print(f"\n📂 SAMPLE DATASET STRUCTURE")
    print(f"-" * 40)
    
    if not os.path.exists(dataset_root):
        print(f"Dataset root not found: {dataset_root}")
        return
    
    # Show root files
    root_files = [f for f in os.listdir(dataset_root) if os.path.isfile(os.path.join(dataset_root, f))]
    print(f"{dataset_root}/")
    for file in root_files[:5]:  # Show first 5 files
        print(f"├── {file}")
    
    # Show sample class directories
    directories = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    excluded_dirs = {'__pycache__', '.git', 'checkpoints', 'logs', 'results', '.ipynb_checkpoints'}
    class_dirs = [d for d in directories if d not in excluded_dirs]
    
    for i, class_dir in enumerate(class_dirs[:num_samples]):
        print(f"├── {class_dir}/")
        
        # Show sample images in this class
        class_path = os.path.join(dataset_root, class_dir)
        try:
            files = os.listdir(class_path)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            for j, img in enumerate(image_files[:3]):  # Show first 3 images
                is_last_img = j == len(image_files[:3]) - 1
                is_last_dir = i == len(class_dirs[:num_samples]) - 1
                
                if is_last_img and is_last_dir:
                    print(f"│   └── {img}")
                else:
                    print(f"│   ├── {img}")
            
            if len(image_files) > 3:
                print(f"│   └── ... ({len(image_files) - 3} more images)")
                
        except PermissionError:
            print(f"│   └── [Permission denied]")
    
    if len(class_dirs) > num_samples:
        print(f"└── ... ({len(class_dirs) - num_samples} more class directories)")

def main():
    """Main function to verify dataset"""
    import sys
    
    # Get dataset root from command line argument or use current directory
    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("🌱 PlantWild Dataset Structure Verification")
    print("=" * 60)
    
    # Verify structure
    results = verify_dataset_structure(dataset_root)
    
    # Show sample structure
    show_sample_structure(dataset_root)
    
    # Return appropriate exit code
    return 0 if results['is_valid'] else 1

if __name__ == "__main__":
    exit(main())