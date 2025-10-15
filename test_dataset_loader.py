#!/usr/bin/env python3
"""
Test script for the updated PlantWild dataset loader
"""

import os
import sys
sys.path.append('.')

def test_dataset_loader():
    """Test the dataset loader with your current structure"""
    
    print("🧪 Testing PlantWild Dataset Loader")
    print("=" * 50)
    
    # Check if required files exist
    required_files = ['dataset_loader.py', 'config_manager.py', 'plantwild_prompts.json']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    try:
        from dataset_loader import PlantDiseaseDataset, create_data_transforms
        from config_manager import ModelConfig
        
        print("✅ Successfully imported dataset modules")
        
        # Test configuration with a few classes
        test_classes = ['apple black rot', 'apple scab', 'tomato leaf mold']
        
        config = ModelConfig(
            dataset_root='.',  # Current directory
            selected_classes=test_classes,
            batch_size=4,
            num_text_per_class=2
        )
        
        print(f"📋 Testing with classes: {test_classes}")
        
        # Create transforms
        train_transform, val_transform = create_data_transforms(image_size=224)
        print("✅ Created data transforms")
        
        # Find prompts file
        prompts_file = 'plantwild_prompts.json'
        if not os.path.exists(prompts_file):
            print(f"❌ Prompts file not found: {prompts_file}")
            return False
        
        # Test dataset creation
        try:
            dataset = PlantDiseaseDataset(
                root_dir='.',
                prompts_file=prompts_file,
                split='train',
                selected_classes=test_classes,
                text_selection='random',
                num_text_per_class=2,
                transform=train_transform,
                max_samples_per_class=5  # Limit for testing
            )
            
            print(f"✅ Successfully created dataset with {len(dataset)} samples")
            print(f"📊 Classes: {len(dataset.class_names)}")
            
            # Test loading a sample
            if len(dataset) > 0:
                image, text, label, class_name = dataset[0]
                print(f"✅ Successfully loaded sample:")
                print(f"   Image shape: {image.shape}")
                print(f"   Text: {text[:100]}...")
                print(f"   Label: {label}")
                print(f"   Class: {class_name}")
                
                # Show class distribution
                distribution = dataset.get_class_distribution()
                print(f"\n📈 Class distribution:")
                for class_name, count in distribution.items():
                    print(f"   {class_name}: {count} samples")
                
                return True
            else:
                print("❌ Dataset is empty")
                return False
                
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_with_all_classes():
    """Test with all available classes"""
    
    print("\n🧪 Testing with All Available Classes")
    print("=" * 50)
    
    try:
        from dataset_loader import PlantDiseaseDataset
        
        config_all = {
            'dataset_root': '.',
            'selected_classes': None,  # Use all classes
            'batch_size': 8,
            'num_text_per_class': 1
        }
        
        # Create dataset with all classes
        dataset_all = PlantDiseaseDataset(
            root_dir='.',
            prompts_file='plantwild_prompts.json',
            split='train',
            selected_classes=None,  # All classes
            text_selection='first',
            num_text_per_class=1,
            transform=None,  # No transform for quick testing
            max_samples_per_class=2  # Limit samples for testing
        )
        
        print(f"✅ Created dataset with ALL classes:")
        print(f"   Total samples: {len(dataset_all)}")
        print(f"   Number of classes: {len(dataset_all.class_names)}")
        
        # Show sample of classes
        sample_classes = dataset_all.class_names[:10]
        print(f"   Sample classes: {sample_classes}")
        
        if len(dataset_all.class_names) > 10:
            print(f"   ... and {len(dataset_all.class_names) - 10} more classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with all classes: {e}")
        return False

def main():
    """Main test function"""
    
    print("🌱 PlantWild Dataset Loader Test")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Dataset structure expected: class folders directly in current directory")
    
    # Test 1: Basic functionality with subset
    success1 = test_dataset_loader()
    
    # Test 2: All classes (if first test passed)
    success2 = test_with_all_classes() if success1 else False
    
    print("\n" + "=" * 60)
    print("🏁 TEST RESULTS")
    print("=" * 60)
    print(f"Basic functionality: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"All classes test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Your dataset loader is ready to use.")
        print("\n📋 Next steps:")
        print("1. Run training with your preferred configuration")
        print("2. Try different class subsets for experiments")
        print("3. Adjust batch size based on your GPU memory")
    else:
        print("\n⚠️ Some tests failed. Please check:")
        print("1. Dataset structure (class folders in current directory)")
        print("2. plantwild_prompts.json file exists")
        print("3. Class folders contain image files")
        print("4. File permissions are correct")
    
    print("=" * 60)
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)