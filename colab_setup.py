#!/usr/bin/env python3
"""
Simple setup script for running Cross-Attention Plant Disease Classification on Google Colab
"""

def setup_colab():
    """Quick setup for Google Colab"""
    
    print("🌱 Setting up Cross-Attention Plant Disease Classification on Google Colab")
    print("=" * 70)
    
    # Install packages
    print("📦 Installing packages...")
    import subprocess
    import sys
    
    packages = [
        "transformers", "torch", "torchvision", "torchaudio",
        "gdown", "pyyaml", "matplotlib", "seaborn", "plotly",
        "scikit-learn", "opencv-python", "albumentations",
        "dataclasses-json", "tqdm"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"  ✅ {package}")
        except:
            print(f"  ❌ Failed to install {package}")
    
    print("\n📁 Creating directories...")
    import os
    os.makedirs("/content/plant_disease_data", exist_ok=True)
    os.makedirs("/content/outputs", exist_ok=True)
    os.makedirs("/content/checkpoints", exist_ok=True)
    print("  ✅ Directories created")
    
    print("\n🚀 Setup complete! You can now:")
    print("1. Upload your dataset")
    print("2. Run the training script")
    print("3. Analyze results")
    
    return True

def quick_test():
    """Run a quick test to verify everything works"""
    
    print("\n🧪 Running quick test...")
    
    try:
        # Test imports
        import torch
        import transformers
        from PIL import Image
        import numpy as np
        
        print(f"  ✅ PyTorch {torch.__version__}")
        print(f"  ✅ Transformers {transformers.__version__}")
        print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file for Colab"""
    
    config_content = """# Cross-Attention Configuration for Google Colab

# Dataset
dataset_path: "/content/plant_disease_data"
selected_classes: []  # Empty = all classes
num_classes: 0  # Auto-detected

# Model Architecture  
image_backbone: "resnet50"
text_encoder: "bert-base-uncased"
feature_dim: 768
fusion_type: "adaptive"
num_cross_attention_layers: 4
num_attention_heads: 12
dropout_rate: 0.1

# Training (Colab optimized)
batch_size: 32  # Reduce if OOM
learning_rate: 0.0001
num_epochs: 20  # Increase for full training
weight_decay: 0.01
mixed_precision: true  # Faster on Colab GPU

# Paths (Colab)
output_dir: "/content/outputs"
checkpoint_dir: "/content/checkpoints"

# Evaluation
eval_frequency: 5
save_frequency: 10
save_predictions: true
"""
    
    with open("/content/colab_config.yaml", "w") as f:
        f.write(config_content)
    
    print("📄 Created sample config: /content/colab_config.yaml")
    return "/content/colab_config.yaml"

def get_dataset_info():
    """Print information about loading datasets on Colab"""
    
    print("\n📊 How to load your dataset on Google Colab:")
    print("=" * 50)
    
    print("\n🔗 Method 1: Google Drive")
    print("  1. Upload dataset.zip to Google Drive")
    print("  2. Get shareable link and extract file ID")
    print("  3. Use: load_from_google_drive('file_id')")
    
    print("\n🌐 Method 2: Direct URL")
    print("  1. Host dataset on service with direct download")
    print("  2. Use: load_from_url('https://example.com/dataset.zip')")
    
    print("\n📤 Method 3: Manual Upload")
    print("  1. Use files.upload() in Colab")
    print("  2. Select and upload dataset.zip")
    print("  3. Extract automatically")
    
    print("\n📁 Expected dataset structure:")
    print("  /content/plant_disease_data/")
    print("  ├── plantwild/")
    print("  │   └── images/")
    print("  │       ├── apple_black_rot/")
    print("  │       ├── tomato_leaf/")
    print("  │       └── ...")
    print("  └── plantwild_prompts.json")

if __name__ == "__main__":
    # Run setup
    setup_colab()
    
    # Run quick test
    if quick_test():
        print("\n✅ All tests passed!")
    
    # Create sample config
    create_sample_config()
    
    # Show dataset info
    get_dataset_info()
    
    print(f"\n🎉 Ready to run Cross-Attention Plant Disease Classification!")
    print(f"💡 Use the Jupyter notebook for step-by-step execution")