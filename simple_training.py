"""
Simple Training Script for PlantWild v2 Cross-Attention Model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json

# Import from local files
from dataset_loader import PlantDiseaseDataset
from cross_attention_model import CrossAttentionClassifier
from trainer import Trainer
from config_manager import ConfigManager

def setup_training_from_loader(loader, config_overrides=None):
    """
    Setup training using the PlantWild v2 loader
    
    Args:
        loader: PlantWildV2Loader instance
        config_overrides: Dictionary of config overrides
    """
    print("🎯 Setting up Cross-Attention Training")
    print("=" * 50)
    
    # Find classes and prompts files
    is_valid, info = loader.verify_dataset()
    if not is_valid:
        raise ValueError("Dataset is not valid for training!")
    
    # Create config manager and get base config
    config_manager = ConfigManager()
    config = config_manager.base_config
    
    # Update config with dataset info
    config.num_classes = info['num_classes']
    config.batch_size = 16
    config.learning_rate = 1e-4
    config.num_epochs = 20
    config.image_backbone = "resnet50"
    config.text_model = "bert-base-uncased"
    config.fusion_type = "attention"
    config.hidden_dim = 512
    
    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Find file paths
    classes_file = None
    prompts_file = None
    
    for file_path in [
        os.path.join(loader.dataset_path, "classes.txt"),
        os.path.join(loader.images_path, "classes.txt"),
        os.path.join(loader.base_path, "classes.txt")
    ]:
        if os.path.exists(file_path):
            classes_file = file_path
            break
    
    for file_path in [
        os.path.join(loader.dataset_path, "plantwild_prompts.json"),
        os.path.join(loader.images_path, "plantwild_prompts.json"),
        os.path.join(loader.base_path, "plantwild_prompts.json")
    ]:
        if os.path.exists(file_path):
            prompts_file = file_path
            break
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📊 Dataset Info:")
    print(f"   Classes: {config.num_classes}")
    print(f"   Images: {info['total_images']}")
    print(f"   Device: {device}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Classes file: {classes_file}")
    print(f"   Prompts file: {prompts_file}")
    
    return config, classes_file, prompts_file, device

def create_datasets_and_model(loader, config, classes_file, prompts_file):
    """Create datasets and model from config"""
    print("\n🔧 Creating Datasets and Model...")
    
    # Create dataset
    dataset = PlantDiseaseDataset(
        root_dir=loader.images_path,
        classes_file=classes_file,
        prompts_file=prompts_file,
        transform=None  # Will be set by trainer
    )
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Create model config dictionary
    model_config = {
        'num_classes': config.num_classes,
        'image_backbone': config.image_backbone,
        'text_model': config.text_model,
        'fusion_type': config.fusion_type,
        'hidden_dim': config.hidden_dim,
        'feature_dim': config.feature_dim,
        'patch_size': config.patch_size,
        'max_text_length': config.max_text_length,
        'num_attention_heads': config.num_attention_heads,
        'num_cross_attention_layers': config.num_cross_attention_layers,
        'dropout': config.dropout
    }
    
    # Create model
    model = CrossAttentionClassifier(model_config)
    
    print(f"✅ Model created: {config.image_backbone} + {config.text_model}")
    
    return dataset, model

def start_training(loader, config_overrides=None):
    """
    Complete training pipeline
    
    Args:
        loader: PlantWildV2Loader instance
        config_overrides: Dict of config overrides
    """
    # Setup configuration
    config, classes_file, prompts_file, device = setup_training_from_loader(loader, config_overrides)
    
    # Create datasets and model
    dataset, model = create_datasets_and_model(loader, config, classes_file, prompts_file)
    
    # Create trainer
    trainer = CrossAttentionTrainer(config)
    
    # Start training
    print("\n🚀 Starting Training...")
    print("=" * 50)
    
    trainer.train(model, dataset)
    
    return trainer, model

# Quick training function
def quick_train(drive_file_id: str, epochs: int = 20, batch_size: int = 16):
    """
    One-liner training setup
    
    Args:
        drive_file_id: Google Drive file ID for plantwild_v2.zip
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    from colab_dataset_setup import quick_setup
    
    # Setup dataset
    print("📥 Setting up dataset...")
    loader = quick_setup(drive_file_id)
    
    if loader is None:
        print("❌ Failed to setup dataset")
        return None
    
    # Training overrides
    overrides = {
        'num_epochs': epochs,
        'batch_size': batch_size
    }
    
    # Start training
    trainer, model = start_training(loader, overrides)
    
    return trainer, model

if __name__ == "__main__":
    print("🎯 Cross-Attention Training Setup")
    print("Usage:")
    print("  trainer, model = quick_train('your_drive_file_id', epochs=20)")