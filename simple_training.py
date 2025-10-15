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
    
    # Add required paths for Trainer class
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.output_dir = "/content/outputs"
    config.checkpoint_dir = "/content/checkpoints"
    
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
        transform=None  # Basic transforms will be added
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
    Complete training pipeline using simplified approach
    
    Args:
        loader: PlantWildV2Loader instance
        config_overrides: Dict of config overrides
    """
    # Setup configuration
    config, classes_file, prompts_file, device = setup_training_from_loader(loader, config_overrides)
    
    # Create datasets and model
    dataset, model = create_datasets_and_model(loader, config, classes_file, prompts_file)
    
    # Use simplified training instead of complex Trainer class
    print("\n🚀 Starting Simplified Training...")
    print("=" * 50)
    
    # Simple training loop
    trained_model = simple_train_loop(model, dataset, config, device)
    
    return trained_model, config

def simple_train_loop(model, dataset, config, device):
    """
    Simplified training loop without complex trainer class
    """
    from torch.utils.data import DataLoader, random_split
    import torch.optim as optimizer_module
    
    # Move model to device
    model = model.to(device)
    
    # Create data splits
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Setup optimizer and loss function
    optimizer = optimizer_module.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"📊 Training setup:")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Device: {device}")
    
    # Training loop
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, texts, labels) in enumerate(train_loader):
            images, texts, labels = images.to(device), texts, labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{config.num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Epoch summary
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{config.num_epochs}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Simple validation every few epochs
        if (epoch + 1) % 5 == 0:
            val_acc = validate_model(model, val_loader, device)
            print(f'Validation Accuracy: {val_acc:.2f}%')
    
    print("✅ Training completed!")
    return model

def validate_model(model, val_loader, device):
    """Simple validation function"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, texts, labels in val_loader:
            images, texts, labels = images.to(device), texts, labels.to(device)
            outputs = model(images, texts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    model.train()
    return 100 * correct / total

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
    model, config = start_training(loader, overrides)
    
    return model, config

if __name__ == "__main__":
    print("🎯 Cross-Attention Training Setup")
    print("Usage:")
    print("  trainer, model = quick_train('your_drive_file_id', epochs=20)")