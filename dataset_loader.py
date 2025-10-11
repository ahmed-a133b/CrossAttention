import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional
import numpy as np

class PlantDiseaseDataset(Dataset):
    """Plant Disease Dataset with support for class subsets"""
    
    def __init__(
        self,
        root_dir: str,
        prompts_file: str,
        split: str = 'train',
        selected_classes: Optional[List[str]] = None,
        text_selection: str = 'random',
        num_text_per_class: int = 5,
        transform: Optional[transforms.Compose] = None,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Args:
            root_dir: Path to dataset root directory
            prompts_file: Path to JSON file with text prompts
            split: 'train', 'val', or 'test'
            selected_classes: List of class names to include (None = all classes)
            text_selection: How to select text prompts ('random', 'first', 'all')
            num_text_per_class: Number of text prompts per class
            transform: Image transformations
            max_samples_per_class: Maximum samples per class (for debugging)
        """
        
        self.root_dir = root_dir
        self.split = split
        self.text_selection = text_selection
        self.num_text_per_class = num_text_per_class
        self.transform = transform
        
        # Load text prompts
        with open(prompts_file, 'r') as f:
            self.prompts_data = json.load(f)
        
        # Get all available classes or use selected subset
        if selected_classes is None:
            self.class_names = sorted(list(self.prompts_data.keys()))
        else:
            self.class_names = [cls for cls in selected_classes if cls in self.prompts_data]
            if len(self.class_names) != len(selected_classes):
                missing = set(selected_classes) - set(self.class_names)
                print(f"Warning: Classes not found in prompts: {missing}")
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Load dataset
        self.samples = self._load_samples(max_samples_per_class)
        
        # Prepare text prompts for each class
        self.class_texts = self._prepare_class_texts()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.class_names)} classes for {split} split")
    
    def _load_samples(self, max_samples_per_class: Optional[int] = None) -> List[Tuple[str, int]]:
        """Load image samples and their labels"""
        samples = []
        
        # Try different possible directory structures
        possible_image_dirs = [
            os.path.join(self.root_dir, 'plantwild', 'images'),
            os.path.join(self.root_dir, 'images'),
            os.path.join(self.root_dir, 'plantwild'),
            self.root_dir
        ]
        
        images_dir = None
        for dir_path in possible_image_dirs:
            if os.path.exists(dir_path):
                # Check if this directory contains class subdirectories
                subdirs = [d for d in os.listdir(dir_path) 
                          if os.path.isdir(os.path.join(dir_path, d)) and d in self.class_names]
                if subdirs:
                    images_dir = dir_path
                    break
        
        if images_dir is None:
            raise FileNotFoundError(f"Images directory not found. Tried: {possible_image_dirs}")
        
        print(f"Using images directory: {images_dir}")
        
        # Check if split file exists
        possible_split_files = [
            os.path.join(self.root_dir, 'plantwild', 'trainval.txt'),
            os.path.join(self.root_dir, 'trainval.txt'),
            os.path.join(images_dir, 'trainval.txt')
        ]
        
        split_file = None
        for file_path in possible_split_files:
            if os.path.exists(file_path):
                split_file = file_path
                break
        
        if split_file:
            print(f"Using split file: {split_file}")
            samples = self._load_from_split_file(split_file, max_samples_per_class)
        else:
            print("No split file found, creating splits from directory structure")
            samples = self._load_from_directories(images_dir, max_samples_per_class)
        
        return samples
    
    def _load_from_split_file(self, split_file: str, max_samples_per_class: Optional[int]) -> List[Tuple[str, int]]:
        """Load samples from existing split file"""
        samples = []
        class_counts = {cls: 0 for cls in self.class_names}
        
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split('=')
                if len(parts) != 3:
                    continue
                
                img_path, label_str, split_type = parts
                label = int(label_str)
                split_type = int(split_type)
                
                # Map split types: 1=train, 2=val, 0=test
                if self.split == 'train' and split_type != 1:
                    continue
                elif self.split == 'val' and split_type != 2:
                    continue
                elif self.split == 'test' and split_type != 0:
                    continue
                
                # Get class name from image path
                class_name = os.path.basename(os.path.dirname(img_path))
                
                if class_name not in self.class_to_idx:
                    continue
                
                # Check max samples per class
                if max_samples_per_class and class_counts[class_name] >= max_samples_per_class:
                    continue
                
                # Use our class index mapping
                new_label = self.class_to_idx[class_name]
                samples.append((img_path, new_label))
                class_counts[class_name] += 1
        
        return samples
    
    def _load_from_directories(self, images_dir: str, max_samples_per_class: Optional[int]) -> List[Tuple[str, int]]:
        """Load samples by scanning directory structure"""
        samples = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(images_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            class_label = self.class_to_idx[class_name]
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Shuffle and limit if needed
            random.shuffle(image_files)
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]
            
            # Split into train/val/test (70/15/15)
            num_images = len(image_files)
            if self.split == 'train':
                selected_files = image_files[:int(0.7 * num_images)]
            elif self.split == 'val':
                selected_files = image_files[int(0.7 * num_images):int(0.85 * num_images)]
            else:  # test
                selected_files = image_files[int(0.85 * num_images):]
            
            for img_file in selected_files:
                img_path = os.path.join(class_dir, img_file)
                samples.append((img_path, class_label))
        
        return samples
    
    def _prepare_class_texts(self) -> Dict[int, List[str]]:
        """Prepare text prompts for each class"""
        class_texts = {}
        
        for class_name, class_idx in self.class_to_idx.items():
            texts = self.prompts_data[class_name]
            
            if self.text_selection == 'first':
                selected_texts = texts[:self.num_text_per_class]
            elif self.text_selection == 'random':
                if len(texts) <= self.num_text_per_class:
                    selected_texts = texts
                else:
                    selected_texts = random.sample(texts, self.num_text_per_class)
            elif self.text_selection == 'all':
                selected_texts = texts
            else:
                raise ValueError(f"Unknown text_selection: {self.text_selection}")
            
            class_texts[class_idx] = selected_texts
        
        return class_texts
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int, str]:
        """
        Returns:
            image: Transformed image tensor
            text: Selected text description
            label: Class label (integer)
            class_name: Class name (string)
        """
        
        img_path, label = self.samples[idx]
        class_name = self.idx_to_class[label]
        
        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        # Select text description
        available_texts = self.class_texts[label]
        if len(available_texts) == 1:
            text = available_texts[0]
        else:
            text = random.choice(available_texts)
        
        return image, text, label, class_name
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per class"""
        distribution = {}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def get_all_texts_for_class(self, class_name: str) -> List[str]:
        """Get all text descriptions for a specific class"""
        if class_name not in self.class_to_idx:
            return []
        class_idx = self.class_to_idx[class_name]
        return self.class_texts[class_idx]

def create_data_transforms(image_size: int = 224, augmentation_strength: float = 0.5) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation transforms"""
    
    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=int(15 * augmentation_strength)),
        transforms.ColorJitter(
            brightness=0.2 * augmentation_strength,
            contrast=0.2 * augmentation_strength,
            saturation=0.2 * augmentation_strength,
            hue=0.1 * augmentation_strength
        ),
        transforms.RandomGrayscale(p=0.1 * augmentation_strength),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation/test transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transforms, val_transforms

def collate_fn(batch):
    """Custom collate function for batching"""
    images, texts, labels, class_names = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Keep texts as list for tokenization in model
    texts = list(texts)
    
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Keep class names as list
    class_names = list(class_names)
    
    return images, texts, labels, class_names

def create_data_loaders(
    config,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        root_dir=config.dataset_root,
        prompts_file='plantwild_prompts.json',
        split='train',
        selected_classes=config.selected_classes,
        text_selection=config.text_selection,
        num_text_per_class=config.num_text_per_class,
        transform=train_transform
    )
    
    val_dataset = PlantDiseaseDataset(
        root_dir=config.dataset_root,
        prompts_file='plantwild_prompts.json',
        split='val',
        selected_classes=config.selected_classes,
        text_selection='first',  # Use consistent text for validation
        num_text_per_class=1,
        transform=val_transform
    )
    
    test_dataset = PlantDiseaseDataset(
        root_dir=config.dataset_root,
        prompts_file='plantwild_prompts.json',
        split='test',
        selected_classes=config.selected_classes,
        text_selection='first',  # Use consistent text for testing
        num_text_per_class=1,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples") 
    print(f"Test: {len(test_dataset)} samples")
    print(f"Classes: {len(train_dataset.class_names)}")
    
    # Print class distribution for training set
    train_dist = train_dataset.get_class_distribution()
    print(f"\nTraining set class distribution:")
    for class_name, count in sorted(train_dist.items()):
        print(f"  {class_name}: {count} samples")
    
    return train_loader, val_loader, test_loader

def analyze_dataset(dataset: PlantDiseaseDataset):
    """Analyze dataset and print statistics"""
    
    print(f"\nDataset Analysis:")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_names)}")
    
    # Class distribution
    distribution = dataset.get_class_distribution()
    print(f"\nClass distribution:")
    for class_name, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} samples")
    
    # Text statistics
    total_texts = 0
    for class_idx, texts in dataset.class_texts.items():
        total_texts += len(texts)
    
    avg_texts_per_class = total_texts / len(dataset.class_names)
    print(f"\nText statistics:")
    print(f"Average texts per class: {avg_texts_per_class:.1f}")
    
    # Sample some texts
    print(f"\nSample texts:")
    for i, (class_name, class_idx) in enumerate(list(dataset.class_to_idx.items())[:3]):
        texts = dataset.class_texts[class_idx]
        print(f"\n{class_name}:")
        for j, text in enumerate(texts[:2]):
            print(f"  {j+1}. {text}")

if __name__ == "__main__":
    # Example usage
    from config_manager import ModelConfig
    
    # Test with a small subset
    config = ModelConfig(
        dataset_root='./data',
        selected_classes=['apple black rot', 'apple leaf', 'tomato leaf'],
        batch_size=8,
        num_text_per_class=3
    )
    
    # Create transforms
    train_transform, val_transform = create_data_transforms()
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(
        root_dir=config.dataset_root,
        prompts_file='plantwild_prompts.json',
        split='train',
        selected_classes=config.selected_classes,
        text_selection=config.text_selection,
        num_text_per_class=config.num_text_per_class,
        transform=train_transform,
        max_samples_per_class=10  # For quick testing
    )
    
    # Analyze dataset
    analyze_dataset(train_dataset)
    
    # Test data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Test one batch
    for batch_idx, (images, texts, labels, class_names) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"Images shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Class names: {class_names}")
        print(f"Sample texts:")
        for i, text in enumerate(texts[:3]):
            print(f"  {i}: {text}")
        break