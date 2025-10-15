import yaml
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import os

@dataclass
class ModelConfig:
    """Configuration for Cross-Attention Model"""
    
    # Model Architecture
    feature_dim: int = 768
    num_attention_heads: int = 8
    num_cross_attention_layers: int = 2
    hidden_dim: int = 512
    dropout: float = 0.1
    
    # Image Encoder
    image_backbone: str = 'resnet50'  # 'resnet50', 'efficientnet_b0'
    patch_size: int = 7
    
    # Text Encoder
    text_model: str = 'bert-base-uncased'
    max_text_length: int = 128
    
    # Fusion
    fusion_type: str = 'adaptive'  # 'concat', 'adaptive', 'bilinear', 'attention'
    
    # Dataset
    num_classes: int = 89
    selected_classes: Optional[List[str]] = None  # Subset of classes to use
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    warmup_epochs: int = 5
    
    # Optimization
    optimizer: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    
    # Loss function
    loss_function: str = 'cross_entropy'  # 'cross_entropy', 'focal', 'label_smoothing'
    label_smoothing: float = 0.1
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_strength: float = 0.5
    
    # Text selection strategy
    text_selection: str = 'random'  # 'random', 'first', 'all'
    num_text_per_class: int = 5
    
    # Regularization
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    use_mixup: bool = False
    use_cutmix: bool = False
    
    # Paths
    dataset_root: str = './data'
    output_dir: str = './outputs'
    checkpoint_dir: str = './checkpoints'
    
    # Hardware
    device: str = 'cuda'
    num_workers: int = 4
    
    # Evaluation
    eval_frequency: int = 5
    save_best_only: bool = True
    
    # Logging
    log_frequency: int = 10
    use_wandb: bool = False
    project_name: str = 'plant-disease-crossattention'
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.selected_classes is not None:
            self.num_classes = len(self.selected_classes)
        
        # Ensure feature_dim is divisible by num_attention_heads
        if self.feature_dim % self.num_attention_heads != 0:
            raise ValueError(f"feature_dim ({self.feature_dim}) must be divisible by num_attention_heads ({self.num_attention_heads})")

class ConfigManager:
    """Manages model configurations and hyperparameter tuning"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.base_config = ModelConfig()
        
    def load_config(self, config_path: str) -> ModelConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update base config with loaded values
        for key, value in config_dict.items():
            if hasattr(self.base_config, key):
                setattr(self.base_config, key, value)
            else:
                print(f"Warning: Unknown configuration key '{key}' ignored")
        
        return self.base_config
    
    def save_config(self, config: ModelConfig, save_path: str):
        """Save configuration to YAML file"""
        config_dict = asdict(config)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def create_hyperparameter_grid(self) -> List[Dict[str, Any]]:
        """Create grid of hyperparameters for tuning"""
        
        param_grid = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [16, 32, 64],
            'feature_dim': [512, 768, 1024],
            'num_attention_heads': [4, 8, 12, 16],
            'num_cross_attention_layers': [1, 2, 3, 4],
            'dropout': [0.1, 0.2, 0.3],
            'fusion_type': ['concat', 'adaptive', 'bilinear', 'attention'],
            'hidden_dim': [256, 512, 768, 1024],
            'weight_decay': [1e-6, 1e-5, 1e-4],
            'warmup_epochs': [3, 5, 10],
            'num_text_per_class': [1, 3, 5, 10],
        }
        
        # Generate all combinations (this could be large!)
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        configurations = []
        for combination in product(*values):
            config_dict = dict(zip(keys, combination))
            
            # Validate feature_dim and num_attention_heads compatibility
            if config_dict['feature_dim'] % config_dict['num_attention_heads'] != 0:
                continue
                
            configurations.append(config_dict)
        
        return configurations
    
    def create_random_search_configs(self, num_configs: int = 50) -> List[Dict[str, Any]]:
        """Create random configurations for hyperparameter search"""
        import random
        
        configurations = []
        
        for _ in range(num_configs):
            # Random sampling
            feature_dim_choices = [512, 768, 1024]
            feature_dim = random.choice(feature_dim_choices)
            
            # Ensure compatibility
            valid_heads = [h for h in [4, 8, 12, 16] if feature_dim % h == 0]
            num_heads = random.choice(valid_heads)
            
            config = {
                'learning_rate': random.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
                'batch_size': random.choice([16, 32, 64]),
                'feature_dim': feature_dim,
                'num_attention_heads': num_heads,
                'num_cross_attention_layers': random.randint(1, 4),
                'dropout': random.uniform(0.1, 0.4),
                'fusion_type': random.choice(['concat', 'adaptive', 'bilinear', 'attention']),
                'hidden_dim': random.choice([256, 512, 768, 1024]),
                'weight_decay': random.choice([1e-6, 1e-5, 1e-4]),
                'warmup_epochs': random.choice([3, 5, 10]),
                'num_text_per_class': random.choice([1, 3, 5, 10]),
            }
            
            configurations.append(config)
        
        return configurations

class ClassSubsetManager:
    """Manages subsets of plant disease classes"""
    
    def __init__(self, all_classes_file: str = 'classes.txt'):
        self.all_classes_file = all_classes_file
        self.all_classes = self.load_all_classes()
    
    def load_all_classes(self) -> List[str]:
        """Load all available classes"""
        try:
            with open(self.all_classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            return classes
        except FileNotFoundError:
            # Fallback to JSON keys if classes.txt not available
            json_file = 'plantwild_prompts.json'
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                return list(data.keys())
            else:
                raise FileNotFoundError("Neither classes.txt nor plantwild_prompts.json found")
    
    def create_subset_by_category(self, categories: List[str]) -> List[str]:
        """Create subset based on plant categories (apple, tomato, etc.)"""
        subset = []
        for category in categories:
            category_classes = [cls for cls in self.all_classes if cls.startswith(category)]
            subset.extend(category_classes)
        return sorted(list(set(subset)))
    
    def create_subset_by_disease_type(self, disease_types: List[str]) -> List[str]:
        """Create subset based on disease types (rust, blight, etc.)"""
        subset = []
        for disease in disease_types:
            disease_classes = [cls for cls in self.all_classes if disease in cls]
            subset.extend(disease_classes)
        return sorted(list(set(subset)))
    
    def create_random_subset(self, num_classes: int) -> List[str]:
        """Create random subset of specified size"""
        import random
        return sorted(random.sample(self.all_classes, min(num_classes, len(self.all_classes))))
    
    def create_balanced_subset(self, categories: List[str], classes_per_category: int = 5) -> List[str]:
        """Create balanced subset with equal representation from each category"""
        subset = []
        for category in categories:
            category_classes = [cls for cls in self.all_classes if cls.startswith(category)]
            if len(category_classes) > classes_per_category:
                import random
                selected = random.sample(category_classes, classes_per_category)
            else:
                selected = category_classes
            subset.extend(selected)
        return sorted(list(set(subset)))
    
    def get_predefined_subsets(self) -> Dict[str, List[str]]:
        """Get predefined useful subsets"""
        subsets = {
            'apple_diseases': self.create_subset_by_category(['apple']),
            'tomato_diseases': self.create_subset_by_category(['tomato']),
            'leaf_diseases': [cls for cls in self.all_classes if 'leaf' in cls],
            'rust_diseases': self.create_subset_by_disease_type(['rust']),
            'blight_diseases': self.create_subset_by_disease_type(['blight']),
            'virus_diseases': self.create_subset_by_disease_type(['virus', 'mosaic']),
            'small_subset_10': self.create_random_subset(10),
            'medium_subset_25': self.create_random_subset(25),
            'large_subset_50': self.create_random_subset(50),
            'major_crops': self.create_subset_by_category(['apple', 'tomato', 'potato', 'corn', 'rice']),
        }
        return subsets
    
    def save_subset_config(self, subset_name: str, classes: List[str], save_dir: str = 'configs'):
        """Save subset configuration"""
        os.makedirs(save_dir, exist_ok=True)
        
        subset_config = {
            'subset_name': subset_name,
            'classes': classes,
            'num_classes': len(classes)
        }
        
        save_path = os.path.join(save_dir, f'subset_{subset_name}.yaml')
        with open(save_path, 'w') as f:
            yaml.dump(subset_config, f, default_flow_style=False, indent=2)
        
        print(f"Saved subset '{subset_name}' with {len(classes)} classes to {save_path}")

def create_experiment_configs():
    """Create a set of experiment configurations for systematic testing"""
    
    config_manager = ConfigManager()
    class_manager = ClassSubsetManager()
    
    # Get predefined subsets
    subsets = class_manager.get_predefined_subsets()
    
    # Create base experiments
    experiments = []
    
    # Experiment 1: Different fusion types on apple diseases
    for fusion_type in ['concat', 'adaptive', 'bilinear', 'attention']:
        config = ModelConfig(
            fusion_type=fusion_type,
            selected_classes=subsets['apple_diseases'],
            num_epochs=30,
            batch_size=32,
            learning_rate=1e-4
        )
        experiments.append((f'apple_fusion_{fusion_type}', config))
    
    # Experiment 2: Different model sizes on medium subset
    for feature_dim, heads in [(512, 8), (768, 12), (1024, 16)]:
        config = ModelConfig(
            feature_dim=feature_dim,
            num_attention_heads=heads,
            selected_classes=subsets['medium_subset_25'],
            num_epochs=40,
            batch_size=32
        )
        experiments.append((f'size_{feature_dim}_{heads}h', config))
    
    # Experiment 3: Learning rate sweep on virus diseases
    for lr in [5e-5, 1e-4, 5e-4]:
        config = ModelConfig(
            learning_rate=lr,
            selected_classes=subsets['virus_diseases'],
            num_epochs=35,
            batch_size=64
        )
        experiments.append((f'virus_lr_{lr}', config))
    
    # Experiment 4: Cross-attention layers comparison
    for num_layers in [1, 2, 3, 4]:
        config = ModelConfig(
            num_cross_attention_layers=num_layers,
            selected_classes=subsets['major_crops'],
            num_epochs=45
        )
        experiments.append((f'layers_{num_layers}', config))
    
    # Save all experiment configs
    os.makedirs('experiments', exist_ok=True)
    
    for exp_name, config in experiments:
        config_manager.save_config(config, f'experiments/{exp_name}.yaml')
    
    print(f"Created {len(experiments)} experiment configurations")
    return experiments

if __name__ == "__main__":
    # Example usage
    
    # Create experiment configurations
    create_experiment_configs()
    
    # Create class subsets
    class_manager = ClassSubsetManager()
    subsets = class_manager.get_predefined_subsets()
    
    for subset_name, classes in subsets.items():
        class_manager.save_subset_config(subset_name, classes)
    
    # Create random hyperparameter configurations
    config_manager = ConfigManager()
    random_configs = config_manager.create_random_search_configs(20)
    
    for i, config in enumerate(random_configs):
        base_config = ModelConfig()
        
        # Update with random config
        for key, value in config.items():
            setattr(base_config, key, value)
        
        config_manager.save_config(base_config, f'experiments/random_search_{i:02d}.yaml')
    
    print("Configuration system setup complete!")