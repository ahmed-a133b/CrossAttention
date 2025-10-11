# 🌱 Cross-Attention Plant Disease Classification on Google Colab

A complete implementation of Cross-Attention Fusion for multimodal plant disease classification that runs seamlessly on Google Colab.

## 🚀 Quick Start on Google Colab

### Option 1: Use the Jupyter Notebook (Recommended)
1. **Open the notebook**: Upload `Cross_Attention_Plant_Disease_Colab.ipynb` to Google Colab
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Run setup cells**: Follow the step-by-step instructions
4. **Upload dataset**: Use one of the provided methods
5. **Train model**: Run the training cells

### Option 2: Python Script Approach
1. **Upload files** to Colab:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload all .py files
   ```

2. **Run setup**:
   ```python
   !python colab_setup.py
   ```

3. **Load dataset** (choose one method):
   ```python
   # Method A: Google Drive
   from colab_dataset_setup import load_from_google_drive
   dataset_manager = load_from_google_drive('your_file_id')
   
   # Method B: Direct URL  
   from colab_dataset_setup import load_from_url
   dataset_manager = load_from_url('https://example.com/dataset.zip')
   
   # Method C: Manual upload
   from google.colab import files
   uploaded = files.upload()  # Upload dataset.zip
   ```

4. **Train model**:
   ```python
   from trainer import Trainer
   from config_manager import ModelConfig
   
   config = ModelConfig(dataset_path="/content/plant_disease_data")
   trainer = Trainer(config)
   results = trainer.train()
   ```

## 📊 Dataset Setup Methods

### 🔗 Method 1: Google Drive (Best for large datasets)
```python
# 1. Upload dataset.zip to Google Drive
# 2. Right-click → Get shareable link
# 3. Extract file ID from URL: https://drive.google.com/file/d/FILE_ID/view
# 4. Use the file ID:

from colab_dataset_setup import load_from_google_drive
dataset_manager = load_from_google_drive('1AbCdEfGhIjKlMnOpQrStUvWxYz')
```

### 🌐 Method 2: Direct URL
```python
# If you have a direct download link:
from colab_dataset_setup import load_from_url
dataset_manager = load_from_url('https://example.com/plantwild_dataset.zip')
```

### 📤 Method 3: Manual Upload (Small datasets < 25MB)
```python
from google.colab import files
import zipfile

# Upload dataset.zip
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('/content/plant_disease_data/')
```

### 🧪 Method 4: Sample Dataset (For testing)
```python
# Create a small sample dataset for testing:
from colab_dataset_setup import create_sample_for_testing
dataset_manager = create_sample_for_testing()
```

## 📁 Required Dataset Structure

Your dataset should have this structure after extraction:
```
/content/plant_disease_data/
├── plantwild/
│   └── images/
│       ├── apple_black_rot/
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       ├── tomato_leaf/
│       │   ├── image1.jpg
│       │   └── ...
│       └── [other_disease_classes]/
└── plantwild_prompts.json
```

## ⚙️ Configuration for Colab

### Basic Configuration
```python
from config_manager import ModelConfig

config = ModelConfig(
    # Colab-specific paths
    dataset_path="/content/plant_disease_data",
    output_dir="/content/outputs",
    checkpoint_dir="/content/checkpoints",
    
    # Colab-optimized settings
    batch_size=32,          # Reduce if OOM
    mixed_precision=True,   # Faster training
    num_epochs=20,          # Adjust as needed
    
    # Model architecture
    image_backbone="resnet50",
    fusion_type="adaptive",
    feature_dim=768
)
```

### Memory-Optimized Configuration (if getting OOM errors)
```python
config = ModelConfig(
    dataset_path="/content/plant_disease_data",
    batch_size=16,          # Smaller batch
    feature_dim=512,        # Smaller features
    num_epochs=15,
    mixed_precision=True,   # Essential for memory
    gradient_clip_norm=1.0  # Stability
)
```

## 🧪 Experiment Examples

### Quick Test (3 classes, 5 epochs)
```python
from config_manager import ModelConfig, ClassSubsetManager
from trainer import Trainer

# Create small subset for testing
class_manager = ClassSubsetManager()
test_classes = class_manager.create_random_subset(3)

config = ModelConfig(
    dataset_path="/content/plant_disease_data",
    selected_classes=test_classes,
    num_classes=3,
    batch_size=16,
    num_epochs=5,
    output_dir="/content/quick_test"
)

trainer = Trainer(config)
results = trainer.train()
print(f"Test accuracy: {results['accuracy']:.4f}")
```

### Compare Fusion Methods
```python
fusion_types = ['concat', 'adaptive', 'bilinear', 'attention']
results = {}

for fusion in fusion_types:
    config = ModelConfig(
        dataset_path="/content/plant_disease_data",
        selected_classes=test_classes,  # Small subset
        fusion_type=fusion,
        num_epochs=10
    )
    
    trainer = Trainer(config)
    result = trainer.train()
    results[fusion] = result['accuracy']
    
    print(f"{fusion}: {result['accuracy']:.4f}")

# Find best fusion method
best_fusion = max(results, key=results.get)
print(f"Best fusion: {best_fusion} ({results[best_fusion]:.4f})")
```

### Hyperparameter Search
```python
from config_manager import ConfigManager

config_manager = ConfigManager()

# Generate random configurations
configs = config_manager.create_random_search_configs(5)
best_acc = 0

for i, config_dict in enumerate(configs):
    print(f"Testing configuration {i+1}/5...")
    
    # Create config with hyperparameters
    config = ModelConfig(dataset_path="/content/plant_disease_data")
    for key, value in config_dict.items():
        setattr(config, key, value)
    
    trainer = Trainer(config)
    results = trainer.train()
    
    if results['accuracy'] > best_acc:
        best_acc = results['accuracy']
        best_config = config_dict
    
    print(f"  Accuracy: {results['accuracy']:.4f}")

print(f"Best accuracy: {best_acc:.4f}")
print(f"Best config: {best_config}")
```

## 💡 Colab Tips

### GPU and Memory Management
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Monitor GPU memory during training
!nvidia-smi

# Clear GPU memory if needed
torch.cuda.empty_cache()
```

### Save Results Permanently
```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Update config to save to Drive
config.output_dir = "/content/drive/MyDrive/plant_disease_results"
config.checkpoint_dir = "/content/drive/MyDrive/plant_disease_checkpoints"
```

### Download Results
```python
# Download results as zip file
from google.colab import files
import zipfile

# Create zip with all results
with zipfile.ZipFile('results.zip', 'w') as zipf:
    # Add model checkpoints
    for root, dirs, files in os.walk('/content/checkpoints'):
        for file in files:
            zipf.write(os.path.join(root, file), file)
    
    # Add plots and metrics
    for root, dirs, files in os.walk('/content/outputs'):
        for file in files:
            if file.endswith(('.png', '.json', '.csv')):
                zipf.write(os.path.join(root, file), file)

files.download('results.zip')
```

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **Out of Memory (OOM) Error**
   ```python
   # Reduce batch size
   config.batch_size = 16  # or 8
   
   # Use smaller model
   config.feature_dim = 512
   config.image_backbone = "resnet18"  # instead of resnet50
   ```

2. **Session Timeout**
   ```python
   # Save checkpoints frequently
   config.save_frequency = 5  # Save every 5 epochs
   
   # Use Google Drive for persistence
   from google.colab import drive
   drive.mount('/content/drive')
   config.checkpoint_dir = "/content/drive/MyDrive/checkpoints"
   ```

3. **Dataset Not Found**
   ```python
   # Verify dataset structure
   import os
   print("Dataset contents:")
   for root, dirs, files in os.walk('/content/plant_disease_data'):
       level = root.replace('/content/plant_disease_data', '').count(os.sep)
       indent = ' ' * 2 * level
       print(f"{indent}{os.path.basename(root)}/")
       subindent = ' ' * 2 * (level + 1)
       for file in files[:5]:  # Show first 5 files
           print(f"{subindent}{file}")
   ```

4. **Slow Training**
   ```python
   # Enable mixed precision
   config.mixed_precision = True
   
   # Use smaller dataset for testing
   config.selected_classes = class_manager.create_random_subset(10)
   
   # Reduce epochs for quick testing  
   config.num_epochs = 10
   ```

## 📈 Expected Results

- **Quick test (3 classes, 5 epochs)**: ~60-80% accuracy
- **Medium subset (10 classes, 20 epochs)**: ~70-85% accuracy  
- **Full dataset (all classes, 50 epochs)**: ~80-95% accuracy

Results depend on:
- Dataset quality and size
- Class balance
- Hyperparameter selection
- Training duration

## 🎯 Next Steps

1. **Scale up**: Use full dataset with more epochs
2. **Optimize**: Run extensive hyperparameter search
3. **Compare**: Implement MVPDR baseline for comparison
4. **Visualize**: Enable attention map visualization
5. **Ensemble**: Combine multiple models for better performance

## 📚 Additional Resources

- [Google Colab Tutorial](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch on Colab](https://pytorch.org/tutorials/beginner/colab.html)
- [Transformers Library](https://huggingface.co/transformers/)

- [Cross-Attention Mechanisms](https://arxiv.org/abs/1706.03762)
