# Cross-Attention Plant Disease Classification - Google Colab Setup

This guide helps you set up and run the Cross-Attention Plant Disease Classification model on Google Colab.

## 🚀 Quick Start

### 1. Dataset Preparation
The model works with the PlantWild dataset. You have several options:

#### Option A: Use Google Drive (Recommended)
1. Download the PlantWild dataset (plantwild_vz.zip or similar)
2. Upload the dataset zip file to your Google Drive
3. Right-click the file → Get shareable link
4. Copy the file ID from the URL (the long string after `/d/` and before `/view`)
5. Use this file ID in the notebook's Google Drive loading section

#### Option B: Use Direct URL
If you have a direct download link to the dataset:
1. Use the URL loading method in the notebook

### 2. Required Files
Upload these files to your Colab environment:
- `cross_attention_model.py` - Main model architecture
- `config_manager.py` - Configuration management  
- `dataset_loader.py` - Dataset loading utilities
- `trainer.py` - Training pipeline
- `colab_dataset_setup.py` - Colab-specific setup
- `fix_plantwild_dataset.py` - Dataset structure fixer
- `plantwild_prompts.json` - Text descriptions for each disease class

### 3. Run the Notebook
1. Open `Cross_Attention_Plant_Disease_Colab.ipynb` in Google Colab
2. Run the setup cells to install dependencies
3. Upload your files or use the file uploader
4. Load your dataset using Google Drive method
5. **IMPORTANT: Run the dataset structure fixer cell after download**
6. Configure your experiment parameters
7. Start training!

## 🔧 Dataset Structure Fix

**Important:** After downloading the PlantWild dataset, you MUST run the dataset structure fixer because:
- The dataset may extract to unexpected locations
- Class folders might be directly in the base directory
- Our code expects a specific folder structure

The fixer automatically:
- Locates class directories wherever they were extracted
- Moves them to the correct location: `/content/plant_disease_data/plantwild/images/`
- Verifies the final structure
- Reports statistics

## 📊 Expected Final Structure

After running the dataset fixer:
```
/content/plant_disease_data/
├── plantwild/
│   ├── images/
│   │   ├── Apple___Apple_scab/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── Apple___Black_rot/
│   │   ├── Tomato___Leaf_Mold/
│   │   └── ... (many disease classes)
├── plantwild_prompts.json
└── dataset.zip (removed after extraction)
```

## 🚨 Common Issues & Solutions

### 1. "Dataset Valid: ❌ NO" after download
**Problem:** Dataset structure doesn't match expectations  
**Solution:** 
1. Run the dataset structure fixer cell
2. Check that class directories contain actual image files
3. Verify prompts file is uploaded

### 2. "No class directories found"
**Problem:** Dataset extracted to unexpected location  
**Solution:**
1. Check `/content/plant_disease_data/` contents
2. Look for folders with disease names (e.g., "Apple___Black_rot")
3. Manually move them to `/content/plant_disease_data/plantwild/images/`

### 3. "Prompts file not found"
**Problem:** `plantwild_prompts.json` not in correct location  
**Solution:**
1. Upload the prompts file to `/content/` directory
2. Or place it in `/content/plant_disease_data/`
3. Re-run the verification cell

### 4. CUDA Out of Memory
**Problem:** GPU memory insufficient  
**Solution:**
- Reduce batch_size to 4 or 8
- Use smaller image_size (224 → 128)
- Enable gradient_checkpointing
- Restart runtime and clear cache

### 5. Slow Training
**Problem:** Training takes too long  
**Solution:**
- Enable GPU: Runtime → Change runtime type → GPU
- Use selected_classes for subset training
- Reduce num_epochs for testing
- Use mixed precision training

## ⚙️ Configuration Examples

### Quick Test (5 classes, small batch)
```python
config = ModelConfig(
    dataset_root='/content/plant_disease_data',
    selected_classes=[
        'Apple___Apple_scab', 'Apple___Black_rot', 
        'Tomato___Leaf_Mold', 'Grape___Leaf_blight', 
        'Corn___Common_rust'
    ],
    batch_size=8,
    num_epochs=5,
    image_size=224
)
```

### Full Training (all classes)
```python
config = ModelConfig(
    dataset_root='/content/plant_disease_data',
    selected_classes=None,  # Use all classes
    batch_size=16,
    num_epochs=20,
    image_size=256,
    use_mixed_precision=True
)
```

### Memory Efficient
```python
config = ModelConfig(
    dataset_root='/content/plant_disease_data',
    batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    gradient_checkpointing=True,
    image_size=224
)
```

## 📈 Monitoring Training

The notebook provides:
- Real-time loss and accuracy plots
- Per-class performance metrics
- Attention weight visualizations
- Learning rate scheduling
- Confusion matrix generation
- Training progress bars

## 💡 Pro Tips

1. **Start Small**: Test with 3-5 classes first
2. **Check GPU**: Ensure GPU is enabled in Colab
3. **Save Regularly**: Download model checkpoints
4. **Monitor Memory**: Watch GPU memory usage
5. **Use Mixed Precision**: Enables larger batch sizes
6. **Validate Structure**: Always run dataset fixer after download

## 🔄 Workflow Summary

1. **Setup Environment**
   ```python
   # Install packages
   !pip install -q transformers torch torchvision
   ```

2. **Upload Files**
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

3. **Download Dataset**
   ```python
   dataset_manager = load_from_google_drive("YOUR_FILE_ID")
   ```

4. **Fix Dataset Structure**
   ```python
   exec(open('fix_plantwild_dataset.py').read())
   fix_plantwild_structure()
   ```

5. **Verify Setup**
   ```python
   dataset_manager.print_dataset_info()
   ```

6. **Configure & Train**
   ```python
   config = ModelConfig(...)
   trainer = CrossAttentionTrainer(config)
   trainer.train()
   ```

## 🎯 Success Indicators

✅ "Dataset Valid: ✅ YES"  
✅ "Number of Classes: > 0"  
✅ "Total Images: > 1000"  
✅ "Prompts File Exists: ✅"  
✅ GPU detected and enabled  
✅ Training loss decreasing  

## 📚 Additional Resources

- [PlantWild Dataset Information](link)
- [Cross-Attention Paper](link)
- [Colab GPU Guide](https://colab.research.google.com/notebooks/gpu.ipynb)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)

## 🆘 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Restart runtime and try again
3. Verify all files are uploaded correctly
4. Check GPU is enabled in runtime settings
5. Review error messages carefully

## 🤝 Contributing

Found a bug or have an improvement? Please open an issue or submit a pull request!

## 📄 License

MIT License - see LICENSE file for details.