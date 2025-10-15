# Cross-Attention Training for PlantWild v2
# Complete training setup in one cell

# Step 1: Install required packages
!pip install -q transformers torch torchvision torchaudio
!pip install -q gdown pyyaml matplotlib seaborn plotly
!pip install -q scikit-learn opencv-python albumentations
!pip install -q dataclasses-json tqdm

# Step 2: Clone the repository (if not already done)
import os
if not os.path.exists('/content/CrossAttention'):
    !git clone https://github.com/ahmed-a133b/CrossAttention.git

# Step 3: Change to repository directory
%cd /content/CrossAttention

# Step 4: Setup dataset and training
from simple_training import quick_train

# Replace 'YOUR_DRIVE_FILE_ID' with your actual Google Drive file ID
DRIVE_FILE_ID = 'YOUR_DRIVE_FILE_ID'  # ⚠️ CHANGE THIS!

# Start training (adjust epochs and batch_size as needed)
trainer, model = quick_train(
    drive_file_id=DRIVE_FILE_ID,
    epochs=20,           # Number of training epochs
    batch_size=16        # Batch size (reduce if you get memory errors)
)

print("🎉 Training completed!")