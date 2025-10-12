# Cross-Attention Fusion for Multimodal Plant Disease Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology Overview](#methodology-overview)
3. [Architecture Components](#architecture-components)
4. [Cross-Attention Mechanism](#cross-attention-mechanism)
5. [Fusion Strategies](#fusion-strategies)
6. [Training Process](#training-process)
7. [Advantages over MVPDR](#advantages-over-mvpdr)
8. [Implementation Details](#implementation-details)
9. [Mathematical Formulation](#mathematical-formulation)
10. [Experimental Configuration](#experimental-configuration)

---

## Introduction

Cross-Attention Fusion is an advanced multimodal learning approach that enables dynamic interaction between visual (image) and textual features for plant disease classification. Unlike traditional fusion methods that simply concatenate or average features, cross-attention allows the model to selectively focus on relevant parts of one modality based on information from the other modality.

### Key Innovation
The method addresses the limitation of static fusion approaches by introducing **learnable attention mechanisms** that can adapt to different disease patterns and their textual descriptions, creating a more sophisticated understanding of multimodal relationships.

---

## Methodology Overview

### Problem Statement
Given an image `I` and its corresponding text descriptions `T`, the goal is to classify the plant disease by effectively combining visual and textual information through dynamic attention mechanisms.

### Core Concept
The Cross-Attention method works by:
1. **Encoding** both image and text into feature representations
2. **Computing attention** between visual and textual features
3. **Fusing** the attended features using learnable fusion strategies
4. **Classifying** the disease based on the fused representation

### Workflow Pipeline
```
Image Input → Image Encoder → Visual Features (V)
                                     ↓
Text Input → Text Encoder → Textual Features (T)
                                     ↓
            Cross-Attention Layers (V ↔ T)
                                     ↓
            Fusion Module → Fused Features (F)
                                     ↓
            Classifier → Disease Prediction
```

---

## Architecture Components

### 1. Image Encoder
- **Backbone Options**: ResNet-50, ResNet-101, EfficientNet-B0/B3/B5
- **Feature Extraction**: Convolutional layers extract spatial features
- **Output**: Visual feature tensor `V ∈ ℝ^(H×W×d_v)`
  - `H×W`: Spatial dimensions (e.g., 14×14 for ResNet-50)
  - `d_v`: Visual feature dimension (e.g., 2048 for ResNet-50)

```python
# Example: ResNet-50 feature extraction
visual_features = resnet50.features(image)  # Shape: [B, 2048, 14, 14]
visual_features = visual_features.flatten(2).transpose(1, 2)  # [B, 196, 2048]
```

### 2. Text Encoder
- **Model**: BERT-based transformers (bert-base-uncased, distilbert, etc.)
- **Input Processing**: Tokenized text with special tokens [CLS], [SEP]
- **Output**: Textual feature tensor `T ∈ ℝ^(L×d_t)`
  - `L`: Sequence length (max 512 tokens)
  - `d_t`: Text feature dimension (768 for BERT-base)

```python
# Example: BERT text encoding
text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
text_features = bert_model(**text_tokens).last_hidden_state  # [B, L, 768]
```

### 3. Feature Projection
- **Purpose**: Align visual and textual feature dimensions
- **Implementation**: Linear projection layers
- **Output**: Aligned features `V' ∈ ℝ^(N_v×d)` and `T' ∈ ℝ^(N_t×d)`

```python
# Feature projection to common dimension
visual_projected = linear_v(visual_features)    # [B, 196, d]
textual_projected = linear_t(textual_features)  # [B, L, d]
```

---

## Cross-Attention Mechanism

### 1. Attention Computation
Cross-attention enables each modality to attend to relevant parts of the other modality:

#### Visual-to-Text Attention
- **Query**: Visual features `V'`
- **Key & Value**: Textual features `T'`
- **Purpose**: Find which text tokens are relevant for each visual region

#### Text-to-Visual Attention
- **Query**: Textual features `T'`
- **Key & Value**: Visual features `V'`
- **Purpose**: Find which visual regions are relevant for each text token

### 2. Multi-Head Attention
```python
def cross_attention(query, key, value, num_heads=12):
    """
    Multi-head cross-attention mechanism
    
    Args:
        query: [B, N_q, d] - Query features
        key:   [B, N_k, d] - Key features  
        value: [B, N_v, d] - Value features
        num_heads: Number of attention heads
    
    Returns:
        attended_features: [B, N_q, d] - Attended query features
        attention_weights: [B, num_heads, N_q, N_k] - Attention maps
    """
    
    # Split into multiple heads
    Q = split_heads(query, num_heads)    # [B, num_heads, N_q, d_head]
    K = split_heads(key, num_heads)      # [B, num_heads, N_k, d_head] 
    V = split_heads(value, num_heads)    # [B, num_heads, N_v, d_head]
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_head)  # [B, h, N_q, N_k]
    attention_weights = softmax(scores, dim=-1)
    
    # Apply attention to values
    attended = torch.matmul(attention_weights, V)  # [B, h, N_q, d_head]
    
    # Concatenate heads
    attended_features = concat_heads(attended)  # [B, N_q, d]
    
    return attended_features, attention_weights
```

### 3. Bidirectional Cross-Attention Layer
```python
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=768, num_heads=12, dropout=0.1):
        super().__init__()
        
        # Visual-to-Text attention
        self.v2t_attention = MultiHeadAttention(d_model, num_heads)
        self.v2t_norm = LayerNorm(d_model)
        
        # Text-to-Visual attention  
        self.t2v_attention = MultiHeadAttention(d_model, num_heads)
        self.t2v_norm = LayerNorm(d_model)
        
        # Feed-forward networks
        self.v_ffn = FeedForward(d_model, dropout)
        self.t_ffn = FeedForward(d_model, dropout)
        
    def forward(self, visual_features, textual_features):
        # Visual attending to text
        v_attended, v2t_weights = self.v2t_attention(
            query=visual_features,
            key=textual_features, 
            value=textual_features
        )
        visual_features = self.v2t_norm(visual_features + v_attended)
        visual_features = visual_features + self.v_ffn(visual_features)
        
        # Text attending to visual
        t_attended, t2v_weights = self.t2v_attention(
            query=textual_features,
            key=visual_features,
            value=visual_features  
        )
        textual_features = self.t2v_norm(textual_features + t_attended)
        textual_features = textual_features + self.t_ffn(textual_features)
        
        return visual_features, textual_features, (v2t_weights, t2v_weights)
```

### 4. Positional Encoding
To preserve spatial relationships in visual features:

```python
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height=14, width=14):
        super().__init__()
        
        # Create 2D positional embeddings
        pe = torch.zeros(height * width, d_model)
        
        for h in range(height):
            for w in range(width):
                pos = h * width + w
                
                # Sine-cosine encoding for height
                for i in range(0, d_model//4, 2):
                    pe[pos, i] = math.sin(h / (10000 ** (i / d_model)))
                    pe[pos, i+1] = math.cos(h / (10000 ** (i / d_model)))
                
                # Sine-cosine encoding for width
                for i in range(d_model//4, d_model//2, 2):
                    pe[pos, i] = math.sin(w / (10000 ** ((i-d_model//4) / d_model)))
                    pe[pos, i+1] = math.cos(w / (10000 ** ((i-d_model//4) / d_model)))
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

## Fusion Strategies

### 1. Concatenation Fusion (Baseline)
```python
def concat_fusion(visual_features, textual_features):
    """Simple concatenation of global features"""
    v_global = visual_features.mean(dim=1)      # [B, d]
    t_global = textual_features.mean(dim=1)     # [B, d]
    fused = torch.cat([v_global, t_global], dim=1)  # [B, 2d]
    return fused
```

### 2. Adaptive Fusion (Learnable Weights)
```python
class AdaptiveFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.visual_gate = nn.Linear(d_model, 1)
        self.textual_gate = nn.Linear(d_model, 1) 
        self.fusion_layer = nn.Linear(d_model, d_model)
        
    def forward(self, visual_features, textual_features):
        # Global pooling
        v_global = visual_features.mean(dim=1)      # [B, d]
        t_global = textual_features.mean(dim=1)     # [B, d]
        
        # Compute adaptive weights
        v_weight = torch.sigmoid(self.visual_gate(v_global))    # [B, 1]
        t_weight = torch.sigmoid(self.textual_gate(t_global))   # [B, 1]
        
        # Normalize weights
        total_weight = v_weight + t_weight
        v_weight = v_weight / total_weight
        t_weight = t_weight / total_weight
        
        # Weighted fusion
        fused = v_weight * v_global + t_weight * t_global  # [B, d]
        fused = self.fusion_layer(fused)
        
        return fused
```

### 3. Bilinear Fusion (Feature Interaction)
```python
class BilinearFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bilinear = nn.Bilinear(d_model, d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, visual_features, textual_features):
        v_global = visual_features.mean(dim=1)      # [B, d]
        t_global = textual_features.mean(dim=1)     # [B, d]
        
        # Bilinear interaction
        interaction = self.bilinear(v_global, t_global)  # [B, d]
        
        # Residual connection
        fused = v_global + t_global + interaction   # [B, d]
        fused = self.layer_norm(fused)
        
        return fused
```

### 4. Attention-Based Fusion
```python
class AttentionFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cross_attention = MultiHeadAttention(d_model, num_heads=8)
        self.fusion_attention = nn.MultiheadAttention(d_model, num_heads=8)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, visual_features, textual_features):
        # Cross-modal attention
        v_attended, _ = self.cross_attention(
            query=visual_features,
            key=textual_features,
            value=textual_features
        )
        
        t_attended, _ = self.cross_attention(
            query=textual_features, 
            key=visual_features,
            value=visual_features
        )
        
        # Concatenate attended features
        all_features = torch.cat([
            v_attended.mean(dim=1, keepdim=True),    # [B, 1, d]
            t_attended.mean(dim=1, keepdim=True)     # [B, 1, d]
        ], dim=1)  # [B, 2, d]
        
        # Self-attention for fusion
        fused, _ = self.fusion_attention(
            all_features, all_features, all_features
        )  # [B, 2, d]
        
        # Final pooling
        fused = fused.mean(dim=1)  # [B, d]
        fused = self.layer_norm(fused)
        
        return fused
```

---

## Training Process

### 1. Loss Functions

#### Cross-Entropy Loss (Standard)
```python
def cross_entropy_loss(predictions, labels):
    return F.cross_entropy(predictions, labels)
```

#### Focal Loss (For Imbalanced Classes)
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions, labels):
        ce_loss = F.cross_entropy(predictions, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

#### Label Smoothing Loss
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, predictions, labels):
        log_probs = F.log_softmax(predictions, dim=-1)
        smooth_labels = (1 - self.smoothing) * F.one_hot(labels, predictions.size(-1)).float()
        smooth_labels += self.smoothing / predictions.size(-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        return loss
```

### 2. Training Algorithm

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, texts, labels) in enumerate(dataloader):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get predictions and attention weights
        outputs, attention_weights = model(images, texts)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(dataloader), 100. * correct / total
```

### 3. Optimization Strategy

```python
# Optimizer configuration
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs,
    eta_min=1e-6
)

# Mixed precision training (optional)
scaler = torch.cuda.amp.GradScaler()
```

---

## Advantages over MVPDR

### 1. Dynamic vs Static Fusion
| Aspect | MVPDR | Cross-Attention |
|--------|-------|-----------------|
| **Fusion Method** | Static prototype averaging | Dynamic attention-based |
| **Text-Image Interaction** | None (separate prototypes) | Direct cross-modal attention |
| **Adaptability** | Fixed weights | Learnable attention patterns |
| **Context Sensitivity** | Low | High |

### 2. Feature Interaction
- **MVPDR**: Creates separate visual and textual prototypes through K-means clustering, then combines with fixed weights
- **Cross-Attention**: Enables direct interaction between image regions and text tokens through learned attention

### 3. Granular Understanding
- **MVPDR**: Works at global feature level (image-level and class-level text)
- **Cross-Attention**: Works at token/region level (spatial regions ↔ text tokens)

### 4. Attention Visualization
Cross-attention provides interpretable attention maps showing:
- Which image regions are important for specific text descriptions
- Which text tokens are relevant for different visual patterns
- How the model makes multimodal decisions

---

## Implementation Details

### 1. Model Architecture
```python
class CrossAttentionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Encoders
        self.image_encoder = ImageEncoder(config.image_backbone)
        self.text_encoder = TextEncoder(config.text_encoder)
        
        # Feature projection
        self.visual_projection = nn.Linear(
            self.image_encoder.feature_dim, 
            config.feature_dim
        )
        self.textual_projection = nn.Linear(
            self.text_encoder.feature_dim,
            config.feature_dim
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(config.feature_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=config.feature_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout_rate
            ) for _ in range(config.num_cross_attention_layers)
        ])
        
        # Fusion module
        if config.fusion_type == 'adaptive':
            self.fusion = AdaptiveFusion(config.feature_dim)
        elif config.fusion_type == 'bilinear':
            self.fusion = BilinearFusion(config.feature_dim)
        elif config.fusion_type == 'attention':
            self.fusion = AttentionFusion(config.feature_dim)
        else:  # concat
            self.fusion = ConcatFusion(config.feature_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.feature_dim // 2, config.num_classes)
        )
    
    def forward(self, images, texts):
        # Encode features
        visual_features = self.image_encoder(images)        # [B, H*W, d_v]
        textual_features = self.text_encoder(texts)         # [B, L, d_t]
        
        # Project to common dimension
        visual_features = self.visual_projection(visual_features)    # [B, H*W, d]
        textual_features = self.textual_projection(textual_features) # [B, L, d]
        
        # Add positional encoding to visual features
        visual_features = self.pos_encoding(visual_features)
        
        # Cross-attention layers
        attention_weights = []
        for layer in self.cross_attention_layers:
            visual_features, textual_features, attn_weights = layer(
                visual_features, textual_features
            )
            attention_weights.append(attn_weights)
        
        # Fusion
        fused_features = self.fusion(visual_features, textual_features)  # [B, d]
        
        # Classification
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        if self.training:
            return logits
        else:
            return logits, attention_weights
```

### 2. Configuration Parameters
```python
@dataclass
class CrossAttentionConfig:
    # Architecture
    image_backbone: str = "resnet50"          # resnet50, efficientnet-b0, etc.
    text_encoder: str = "bert-base-uncased"   # BERT model
    feature_dim: int = 768                    # Common feature dimension
    num_cross_attention_layers: int = 4       # Number of cross-attention layers
    num_attention_heads: int = 12             # Multi-head attention heads
    dropout_rate: float = 0.1                 # Dropout probability
    fusion_type: str = "adaptive"             # Fusion strategy
    
    # Training
    batch_size: int = 32                      # Batch size
    learning_rate: float = 1e-4               # Learning rate
    num_epochs: int = 50                      # Training epochs
    weight_decay: float = 1e-2                # L2 regularization
    
    # Dataset
    num_classes: int = 0                      # Number of disease classes
    selected_classes: List[str] = None        # Subset of classes
    max_text_length: int = 512                # Maximum text tokens
    
    # Optimization
    mixed_precision: bool = True              # Use mixed precision
    gradient_clip_norm: float = 1.0           # Gradient clipping
    scheduler_type: str = "cosine"            # LR scheduler
```

---

## Mathematical Formulation

### 1. Attention Mechanism
Given visual features `V ∈ ℝ^(N_v × d)` and textual features `T ∈ ℝ^(N_t × d)`:

**Visual-to-Text Attention:**
```
Q_v = V W_q^v,  K_t = T W_k^t,  V_t = T W_v^t

A_v→t = softmax(Q_v K_t^T / √d_k)

V' = V + A_v→t V_t
```

**Text-to-Visual Attention:**
```
Q_t = T W_q^t,  K_v = V' W_k^v,  V_v = V' W_v^v

A_t→v = softmax(Q_t K_v^T / √d_k)

T' = T + A_t→v V_v
```

### 2. Multi-Head Attention
For `h` attention heads:
```
head_i = Attention(Q W_q^i, K W_k^i, V W_v^i)

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_o
```

### 3. Fusion Functions
**Adaptive Fusion:**
```
α_v = σ(W_v^g · pool(V'))
α_t = σ(W_t^g · pool(T'))
α_v, α_t = normalize(α_v, α_t)

F = α_v · pool(V') + α_t · pool(T')
```

**Bilinear Fusion:**
```
F = pool(V') + pool(T') + W_b(pool(V') ⊗ pool(T'))
```

Where `⊗` denotes element-wise multiplication.

---

## Experimental Configuration

### 1. Dataset Splits
- **Training**: 70% of images
- **Validation**: 15% of images  
- **Testing**: 15% of images

### 2. Hyperparameter Ranges
```python
hyperparameter_grid = {
    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    'batch_size': [16, 32, 64],
    'feature_dim': [256, 512, 768, 1024],
    'num_attention_heads': [4, 8, 12, 16],
    'num_cross_attention_layers': [2, 4, 6],
    'dropout_rate': [0.1, 0.2, 0.3],
    'weight_decay': [1e-4, 1e-3, 1e-2],
    'fusion_type': ['concat', 'adaptive', 'bilinear', 'attention']
}
```

### 3. Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged precision  
- **Recall**: Per-class and macro-averaged recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results
- **Attention Visualization**: Qualitative analysis of attention patterns

### 4. Comparison Baselines
1. **Image-only**: ResNet-50 classifier
2. **Text-only**: BERT classifier  
3. **Simple Concatenation**: Baseline multimodal fusion
4. **MVPDR**: Original prototype-based method
5. **Cross-Attention Variants**: Different fusion strategies

---

## Conclusion

The Cross-Attention Fusion method represents a significant advancement over traditional multimodal fusion approaches by enabling dynamic, learnable interactions between visual and textual modalities. Through its sophisticated attention mechanisms and multiple fusion strategies, it can adaptively focus on relevant cross-modal relationships, leading to more accurate and interpretable plant disease classification.

### Key Benefits:
1. **Dynamic Interaction**: Enables adaptive cross-modal attention
2. **Interpretability**: Provides attention visualization capabilities  
3. **Flexibility**: Supports multiple fusion strategies
4. **Scalability**: Can handle various numbers of disease classes
5. **Performance**: Achieves superior accuracy compared to static fusion methods

This methodology forms the foundation for the SecuriLeaf project, providing a robust framework for multimodal plant disease classification with enhanced interpretability and performance.