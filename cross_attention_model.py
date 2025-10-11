import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-like architectures"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ImageEncoder(nn.Module):
    """Enhanced image encoder with patch-based features"""
    def __init__(self, backbone='resnet50', feature_dim=768, patch_size=7):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            backbone_dim = 2048
            # Remove final layers
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            backbone_dim = 1280
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        
        # Adaptive pooling to get consistent patch grid
        self.adaptive_pool = nn.AdaptiveAvgPool2d((patch_size, patch_size))
        
        # Project to common dimension
        self.projection = nn.Linear(backbone_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Positional encoding for patches
        self.pos_encoding = PositionalEncoding(feature_dim, max_len=patch_size**2)

    def forward(self, x):
        # Extract features: [B, C, H, W]
        features = self.backbone(x)
        
        # Adaptive pooling to fixed patch grid: [B, C, patch_size, patch_size]
        features = self.adaptive_pool(features)
        
        # Reshape to patches: [B, patch_size*patch_size, C]
        B, C, H, W = features.shape
        patches = features.view(B, C, -1).transpose(1, 2)  # [B, num_patches, C]
        
        # Project to common dimension
        patches = self.projection(patches)  # [B, num_patches, feature_dim]
        patches = self.layer_norm(patches)
        patches = self.dropout(patches)
        
        # Add positional encoding
        patches = patches.transpose(0, 1)  # [num_patches, B, feature_dim]
        patches = self.pos_encoding(patches)
        patches = patches.transpose(0, 1)  # [B, num_patches, feature_dim]
        
        return patches

class TextEncoder(nn.Module):
    """Text encoder using BERT-like models"""
    def __init__(self, model_name='bert-base-uncased', feature_dim=768, max_length=128):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        bert_dim = self.bert.config.hidden_size
        
        # Project to common dimension if needed
        if bert_dim != feature_dim:
            self.projection = nn.Linear(bert_dim, feature_dim)
        else:
            self.projection = nn.Identity()
            
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, texts):
        """
        Args:
            texts: List of text strings or pre-tokenized inputs
        """
        if isinstance(texts, list):
            # Tokenize texts
            encoded = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to same device as model
            device = next(self.parameters()).device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
        else:
            input_ids = texts['input_ids']
            attention_mask = texts['attention_mask']
        
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use all token embeddings, not just [CLS]
        embeddings = outputs.last_hidden_state  # [B, seq_len, bert_dim]
        
        # Project to common dimension
        embeddings = self.projection(embeddings)  # [B, seq_len, feature_dim]
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings, attention_mask

class CrossModalAttention(nn.Module):
    """Cross-modal attention between image and text features"""
    def __init__(self, feature_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        # Image to Text attention
        self.img2text_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Text to Image attention  
        self.text2img_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for enhanced representations
        self.img_self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.text_self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward networks
        self.img_ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
        self.text_ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
        # Layer normalization
        self.img_norm1 = nn.LayerNorm(feature_dim)
        self.img_norm2 = nn.LayerNorm(feature_dim)
        self.img_norm3 = nn.LayerNorm(feature_dim)
        
        self.text_norm1 = nn.LayerNorm(feature_dim)
        self.text_norm2 = nn.LayerNorm(feature_dim)
        self.text_norm3 = nn.LayerNorm(feature_dim)

    def forward(self, img_features, text_features, text_mask=None):
        """
        Args:
            img_features: [B, num_patches, feature_dim]
            text_features: [B, seq_len, feature_dim] 
            text_mask: [B, seq_len] attention mask for text
        """
        
        # Self-attention for both modalities
        img_self_attended, _ = self.img_self_attention(
            img_features, img_features, img_features
        )
        img_features = self.img_norm1(img_features + img_self_attended)
        
        if text_mask is not None:
            # Convert mask for attention (True = keep, False = mask)
            text_key_padding_mask = ~text_mask.bool()
        else:
            text_key_padding_mask = None
            
        text_self_attended, _ = self.text_self_attention(
            text_features, text_features, text_features,
            key_padding_mask=text_key_padding_mask
        )
        text_features = self.text_norm1(text_features + text_self_attended)
        
        # Cross-modal attention: Image attends to Text
        img_cross_attended, img_attention_weights = self.img2text_attention(
            query=img_features,
            key=text_features,
            value=text_features,
            key_padding_mask=text_key_padding_mask
        )
        img_features = self.img_norm2(img_features + img_cross_attended)
        
        # Cross-modal attention: Text attends to Image  
        text_cross_attended, text_attention_weights = self.text2img_attention(
            query=text_features,
            key=img_features,
            value=img_features
        )
        text_features = self.text_norm2(text_features + text_cross_attended)
        
        # Feed-forward networks
        img_ffn_out = self.img_ffn(img_features)
        img_features = self.img_norm3(img_features + img_ffn_out)
        
        text_ffn_out = self.text_ffn(text_features)
        text_features = self.text_norm3(text_features + text_ffn_out)
        
        return img_features, text_features, (img_attention_weights, text_attention_weights)

class FusionModule(nn.Module):
    """Advanced fusion module with multiple fusion strategies"""
    def __init__(self, feature_dim=768, fusion_type='adaptive', num_classes=89):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.feature_dim = feature_dim
        
        if fusion_type == 'concat':
            # Simple concatenation
            self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)
            
        elif fusion_type == 'adaptive':
            # Adaptive gating mechanism
            self.img_gate = nn.Linear(feature_dim, feature_dim)
            self.text_gate = nn.Linear(feature_dim, feature_dim)
            self.fusion_gate = nn.Linear(feature_dim * 2, 2)
            
        elif fusion_type == 'bilinear':
            # Bilinear fusion
            self.bilinear = nn.Bilinear(feature_dim, feature_dim, feature_dim)
            
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.combine = nn.Linear(feature_dim * 2, feature_dim)
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, img_features, text_features):
        """
        Args:
            img_features: [B, num_patches, feature_dim]
            text_features: [B, seq_len, feature_dim]
        """
        
        # Global pooling to get single representation per modality
        img_pooled = torch.mean(img_features, dim=1)  # [B, feature_dim]
        text_pooled = torch.mean(text_features, dim=1)  # [B, feature_dim]
        
        if self.fusion_type == 'concat':
            fused = torch.cat([img_pooled, text_pooled], dim=1)
            fused = self.fusion_layer(fused)
            
        elif self.fusion_type == 'adaptive':
            # Compute gates
            img_gated = torch.sigmoid(self.img_gate(img_pooled)) * img_pooled
            text_gated = torch.sigmoid(self.text_gate(text_pooled)) * text_pooled
            
            # Adaptive weighting
            combined = torch.cat([img_gated, text_gated], dim=1)
            weights = torch.softmax(self.fusion_gate(combined), dim=1)
            
            fused = weights[:, 0:1] * img_gated + weights[:, 1:2] * text_gated
            
        elif self.fusion_type == 'bilinear':
            fused = self.bilinear(img_pooled, text_pooled)
            
        elif self.fusion_type == 'attention':
            # Use text as query, image as key/value
            attended_text, _ = self.attention(
                text_pooled.unsqueeze(1), 
                img_pooled.unsqueeze(1), 
                img_pooled.unsqueeze(1)
            )
            attended_text = attended_text.squeeze(1)
            
            combined = torch.cat([img_pooled, attended_text], dim=1)
            fused = self.combine(combined)
        
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused

class CrossAttentionClassifier(nn.Module):
    """Complete Cross-Attention Multimodal Classifier"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Encoders
        self.image_encoder = ImageEncoder(
            backbone=config['image_backbone'],
            feature_dim=config['feature_dim'],
            patch_size=config['patch_size']
        )
        
        self.text_encoder = TextEncoder(
            model_name=config['text_model'],
            feature_dim=config['feature_dim'],
            max_length=config['max_text_length']
        )
        
        # Cross-modal attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttention(
                feature_dim=config['feature_dim'],
                num_heads=config['num_attention_heads'],
                dropout=config['dropout']
            ) for _ in range(config['num_cross_attention_layers'])
        ])
        
        # Fusion module
        self.fusion = FusionModule(
            feature_dim=config['feature_dim'],
            fusion_type=config['fusion_type'],
            num_classes=config['num_classes']
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config['feature_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['num_classes'])
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, images, texts):
        """
        Args:
            images: [B, 3, H, W] image tensors
            texts: List of text strings or tokenized inputs
        """
        
        # Encode modalities
        img_features = self.image_encoder(images)  # [B, num_patches, feature_dim]
        text_features, text_mask = self.text_encoder(texts)  # [B, seq_len, feature_dim]
        
        attention_weights = []
        
        # Apply cross-attention layers
        for cross_attention in self.cross_attention_layers:
            img_features, text_features, att_weights = cross_attention(
                img_features, text_features, text_mask
            )
            attention_weights.append(att_weights)
        
        # Fuse modalities
        fused_features = self.fusion(img_features, text_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'img_features': img_features,
            'text_features': text_features,
            'fused_features': fused_features,
            'attention_weights': attention_weights
        }