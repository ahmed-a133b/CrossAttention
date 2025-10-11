import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from cross_attention_model import CrossAttentionClassifier
from dataset_loader import create_data_loaders, create_data_transforms
from config_manager import ModelConfig

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return focal_loss.mean()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = nn.functional.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class MetricsCalculator:
    """Calculate and track training metrics"""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.all_predictions = []
        self.all_labels = []
        self.all_losses = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, loss: float):
        """Update metrics with batch results"""
        pred_classes = torch.argmax(predictions, dim=1)
        
        self.all_predictions.extend(pred_classes.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        self.all_losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics"""
        if not self.all_predictions:
            return {}
        
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        # Overall metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Average loss
        avg_loss = np.mean(self.all_losses) if self.all_losses else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'avg_loss': avg_loss
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'{class_name}_precision'] = precision_per_class[i]
                metrics[f'{class_name}_recall'] = recall_per_class[i]
                metrics[f'{class_name}_f1'] = f1_per_class[i]
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        if not self.all_predictions:
            return np.zeros((self.num_classes, self.num_classes))
        
        return confusion_matrix(
            self.all_labels, 
            self.all_predictions, 
            labels=list(range(self.num_classes))
        )

class Trainer:
    """Training manager for Cross-Attention model"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize model
        self.model = CrossAttentionClassifier(config.__dict__).to(self.device)
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup data loaders
        self.train_loader, self.val_loader, self.test_loader = self._setup_data_loaders()
        
        # Metrics tracking
        self.train_metrics = MetricsCalculator(
            config.num_classes, 
            self.train_loader.dataset.class_names
        )
        self.val_metrics = MetricsCalculator(
            config.num_classes,
            self.val_loader.dataset.class_names
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        
        print(f"Training setup complete. Using device: {self.device}")
        print(f"Model parameters: {self._count_parameters():,}")
    
    def _setup_loss_function(self):
        """Setup loss function based on config"""
        if self.config.loss_function == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.config.loss_function == 'focal':
            return FocalLoss(alpha=self.config.focal_alpha, gamma=self.config.focal_gamma)
        elif self.config.loss_function == 'label_smoothing':
            return LabelSmoothingCrossEntropy(smoothing=self.config.label_smoothing)
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
    
    def _setup_optimizer(self):
        """Setup optimizer based on config"""
        if self.config.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def _setup_data_loaders(self):
        """Setup data loaders"""
        train_transform, val_transform = create_data_transforms(
            augmentation_strength=self.config.augmentation_strength
        )
        
        return create_data_loaders(self.config, train_transform, val_transform)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        total_batches = len(self.train_loader)
        
        for batch_idx, (images, texts, labels, class_names) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, texts)
            logits = outputs['logits']
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(logits, labels, loss.item())
            
            # Log progress
            if batch_idx % self.config.log_frequency == 0:
                print(f'Epoch {epoch+1}/{self.config.num_epochs} '
                      f'[{batch_idx}/{total_batches}] '
                      f'Loss: {loss.item():.4f} '
                      f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        # Update scheduler
        if self.scheduler and self.config.scheduler != 'plateau':
            self.scheduler.step()
        
        return self.train_metrics.compute_metrics()
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for images, texts, labels, class_names in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, texts)
                logits = outputs['logits']
                
                # Compute loss
                loss = self.criterion(logits, labels)
                
                # Update metrics
                self.val_metrics.update(logits, labels, loss.item())
        
        metrics = self.val_metrics.compute_metrics()
        
        # Update scheduler if using plateau
        if self.scheduler and self.config.scheduler == 'plateau':
            self.scheduler.step(metrics['accuracy'])
        
        return metrics
    
    def test(self) -> Dict[str, float]:
        """Test the model"""
        self.model.eval()
        test_metrics = MetricsCalculator(
            self.config.num_classes,
            self.test_loader.dataset.class_names
        )
        
        with torch.no_grad():
            for images, texts, labels, class_names in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, texts)
                logits = outputs['logits']
                
                # Compute loss
                loss = self.criterion(logits, labels)
                
                # Update metrics
                test_metrics.update(logits, labels, loss.item())
        
        return test_metrics.compute_metrics(), test_metrics.get_confusion_matrix()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'config': self.config.__dict__,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with val_acc: {self.best_val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', color='green')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot (if available)
        if hasattr(self.optimizer, 'param_groups'):
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lrs, label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to {plot_path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        cm_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {cm_path}")
    
    def train(self):
        """Complete training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.eval_frequency == 0 or epoch == self.config.num_epochs - 1:
                val_metrics = self.validate(epoch)
                
                # Update history
                self.history['train_loss'].append(train_metrics['avg_loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_loss'].append(val_metrics['avg_loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['macro_f1'])
                
                # Check for best model
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_val_f1 = val_metrics['macro_f1']
                
                # Save checkpoint
                if self.config.save_best_only and is_best:
                    self.save_checkpoint(epoch, is_best=True)
                elif not self.config.save_best_only:
                    self.save_checkpoint(epoch, is_best=is_best)
                
                epoch_time = time.time() - epoch_start_time
                
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} Summary:")
                print(f"Train - Loss: {train_metrics['avg_loss']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val   - Loss: {val_metrics['avg_loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"F1: {val_metrics['macro_f1']:.4f}")
                print(f"Time: {epoch_time:.2f}s")
                if is_best:
                    print("*** New best model! ***")
                print("-" * 50)
        
        # Final testing
        print("\nTesting best model...")
        best_checkpoint_path = os.path.join(self.config.checkpoint_dir, 'best_checkpoint.pth')
        if os.path.exists(best_checkpoint_path):
            self.load_checkpoint(best_checkpoint_path)
        
        test_metrics, confusion_matrix = self.test()
        
        total_time = time.time() - start_time
        
        # Final results
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"\nFinal test results:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['macro_precision']:.4f}")
        print(f"Test Recall: {test_metrics['macro_recall']:.4f}")
        print(f"Test F1: {test_metrics['macro_f1']:.4f}")
        
        # Save final results
        results = {
            'config': self.config.__dict__,
            'training_time_hours': total_time / 3600,
            'best_val_accuracy': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'test_metrics': test_metrics,
            'history': self.history
        }
        
        results_path = os.path.join(self.config.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate plots
        self.plot_training_history()
        self.plot_confusion_matrix(confusion_matrix, self.test_loader.dataset.class_names)
        
        print(f"\nResults saved to {self.config.output_dir}")
        
        return test_metrics

if __name__ == "__main__":
    # Example training run
    from config_manager import ModelConfig
    
    # Create a test configuration
    config = ModelConfig(
        # Dataset
        dataset_root='./data',
        selected_classes=['apple black rot', 'apple leaf', 'apple scab', 'tomato leaf'],
        num_classes=4,
        
        # Model
        feature_dim=768,
        num_attention_heads=12,
        num_cross_attention_layers=2,
        fusion_type='adaptive',
        
        # Training
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=20,
        
        # Output
        output_dir='./outputs/test_run',
        checkpoint_dir='./checkpoints/test_run'
    )
    
    # Create trainer and start training
    trainer = Trainer(config)
    test_metrics = trainer.train()
    
    print("Training completed successfully!")