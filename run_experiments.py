#!/usr/bin/env python3
"""
Main script for running Cross-Attention Plant Disease Classification experiments
"""

import argparse
import os
import sys
import json
import yaml
from datetime import datetime
from typing import List, Dict, Any

import torch
import numpy as np
import random

from config_manager import ModelConfig, ConfigManager, ClassSubsetManager
from trainer import Trainer

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_experiment_name(config: ModelConfig) -> str:
    """Create a unique experiment name based on configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Key configuration elements for naming
    backbone = config.image_backbone.replace('-', '')
    fusion = config.fusion_type
    classes = f"{config.num_classes}cls"
    layers = f"{config.num_cross_attention_layers}layers"
    dim = f"{config.feature_dim}dim"
    
    # If using subset, add subset info
    subset_info = ""
    if config.selected_classes:
        if len(config.selected_classes) <= 5:
            # Show first few class names if small subset
            first_classes = "_".join(config.selected_classes[:2])
            subset_info = f"_{first_classes}"
        else:
            # Just show count for large subsets
            subset_info = f"_{len(config.selected_classes)}subset"
    
    exp_name = f"{backbone}_{fusion}_{classes}_{layers}_{dim}{subset_info}_{timestamp}"
    return exp_name

def run_single_experiment(config_path: str, experiment_name: str = None) -> Dict[str, Any]:
    """Run a single experiment with given configuration"""
    
    print(f"Loading configuration from {config_path}")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # Create experiment name if not provided
    if experiment_name is None:
        experiment_name = create_experiment_name(config)
    
    # Update output directories with experiment name
    config.output_dir = os.path.join(config.output_dir, experiment_name)
    config.checkpoint_dir = os.path.join(config.checkpoint_dir, experiment_name)
    
    print(f"Running experiment: {experiment_name}")
    print(f"Output directory: {config.output_dir}")
    
    # Set random seed
    set_seed(42)
    
    # Create trainer and run training
    try:
        trainer = Trainer(config)
        test_metrics = trainer.train()
        
        # Return results
        return {
            'experiment_name': experiment_name,
            'config_path': config_path,
            'config': config.__dict__,
            'test_metrics': test_metrics,
            'status': 'completed',
            'error': None
        }
        
    except Exception as e:
        print(f"Error in experiment {experiment_name}: {str(e)}")
        return {
            'experiment_name': experiment_name,
            'config_path': config_path,
            'config': config.__dict__,
            'test_metrics': {},
            'status': 'failed',
            'error': str(e)
        }

def run_hyperparameter_search(search_type: str = 'random', num_configs: int = 10) -> List[Dict[str, Any]]:
    """Run hyperparameter search experiments"""
    
    print(f"Running {search_type} hyperparameter search with {num_configs} configurations")
    
    config_manager = ConfigManager()
    
    if search_type == 'random':
        configurations = config_manager.create_random_search_configs(num_configs)
    elif search_type == 'grid':
        configurations = config_manager.create_hyperparameter_grid()[:num_configs]
    else:
        raise ValueError(f"Unknown search type: {search_type}")
    
    results = []
    
    for i, config_dict in enumerate(configurations):
        print(f"\n{'-'*60}")
        print(f"Hyperparameter search: {i+1}/{len(configurations)}")
        print(f"{'-'*60}")
        
        # Create config object
        base_config = ModelConfig()
        
        # Update with hyperparameter values
        for key, value in config_dict.items():
            setattr(base_config, key, value)
        
        # Create temporary config file
        temp_config_path = f'temp_config_{i}.yaml'
        config_manager.save_config(base_config, temp_config_path)
        
        # Run experiment
        try:
            result = run_single_experiment(temp_config_path, f'hp_search_{search_type}_{i:03d}')
            results.append(result)
        except Exception as e:
            print(f"Failed hyperparameter config {i}: {e}")
            results.append({
                'experiment_name': f'hp_search_{search_type}_{i:03d}',
                'config': config_dict,
                'status': 'failed',
                'error': str(e)
            })
        
        # Clean up temp file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    # Save hyperparameter search results
    search_results = {
        'search_type': search_type,
        'num_configurations': len(configurations),
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    results_file = f'hyperparameter_search_{search_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(search_results, f, indent=2, default=str)
    
    print(f"\nHyperparameter search results saved to {results_file}")
    
    # Print summary
    successful_runs = [r for r in results if r['status'] == 'completed']
    if successful_runs:
        best_result = max(successful_runs, key=lambda x: x['test_metrics'].get('accuracy', 0))
        print(f"\nBest configuration achieved {best_result['test_metrics']['accuracy']:.4f} test accuracy")
        print(f"Best experiment: {best_result['experiment_name']}")
    
    return results

def run_class_subset_experiments() -> List[Dict[str, Any]]:
    """Run experiments with different class subsets"""
    
    print("Running class subset experiments")
    
    class_manager = ClassSubsetManager()
    subsets = class_manager.get_predefined_subsets()
    
    results = []
    
    for subset_name, classes in subsets.items():
        if len(classes) == 0:
            continue
            
        print(f"\n{'-'*60}")
        print(f"Running experiment with {subset_name}: {len(classes)} classes")
        print(f"{'-'*60}")
        
        # Create configuration for this subset
        config = ModelConfig(
            selected_classes=classes,
            num_classes=len(classes),
            batch_size=32,
            learning_rate=1e-4,
            num_epochs=30,
            fusion_type='adaptive',
            feature_dim=768,
            num_attention_heads=12
        )
        
        # Save temporary config
        temp_config_path = f'temp_subset_{subset_name}.yaml'
        config_manager = ConfigManager()
        config_manager.save_config(config, temp_config_path)
        
        # Run experiment
        try:
            result = run_single_experiment(temp_config_path, f'subset_{subset_name}')
            results.append(result)
        except Exception as e:
            print(f"Failed subset experiment {subset_name}: {e}")
            results.append({
                'experiment_name': f'subset_{subset_name}',
                'subset_name': subset_name,
                'num_classes': len(classes),
                'status': 'failed',
                'error': str(e)
            })
        
        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    # Save results
    subset_results = {
        'experiment_type': 'class_subsets',
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    results_file = f'subset_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(subset_results, f, indent=2, default=str)
    
    print(f"\nSubset experiment results saved to {results_file}")
    
    return results

def compare_fusion_methods() -> List[Dict[str, Any]]:
    """Compare different fusion methods"""
    
    print("Comparing fusion methods")
    
    fusion_types = ['concat', 'adaptive', 'bilinear', 'attention']
    results = []
    
    # Use a medium-sized subset for comparison
    class_manager = ClassSubsetManager()
    test_classes = class_manager.create_subset_by_category(['apple', 'tomato'])[:10]
    
    for fusion_type in fusion_types:
        print(f"\n{'-'*60}")
        print(f"Testing fusion method: {fusion_type}")
        print(f"{'-'*60}")
        
        # Create configuration
        config = ModelConfig(
            selected_classes=test_classes,
            num_classes=len(test_classes),
            fusion_type=fusion_type,
            batch_size=32,
            learning_rate=1e-4,
            num_epochs=25,
            feature_dim=768,
            num_attention_heads=12
        )
        
        # Save temporary config
        temp_config_path = f'temp_fusion_{fusion_type}.yaml'
        config_manager = ConfigManager()
        config_manager.save_config(config, temp_config_path)
        
        # Run experiment
        try:
            result = run_single_experiment(temp_config_path, f'fusion_{fusion_type}')
            results.append(result)
        except Exception as e:
            print(f"Failed fusion experiment {fusion_type}: {e}")
            results.append({
                'experiment_name': f'fusion_{fusion_type}',
                'fusion_type': fusion_type,
                'status': 'failed',
                'error': str(e)
            })
        
        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    # Save results
    fusion_results = {
        'experiment_type': 'fusion_comparison',
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    results_file = f'fusion_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(fusion_results, f, indent=2, default=str)
    
    print(f"\nFusion comparison results saved to {results_file}")
    
    # Print comparison
    successful_runs = [r for r in results if r['status'] == 'completed']
    if successful_runs:
        print("\nFusion Method Comparison:")
        print("=" * 50)
        for result in successful_runs:
            fusion = result['config']['fusion_type']
            acc = result['test_metrics']['accuracy']
            f1 = result['test_metrics']['macro_f1']
            print(f"{fusion:10s}: Acc={acc:.4f}, F1={f1:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Cross-Attention Plant Disease Classification')
    
    parser.add_argument('--mode', type=str, choices=[
        'single', 'hp_search', 'subsets', 'fusion_comparison', 'all'
    ], default='single', help='Experiment mode')
    
    parser.add_argument('--config', type=str, help='Path to configuration file (for single mode)')
    parser.add_argument('--name', type=str, help='Experiment name (optional)')
    
    # Hyperparameter search options
    parser.add_argument('--search_type', type=str, choices=['random', 'grid'], 
                       default='random', help='Hyperparameter search type')
    parser.add_argument('--num_configs', type=int, default=10, 
                       help='Number of configurations for hyperparameter search')
    
    # Quick test options
    parser.add_argument('--quick_test', action='store_true', 
                       help='Run quick test with small subset and few epochs')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        print("Running quick test...")
        config = ModelConfig(
            selected_classes=['apple black rot', 'apple leaf', 'tomato leaf'],
            num_classes=3,
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=3,
            eval_frequency=1,
            output_dir='./outputs/quick_test',
            checkpoint_dir='./checkpoints/quick_test'
        )
        
        config_manager = ConfigManager()
        temp_config = 'quick_test_config.yaml'
        config_manager.save_config(config, temp_config)
        
        try:
            result = run_single_experiment(temp_config, 'quick_test')
            print(f"Quick test completed: {result['status']}")
            if result['status'] == 'completed':
                print(f"Test accuracy: {result['test_metrics']['accuracy']:.4f}")
        finally:
            if os.path.exists(temp_config):
                os.remove(temp_config)
        
        return
    
    # Main experiment modes
    if args.mode == 'single':
        if not args.config:
            print("Error: --config required for single mode")
            sys.exit(1)
        
        result = run_single_experiment(args.config, args.name)
        print(f"\nExperiment completed: {result['status']}")
        
        if result['status'] == 'completed':
            metrics = result['test_metrics']
            print(f"Final Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['macro_precision']:.4f}")
            print(f"  Recall: {metrics['macro_recall']:.4f}")
            print(f"  F1 Score: {metrics['macro_f1']:.4f}")
    
    elif args.mode == 'hp_search':
        results = run_hyperparameter_search(args.search_type, args.num_configs)
        print(f"\nHyperparameter search completed. {len(results)} configurations tested.")
    
    elif args.mode == 'subsets':
        results = run_class_subset_experiments()
        print(f"\nClass subset experiments completed. {len(results)} subsets tested.")
    
    elif args.mode == 'fusion_comparison':
        results = compare_fusion_methods()
        print(f"\nFusion method comparison completed. {len(results)} methods tested.")
    
    elif args.mode == 'all':
        print("Running all experiment types...")
        
        # Run fusion comparison
        print("\n1. Fusion Method Comparison")
        fusion_results = compare_fusion_methods()
        
        # Run subset experiments (limited)
        print("\n2. Class Subset Experiments (selected subsets)")
        class_manager = ClassSubsetManager()
        selected_subsets = {
            'apple_diseases': class_manager.create_subset_by_category(['apple']),
            'small_subset_10': class_manager.create_random_subset(10),
        }
        
        subset_results = []
        for subset_name, classes in selected_subsets.items():
            config = ModelConfig(
                selected_classes=classes,
                num_classes=len(classes),
                num_epochs=20,
                batch_size=32
            )
            
            temp_config = f'temp_{subset_name}.yaml'
            ConfigManager().save_config(config, temp_config)
            
            try:
                result = run_single_experiment(temp_config, f'all_mode_{subset_name}')
                subset_results.append(result)
            finally:
                if os.path.exists(temp_config):
                    os.remove(temp_config)
        
        # Run small hyperparameter search
        print("\n3. Hyperparameter Search (5 configurations)")
        hp_results = run_hyperparameter_search('random', 5)
        
        print("\nAll experiments completed!")
        print(f"Fusion experiments: {len(fusion_results)}")
        print(f"Subset experiments: {len(subset_results)}")
        print(f"Hyperparameter experiments: {len(hp_results)}")

if __name__ == "__main__":
    main()