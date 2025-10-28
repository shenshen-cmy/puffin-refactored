# scripts/train_puffin.py
#!/usr/bin/env python3

import argparse
import yaml
import torch
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.puffin import Puffin
from data.dataset import create_data_loaders
from training.trainer_puffin import PuffinTrainer
from utils.path_utils import ensure_absolute_path, create_directory


def load_config(config_path):
    """Safely load YAML configuration, handling encoding issues"""
    config_path = ensure_absolute_path(config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except UnicodeDecodeError:
        try:
            with open(config_path, 'r', encoding='gbk') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            encodings = ['utf-8-sig', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(config_path, 'r', encoding=encoding) as f:
                        return yaml.safe_load(f)
                except UnicodeDecodeError:
                    continue
            raise Exception(f"Could not decode config file with encodings: {encodings}")


def main():
    print("=== Puffin Training Script ===")

    parser = argparse.ArgumentParser(description='Train Puffin model on custom data')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--auto_split', action='store_true', help='Use automatic data splitting')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/validation split ratio (for auto split)')
    parser.add_argument('--split_by', type=str, choices=['random', 'chromosome'], default='random',
                        help='Split method: random or by chromosome (for auto split)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    print(f"Arguments: config={args.config}, auto_split={args.auto_split}")
    print(f"           split_ratio={args.split_ratio}, split_by={args.split_by}")

    # Load configuration
    try:
        config = load_config(args.config)
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading config file: {e}")
        return

    # Fix all paths to absolute paths
    config['data']['train_tsv'] = ensure_absolute_path(config['data']['train_tsv'])
    if 'val_tsv' in config['data'] and config['data']['val_tsv']:
        config['data']['val_tsv'] = ensure_absolute_path(config['data']['val_tsv'])

    # Fix output paths
    config['paths']['model_dir'] = ensure_absolute_path(config['paths']['model_dir'])
    config['paths']['log_dir'] = ensure_absolute_path(config['paths']['log_dir'])
    config['paths']['output_dir'] = ensure_absolute_path(config['paths']['output_dir'])

    # Create necessary directories
    create_directory(config['paths']['model_dir'])
    create_directory(config['paths']['log_dir'])
    create_directory(config['paths']['output_dir'])

    print("Initializing Puffin model...")
    print(f"Target method: {config['model']['target_method']}")
    print(f"Sequence length: {config['data']['sequence_length']}")
    print(f"Overlap: {config['data']['overlap']}")

    # Check if training data file exists
    if not os.path.exists(config['data']['train_tsv']):
        print(f"Training data file not found: {config['data']['train_tsv']}")
        return

    # Create data loaders
    print("Loading data...")
    try:
        train_loader, val_loader = create_data_loaders(
            config['data']['train_tsv'],
            val_tsv=config['data'].get('val_tsv'),
            config=config,
            auto_split=args.auto_split,
            split_ratio=args.split_ratio,
            split_by=args.split_by,
            random_seed=args.random_seed
        )

        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize Puffin model
    model = Puffin(config)

    print(f"Puffin model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize Puffin trainer
    trainer = PuffinTrainer(model, train_loader, val_loader, config)

    # Start training
    trainer.train()

    print("Puffin training completed successfully!")
    print(f"Best model saved to: {os.path.join(config['paths']['model_dir'], 'puffin_best_model.pth')}")


if __name__ == "__main__":
    main()