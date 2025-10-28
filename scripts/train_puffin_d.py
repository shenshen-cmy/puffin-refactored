# scripts/train_puffin_d.py
#!/usr/bin/env python3

import argparse
import yaml
import torch
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.puffin_D import Puffin_D
from data.dataset import create_data_loaders
from training.trainer_puffin_d import PuffinDTrainer
from utils.path_utils import ensure_absolute_path, create_directory, find_file_in_project


def load_config(config_path):
    """Safely load YAML configuration, handling encoding and path issues"""
    config_path = ensure_absolute_path(config_path)
    print(f"Loading configuration from: {config_path}")

    # Check if file exists
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        # Try to find in project
        found_path = find_file_in_project(os.path.basename(config_path))
        if found_path:
            print(f"Found alternative config file: {found_path}")
            config_path = found_path
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()

        # Handle Windows path backslash issues
        config_content = config_content.replace('\\', '/')

        import io
        config = yaml.safe_load(io.StringIO(config_content))
        print(f"Configuration loaded successfully: {config_path}")
        return config

    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
        # Try direct loading
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except:
            raise Exception(f"Could not parse config file: {e}")
    except UnicodeDecodeError:
        try:
            with open(config_path, 'r', encoding='gbk') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded successfully (GBK encoding): {config_path}")
            return config
        except UnicodeDecodeError:
            encodings = ['utf-8-sig', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(config_path, 'r', encoding=encoding) as f:
                        config = yaml.safe_load(f)
                    print(f"Configuration loaded successfully ({encoding} encoding): {config_path}")
                    return config
                except UnicodeDecodeError:
                    continue
            raise Exception(f"Could not decode config file with encodings: {encodings}")


def validate_data_paths(config, auto_split):
    """Validate data paths"""
    # Fix training data path
    config['data']['train_tsv'] = ensure_absolute_path(config['data']['train_tsv'])

    if not os.path.exists(config['data']['train_tsv']):
        print(f"Training data file not found: {config['data']['train_tsv']}")
        # Try to find in project
        found_path = find_file_in_project(os.path.basename(config['data']['train_tsv']))
        if found_path:
            print(f"Found alternative training data file: {found_path}")
            config['data']['train_tsv'] = found_path
        else:
            raise FileNotFoundError(f"Training data file not found: {config['data']['train_tsv']}")

    # Validation set path handling
    if not auto_split and 'val_tsv' in config['data'] and config['data']['val_tsv']:
        config['data']['val_tsv'] = ensure_absolute_path(config['data']['val_tsv'])
        if not os.path.exists(config['data']['val_tsv']):
            print(f"Validation data file not found: {config['data']['val_tsv']}")
            print("Switching to auto-split mode...")
            auto_split = True

    return config, auto_split


def main():
    print("=== Puffin_D Training Script ===")

    parser = argparse.ArgumentParser(description='Train Puffin_D model on custom data')
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

    # Validate and fix data paths
    try:
        config, final_auto_split = validate_data_paths(config, args.auto_split)
        print(f"Data splitting mode: {'Auto-split' if final_auto_split else 'Manual split'}")
    except Exception as e:
        print(f"Error validating data paths: {e}")
        return

    # Fix output paths
    config['paths']['model_dir'] = ensure_absolute_path(config['paths']['model_dir'])
    config['paths']['log_dir'] = ensure_absolute_path(config['paths']['log_dir'])
    config['paths']['output_dir'] = ensure_absolute_path(config['paths']['output_dir'])

    # Create necessary directories
    create_directory(config['paths']['model_dir'])
    create_directory(config['paths']['log_dir'])
    create_directory(config['paths']['output_dir'])

    print("Training Puffin_D model...")
    print(f"Target method: {config['model']['target_method']}")
    print(f"Sequence length: {config['data']['sequence_length']} (100kb)")
    print(f"Overlap: {config['data']['overlap']}")

    # Create data loaders
    print("Loading data...")
    try:
        train_loader, val_loader = create_data_loaders(
            config['data']['train_tsv'],
            val_tsv=config['data'].get('val_tsv') if not final_auto_split else None,
            config=config,
            auto_split=final_auto_split,
            split_ratio=args.split_ratio,
            split_by=args.split_by,
            random_seed=args.random_seed
        )

        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")

        if len(train_loader.dataset) == 0:
            print("No training samples generated!")
            print("Possible reasons:")
            print("  1. Sequence length too short (needs at least 100kb)")
            print("  2. Incorrect data format")
            print("  3. Sequence length mismatch with configuration")
            return

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize Puffin_D model
    model = Puffin_D(config)

    print(f"Puffin_D model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize Puffin_D trainer
    trainer = PuffinDTrainer(model, train_loader, val_loader, config)

    # Start training
    trainer.train()

    print("Puffin_D training completed successfully!")
    print(f"Best model saved to: {os.path.join(config['paths']['model_dir'], 'puffin_d_best_model.pth')}")


if __name__ == "__main__":
    main()