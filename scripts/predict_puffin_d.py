# scripts/predict_puffin_d.py
#!/usr/bin/env python3

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
from Bio import SeqIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.puffin_D import Puffin_D
from utils.sequence_utils import sequence_to_onehot
from utils.path_utils import ensure_absolute_path, create_directory


def load_model(model_path):
    """Load trained Puffin_D model with PyTorch 2.6+ compatibility"""
    model_path = ensure_absolute_path(model_path)
    print(f"Loading Puffin_D model from {model_path}...")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        from utils.path_utils import find_file_in_project
        found_path = find_file_in_project(os.path.basename(model_path))
        if found_path:
            model_path = found_path
            print(f"Found model at: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        # First try safe loading
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"Safe load failed, using weights_only=False: {e}")
        try:
            # Use non-safe loading (only for trusted sources)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e2:
            print(f"Standard load failed: {e2}")
            # Try using pickle
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            except Exception as e3:
                raise Exception(f"All loading methods failed: {e3}")

    config = checkpoint['config']
    print("Model checkpoint loaded successfully")

    model = Puffin_D(config)

    # Load model state
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading state dict: {e}")
        # Try partial loading
        print("Attempting partial state dict loading...")
        model_state = model.state_dict()
        pretrained_state = checkpoint['model_state_dict']

        # Only load matching keys
        loaded_keys = []
        for key in model_state.keys():
            if key in pretrained_state:
                if model_state[key].shape == pretrained_state[key].shape:
                    model_state[key] = pretrained_state[key]
                    loaded_keys.append(key)

        model.load_state_dict(model_state)
        print(f"Partially loaded {len(loaded_keys)}/{len(model_state)} parameters")

    model.eval()
    print("Model state dict loaded successfully")

    return model, config


def predict_sequences(model, sequences, use_cuda=False):
    """Predict transcription initiation signals for sequences"""
    predictions = []

    print(f"Predicting {len(sequences)} sequences...")

    for i, sequence in enumerate(sequences):
        print(f"Processing sequence {i + 1}/{len(sequences)}...")

        # Convert to one-hot encoding
        try:
            encoded = sequence_to_onehot(sequence)
        except Exception as e:
            print(f"Error encoding sequence {i + 1}: {e}")
            predictions.append(np.zeros(len(sequence)))
            continue

        if use_cuda and torch.cuda.is_available():
            input_tensor = torch.FloatTensor(encoded).unsqueeze(0).cuda()
        else:
            input_tensor = torch.FloatTensor(encoded).unsqueeze(0)

        with torch.no_grad():
            try:
                prediction = model(input_tensor)
                pred_np = prediction.cpu().numpy()[0, 0]  # Single channel output
                predictions.append(pred_np)
                print(f"Prediction completed, length: {len(pred_np)}")
            except Exception as e:
                print(f"Prediction failed for sequence {i + 1}: {e}")
                predictions.append(np.zeros(len(sequence)))

    return predictions


def save_results_to_csv(results, sequence_names, sequences, output_file):
    """Save results to CSV files"""
    output_file = ensure_absolute_path(output_file)
    create_directory(os.path.dirname(output_file))

    # Create detailed results DataFrame
    detailed_results = []

    for name, seq, pred in zip(sequence_names, sequences, results):
        # Create one row per position
        for pos in range(len(pred)):
            detailed_results.append({
                'sequence_id': name,
                'position': pos,
                'prediction': pred[pos],
                'sequence_base': seq[pos] if pos < len(seq) else 'N'
            })

    # Save detailed results
    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv(output_file, index=False)

    # Create summary file
    summary_file = output_file.replace('.csv', '_summary.csv')
    summary_data = []

    for name, seq, pred in zip(sequence_names, sequences, results):
        if len(pred) > 0:
            summary_data.append({
                'sequence_id': name,
                'length': len(seq),
                'max_prediction': np.max(pred),
                'mean_prediction': np.mean(pred),
                'std_prediction': np.std(pred),
                'tss_position': np.argmax(pred) if len(pred) > 0 else -1,
                'tss_strength': np.max(pred) if len(pred) > 0 else 0
            })
        else:
            summary_data.append({
                'sequence_id': name,
                'length': len(seq),
                'max_prediction': 0,
                'mean_prediction': 0,
                'std_prediction': 0,
                'tss_position': -1,
                'tss_strength': 0
            })

    pd.DataFrame(summary_data).to_csv(summary_file, index=False)

    return output_file, summary_file


def main():
    parser = argparse.ArgumentParser(description='Predict transcription initiation signals using Puffin_D')
    parser.add_argument('--model', type=str, required=True, help='Path to trained Puffin_D model')
    parser.add_argument('--sequences', type=str, required=True, help='FASTA file with sequences')
    parser.add_argument('--output', type=str, default='puffin_d_predictions.csv', help='Output CSV file')
    parser.add_argument('--use_cuda', action='store_true', help='Use GPU for prediction')

    args = parser.parse_args()

    print("=== Puffin_D Prediction Script ===")

    # Load model
    try:
        model, config = load_model(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    if args.use_cuda and torch.cuda.is_available():
        model.cuda()
        print("Using GPU for prediction")
    else:
        print("Using CPU for prediction")

    # Fix sequence file path
    sequences_path = ensure_absolute_path(args.sequences)

    # Check if sequence file exists
    if not os.path.exists(sequences_path):
        print(f"Sequences file not found: {sequences_path}")
        from utils.path_utils import find_file_in_project
        found_path = find_file_in_project(os.path.basename(sequences_path))
        if found_path:
            sequences_path = found_path
            print(f"Found sequences at: {sequences_path}")
        else:
            print("Available FASTA files in project:")
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for root, dirs, files in os.walk(project_root):
                for file in files:
                    if file.endswith(('.fasta', '.fa')):
                        print(f"  - {os.path.join(root, file)}")
            return

    # Load sequences
    print(f"Loading sequences from {sequences_path}...")
    sequences = []
    sequence_names = []

    try:
        for record in SeqIO.parse(sequences_path, "fasta"):
            sequences.append(str(record.seq))
            sequence_names.append(record.id)
    except Exception as e:
        print(f"Error loading sequences: {e}")
        return

    print(f"Loaded {len(sequences)} sequences")

    # Validate sequence lengths
    for i, seq in enumerate(sequences):
        if len(seq) < 100000:
            print(f"Warning: Sequence {sequence_names[i]} is shorter than recommended 100kb for Puffin_D ({len(seq)} bp).")
        elif len(seq) > 100000:
            print(f"Warning: Sequence {sequence_names[i]} is longer than 100kb ({len(seq)} bp). Only first 100kb will be used.")
            sequences[i] = seq[:100000]  # Truncate to 100kb

    # Perform prediction
    predictions = predict_sequences(model, sequences, args.use_cuda)

    # Save results
    output_path = ensure_absolute_path(args.output)
    print(f"Saving results to {output_path}...")

    try:
        detailed_file, summary_file = save_results_to_csv(predictions, sequence_names, sequences, output_path)

        print("Puffin_D prediction completed!")
        print(f"Detailed predictions saved to: {detailed_file}")
        print(f"Summary saved to: {summary_file}")

        # Display summary information
        print("\nPrediction Summary:")
        for i, (name, pred) in enumerate(zip(sequence_names, predictions)):
            if len(pred) > 0:
                tss_pos = np.argmax(pred)
                tss_strength = np.max(pred)
                print(f"  {name}: TSS at position {tss_pos}, strength {tss_strength:.4f}")
            else:
                print(f"  {name}: No valid prediction")

    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()