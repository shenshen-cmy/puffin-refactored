# scripts/puffin.py
#!/usr/bin/env python3

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
from Bio import SeqIO

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.puffin import Puffin
from utils.sequence_utils import sequence_to_onehot
from utils.path_utils import ensure_absolute_path, create_directory, find_file_in_project


def load_model(model_path):
    """Load trained Puffin model with PyTorch 2.6+ compatibility"""
    model_path = ensure_absolute_path(model_path)
    print(f"Loading model from: {model_path}")

    # If file doesn't exist, try to find in project
    if not os.path.exists(model_path):
        print(f"Model file not found at specified path, searching in project...")
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
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    config = checkpoint['config']
    print("Model checkpoint loaded successfully")

    model = Puffin(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model state dict loaded successfully")

    return model, config


def predict_and_interpret(model, sequences, sequence_names, use_cuda=False, interpret=True):
    """Predict and interpret sequences"""
    results = {}

    for name, sequence in zip(sequence_names, sequences):
        print(f"Processing {name}...")

        if interpret:
            # Full interpretability analysis
            interpretation_df = model.interpret(sequence)
            results[name] = {
                'dataframe': interpretation_df,
                'sequence': sequence
            }
            print(f"Interpretation completed for {name}")
        else:
            # Prediction only
            encoded = sequence_to_onehot(sequence)

            if use_cuda:
                input_tensor = torch.FloatTensor(encoded).unsqueeze(0).cuda()
            else:
                input_tensor = torch.FloatTensor(encoded).unsqueeze(0)

            with torch.no_grad():
                prediction = model(input_tensor)
                prediction_np = prediction.cpu().numpy()[0, 0]

            # Create simple prediction DataFrame
            interpretation_df = pd.DataFrame({
                'Coordinate': list(range(len(sequence))),
                'Sequence': list(sequence),
                'Prediction': prediction_np
            }).T

            results[name] = {
                'dataframe': interpretation_df,
                'sequence': sequence
            }
            print(f"Prediction completed for {name}")

    return results


def save_results_to_csv(results, output_file):
    """Save results to CSV file"""
    output_file = ensure_absolute_path(output_file)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        create_directory(output_dir)

    print(f"Saving results to {output_file}...")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for name, result in results.items():
                df = result['dataframe']
                sequence = result['sequence']

                print(f"Saving {name}...")

                # Write sequence information
                f.write(f"# Sequence: {name}\n")
                f.write(f"# Length: {len(sequence)}\n")
                f.write(f"# Sequence (first 100bp): {sequence[:100]}...\n")
                f.write("#" + "=" * 50 + "\n")

                # Manually write DataFrame content
                columns = [str(col) for col in df.columns]
                f.write("," + ",".join(columns) + "\n")

                # Write each row of data
                for idx in df.index:
                    row_data = df.loc[idx]
                    if hasattr(row_data, '__iter__') and not isinstance(row_data, str):
                        row_values = [str(x) for x in row_data]
                    else:
                        row_values = [str(row_data)]

                    f.write(f"{idx}," + ",".join(row_values) + "\n")

                f.write("\n\n")
                print(f"Saved {name} with {len(df.index)} rows")

        print(f"All results saved to: {output_file}")

    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Puffin: Predict and interpret transcription initiation signals')
    parser.add_argument('--model', type=str, required=True, help='Path to trained Puffin model')
    parser.add_argument('--sequences', type=str, required=True, help='FASTA file with sequences')
    parser.add_argument('--output', type=str, default='puffin_results.csv', help='Output CSV file')
    parser.add_argument('--no_interpret', action='store_true', help='Only predict, no interpretation (faster)')
    parser.add_argument('--use_cuda', action='store_true', help='Use GPU for prediction')

    args = parser.parse_args()

    print("=== Puffin Prediction Script ===")

    # Load model
    print(f"Loading Puffin model from {args.model}...")
    model, config = load_model(args.model)

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
        found_path = find_file_in_project(os.path.basename(sequences_path))
        if found_path:
            sequences_path = found_path
            print(f"Found sequences at: {sequences_path}")
        else:
            print("Available FASTA files in project:")
            for root, dirs, files in os.walk(project_root):
                for file in files:
                    if file.endswith(('.fasta', '.fa')):
                        print(f"  - {os.path.join(root, file)}")
            return

    # Load sequences
    print(f"Loading sequences from {sequences_path}...")
    sequences = []
    sequence_names = []

    for record in SeqIO.parse(sequences_path, "fasta"):
        sequences.append(str(record.seq))
        sequence_names.append(record.id)

    print(f"Loaded {len(sequences)} sequences")

    # Validate sequence lengths
    for i, seq in enumerate(sequences):
        if len(seq) < 651:
            print(f"Warning: Sequence {sequence_names[i]} is too short ({len(seq)} bp). Minimum required: 651 bp.")

    # Perform prediction and interpretation
    print("Running Puffin analysis...")
    if args.no_interpret:
        print("Mode: Prediction only (no interpretation)")
    else:
        print("Mode: Full prediction with interpretation")

    results = predict_and_interpret(model, sequences, sequence_names, args.use_cuda, not args.no_interpret)

    # Save results
    output_path = ensure_absolute_path(args.output)
    save_results_to_csv(results, output_path)

    print("Puffin analysis completed!")
    if not args.no_interpret:
        print("Use scripts/plot_puffin.py to generate visualization plots from the CSV results")


if __name__ == "__main__":
    main()