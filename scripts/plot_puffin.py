# scripts/plot_puffin.py
#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.visualization import (
    plot_transcription_profile,
    plot_contribution_breakdown,
    plot_motif_contributions,
    plot_contribution_heatmap,
    plot_interpretation_summary
)


def parse_puffin_csv(csv_file):
    """Parse Puffin output CSV file"""
    sequences_data = {}

    # Fix input file path
    if not os.path.isabs(csv_file):
        csv_file = os.path.join(project_root, csv_file)

    print(f"Reading CSV file from: {csv_file}")

    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return sequences_data

    with open(csv_file, 'r', encoding='utf-8') as f:
        current_sequence = None
        current_data = []

        for line in f:
            line = line.strip()

            if line.startswith('# Sequence:'):
                # New sequence starts
                if current_sequence is not None and current_data:
                    # Save previous sequence data
                    try:
                        df = pd.DataFrame(current_data[1:], columns=current_data[0])
                        df = df.set_index(df.columns[0])
                        sequences_data[current_sequence] = df
                    except Exception as e:
                        print(f"Warning: Error parsing data for {current_sequence}: {e}")

                current_sequence = line.replace('# Sequence: ', '').strip()
                current_data = []

            elif line.startswith('#') or not line:
                # Comment line or empty line, skip
                continue

            elif current_sequence is not None:
                # Data line
                current_data.append(line.split(','))

    # Save last sequence data
    if current_sequence is not None and current_data:
        try:
            df = pd.DataFrame(current_data[1:], columns=current_data[0])
            df = df.set_index(df.columns[0])
            sequences_data[current_sequence] = df
        except Exception as e:
            print(f"Warning: Error parsing data for {current_sequence}: {e}")

    return sequences_data


def main():
    parser = argparse.ArgumentParser(description='Generate plots from Puffin analysis results')
    parser.add_argument('--input', type=str, required=True, help='Puffin output CSV file')
    parser.add_argument('--output_dir', type=str, default='puffin_plots', help='Output directory for plots')
    parser.add_argument('--plot_type', type=str, required=True,
                        choices=['all', 'profile', 'breakdown', 'motif_effects', 'heatmap', 'summary'],
                        help='Type of plot to generate')
    parser.add_argument('--sequence', type=str, help='Specific sequence to plot (if not specified, plot all)')

    args = parser.parse_args()

    # Fix output directory path
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(project_root, args.output_dir)
    else:
        output_dir = args.output_dir

    print(f"Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Parse CSV file
    print(f"Parsing Puffin results from {args.input}...")
    sequences_data = parse_puffin_csv(args.input)

    print(f"Found {len(sequences_data)} sequences in results")

    if not sequences_data:
        print("No valid sequences found. Please check the input file.")
        return

    # Determine which sequences to plot
    if args.sequence:
        if args.sequence in sequences_data:
            sequences_to_plot = {args.sequence: sequences_data[args.sequence]}
        else:
            print(f"Error: Sequence '{args.sequence}' not found in results")
            print(f"Available sequences: {list(sequences_data.keys())}")
            return
    else:
        sequences_to_plot = sequences_data

    # Generate plots
    for seq_name, df in sequences_to_plot.items():
        print(f"Generating plots for {seq_name}...")

        # Clean sequence name for filename
        clean_name = "".join(c for c in seq_name if c.isalnum() or c in ('-', '_'))

        # Get sequence information (if available)
        sequence_info = None
        if 'Sequence' in df.index:
            try:
                sequence_info = ''.join(df.loc['Sequence'].values.astype(str))
            except:
                sequence_info = None

        if args.plot_type in ['all', 'profile']:
            # Transcription signal profile
            if 'Prediction' in df.index:
                try:
                    plot_transcription_profile(
                        df.loc['Prediction'].values,
                        sequence=sequence_info,
                        title=f'Transcription Profile - {seq_name}',
                        save_path=os.path.join(output_dir, f'{clean_name}_profile.png')
                    )
                except Exception as e:
                    print(f"Error generating profile plot for {seq_name}: {e}")

        if args.plot_type in ['all', 'breakdown']:
            # Contribution breakdown
            try:
                plot_contribution_breakdown(
                    df,
                    sequence=sequence_info,
                    title=f'Contribution Breakdown - {seq_name}',
                    save_path=os.path.join(output_dir, f'{clean_name}_breakdown.png')
                )
            except Exception as e:
                print(f"Error generating breakdown plot for {seq_name}: {e}")

        if args.plot_type in ['all', 'motif_effects']:
            # Motif effect curves
            try:
                plot_motif_contributions(
                    df,
                    title=f'Motif Effects - {seq_name}',
                    save_path=os.path.join(output_dir, f'{clean_name}_motif_effects.png')
                )
            except Exception as e:
                print(f"Error generating motif effects plot for {seq_name}: {e}")

        if args.plot_type in ['all', 'heatmap']:
            # Motif contribution heatmap
            try:
                plot_contribution_heatmap(
                    df,
                    title=f'Motif Contribution Heatmap - {seq_name}',
                    save_path=os.path.join(output_dir, f'{clean_name}_heatmap.png')
                )
            except Exception as e:
                print(f"Error generating heatmap plot for {seq_name}: {e}")

        if args.plot_type in ['all', 'summary']:
            # Complete summary plot
            try:
                plot_interpretation_summary(
                    df,
                    title=f'Interpretation Summary - {seq_name}',
                    save_path=os.path.join(output_dir, f'{clean_name}_summary.png')
                )
            except Exception as e:
                print(f"Error generating summary plot for {seq_name}: {e}")

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()