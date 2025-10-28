# Puffin Usage Guide

Complete documentation for using the Puffin framework for transcription initiation analysis.

## 📋 Table of Contents

- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Prediction & Interpretation](#prediction--interpretation)
- [Visualization](#visualization)
- [Complete Examples](#complete-examples)

## 🗃️ Data Preparation

### Training Data Format

Create a TSV file with the following columns:

```tsv
sequence_id    sequence    signal_file    length    chromosome
promoter_1     ATCG...     signals/1.npy  2000      chr1
promoter_2     GCTA...     signals/2.npy  2000      chr2
Required columns:

sequence_id: Unique identifier

sequence: DNA sequence string

signal_file: Path to corresponding signal file

length: Sequence length

Optional columns:

chromosome: For chromosome-based splitting

Signal Files
Each sequence should have a corresponding NPY file with transcription signals:

python
import numpy as np

# Create signal array (same length as sequence)
signal = np.array([...], dtype=np.float32)
np.save("signals/promoter_1.npy", signal)
Generate Example Data
bash
# Generate Puffin example data (650bp sequences)
python scripts/create_example_data_puffin.py \
  --output_dir my_puffin_data \
  --num_sequences 50 \
  --sequence_length 2000

# Generate Puffin-D example data (100kb sequences)
python scripts/create_example_data_puffin_d.py \
  --output_dir my_puffin_d_data \
  --num_sequences 20 \
  --sequence_length 500000
🏋️ Model Training
Puffin Training
Configuration file (config/train_puffin.yaml):

yaml
data:
  train_tsv: "example_puffin_data/training_data.tsv"
  sequence_length: 650
  overlap: 325

model:
  num_motifs: 10
  target_method: "FANTOM_CAGE"
  use_cuda: true

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 20
Training command:

bash
python scripts/train_puffin.py \
  --config config/train_puffin.yaml \
  --auto_split \
  --split_ratio 0.8 \
  --split_by random \
  --random_seed 42
Puffin-D Training
Configuration file (config/train_puffin_d.yaml):

yaml
data:
  train_tsv: "example_puffin_d_data/training_data.tsv"
  sequence_length: 100000
  overlap: 50000

model:
  use_cuda: true
  target_method: "FANTOM_CAGE"

training:
  batch_size: 4
  learning_rate: 0.0001
  num_epochs: 50
Training command:

bash
python scripts/train_puffin_d.py \
  --config config/train_puffin_d.yaml \
  --auto_split \
  --split_ratio 0.8 \
  --split_by chromosome \
  --random_seed 42
Training Options
--auto_split: Automatically split data (no separate validation file needed)

--split_ratio: Train/validation split ratio (default: 0.8)

--split_by: Split method: random or chromosome

--random_seed: Random seed for reproducibility

🔮 Prediction & Interpretation
Puffin Prediction with Interpretation
bash
python scripts/puffin.py \
  --model models/saved/puffin_best_model.pth \
  --sequences prediction_example_puffin/test_sequences.fasta \
  --output puffin_analysis_results.csv \
  --use_cuda
Options:

--no_interpret: Prediction only (faster, no contribution analysis)

--use_cuda: Use GPU acceleration

Puffin-D Prediction
bash
python scripts/predict_puffin_d.py \
  --model models/saved/puffin_d_best_model.pth \
  --sequences prediction_example_puffin_d/test_sequences.fasta \
  --output puffin_d_predictions.csv \
  --use_cuda
Output Formats
Puffin interpretation output includes:

Sequence coordinates and bases

Motif activation scores for each position

Motif effect scores

Basepair contribution scores to transcription initiation

Basepair contribution scores to motif activation

Prediction signals

Puffin-D prediction output includes:

Detailed per-position predictions

Summary statistics (TSS positions, strengths)

📊 Visualization
Generate Analysis Plots
bash
python scripts/plot_puffin.py \
  --input puffin_analysis_results.csv \
  --output_dir puffin_plots \
  --plot_type all \
  --sequence simulated_strong_promoter_001
Plot types:

profile: Transcription signal profile

breakdown: Contribution breakdown

motif_effects: Individual motif effects

heatmap: Motif contribution heatmap

summary: Complete interpretation summary

all: Generate all plot types

Custom Visualization
python
import pandas as pd
from utils.visualization import plot_transcription_profile

# Load results
df = pd.read_csv("puffin_analysis_results.csv", index_col=0)

# Create custom plot
fig = plot_transcription_profile(
    df.loc['Prediction'].values,
    title="Custom Transcription Profile",
    save_path="custom_plot.png"
)
🎯 Complete Examples
Example 1: Full Puffin Pipeline
bash
# 1. Generate example data
python scripts/create_example_data_puffin.py

# 2. Train model
python scripts/train_puffin.py --config config/train_puffin.yaml --auto_split

# 3. Run prediction with interpretation
python scripts/puffin.py \
  --model models/saved/puffin_best_model.pth \
  --sequences prediction_example_puffin/test_sequences.fasta \
  --output full_analysis.csv

# 4. Generate visualizations
python scripts/plot_puffin.py \
  --input full_analysis.csv \
  --output_dir results_plots \
  --plot_type all
Example 2: Puffin-D High-Resolution Analysis
bash
# 1. Generate long sequence data
python scripts/create_example_data_puffin_d.py

# 2. Train Puffin-D model
python scripts/train_puffin_d.py --config config/train_puffin_d.yaml --auto_split

# 3. Predict on test sequences
python scripts/predict_puffin_d.py \
  --model models/saved/puffin_d_best_model.pth \
  --sequences prediction_example_puffin_d/test_sequences.fasta \
  --output long_sequence_predictions.csv
Example 3: Batch Processing
bash
# Process multiple FASTA files
for fasta_file in data/*.fasta; do
    base_name=$(basename "$fasta_file" .fasta)
    python scripts/puffin.py \
        --model models/saved/puffin_best_model.pth \
        --sequences "$fasta_file" \
        --output "results/${base_name}_analysis.csv"
done
⚙️ Configuration Details
Puffin Model Parameters
yaml
model:
  motif_kernel_size: 51      # Motif detection kernel size
  initiator_kernel_size: 15  # Initiator detection kernel size  
  trinucleotide_kernel_size: 3  # Trinucleotide detection kernel size
  motif_effect_range: 601    # Motif effect range
  num_motifs: 10             # Number of motif kernels
  num_initiators: 20         # Number of initiator kernels
  num_trinucleotides: 64     # Number of trinucleotide kernels
Training Parameters
yaml
training:
  batch_size: 32             # Batch size (adjust based on memory)
  learning_rate: 0.001       # Learning rate
  num_epochs: 20             # Training epochs
  early_stopping_patience: 20 # Early stopping patience
  validation_freq: 2         # Validation frequency
  save_freq: 5               # Model saving frequency
🐛 Troubleshooting
Common Issues
CUDA out of memory

Reduce batch size in configuration

Use --use_cuda only if GPU available

File not found errors

Use absolute paths in configuration files

Check file permissions

Sequence length mismatches

Ensure all sequences match specified length in config

Check signal file dimensions

Training instability

Adjust learning rate

Check data normalization

Verify random seed consistency

Performance Tips
Use --use_cuda for GPU acceleration

Adjust num_workers in data loaders for I/O performance

Use chromosome-based splitting for genomic data

Pre-process large datasets for faster loading