# Puffin: Interpretable Transcription Initiation Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Puffin is a deep learning framework for predicting and interpreting transcription initiation signals from DNA sequences. This repository provides a simplified implementation of the methodology described in the original research paper.

> **Original Research**: Kseniia Dudnyk et al., "Sequence basis of transcription initiation in the human genome". *Science* 384, eadj0116 (2024). DOI: [10.1126/science.adj0116](https://doi.org/10.1126/science.adj0116)

## 🎯 Project Philosophy

This project demonstrates the **methodology** for interpretable transcription initiation analysis. It provides:

- **Simplified training pipelines** for educational and research purposes
- **Complete interpretability tools** for analyzing sequence contributions
- **Cross-platform compatibility** for easy deployment
- **Comprehensive documentation** for researchers

## ⚠️ Critical Limitations - Important Notice

**This implementation demonstrates the methodology but has key limitations for real biological discovery:**

### 🚫 Single Training Cannot Discover Real Biological Motifs

The models trained with this code **will not automatically discover real biological motifs** like TATA, YY1, NFY, etc. The convolutional kernels will learn generic patterns labeled as `Learned_Motif_00`, `Learned_Motif_01`, etc.

### 🔬 Real Motif Discovery Requirements

To discover actual biological motifs as in the original paper, you must:

1. **Train multiple independent replicas** (12+ models)
2. **Perform motif stability analysis** (correlation > 0.95 across replicas)
3. **Validate against known motif databases** (JASPAR, CIS-BP, etc.)
4. **Implement multi-stage training** with motif initialization
5. **Conduct experimental validation** (CRISPR, TF knockdowns)

### 📚 Reference Implementation

This code provides the **infrastructure** for motif discovery research. For the complete methodology, refer to the original paper's supplementary materials and consider implementing the full three-stage training protocol.

## 🏗️ Model Architecture

### Puffin (Interpretable Model) - 650bp Input

**Input Format:**
- **Sequence**: 650bp DNA sequence (ACGT characters)
- **Encoding**: One-hot encoded (4 channels × 650 positions)
- **Required**: Minimum 651bp for proper analysis (325bp padding each side)

**Output Components:**
1. **Prediction Signal**: Log-scale transcription initiation probability at each position
2. **Motif Activations**: 20 learned motif detection scores (10 forward + 10 reverse strand)
3. **Motif Effects**: Contribution of each motif to final prediction
4. **Basepair Contributions**: Per-basepair impact on transcription initiation
5. **Total Effects**: Combined effects from all sequence patterns

**Architecture Details:**
- **Motif Detection**: 51bp convolutional kernels detecting sequence patterns
- **Initiator Detection**: 15bp kernels fine-tuning local base preferences  
- **Trinucleotide Detection**: 3bp kernels capturing residual dependencies
- **Effect Ranges**: Motif effects span ±300bp, initiator effects ±7bp
- **Activation**: Softplus for interpretable non-negative outputs

### Puffin-D (Performance Model) - 100kb Input

**Input Format:**
- **Sequence**: 100,000bp genomic region
- **Encoding**: One-hot encoded (4 channels × 100,000 positions)
- **Recommended**: Exact 100kb length for optimal performance

**Output:**
- **Prediction Signal**: High-resolution transcription initiation profile across 100kb
- **Single Target**: Focused prediction for one experimental method

**Architecture Details:**
- **Encoder-Decoder**: Deep network with multiple downsampling/upsampling stages
- **Skip Connections**: Preserve information across scales
- **Multi-scale Processing**: Handles both local motifs and long-range dependencies
- **Final Output**: Single-channel prediction through 1x1 convolution + Softplus

## 🔄 Key Modifications from Original Implementation

### Single-Target Output Architecture

**Original Version:**
- Simultaneously predicted 5 experimental methods × 2 strands = 10 output channels
- Methods: FANTOM_CAGE, ENCODE_CAGE, ENCODE_RAMPAGE, GRO_CAP, PRO_CAP

**Our Version:**
- **Single target output**: One experimental method, single channel prediction
- **Configurable target**: Set in configuration file (`target_method` parameter)
- **Algorithm preservation**: All interpretability algorithms identical to original paper

**Rationale for Modification:**
1. **Simplified Usage**: Most users have single data type, not multiple experimental methods
2. **Reduced Complexity**: Easier model training and interpretation
3. **Focused Analysis**: Better understanding of specific experimental conditions
4. **Resource Efficiency**: Lower memory usage and faster training
5. **Methodological Focus**: Demonstrates core algorithms without multi-target complexity

### Generic Motif Naming

**Original Version:** Hard-coded biological motif names (TATA+, YY1+, NFY+, etc.)

**Our Version:** Generic identifiers (`Learned_Motif_00`, `Learned_Motif_01`, etc.)

**Purpose:** Clearly indicate that learned patterns may not correspond to real biological motifs without proper validation.

## 📊 Understanding Puffin Output Interpretation

### Output File Structure

The Puffin interpretation CSV contains these key sections:

**Basic Information:**
- `Coordinate`: Position in sequence (0 to sequence_length-1)
- `Sequence`: DNA bases at each position
- `Prediction`: Transcription initiation probability (log scale)

**Motif Analysis (20 motifs total):**
- `Learned_Motif_00 motif activation`: Detection strength of motif 00 at each position
- `Learned_Motif_00 motif effect`: Contribution of motif 00 to final prediction
- `Learned_Motif_00 Basepair contribution score to transcription initiation`: Per-basepair impact through motif 00
- `Learned_Motif_00 Basepair contribution score to motif activation`: Per-basepair impact on motif detection

**Summary Effects:**
- `Sum of motif effect`: Combined effects from all motifs
- `Sum of initiator effect`: Local initiation preferences
- `Sum of trinucleotide effect`: Residual sequence dependencies
- `Sum of total effect`: Complete sequence contribution

**Overall Contributions:**
- `Basepair contribution score to transcription initiation`: Total per-basepair impact

### How to Interpret Results

1. **Prediction Peaks**: High values indicate likely transcription start sites
2. **Positive Motif Effects**: Motifs increasing transcription probability
3. **Negative Motif Effects**: Motifs decreasing transcription probability  
4. **Basepair Contributions**: Individual bases promoting (blue) or inhibiting (red) transcription
5. **Motif Activation Peaks**: Positions where motifs are strongly detected

## 📁 Project Structure

```
puffin/
├── config/                 # Configuration files
│   ├── train_puffin.yaml
│   └── train_puffin_d.yaml
├── models/                 # Model architectures
│   ├── puffin.py
│   └── puffin_d.py
├── data/                   # Data loading utilities
│   └── dataset.py
├── training/               # Training strategies
│   ├── trainer_puffin.py
│   ├── trainer_puffin_d.py
│   └── losses.py
├── scripts/                # User interfaces
│   ├── train_puffin.py
│   ├── train_puffin_d.py
│   ├── puffin.py
│   ├── predict_puffin_d.py
│   ├── plot_puffin.py
│   ├── create_example_data_puffin.py
│   └── create_example_data_puffin_d.py
├── utils/                  # Utility functions
│   ├── sequence_utils.py
│   ├── visualization.py
│   └── path_utils.py
└── example_data/           # Generated example data
```

## Quick Start

### Step 1: Install Miniconda (Skip if Already Installed)

**Windows:**
```bash
# Download Miniconda from https://docs.conda.io/en/latest/miniconda.html
# Run the installer and follow prompts
# Open Anaconda Prompt from Start Menu
```
**Mac/Linux:**
```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow installation prompts, restart terminal after installation
```

### Step 2: Create and Activate Conda Environment
```bash
# Create new environment with Python 3.9
conda create -n puffin python=3.9

# Activate the environment
conda activate puffin

# Verify Python version
python --version  # Should show Python 3.9.x
```
### Step 3: Install Dependencies
```bash
# Clone the repository
git clone https://github.com/shenshen-cmy/puffin-refactored
cd puffin-refactored

# Install PyTorch (choose appropriate version for your system)
# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# For GPU with CUDA 11.3:
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Install other dependencies
pip install -r requirements.txt
```
### Step 4: Generate Example Data
```bash
# Generate Puffin training data (650bp sequences)
python scripts/create_example_data_puffin.py

# Generate Puffin-D training data (100kb sequences)  
python scripts/create_example_data_puffin_d.py
```
You should see output like:

```text
Generating biologically meaningful promoter sequences...
Generated 5/30 sequences
...
Created example data to: /path/to/puffin/example_puffin_data/
Contains 30 biologically meaningful promoter sequences
```
### Step 5: Train Models
Train Puffin Model:

```bash
python scripts/train_puffin.py --config config/train_puffin.yaml --auto_split
```
Train Puffin-D Model:

```bash
python scripts/train_puffin_d.py --config config/train_puffin_d.yaml --auto_split
```
### Step 6: Run Analysis on Test Sequences
Puffin with Full Interpretation:

```bash
python scripts/puffin.py /
  --model models/saved/puffin_best_model.pth /
  --sequences prediction_example_puffin/test_sequences.fasta /
  --output my_analysis.csv
```
Puffin-D Prediction Only:
```bash
python scripts/predict_puffin_d.py \
  --model models/saved/puffin_d_best_model.pth \
  --sequences prediction_example_puffin_d/test_sequences.fasta \
  --output my_predictions.csv
```

### Step 7: Generate Visualizations
```bash
python scripts/plot_puffin.py \
  --input my_analysis.csv \
  --output_dir my_plots \
  --plot_type all
```

### Step 8: Run Complete Pipeline (Alternative)
Or run everything with one command:
```bash
bash run_example_pipeline.sh
```

## 🗃️ Data Preparation
Training Data Format

Create a TSV file with the following structure:

### **training_data.tsv:**

```tsv
sequence_id    sequence    signal_file    length    chromosome
promoter_1     ATCG...     signals/1.npy  2000      chr1
promoter_2     GCTA...     signals/2.npy  2000      chr2
```
Required columns:
```
sequence_id: Unique identifier for each sequence
sequence: DNA sequence string (ACGT characters only)
signal_file: Relative path to corresponding signal file
length: Sequence length in basepairs
```
Optional columns:
```
chromosome: For chromosome-based data splitting
start/end: Genomic coordinates (for reference)
strand: DNA strand information
```
### **Signal Files：**
Each sequence must have a corresponding NPY file with transcription signals:

```python
import numpy as np

# Create signal array (must match sequence length)
signal = np.random.random(2000).astype(np.float32)  # Example random signal
np.save("signals/promoter_1.npy", signal)
```
Signal Requirements:
```
Same length as corresponding sequence
Float32 data type
Values represent transcription initiation probability
Log10 scale recommended for better numerical stability
```

## Dataset Splitting Options
### Automatic Splitting (Recommended for Beginners):
```bash
python scripts/train_puffin.py --config config.yaml --auto_split --split_ratio 0.8
```
Automatically splits single TSV file into training/validation

Options: --split_by random (default) or --split_by chromosome

Set ratio with --split_ratio (default: 0.8 for 80% training)

### Manual Splitting (Advanced Users):
```yaml
# In config file:
data:
  train_tsv: "data/train_data.tsv"
  val_tsv: "data/val_data.tsv"
```
Provide separate files for training and validation

More control over specific sequences in each set

## ⚙️ Configuration Files
Puffin Configuration (config/train_puffin.yaml)
```yaml
data:
  train_tsv: "example_puffin_data/training_data.tsv"
  sequence_length: 650      # Input sequence length
  overlap: 325              # Sliding window overlap

model:
  motif_kernel_size: 51     # Motif detection kernel size
  initiator_kernel_size: 15 # Initiator detection kernel size
  trinucleotide_kernel_size: 3
  motif_effect_range: 601   # Effect range in basepairs
  num_motifs: 10            # Number of motif kernels (5 forward + 5 reverse)
  num_initiators: 20
  num_trinucleotides: 64
  target_method: "FANTOM_CAGE"  # Single target method
  use_cuda: true

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 20
  early_stopping_patience: 20
...
```
Puffin-D Configuration (config/train_puffin_d.yaml)
```yaml
data:
  train_tsv: "example_puffin_d_data/training_data.tsv"
  sequence_length: 100000   # 100kb sequences
  overlap: 50000            # 50% overlap for sliding windows

model:
  use_cuda: true
  target_method: "FANTOM_CAGE"

training:
  batch_size: 4             # Smaller batches for long sequences
  learning_rate: 0.0001     # Lower learning rate
  num_epochs: 50            # More epochs for convergence
...
```

## 📊 Output Interpretation Guide
Puffin Output Columns Explained
Basic Sequence Information:
```
Coordinate: 0-based position along the sequence
Sequence: DNA base (A,C,G,T) at each position
Prediction: Model's transcription initiation probability
```
For Each Learned Motif (20 total):
```
Learned_Motif_XX motif activation: How strongly the motif is detected
Learned_Motif_XX motif effect: How much the motif influences transcription
Learned_Motif_XX Basepair contribution...: Individual base contributions
```
Key Interpretation Points:
```
High Prediction values = Likely transcription start sites
Positive motif effects = Motifs that promote transcription
Negative motif effects = Motifs that inhibit transcription
Blue bars in contribution plots = Bases promoting transcription
Red bars in contribution plots = Bases inhibiting transcription
```
Example Analysis Workflow
```
Identify TSS candidates: Look for peaks in Prediction column
Check motif context: See which motifs have high activation near TSS
Analyze contributions: Understand which bases drive transcription
Compare sequences: See how different sequences produce different patterns
```

## 🐛 Troubleshooting
Common Issues and Solutions

CUDA Out of Memory:
```bash
# Reduce batch size in config file
training:
  batch_size: 16  # Instead of 32
```
File Not Found Errors:
```bash
# Use absolute paths in configuration
train_tsv: "/full/path/to/your/data/training_data.tsv"
```
Training Instability:
```bash
# Reduce learning rate
training:
  learning_rate: 0.0005
```
Sequence Length Mismatches:
```
Ensure all sequences match the length specified in config
Check signal files have exactly the same length as sequences
```
Performance Tips
```
Use --use_cuda if you have NVIDIA GPU with CUDA
Adjust num_workers in data loaders for faster loading
Use chromosome-based splitting for genomic data
Pre-process large datasets to avoid memory issues
```

## 📋 Requirements
See requirements.txt for complete dependency list.

## 🙏 Citation
If you use this code in your research, please cite the original paper:
> **Original Research**: Kseniia Dudnyk et al., "Sequence basis of transcription initiation in the human genome". *Science* 384, eadj0116 (2024). DOI: [10.1126/science.adj0116](https://doi.org/10.1126/science.adj0116)

## 🤝 Contributing
We welcome contributions! Please feel free to submit issues and pull requests.

## 📞 Support
For questions and support:

Open an issue on GitHub

Check the detailed USAGE.md documentation

Refer to the original paper for methodological questions
