# Puffin: Interpretable Transcription Initiation Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Puffin is a deep learning framework for predicting and interpreting transcription initiation signals from DNA sequences. This repository provides a simplified implementation of the methodology described in the original research paper.

> **Original Research**: Kseniia Dudnyk et al., "Sequence basis of transcription initiation in the human genome". *Science* 384, eadj0116 (2024). DOI: [10.1126/science.adj0116](https://doi.org/10.1126/science.adj0116)

## Project Philosophy

This project demonstrates the **methodology** for interpretable transcription initiation analysis. It provides:

- **Simplified training pipelines** for educational and research purposes
- **Complete interpretability tools** for analyzing sequence contributions
- **Cross-platform compatibility** for easy deployment
- **Comprehensive documentation** for researchers

## Critical Limitations - Important Notice

**This implementation demonstrates the methodology but has key limitations for real biological discovery:**

### Single Training Cannot Discover Real Biological Motifs

The models trained with this code **will not automatically discover real biological motifs** like TATA, YY1, NFY, etc. The convolutional kernels will learn generic patterns labeled as `Learned_Motif_00`, `Learned_Motif_01`, etc.

### Real Motif Discovery Requirements

To discover actual biological motifs as in the original paper, you must:

1. **Train multiple independent replicas** (12+ models)
2. **Perform motif stability analysis** (correlation > 0.95 across replicas)
3. **Validate against known motif databases** (JASPAR, CIS-BP, etc.)
4. **Implement multi-stage training** with motif initialization
5. **Conduct experimental validation** (CRISPR, TF knockdowns)

### Reference Implementation

This code provides the **infrastructure** for motif discovery research. For the complete methodology, refer to the original paper's supplementary materials and consider implementing the full three-stage training protocol.

## Model Architecture

### Puffin (Interpretable Model)
- **Input**: 650bp sequences (±325bp around TSS)
- **Output**: Single-target transcription initiation signals
- **Components**:
  - Motif detection (51bp kernels)
  - Initiator detection (15bp kernels) 
  - Trinucleotide detection (3bp kernels)
- **Focus**: Sequence interpretability and contribution analysis

### Puffin-D (Performance Model)
- **Input**: 100kb genomic regions
- **Output**: High-resolution transcription initiation profiles
- **Architecture**: Deep encoder-decoder with skip connections
- **Focus**: Prediction accuracy and long-range dependencies

## Project Structure
puffin/├── config/ # Configuration files│ ├── train_puffin.yaml│ └── train_puffin_d.yaml├── models/ # Model architectures│ ├── puffin.py│ └── puffin_d.py├── data/ # Data loading utilities│ └── dataset.py├── training/ # Training strategies│ ├── trainer_puffin.py│ ├── trainer_puffin_d.py│ └── losses.py├── scripts/ # User interfaces│ ├── train_puffin.py│ ├── train_puffin_d.py│ ├── puffin.py│ ├── predict_puffin_d.py│ ├── plot_puffin.py│ ├── create_example_data_puffin.py│ └── create_example_data_puffin_d.py├── utils/ # Utility functions│ ├── sequence_utils.py│ ├── visualization.py│ └── path_utils.py└── example_data/ # Generated example data
plaintext

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/puffin.git
cd puffin

# Install dependencies
pip install -r requirements.txt
Generate Example Data
bash
# Generate Puffin training data (650bp sequences)
python scripts/create_example_data_puffin.py

# Generate Puffin-D training data (100kb sequences)  
python scripts/create_example_data_puffin_d.py
Train Models
bash
# Train Puffin model
python scripts/train_puffin.py --config config/train_puffin.yaml --auto_split

# Train Puffin-D model
python scripts/train_puffin_d.py --config config/train_puffin_d.yaml --auto_split
Run Complete Example
bash
# Run the complete example pipeline
bash run_example_pipeline.sh
Data Format Requirements
Training Data (TSV + NPY)
training_data.tsv:
tsv
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
运行
import numpy as np

# Create signal array (same length as sequence)
signal = np.array([...], dtype=np.float32)
np.save("signals/promoter_1.npy", signal)
Data Splitting Options
We provide flexible data splitting methods for training and validation:
Automatic Splitting
bash
# Random splitting (default)
python scripts/train_puffin.py --config config.yaml --auto_split --split_by random

# Chromosome-based splitting (requires chromosome column)
python scripts/train_puffin.py --config config.yaml --auto_split --split_by chromosome
Manual Splitting
bash
# Provide separate training and validation files
python scripts/train_puffin.py --config config.yaml
In your config file, specify both files:
yaml
data:
  train_tsv: "data/training_data.tsv"
  val_tsv: "data/validation_data.tsv"
Splitting Parameters
--split_ratio: Train/validation ratio (default: 0.8)
--split_by: random or chromosome
--random_seed: For reproducible splits
Usage
See USAGE.md for detailed usage instructions and command-line examples.
Requirements
See requirements.txt for complete dependency list.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Citation
If you use this code in your research, please cite the original paper:
bibtex
@article{dudnyk2024sequence,
  title={Sequence basis of transcription initiation in the human genome},
  author={Dudnyk, Kseniia and Chen, Shilu and Qiu, Yupeng and Zhang, Jun and Qiu, Yuchen and Yang, Chen and Zhou, Jian},
  journal={Science},
  volume={384},
  number={6697},
  pages={eadj0116},
  year={2024},
  publisher={American Association for the Advancement of Science}
}
Contributing
We welcome contributions! Please feel free to submit issues and pull requests.
Support
For questions and support:
Open an issue on GitHub
Check the detailed USAGE.md documentation
Refer to the original paper for methodological questions