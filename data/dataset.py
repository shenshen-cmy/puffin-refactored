# data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split


class SequenceDataset(Dataset):
    """Base sequence dataset for DNA sequences and transcription signals"""

    def __init__(self, sequences, signals):
        self.sequences = sequences
        self.signals = signals

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        signal = self.signals[idx]

        # Convert sequence to one-hot encoding
        sequence_encoded = self.sequence_to_onehot(sequence)

        return torch.FloatTensor(sequence_encoded), torch.FloatTensor(signal)

    def sequence_to_onehot(self, sequence):
        """Convert DNA sequence to one-hot encoding"""
        base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        onehot = np.zeros((4, len(sequence)), dtype=np.float32)

        for i, base in enumerate(sequence.upper()):
            if base in base_to_index:
                onehot[base_to_index[base], i] = 1.0

        return onehot


class TSVDataset(SequenceDataset):
    """Unified dataset based on TSV files"""

    def __init__(self, tsv_file, data_dir="", seq_length=650, overlap=325, max_samples=None):
        self.tsv_file = tsv_file
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.overlap = overlap

        # Load and process data
        sequences, signals = self.load_and_process_data()

        if max_samples:
            sequences = sequences[:max_samples]
            signals = signals[:max_samples]

        super().__init__(sequences, signals)

    def load_and_process_data(self):
        """Load and process data from TSV file"""
        sequences = []
        signals = []

        # Read TSV file
        metadata = pd.read_csv(self.tsv_file, sep='\t')

        print(f"Loaded {len(metadata)} entries from {self.tsv_file}")

        for idx, row in metadata.iterrows():
            sequence_id = row.get('sequence_id', f'entry_{idx}')
            sequence = row['sequence']

            # Load signal data
            if 'signal_file' in row and pd.notna(row['signal_file']):
                signal_file = os.path.join(self.data_dir, row['signal_file'])
                try:
                    signal = np.load(signal_file)
                except Exception as e:
                    print(f"Warning: Could not load signal file {signal_file}: {e}")
                    continue
            elif 'signal' in row and pd.notna(row['signal']):
                # If signal is stored directly in TSV
                try:
                    if isinstance(row['signal'], str):
                        signal = np.array(eval(row['signal']))
                    else:
                        signal = np.array(row['signal'])
                except:
                    print(f"Warning: Could not parse signal for {sequence_id}")
                    continue
            else:
                print(f"Warning: No signal data for {sequence_id}")
                continue

            # Validate data integrity
            if len(sequence) != len(signal):
                print(f"Warning: Sequence and signal length mismatch for {sequence_id}")
                continue

            # Use sliding window to split data
            seq_fragments, signal_fragments = self.sliding_window(
                sequence, signal, self.seq_length, self.overlap
            )

            sequences.extend(seq_fragments)
            signals.extend(signal_fragments)

        print(f"Generated {len(sequences)} fragments from {len(metadata)} entries")
        return sequences, signals

    def sliding_window(self, sequence, signal, window_size, overlap):
        """Split sequence and signal using sliding window"""
        sequences = []
        signals = []

        step_size = window_size - overlap
        seq_len = len(sequence)

        # Ensure sequence is long enough
        if seq_len < window_size:
            print(f"Warning: Sequence too short for window size {window_size}")
            return sequences, signals

        for start in range(0, seq_len - window_size + 1, step_size):
            end = start + window_size

            seq_fragment = sequence[start:end]
            signal_fragment = signal[start:end]

            # Ensure length matches
            if len(seq_fragment) == window_size and len(signal_fragment) == window_size:
                sequences.append(seq_fragment)
                signals.append(signal_fragment)

        return sequences, signals

    def shuffle(self):
        """Shuffle the dataset"""
        combined = list(zip(self.sequences, self.signals))
        random.shuffle(combined)
        self.sequences, self.signals = zip(*combined)
        self.sequences = list(self.sequences)
        self.signals = list(self.signals)


def create_data_loaders(train_tsv, val_tsv=None, config=None, auto_split=False, split_ratio=0.8, split_by='random',
                        random_seed=42):
    """Create training and validation data loaders with support for automatic and manual splitting"""
    from torch.utils.data import DataLoader

    # Set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if auto_split:
        # Automatic splitting mode
        print("Using automatic data splitting...")
        print(f"Split ratio: {split_ratio}, Split method: {split_by}")

        # Load full data
        full_dataset = TSVDataset(
            train_tsv,
            data_dir=os.path.dirname(train_tsv),
            seq_length=config['data']['sequence_length'],
            overlap=config['data']['overlap']
        )

        # Split data based on strategy
        if split_by == 'random':
            # Random split
            train_size = int(split_ratio * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            print(f"Random split: {len(train_dataset)} training, {len(val_dataset)} validation samples")

        elif split_by == 'chromosome':
            # Split by chromosome - requires chromosome info from original TSV
            metadata = pd.read_csv(train_tsv, sep='\t')

            if 'chromosome' not in metadata.columns:
                print("Warning: 'chromosome' column not found, falling back to random split")
                train_size = int(split_ratio * len(full_dataset))
                val_size = len(full_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(
                    full_dataset, [train_size, val_size]
                )
            else:
                chromosomes = metadata['chromosome'].unique()
                print(f"Found {len(chromosomes)} chromosomes: {chromosomes}")

                # Randomly select chromosomes for validation set
                num_val_chromosomes = max(1, int(len(chromosomes) * (1 - split_ratio)))
                val_chromosomes = set(random.sample(list(chromosomes), num_val_chromosomes))

                # Split dataset by chromosome
                train_indices = []
                val_indices = []

                # Build index mapping: TSV entries -> sliding window fragments
                fragment_to_entry = []
                current_idx = 0

                for idx, row in metadata.iterrows():
                    sequence = row['sequence']
                    step_size = config['data']['sequence_length'] - config['data']['overlap']
                    num_fragments = max(1, (len(sequence) - config['data']['sequence_length']) // step_size + 1)

                    chrom = row.get('chromosome', 'unknown')
                    if chrom in val_chromosomes:
                        # All fragments from this entry go to validation set
                        val_indices.extend(range(current_idx, current_idx + num_fragments))
                    else:
                        # All fragments from this entry go to training set
                        train_indices.extend(range(current_idx, current_idx + num_fragments))

                    current_idx += num_fragments

                train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
                val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

                print(f"Chromosome split: {len(train_dataset)} training, {len(val_dataset)} validation samples")
                print(
                    f"Training chromosomes: {set(metadata[~metadata['chromosome'].isin(val_chromosomes)]['chromosome'].unique())}")
                print(f"Validation chromosomes: {val_chromosomes}")

    else:
        # Manual splitting mode
        print("Using manual data splitting...")
        train_dataset = TSVDataset(
            train_tsv,
            data_dir=os.path.dirname(train_tsv),
            seq_length=config['data']['sequence_length'],
            overlap=config['data']['overlap']
        )

        if val_tsv is None:
            raise ValueError("In manual split mode, val_tsv must be provided")

        val_dataset = TSVDataset(
            val_tsv,
            data_dir=os.path.dirname(val_tsv),
            seq_length=config['data']['sequence_length'],
            overlap=config['data']['overlap']
        )

        print(f"Manual split: {len(train_dataset)} training, {len(val_dataset)} validation samples")

    # Shuffle training set
    if not auto_split or split_by == 'random':
        # For manual split or random auto split, shuffle training set
        if hasattr(train_dataset, 'shuffle'):
            # If it's a direct TSVDataset instance, can shuffle
            train_dataset.shuffle()
        elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'shuffle'):
            # If it's a Subset, shuffle underlying dataset
            train_dataset.dataset.shuffle()

    # For chromosome split, already grouped by chromosome, no additional shuffling needed

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 2)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 2)
    )

    return train_loader, val_loader