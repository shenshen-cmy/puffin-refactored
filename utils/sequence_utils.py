# utils/sequence_utils.py
import numpy as np
import torch


def sequence_to_onehot(sequence):
    """
    Convert DNA sequence to one-hot encoding

    Args:
        sequence: DNA sequence string

    Returns:
        onehot: 4 x L numpy array
    """
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    onehot = np.zeros((4, len(sequence)), dtype=np.float32)

    for i, base in enumerate(sequence.upper()):
        if base in base_to_index:
            onehot[base_to_index[base], i] = 1.0

    return onehot


def reverse_complement(sequence):
    """
    Get reverse complement of DNA sequence

    Args:
        sequence: DNA sequence string

    Returns:
        rev_comp: Reverse complement sequence
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence))


def validate_sequence(sequence):
    """
    Validate if DNA sequence contains only valid bases

    Args:
        sequence: DNA sequence string

    Returns:
        valid: Whether sequence is valid
    """
    valid_bases = {'A', 'C', 'G', 'T', 'a', 'c', 'g', 't', 'N', 'n'}
    return all(base in valid_bases for base in sequence)


def calculate_gc_content(sequence):
    """
    Calculate GC content of sequence

    Args:
        sequence: DNA sequence string

    Returns:
        gc_content: GC content (0-1)
    """
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence) if len(sequence) > 0 else 0


def pad_sequence(sequence, target_length, pad_char='N'):
    """
    Pad sequence to target length

    Args:
        sequence: DNA sequence string
        target_length: Target length
        pad_char: Padding character

    Returns:
        padded_sequence: Padded sequence
    """
    if len(sequence) >= target_length:
        return sequence[:target_length]
    else:
        return sequence + pad_char * (target_length - len(sequence))


def random_dna_sequence(length):
    """
    Generate random DNA sequence

    Args:
        length: Sequence length

    Returns:
        sequence: Random DNA sequence
    """
    bases = ['A', 'C', 'G', 'T']
    return ''.join(np.random.choice(bases) for _ in range(length))


def sequences_to_tensor(sequences, device='cpu'):
    """
    Convert multiple sequences to tensor in batch

    Args:
        sequences: List of sequences
        device: Device

    Returns:
        tensor: Batch sequence tensor
    """
    encoded_sequences = [sequence_to_onehot(seq) for seq in sequences]
    tensor = torch.FloatTensor(np.stack(encoded_sequences))

    if device == 'cuda':
        tensor = tensor.cuda()

    return tensor