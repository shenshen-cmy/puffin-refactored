# scripts/create_example_data_puffin.py
#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from Bio.Seq import Seq


def generate_biological_sequence(length, gc_content=0.4):
    """Generate DNA sequence with biological characteristics"""
    # Adjust base probabilities based on GC content
    at_prob = (1 - gc_content) / 2
    gc_prob = gc_content / 2

    bases = ['A', 'C', 'G', 'T']
    probs = [at_prob, gc_prob, gc_prob, at_prob]

    sequence = ''.join(np.random.choice(bases, length, p=probs))
    return sequence


def insert_motif(sequence, motif, position):
    """Insert motif at specified position"""
    seq_list = list(sequence)
    for i, base in enumerate(motif):
        if position + i < len(seq_list):
            seq_list[position + i] = base
    return ''.join(seq_list)


def create_gaussian_peak(length, center, width, height):
    """Create Gaussian peak"""
    x = np.arange(length)
    peak = np.exp(-(x - center) ** 2 / (2 * width ** 2))
    return peak * height


def create_realistic_signal(sequence_length, promoter_positions, motif_effects):
    """Create realistic transcription initiation signal"""
    signal = np.zeros(sequence_length)

    # Create signal for each promoter position
    for promoter_pos in promoter_positions:
        # Main TSS peak
        main_peak = create_gaussian_peak(sequence_length, promoter_pos, width=5, height=2.0)
        signal += main_peak

        # Minor TSS peaks (real promoters often have multiple TSS)
        for offset in [-10, -5, 5, 10, 15]:
            minor_pos = promoter_pos + offset
            if 0 <= minor_pos < sequence_length:
                minor_peak = create_gaussian_peak(sequence_length, minor_pos, width=3, height=0.5)
                signal += minor_peak

    # Add motif effects
    for motif_info in motif_effects:
        pos = motif_info['position']
        effect_strength = motif_info['effect']
        effect_width = motif_info.get('width', 50)

        # Motif effects are typically bidirectional
        if effect_strength > 0:
            effect_peak = create_gaussian_peak(sequence_length, pos, width=effect_width, height=effect_strength)
            signal += effect_peak

    # Add background noise
    background = np.random.normal(0, 0.05, sequence_length)
    signal += background

    # Ensure non-negative
    signal = np.clip(signal, 0, None)

    return signal


def create_biological_promoter_sequence(length=2000, gc_content=0.45):
    """Create promoter sequence with real biological features"""

    # Define real promoter motifs
    motifs = {
        'TATA_box': 'TATAAA',
        'Inr': 'YYANWYY',  # Y = C/T, N = any, W = A/T
        'DPE': 'RGWYVT',  # Downstream promoter element
        'GC_box': 'GGCGGG',
        'CAAT_box': 'CCAAT',
        'SP1': 'GGGCGG',
        'NFY': 'CCAAT',
        'YY1': 'CGCCAT',
        'ETS': 'GGAWTS',  # W = A/T, S = C/G
    }

    # Generate base sequence
    sequence = generate_biological_sequence(length, gc_content)

    # Insert motifs at reasonable positions
    motif_positions = {}

    # TATA box typically at -30 position
    tata_pos = length // 2 - 30
    sequence = insert_motif(sequence, motifs['TATA_box'], tata_pos)
    motif_positions['TATA_box'] = tata_pos

    # Inr near transcription start site
    inr_pos = length // 2
    inr_sequence = motifs['Inr'].replace('Y', 'C').replace('Y', 'T').replace('N', 'A').replace('W', 'A')
    sequence = insert_motif(sequence, inr_sequence, inr_pos)
    motif_positions['Inr'] = inr_pos

    # DPE at +30 position
    dpe_pos = length // 2 + 28
    dpe_sequence = motifs['DPE'].replace('R', 'G').replace('W', 'A').replace('Y', 'C').replace('V', 'A').replace('T', 'T')
    sequence = insert_motif(sequence, dpe_sequence, dpe_pos)
    motif_positions['DPE'] = dpe_pos

    # Randomly insert other motifs
    other_motifs = ['GC_box', 'CAAT_box', 'SP1', 'NFY', 'YY1', 'ETS']
    for motif_name in other_motifs:
        if np.random.random() > 0.3:  # 70% probability to insert
            pos = np.random.randint(100, length - 100)
            motif_seq = motifs[motif_name]
            # Handle degenerate bases
            clean_motif = motif_seq.replace('W', np.random.choice(['A', 'T']))
            clean_motif = clean_motif.replace('S', np.random.choice(['C', 'G']))
            clean_motif = clean_motif.replace('R', np.random.choice(['A', 'G']))
            clean_motif = clean_motif.replace('Y', np.random.choice(['C', 'T']))
            clean_motif = clean_motif.replace('K', np.random.choice(['G', 'T']))
            clean_motif = clean_motif.replace('M', np.random.choice(['A', 'C']))
            clean_motif = clean_motif.replace('B', np.random.choice(['C', 'G', 'T']))
            clean_motif = clean_motif.replace('D', np.random.choice(['A', 'G', 'T']))
            clean_motif = clean_motif.replace('H', np.random.choice(['A', 'C', 'T']))
            clean_motif = clean_motif.replace('V', np.random.choice(['A', 'C', 'G']))
            clean_motif = clean_motif.replace('N', np.random.choice(['A', 'C', 'G', 'T']))

            sequence = insert_motif(sequence, clean_motif, pos)
            motif_positions[motif_name] = pos

    return sequence, motif_positions


def create_example_data(output_dir="example_puffin_data", num_sequences=20, sequence_length=2000):
    """Create biologically meaningful example data in TSV format"""

    # Ensure output directory is in project root
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "signals"), exist_ok=True)

    data_records = []

    print("Generating biologically meaningful promoter sequences...")

    chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5']
    gene_types = ['housekeeping', 'tissue_specific', 'developmental', 'stress_response']

    for i in range(num_sequences):
        chrom = np.random.choice(chromosomes)
        gene_type = np.random.choice(gene_types)

        # Adjust GC content based on gene type
        if gene_type == 'housekeeping':
            gc_content = 0.55  # Housekeeping genes typically have higher GC content
        elif gene_type == 'tissue_specific':
            gc_content = 0.45
        else:
            gc_content = 0.5

        # Generate biological sequence
        sequence, motif_positions = create_biological_promoter_sequence(sequence_length, gc_content)

        # Define motif effects
        motif_effects = []

        # TATA box effect - strong but narrow
        if 'TATA_box' in motif_positions:
            motif_effects.append({
                'position': motif_positions['TATA_box'],
                'effect': np.random.uniform(1.5, 3.0),
                'width': 10
            })

        # Inr effect - at TSS position
        if 'Inr' in motif_positions:
            motif_effects.append({
                'position': motif_positions['Inr'],
                'effect': np.random.uniform(1.0, 2.0),
                'width': 5
            })

        # GC box effect - wider region
        if 'GC_box' in motif_positions:
            motif_effects.append({
                'position': motif_positions['GC_box'],
                'effect': np.random.uniform(0.5, 1.5),
                'width': 30
            })

        # Other motif effects
        for motif_name in ['SP1', 'NFY', 'YY1', 'ETS']:
            if motif_name in motif_positions:
                motif_effects.append({
                    'position': motif_positions[motif_name],
                    'effect': np.random.uniform(0.3, 1.2),
                    'width': np.random.randint(20, 60)
                })

        # Main TSS position (near Inr)
        main_tss = sequence_length // 2
        promoter_positions = [main_tss]

        # Add some minor TSS
        for _ in range(np.random.randint(1, 4)):
            offset = np.random.choice([-15, -10, -5, 5, 10, 15])
            promoter_positions.append(main_tss + offset)

        # Create signal
        signal = create_realistic_signal(sequence_length, promoter_positions, motif_effects)

        # Save signal file
        seq_id = f"{chrom}_promoter_{i + 1:03d}_{gene_type}"
        signal_file = f"signals/{seq_id}.npy"
        np.save(os.path.join(output_dir, signal_file), signal.astype(np.float32))

        # Record metadata
        data_records.append({
            'sequence_id': seq_id,
            'sequence': sequence,
            'signal_file': signal_file,
            'length': sequence_length,
            'chromosome': chrom,
            'gene_type': gene_type,
            'start': 0,
            'end': sequence_length,
            'main_tss': main_tss,
            'has_tata': 'TATA_box' in motif_positions,
            'has_gc_box': 'GC_box' in motif_positions,
            'num_motifs': len(motif_positions)
        })

        if (i + 1) % 5 == 0:
            print(f"Generated {i + 1}/{num_sequences} sequences")

    # Save TSV file
    df = pd.DataFrame(data_records)
    output_tsv = os.path.join(output_dir, "training_data.tsv")
    df.to_csv(output_tsv, sep='\t', index=False)

    print(f"Created example data to: {output_dir}/")
    print(f"Contains {len(data_records)} biologically meaningful promoter sequences")
    print("\nSequence statistics:")
    print(f"  Average length: {sequence_length} bp")
    print(f"  Sequences with TATA box: {df['has_tata'].sum()}/{len(df)}")
    print(f"  Sequences with GC box: {df['has_gc_box'].sum()}/{len(df)}")
    print(f"  Average motif count: {df['num_motifs'].mean():.1f}")
    print(f"  Gene type distribution: {df['gene_type'].value_counts().to_dict()}")

    return output_dir


def create_prediction_example(output_dir="prediction_example_puffin"):
    """Create example FASTA file for prediction - using generic descriptive names"""

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create FASTA file
    fasta_file = os.path.join(output_dir, "test_sequences.fasta")

    # Create several test sequences with different characteristics
    test_sequences = []

    # Use generic descriptions, avoid specific motif names
    sequence1, features1 = create_biological_promoter_sequence(1000, gc_content=0.4)
    test_sequences.append(("simulated_strong_promoter_001", sequence1))

    sequence2, features2 = create_biological_promoter_sequence(1000, gc_content=0.65)
    test_sequences.append(("simulated_GCrich_promoter_002", sequence2))

    sequence3, features3 = create_biological_promoter_sequence(1000, gc_content=0.45)
    test_sequences.append(("simulated_weak_promoter_003", sequence3))

    sequence4, features4 = create_biological_promoter_sequence(1000, gc_content=0.5)
    test_sequences.append(("simulated_variable_promoter_004", sequence4))

    with open(fasta_file, 'w') as f:
        for name, seq in test_sequences:
            f.write(f">{name}\n")
            # 80 characters per line
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")

    print(f"Created prediction example file to: {fasta_file}")
    print("Contains 4 simulated promoter sequences:")
    print("  1. simulated_strong_promoter_001")
    print("  2. simulated_GCrich_promoter_002")
    print("  3. simulated_weak_promoter_003")
    print("  4. simulated_variable_promoter_004")

    return fasta_file


def analyze_biological_features(sequence):
    """Analyze biological features of sequence"""
    features = {}

    # GC content
    gc_count = sequence.count('G') + sequence.count('C')
    features['gc_content'] = gc_count / len(sequence)

    # Detect common motifs
    motifs = {
        'TATA_box': 'TATAAA',
        'GC_box': 'GGCGGG',
        'CAAT_box': 'CCAAT',
        'Inr_like': 'YYANWYY',
    }

    for motif_name, motif_seq in motifs.items():
        if motif_seq in sequence:
            features[motif_name] = True
        else:
            features[motif_name] = False

    return features


if __name__ == "__main__":
    print("=" * 60)
    print("Generating biologically meaningful training and test data")
    print("=" * 60)

    # Create training data example
    train_dir = create_example_data("example_puffin_data", num_sequences=30, sequence_length=2000)

    # Create prediction data example
    pred_file = create_prediction_example()

    print("\n" + "=" * 60)
    print("Usage instructions:")
    print("=" * 60)
    print("1. Training data location:", train_dir)
    print("2. Prediction data location:", pred_file)
    print("3. Modify config/train_puffin.yaml data/train_tsv path to:",
          os.path.join(train_dir, "training_data.tsv"))
    print("\nData characteristics:")
    print("  • Contains real promoter motifs (TATA box, Inr, GC box, etc.)")
    print("  • Simulates GC content features for different gene types")
    print("  • Contains multiple transcription start sites")
    print("  • Signals include motif position effects")
    print("  • Suitable for training biologically meaningful models")