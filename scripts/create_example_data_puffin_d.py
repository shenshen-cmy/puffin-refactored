# scripts/create_example_data_puffin_d.py
import pandas as pd
import numpy as np
import os
from Bio.Seq import Seq


def generate_long_biological_sequence(length, gc_content=0.4, variation_interval=5000):
    """Generate long biological sequence with regional variations"""
    sequence = ""

    # Generate in segments with varying GC content
    segment_length = min(variation_interval, length // 10)
    remaining_length = length

    while remaining_length > 0:
        current_length = min(segment_length, remaining_length)

        # Slightly varying GC content for each segment
        segment_gc = max(0.3, min(0.7, gc_content + np.random.normal(0, 0.1)))

        # Adjust base probabilities based on GC content
        at_prob = (1 - segment_gc) / 2
        gc_prob = segment_gc / 2

        bases = ['A', 'C', 'G', 'T']
        probs = [at_prob, gc_prob, gc_prob, at_prob]

        segment = ''.join(np.random.choice(bases, current_length, p=probs))
        sequence += segment
        remaining_length -= current_length

    return sequence


def insert_motif_clusters(sequence, num_clusters=50):
    """Insert motif clusters in long sequence"""
    sequence_length = len(sequence)
    motif_pool = [
        'TATAAA', 'CCAAT', 'GGCGGG', 'GGGCGG', 'CGCCAT', 'GGAATG',
        'ATTGGG', 'CCGCCC', 'ATGCAA', 'GGGGGG', 'CCCCCC', 'AAAAAA',
        'TTTTTT', 'CACGTG', 'GGCCGG', 'AAGTGA', 'TCGCGA', 'GATTGG'
    ]

    motif_positions = {}

    # Dynamically adjust cluster center range based on sequence length
    min_cluster_center = 10000  # Minimum start position
    max_cluster_center = sequence_length - 10000  # Maximum start position

    # Ensure enough space to insert clusters
    if max_cluster_center <= min_cluster_center:
        # If sequence is too short, reduce cluster count or adjust range
        num_clusters = min(num_clusters, 5)
        max_cluster_center = sequence_length - 1000
        min_cluster_center = 1000

    for cluster_id in range(num_clusters):
        # Select cluster center position
        cluster_center = np.random.randint(min_cluster_center, max_cluster_center)

        # Insert 3-8 motifs in cluster
        num_motifs_in_cluster = np.random.randint(3, 9)

        for i in range(num_motifs_in_cluster):
            motif = np.random.choice(motif_pool)
            # Insert at random position near cluster center
            offset = np.random.randint(-2000, 2000)
            position = cluster_center + offset

            # Ensure position is valid
            if 0 <= position < sequence_length - len(motif):
                # Insert motif
                seq_list = list(sequence)
                for j, base in enumerate(motif):
                    if position + j < len(seq_list):
                        seq_list[position + j] = base
                sequence = ''.join(seq_list)

                motif_name = f"cluster_{cluster_id}_motif_{i}"
                motif_positions[motif_name] = position

    return sequence, motif_positions


def create_long_signal(sequence_length, active_regions):
    """Create transcription initiation signal for long sequence"""
    signal = np.zeros(sequence_length)

    # Add background signal (simulating open chromatin regions)
    for region in active_regions:
        start, end, strength = region
        region_length = end - start

        # Create broad peaks representing active regions
        x = np.arange(sequence_length)
        region_center = (start + end) // 2
        region_signal = np.exp(-(x - region_center) ** 2 / (2 * (region_length / 4) ** 2))
        signal += region_signal * strength

    # Add specific TSS peaks within active regions
    for region in active_regions:
        start, end, strength = region
        # Add multiple TSS within region
        num_tss = np.random.randint(3, 8)
        for _ in range(num_tss):
            tss_pos = np.random.randint(start, end)
            tss_width = np.random.randint(3, 10)
            tss_strength = strength * np.random.uniform(0.5, 2.0)

            tss_signal = np.exp(-(np.arange(sequence_length) - tss_pos) ** 2 / (2 * tss_width ** 2))
            signal += tss_signal * tss_strength

    # Add long-range correlation noise
    long_noise = np.random.normal(0, 0.02, sequence_length)
    # Smooth noise
    try:
        from scipy.ndimage import gaussian_filter1d
        long_noise = gaussian_filter1d(long_noise, sigma=100)
    except ImportError:
        # If scipy not available, use simple moving average
        window_size = 100
        long_noise_smooth = np.convolve(long_noise, np.ones(window_size) / window_size, mode='same')
        long_noise = long_noise_smooth

    signal += long_noise

    # Add short noise
    short_noise = np.random.normal(0, 0.05, sequence_length)
    signal += short_noise

    # Ensure non-negative
    signal = np.clip(signal, 0, None)

    return signal


def create_puffin_d_example_data(output_dir="example_puffin_d_data", num_sequences=10, sequence_length=500000):
    """Create long sequence example data specifically for Puffin_D"""

    # Ensure output directory is in project root
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "signals"), exist_ok=True)

    data_records = []

    print("Generating long sequence data for Puffin_D...")
    print(f"Sequence length: {sequence_length:,} bp")

    chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10']
    genomic_contexts = ['gene_rich', 'gene_poor', 'heterochromatin', 'euchromatin']

    for i in range(num_sequences):
        chrom = np.random.choice(chromosomes)
        context = np.random.choice(genomic_contexts)

        # Adjust GC content based on genomic context
        if context == 'gene_rich':
            gc_content = 0.45
            num_active_regions = np.random.randint(8, 15)
        elif context == 'gene_poor':
            gc_content = 0.38
            num_active_regions = np.random.randint(2, 6)
        elif context == 'heterochromatin':
            gc_content = 0.35
            num_active_regions = np.random.randint(1, 4)
        else:  # euchromatin
            gc_content = 0.42
            num_active_regions = np.random.randint(5, 10)

        # Generate long sequence
        sequence = generate_long_biological_sequence(sequence_length, gc_content)

        # Insert motif clusters
        sequence, motif_positions = insert_motif_clusters(sequence)

        # Define active regions
        active_regions = []
        for _ in range(num_active_regions):
            region_start = np.random.randint(50000, sequence_length - 100000)
            region_length = np.random.randint(20000, 80000)
            region_end = min(region_start + region_length, sequence_length - 1)
            region_strength = np.random.uniform(0.5, 3.0)
            active_regions.append((region_start, region_end, region_strength))

        # Create signal
        signal = create_long_signal(sequence_length, active_regions)

        # Save signal file
        seq_id = f"{chrom}_long_region_{i + 1:03d}_{context}"
        signal_file = f"signals/{seq_id}.npy"
        np.save(os.path.join(output_dir, signal_file), signal.astype(np.float32))

        # Record metadata
        data_records.append({
            'sequence_id': seq_id,
            'sequence': sequence,
            'signal_file': signal_file,
            'length': sequence_length,
            'chromosome': chrom,
            'genomic_context': context,
            'num_active_regions': num_active_regions,
            'num_motif_clusters': len(motif_positions) // 10,  # Approximate cluster count
            'gc_content': gc_content
        })

        if (i + 1) % 2 == 0:
            print(f"Generated {i + 1}/{num_sequences} long sequences")

    # Save TSV file
    df = pd.DataFrame(data_records)
    output_tsv = os.path.join(output_dir, "training_data.tsv")
    df.to_csv(output_tsv, sep='\t', index=False)

    print(f"Created Puffin_D example data to: {output_dir}/")
    print(f"Contains {len(data_records)} long sequences")
    print("\nSequence statistics:")
    print(f"  Sequence length: {sequence_length:,} bp")
    print(f"  Total data volume: {sequence_length * len(data_records):,} bp")
    print(f"  Genomic context distribution: {df['genomic_context'].value_counts().to_dict()}")
    print(f"  Average active regions: {df['num_active_regions'].mean():.1f}")
    print(f"  Average motif clusters: {df['num_motif_clusters'].mean():.1f}")

    return output_dir


def create_puffin_d_prediction_example(output_dir="prediction_example_puffin_d"):
    """Create prediction example files specifically for Puffin_D"""

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create FASTA file
    fasta_file = os.path.join(output_dir, "test_sequences.fasta")

    # Create several 100kb test sequences
    test_sequences = []

    # Generate 4 test sequences of 100kb each
    for i in range(4):
        context_types = ['gene_rich', 'gene_poor', 'mixed', 'variable']
        context = context_types[i]

        sequence_length = 100000  # 100kb

        # Generate long sequence
        sequence = generate_long_biological_sequence(sequence_length, gc_content=0.42)

        # Adjust cluster count for 100kb sequences
        sequence, _ = insert_motif_clusters(sequence, num_clusters=10)  # Reduced cluster count

        test_sequences.append((f"puffin_d_test_sequence_{i + 1:02d}_{context}", sequence))

    with open(fasta_file, 'w') as f:
        for name, seq in test_sequences:
            f.write(f">{name}\n")
            # 80 characters per line
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")

    print(f"Created Puffin_D prediction example file to: {fasta_file}")
    print("Contains 4 test sequences of 100kb:")
    for name, _ in test_sequences:
        print(f"  - {name}")

    return fasta_file


if __name__ == "__main__":
    print("=" * 60)
    print("Generating long sequence training and test data for Puffin_D")
    print("=" * 60)

    # Create training data example - 500kb long sequences
    train_dir = create_puffin_d_example_data("example_puffin_d_data", num_sequences=8, sequence_length=500000)

    # Create prediction data example - 100kb sequences
    pred_file = create_puffin_d_prediction_example()

    print("\n" + "=" * 60)
    print("Usage instructions:")
    print("=" * 60)
    print("1. Puffin_D training data location:", train_dir)
    print("2. Puffin_D prediction data location:", pred_file)
    print("3. Modify config/train_puffin_d.yaml data/train_tsv path to:",
          os.path.join(train_dir, "training_data.tsv"))
    print("\nData characteristics:")
    print("  • Training data: 500kb long sequences")
    print("  • Test data: 100kb sequences")
    print("  • Contains motif clusters and active regions")
    print("  • Simulates different genomic environments")
    print("  • Suitable for training Puffin_D model to capture long-range dependencies")