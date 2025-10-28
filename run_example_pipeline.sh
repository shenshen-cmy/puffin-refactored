#!/bin/bash

# Puffin Complete Example Pipeline
# This script demonstrates the complete workflow from data generation to analysis

set -e  # Exit on error

echo "==========================================="
echo "Puffin Complete Example Pipeline"
echo "==========================================="

# Configuration
PUFFIN_DATA_DIR="example_puffin_data"
PUFFIN_D_DATA_DIR="example_puffin_d_data"
MODEL_DIR="models/saved"
RESULTS_DIR="pipeline_results"
PLOTS_DIR="${RESULTS_DIR}/plots"

# Create directories
echo "Creating directories..."
mkdir -p ${MODEL_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p ${PLOTS_DIR}

# Step 1: Generate Example Data
echo "Step 1: Generating example data..."
echo "-------------------------------------------"

echo "Generating Puffin data (650bp sequences)..."
python scripts/create_example_data_puffin.py \
    --output_dir ${PUFFIN_DATA_DIR} \
    --num_sequences 30 \
    --sequence_length 2000

echo "Generating Puffin-D data (100kb sequences)..."
python scripts/create_example_data_puffin_d.py \
    --output_dir ${PUFFIN_D_DATA_DIR} \
    --num_sequences 8 \
    --sequence_length 500000

# Step 2: Train Models
echo ""
echo "Step 2: Training models..."
echo "-------------------------------------------"

echo "Training Puffin model..."
python scripts/train_puffin.py \
    --config config/train_puffin.yaml \
    --auto_split \
    --split_ratio 0.8 \
    --split_by random \
    --random_seed 42

echo "Training Puffin-D model..."
python scripts/train_puffin_d.py \
    --config config/train_puffin_d.yaml \
    --auto_split \
    --split_ratio 0.8 \
    --split_by random \
    --random_seed 42

# Step 3: Run Predictions and Analysis
echo ""
echo "Step 3: Running predictions and analysis..."
echo "-------------------------------------------"

# Check if models were created
if [ ! -f "${MODEL_DIR}/puffin_best_model.pth" ]; then
    echo "Error: Puffin model not found. Training may have failed."
    exit 1
fi

if [ ! -f "${MODEL_DIR}/puffin_d_best_model.pth" ]; then
    echo "Error: Puffin-D model not found. Training may have failed."
    exit 1
fi

echo "Running Puffin analysis with interpretation..."
python scripts/puffin.py \
    --model ${MODEL_DIR}/puffin_best_model.pth \
    --sequences prediction_example_puffin/test_sequences.fasta \
    --output ${RESULTS_DIR}/puffin_analysis.csv

echo "Running Puffin-D prediction..."
python scripts/predict_puffin_d.py \
    --model ${MODEL_DIR}/puffin_d_best_model.pth \
    --sequences prediction_example_puffin_d/test_sequences.fasta \
    --output ${RESULTS_DIR}/puffin_d_predictions.csv

# Step 4: Generate Visualizations
echo ""
echo "Step 4: Generating visualizations..."
echo "-------------------------------------------"

echo "Creating Puffin analysis plots..."
python scripts/plot_puffin.py \
    --input ${RESULTS_DIR}/puffin_analysis.csv \
    --output_dir ${PLOTS_DIR} \
    --plot_type all

# Step 5: Generate Summary Report
echo ""
echo "Step 5: Generating summary report..."
echo "-------------------------------------------"

# Create summary file
SUMMARY_FILE="${RESULTS_DIR}/pipeline_summary.txt"

cat > ${SUMMARY_FILE} << EOF
Puffin Pipeline Summary
=======================

Generated: $(date)

Output Files:
-------------
1. Models:
   - Puffin: ${MODEL_DIR}/puffin_best_model.pth
   - Puffin-D: ${MODEL_DIR}/puffin_d_best_model.pth

2. Results:
   - Puffin analysis: ${RESULTS_DIR}/puffin_analysis.csv
   - Puffin-D predictions: ${RESULTS_DIR}/puffin_d_predictions.csv
   - Puffin-D summary: ${RESULTS_DIR}/puffin_d_predictions_summary.csv

3. Visualizations: ${PLOTS_DIR}/
   - Transcription profiles
   - Contribution breakdowns
   - Motif effect plots
   - Contribution heatmaps
   - Interpretation summaries

Next Steps:
-----------
1. Examine the plots in ${PLOTS_DIR}/
2. Check the analysis results in ${RESULTS_DIR}/
3. Use trained models on your own data
4. Refer to USAGE.md for advanced usage

Important Notes:
----------------
- This demonstrates the methodology for interpretable analysis
- Models learn generic patterns (not real biological motifs)
- For real motif discovery, implement multi-stage training as in original paper

EOF

echo "Pipeline completed successfully!"
echo "Summary saved to: ${SUMMARY_FILE}"
echo ""
echo "Generated files:"
echo "- Models: ${MODEL_DIR}/"
echo "- Results: ${RESULTS_DIR}/"
echo "- Plots: ${PLOTS_DIR}/"
echo "- Summary: ${SUMMARY_FILE}"

# Display Puffin-D prediction summary if available
if [ -f "${RESULTS_DIR}/puffin_d_predictions_summary.csv" ]; then
    echo ""
    echo "Puffin-D Prediction Summary:"
    echo "----------------------------"
    cat ${RESULTS_DIR}/puffin_d_predictions_summary.csv
fi

echo ""
echo "==========================================="
echo "Pipeline completed! Check results above."
echo "==========================================="