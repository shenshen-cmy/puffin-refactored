# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


def plot_transcription_profile(prediction, sequence=None, title="Transcription Profile", save_path=None):
    """Plot transcription initiation signal profile"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # Ensure prediction is numeric type
    try:
        if hasattr(prediction, 'values'):
            prediction = prediction.values
        prediction = np.array(prediction, dtype=float)

        # Check if there is valid data
        if len(prediction) == 0 or np.all(np.isnan(prediction)):
            print("Warning: No valid prediction data for plotting")
            return fig
    except Exception as e:
        print(f"Warning: Error processing prediction data: {e}")
        return fig

    # Plot prediction signal
    ax.plot(prediction, 'b-', linewidth=2, label='Prediction')
    ax.set_xlabel('Position (bp)', fontsize=12)
    ax.set_ylabel('Transcription Signal', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Mark peak position
    if len(prediction) > 0 and not np.all(np.isnan(prediction)):
        valid_indices = ~np.isnan(prediction)
        if np.any(valid_indices):
            valid_prediction = prediction[valid_indices]
            peak_pos = np.argmax(valid_prediction)
            peak_val = valid_prediction[peak_pos]

            # Only mark if within reasonable range
            if peak_pos < len(valid_prediction) - 50:
                ax.annotate(f'TSS: {peak_pos}bp',
                            xy=(peak_pos, peak_val),
                            xytext=(peak_pos + 50, peak_val * 0.8),
                            arrowprops=dict(arrowstyle='->', color='red'))

    # If sequence is provided, display first 100bp at top
    if sequence is not None:
        display_seq = sequence[:100] + "..." if len(sequence) > 100 else sequence
        ax.text(0.02, 0.98, f'Sequence: {display_seq}', transform=ax.transAxes,
                fontfamily='monospace', fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig


def plot_contribution_breakdown(interpretation_df, sequence=None, title="Contribution Breakdown", save_path=None):
    """Plot contribution breakdown with data compatibility fixes"""
    # Ensure there is data
    if interpretation_df is None or interpretation_df.empty:
        print("Warning: No data for contribution breakdown plot")
        return None

    # Check for required data columns
    required_columns = ['Basepair contribution score to transcription initiation']
    available_columns = [col for col in required_columns if col in interpretation_df.index]

    if not available_columns:
        print("Warning: No contribution data available for plotting")
        return None

    if sequence is not None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2, 2]})
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes = [axes[0], axes[1]]

    current_ax = 0

    # Plot sequence (if provided)
    if sequence is not None:
        axes[0].text(0.02, 0.5, sequence, fontfamily='monospace', fontsize=6,
                     ha='left', va='center', transform=axes[0].transAxes)
        axes[0].set_title('DNA Sequence', fontsize=10)
        axes[0].axis('off')
        current_ax += 1

    # Plot total effect
    effect_columns = ['Sum of total effect', 'Prediction']
    effect_col = None
    for col in effect_columns:
        if col in interpretation_df.index:
            effect_col = col
            break

    if effect_col:
        try:
            total_effect = interpretation_df.loc[effect_col].values
            total_effect = np.array(total_effect, dtype=float)

            # Check for valid data
            if len(total_effect) > 0 and not np.all(np.isnan(total_effect)):
                axes[current_ax].plot(total_effect, linewidth=2, color='black', label='Total Effect')
                axes[current_ax].set_ylabel('Total Effect', fontsize=12)
                axes[current_ax].legend()
                axes[current_ax].grid(True, alpha=0.3)
                current_ax += 1
        except Exception as e:
            print(f"Warning: Error plotting total effect: {e}")

    # Plot basepair contributions
    contribution_col = 'Basepair contribution score to transcription initiation'
    if contribution_col in interpretation_df.index:
        try:
            basepair_scores = interpretation_df.loc[contribution_col].values
            basepair_scores = np.array(basepair_scores, dtype=float)

            # Check for valid data
            if len(basepair_scores) > 0 and not np.all(np.isnan(basepair_scores)):
                positions = np.arange(len(basepair_scores))
                colors = ['red' if x < 0 else 'blue' for x in basepair_scores]
                axes[current_ax].bar(positions, basepair_scores, color=colors, alpha=0.7, width=1.0)
                axes[current_ax].set_xlabel('Position (bp)', fontsize=12)
                axes[current_ax].set_ylabel('Basepair Contribution', fontsize=12)
                axes[current_ax].grid(True, alpha=0.3)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='blue', label='Positive Contribution (Promotes)'),
                    Patch(facecolor='red', label='Negative Contribution (Inhibits)')
                ]
                axes[current_ax].legend(handles=legend_elements)
        except Exception as e:
            print(f"Warning: Error plotting basepair contribution: {e}")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig


def plot_motif_contributions(interpretation_df, title="Motif Contributions", save_path=None):
    """Plot individual motif contributions"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Get all motif effect data
    motif_effect_cols = [col for col in interpretation_df.index
                         if 'motif effect' in col and 'Basepair' not in col and 'Learned_Motif' in col]

    if not motif_effect_cols:
        print("Warning: No motif effect data available for plotting")
        ax.text(0.5, 0.5, 'No motif effect data available',
                ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
        return fig

    plotted_count = 0
    for col in motif_effect_cols:
        try:
            motif_name = col.replace(' motif effect', '')
            effects = interpretation_df.loc[col].values
            effects = np.array(effects, dtype=float)

            # Check for valid data
            if len(effects) > 0 and not np.all(np.isnan(effects)):
                ax.plot(effects, label=motif_name, linewidth=1.5, alpha=0.8)
                plotted_count += 1
        except Exception as e:
            print(f"Warning: Error plotting motif {col}: {e}")

    if plotted_count > 0:
        ax.set_xlabel('Position (bp)', fontsize=12)
        ax.set_ylabel('Motif Effect', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No valid motif effect data',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig


def plot_contribution_heatmap(interpretation_df, title="Motif Contribution Heatmap", save_path=None):
    """Plot motif contribution heatmap"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Get motif contribution data
    motif_contr_cols = [col for col in interpretation_df.index
                        if 'Basepair contribution score to transcription initiation' in col
                        and 'Learned_Motif' in col]

    motif_data = []
    motif_names = []

    for col in motif_contr_cols:
        try:
            motif_name = col.replace(' Basepair contribution score to transcription initiation', '')
            contribution = interpretation_df.loc[col].values
            contribution = np.array(contribution, dtype=float)

            # Check for valid data
            if len(contribution) > 0 and not np.all(np.isnan(contribution)):
                motif_data.append(contribution)
                motif_names.append(motif_name)
        except Exception as e:
            print(f"Warning: Error processing motif {col}: {e}")

    if motif_data:
        try:
            # Create heatmap
            heatmap_data = np.array(motif_data)
            im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                           interpolation='nearest')
            ax.set_yticks(range(len(motif_names)))
            ax.set_yticklabels(motif_names)
            ax.set_xlabel('Position (bp)', fontsize=12)
            ax.set_ylabel('Motifs', fontsize=12)
            ax.set_title(title, fontsize=14)

            plt.colorbar(im, ax=ax, label='Contribution Score')
        except Exception as e:
            print(f"Warning: Error creating heatmap: {e}")
            ax.text(0.5, 0.5, 'Error creating heatmap',
                    ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No motif contribution data available',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig


def plot_interpretation_summary(interpretation_df, title="Interpretation Summary", save_path=None):
    """Plot complete interpretability analysis summary"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Subplot 1: Prediction signal
    if 'Prediction' in interpretation_df.index:
        try:
            prediction = interpretation_df.loc['Prediction'].values
            prediction = np.array(prediction, dtype=float)
            if len(prediction) > 0 and not np.all(np.isnan(prediction)):
                axes[0].plot(prediction, 'b-', linewidth=2)
                axes[0].set_ylabel('Prediction', fontsize=12)
                axes[0].set_title('Transcription Initiation Signal', fontsize=12)
                axes[0].grid(True, alpha=0.3)
        except Exception as e:
            print(f"Warning: Error plotting prediction: {e}")

    # Subplot 2: Total effect or contribution
    effect_columns = ['Sum of total effect', 'Basepair contribution score to transcription initiation']
    effect_data = None

    for col in effect_columns:
        if col in interpretation_df.index:
            try:
                effect_data = interpretation_df.loc[col].values
                effect_data = np.array(effect_data, dtype=float)
                if len(effect_data) > 0 and not np.all(np.isnan(effect_data)):
                    axes[1].plot(effect_data, 'k-', linewidth=2)
                    axes[1].set_ylabel(col, fontsize=12)
                    axes[1].set_title(f'{col}', fontsize=12)
                    axes[1].grid(True, alpha=0.3)
                    break
            except Exception as e:
                print(f"Warning: Error plotting {col}: {e}")

    # Subplot 3: Motif contribution heatmap
    motif_contr_cols = [col for col in interpretation_df.index
                        if 'Basepair contribution score to transcription initiation' in col
                        and 'Learned_Motif' in col]

    if motif_contr_cols:
        try:
            motif_data = []
            motif_names = []

            for col in motif_contr_cols:
                motif_name = col.replace(' Basepair contribution score to transcription initiation', '')
                contribution = interpretation_df.loc[col].values
                contribution = np.array(contribution, dtype=float)
                if len(contribution) > 0 and not np.all(np.isnan(contribution)):
                    motif_data.append(contribution)
                    motif_names.append(motif_name)

            if motif_data:
                heatmap_data = np.array(motif_data)
                im = axes[2].imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                                    interpolation='nearest')
                axes[2].set_yticks(range(len(motif_names)))
                axes[2].set_yticklabels(motif_names)
                axes[2].set_xlabel('Position (bp)', fontsize=12)
                axes[2].set_ylabel('Motifs', fontsize=12)
                axes[2].set_title('Motif Contributions (Heatmap)', fontsize=12)
                plt.colorbar(im, ax=axes[2])
            else:
                axes[2].text(0.5, 0.5, 'No valid motif contribution data',
                             ha='center', va='center', transform=axes[2].transAxes)
        except Exception as e:
            print(f"Warning: Error creating motif heatmap: {e}")
            axes[2].text(0.5, 0.5, 'Error creating motif heatmap',
                         ha='center', va='center', transform=axes[2].transAxes)
    else:
        axes[2].text(0.5, 0.5, 'No motif contribution data available',
                     ha='center', va='center', transform=axes[2].transAxes)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig