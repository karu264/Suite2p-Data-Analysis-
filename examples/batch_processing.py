"""
Batch processing example for multiple Suite2p datasets.

This script demonstrates how to:
1. Process multiple experiments
2. Compare results across datasets
3. Generate summary statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import suite2p_analysis functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from suite2p_analysis import (
    load_suite2p_data,
    calculate_dff,
    detect_events,
    calculate_correlations,
)


def analyze_experiment(data_path):
    """
    Analyze a single experiment.
    
    Parameters
    ----------
    data_path : Path
        Path to Suite2p plane folder
        
    Returns
    -------
    dict
        Dictionary with analysis results
    """
    print(f"\nProcessing: {data_path}")
    
    # Load data
    data = load_suite2p_data(data_path)
    F = data["F"]
    Fneu = data["Fneu"]
    iscell = data["iscell"]
    
    # Filter for cells
    cell_indices = np.where(iscell[:, 0] == 1)[0]
    F_cells = F[cell_indices]
    Fneu_cells = Fneu[cell_indices]
    
    # Calculate dF/F
    dff = calculate_dff(F_cells, Fneu_cells)
    
    # Detect events
    events = detect_events(dff, threshold=2.0)
    n_events = [len(e["onset"]) for e in events]
    
    # Calculate correlations
    correlations = calculate_correlations(dff)
    
    # Compute summary statistics
    results = {
        "path": str(data_path),
        "n_rois": F.shape[0],
        "n_cells": len(cell_indices),
        "n_frames": F.shape[1],
        "mean_events_per_cell": np.mean(n_events),
        "std_events_per_cell": np.std(n_events),
        "mean_correlation": np.mean(correlations[np.triu_indices_from(correlations, k=1)]),
        "mean_activity": np.mean(dff),
        "std_activity": np.std(dff),
    }
    
    print(f"  - Cells: {results['n_cells']}")
    print(f"  - Mean events per cell: {results['mean_events_per_cell']:.2f}")
    print(f"  - Mean correlation: {results['mean_correlation']:.3f}")
    
    return results


def main():
    """Run batch processing pipeline."""
    
    # Define experiment paths
    # Replace with your actual data paths
    experiment_paths = [
        Path("data/experiment1/suite2p/plane0"),
        Path("data/experiment2/suite2p/plane0"),
        Path("data/experiment3/suite2p/plane0"),
    ]
    
    # Check if paths exist
    existing_paths = [p for p in experiment_paths if p.exists()]
    
    if not existing_paths:
        print("Warning: No experiment paths exist.")
        print("This is a template script. Please update the experiment_paths list.")
        print("\nExpected structure for each experiment:")
        print("data/experimentN/suite2p/plane0/")
        print("  ├── F.npy")
        print("  ├── Fneu.npy")
        print("  ├── iscell.npy")
        print("  └── ...")
        return
    
    print(f"Found {len(existing_paths)} experiments to process")
    
    # Process all experiments
    all_results = []
    for path in existing_paths:
        try:
            results = analyze_experiment(path)
            all_results.append(results)
        except Exception as e:
            print(f"  Error processing {path}: {e}")
    
    if not all_results:
        print("\nNo experiments were successfully processed.")
        return
    
    # Create summary DataFrame
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save results
    output_file = "batch_processing_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Number of cells
    axes[0, 0].bar(range(len(df)), df["n_cells"])
    axes[0, 0].set_xlabel("Experiment")
    axes[0, 0].set_ylabel("Number of Cells")
    axes[0, 0].set_title("Cells per Experiment")
    
    # Events per cell
    axes[0, 1].bar(range(len(df)), df["mean_events_per_cell"])
    axes[0, 1].errorbar(
        range(len(df)),
        df["mean_events_per_cell"],
        yerr=df["std_events_per_cell"],
        fmt="none",
        color="black",
        capsize=5
    )
    axes[0, 1].set_xlabel("Experiment")
    axes[0, 1].set_ylabel("Events per Cell")
    axes[0, 1].set_title("Activity Level")
    
    # Correlations
    axes[1, 0].bar(range(len(df)), df["mean_correlation"])
    axes[1, 0].set_xlabel("Experiment")
    axes[1, 0].set_ylabel("Mean Correlation")
    axes[1, 0].set_title("Cell-Cell Correlations")
    
    # Activity variance
    axes[1, 1].bar(range(len(df)), df["std_activity"])
    axes[1, 1].set_xlabel("Experiment")
    axes[1, 1].set_ylabel("Activity Std Dev")
    axes[1, 1].set_title("Activity Variability")
    
    plt.tight_layout()
    plt.savefig("batch_comparison.png", dpi=150, bbox_inches="tight")
    print("Comparison plot saved to: batch_comparison.png")
    
    plt.show()
    
    print("\nBatch processing complete!")


if __name__ == "__main__":
    main()
