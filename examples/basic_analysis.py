"""
Basic analysis example for Suite2p data.

This script demonstrates how to:
1. Load Suite2p data
2. Calculate dF/F
3. Detect calcium events
4. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import suite2p_analysis functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from suite2p_analysis import (
    load_suite2p_data,
    calculate_dff,
    detect_events,
    plot_traces,
    plot_correlation_matrix,
    calculate_correlations,
)


def main():
    """Run basic analysis pipeline."""
    
    # Path to Suite2p output
    # Replace with your actual data path
    data_path = Path("data/suite2p/plane0")
    
    if not data_path.exists():
        print(f"Warning: Data path does not exist: {data_path}")
        print("This is a template script. Please update the data_path variable.")
        print("\nExpected structure:")
        print("data/suite2p/plane0/")
        print("  ├── F.npy")
        print("  ├── Fneu.npy")
        print("  ├── iscell.npy")
        print("  ├── spks.npy")
        print("  └── stat.npy")
        return
    
    print("Loading Suite2p data...")
    data = load_suite2p_data(data_path)
    
    # Extract data
    F = data["F"]
    Fneu = data["Fneu"]
    iscell = data["iscell"]
    
    print(f"Loaded data:")
    print(f"  - Total ROIs: {F.shape[0]}")
    print(f"  - Frames: {F.shape[1]}")
    print(f"  - Cells: {np.sum(iscell[:, 0])}")
    
    # Filter for cells only
    cell_indices = np.where(iscell[:, 0] == 1)[0]
    F_cells = F[cell_indices]
    Fneu_cells = Fneu[cell_indices]
    
    print("\nCalculating dF/F...")
    dff = calculate_dff(F_cells, Fneu_cells)
    
    print("Detecting calcium events...")
    events = detect_events(dff, threshold=2.0)
    
    # Print event statistics
    n_events_per_cell = [len(e["onset"]) for e in events]
    print(f"\nEvent statistics:")
    print(f"  - Total events: {sum(n_events_per_cell)}")
    print(f"  - Mean events per cell: {np.mean(n_events_per_cell):.1f}")
    print(f"  - Max events in a cell: {max(n_events_per_cell)}")
    
    print("\nCalculating correlations...")
    correlations = calculate_correlations(dff)
    
    # Visualize results
    print("Creating visualizations...")
    
    # Plot traces
    fig1 = plot_traces(
        dff,
        n_cells=10,
        frame_rate=30.0,
        title="Example Calcium Activity (dF/F)"
    )
    plt.savefig("output_traces.png", dpi=150, bbox_inches="tight")
    print("  - Saved: output_traces.png")
    
    # Plot correlation matrix
    fig2 = plot_correlation_matrix(
        correlations,
        title="Cell-Cell Correlations"
    )
    plt.savefig("output_correlations.png", dpi=150, bbox_inches="tight")
    print("  - Saved: output_correlations.png")
    
    # Show plots
    plt.show()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
