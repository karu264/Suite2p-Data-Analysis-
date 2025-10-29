"""
Visualization examples for Suite2p data.

This script demonstrates various visualization options:
1. Fluorescence traces
2. Correlation matrices
3. Cell spatial maps
4. Event raster plots
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
    calculate_correlations,
    plot_traces,
    plot_correlation_matrix,
    plot_cell_map,
    plot_raster,
)


def main():
    """Create various visualizations of Suite2p data."""
    
    # Path to Suite2p output
    data_path = Path("data/suite2p/plane0")
    
    if not data_path.exists():
        print(f"Warning: Data path does not exist: {data_path}")
        print("This is a template script. Please update the data_path variable.")
        return
    
    print("Loading Suite2p data...")
    data = load_suite2p_data(data_path)
    
    F = data["F"]
    Fneu = data["Fneu"]
    iscell = data["iscell"]
    stat = data.get("stat")
    ops = data.get("ops")
    
    # Filter for cells
    cell_indices = np.where(iscell[:, 0] == 1)[0]
    F_cells = F[cell_indices]
    Fneu_cells = Fneu[cell_indices]
    
    # Calculate dF/F
    print("Calculating dF/F...")
    dff = calculate_dff(F_cells, Fneu_cells)
    
    # Detect events
    print("Detecting events...")
    events = detect_events(dff, threshold=2.0)
    
    # Calculate correlations
    print("Calculating correlations...")
    correlations = calculate_correlations(dff)
    
    print("\nCreating visualizations...")
    
    # 1. Plot raw fluorescence traces
    fig1 = plot_traces(
        F_cells,
        n_cells=10,
        frame_rate=30.0,
        title="Raw Fluorescence (F)"
    )
    plt.savefig("viz_raw_traces.png", dpi=150, bbox_inches="tight")
    print("  - Saved: viz_raw_traces.png")
    
    # 2. Plot dF/F traces
    fig2 = plot_traces(
        dff,
        n_cells=10,
        frame_rate=30.0,
        title="Normalized Activity (dF/F)"
    )
    plt.savefig("viz_dff_traces.png", dpi=150, bbox_inches="tight")
    print("  - Saved: viz_dff_traces.png")
    
    # 3. Plot traces for a specific time window
    fig3 = plot_traces(
        dff,
        n_cells=5,
        time_range=(0, 300),  # First 10 seconds at 30 Hz
        frame_rate=30.0,
        title="Activity (First 10 seconds)"
    )
    plt.savefig("viz_traces_window.png", dpi=150, bbox_inches="tight")
    print("  - Saved: viz_traces_window.png")
    
    # 4. Plot correlation matrix
    fig4 = plot_correlation_matrix(
        correlations,
        title="Cell-Cell Correlations"
    )
    plt.savefig("viz_correlations.png", dpi=150, bbox_inches="tight")
    print("  - Saved: viz_correlations.png")
    
    # 5. Plot event raster
    fig5 = plot_raster(
        events,
        frame_rate=30.0,
        title="Calcium Event Raster"
    )
    plt.savefig("viz_raster.png", dpi=150, bbox_inches="tight")
    print("  - Saved: viz_raster.png")
    
    # 6. Plot cell spatial map (if stat and ops are available)
    if stat is not None and ops is not None:
        fig6 = plot_cell_map(
            stat[cell_indices],
            ops,
            title="Spatial Distribution of Cells"
        )
        plt.savefig("viz_cell_map.png", dpi=150, bbox_inches="tight")
        print("  - Saved: viz_cell_map.png")
    
    # 7. Create a summary figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Example traces
    time = np.arange(dff.shape[1]) / 30.0
    for i in range(min(5, dff.shape[0])):
        axes[0, 0].plot(time, dff[i] + i*2, linewidth=0.5, label=f"Cell {i}")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("dF/F + offset")
    axes[0, 0].set_title("Example Calcium Traces")
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)
    
    # Subplot 2: Event counts histogram
    n_events = [len(e["onset"]) for e in events]
    axes[0, 1].hist(n_events, bins=20, edgecolor="black")
    axes[0, 1].set_xlabel("Number of Events")
    axes[0, 1].set_ylabel("Number of Cells")
    axes[0, 1].set_title("Distribution of Event Counts")
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)
    
    # Subplot 3: Correlation distribution
    corr_values = correlations[np.triu_indices_from(correlations, k=1)]
    axes[1, 0].hist(corr_values, bins=50, edgecolor="black")
    axes[1, 0].set_xlabel("Correlation Coefficient")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Distribution of Pairwise Correlations")
    axes[1, 0].axvline(np.mean(corr_values), color="red", linestyle="--", label="Mean")
    axes[1, 0].legend()
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)
    
    # Subplot 4: Activity statistics
    mean_activity = np.mean(dff, axis=1)
    std_activity = np.std(dff, axis=1)
    axes[1, 1].scatter(mean_activity, std_activity, alpha=0.5)
    axes[1, 1].set_xlabel("Mean Activity")
    axes[1, 1].set_ylabel("Std Dev Activity")
    axes[1, 1].set_title("Activity Variability")
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("viz_summary.png", dpi=150, bbox_inches="tight")
    print("  - Saved: viz_summary.png")
    
    print("\nAll visualizations complete!")
    print("\nTo view the figures interactively, call plt.show()")


if __name__ == "__main__":
    main()
