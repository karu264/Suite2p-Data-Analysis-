"""
Suite2p Data Analysis Toolkit

A comprehensive toolkit for analyzing calcium imaging data processed with Suite2p.
"""

__version__ = "0.1.0"

from .loader import load_suite2p_data, load_multiple_planes
from .analysis import (
    calculate_dff,
    detect_events,
    calculate_correlations,
    analyze_traces,
)
from .visualization import (
    plot_traces,
    plot_correlation_matrix,
    plot_cell_map,
    plot_raster,
)
from .utils import filter_cells, normalize_traces, baseline_correction

__all__ = [
    # Loader functions
    "load_suite2p_data",
    "load_multiple_planes",
    # Analysis functions
    "calculate_dff",
    "detect_events",
    "calculate_correlations",
    "analyze_traces",
    # Visualization functions
    "plot_traces",
    "plot_correlation_matrix",
    "plot_cell_map",
    "plot_raster",
    # Utility functions
    "filter_cells",
    "normalize_traces",
    "baseline_correction",
]
