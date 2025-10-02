"""
Analysis functions for Suite2p calcium imaging data.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import signal, stats


def calculate_dff(
    F: np.ndarray,
    Fneu: np.ndarray,
    neuropil_coefficient: float = 0.7,
    percentile: int = 10,
) -> np.ndarray:
    """
    Calculate delta F/F (dF/F) from raw fluorescence traces.

    Parameters
    ----------
    F : np.ndarray
        Raw fluorescence traces (n_cells, n_frames)
    Fneu : np.ndarray
        Neuropil fluorescence traces (n_cells, n_frames)
    neuropil_coefficient : float, optional
        Coefficient for neuropil subtraction (default: 0.7)
    percentile : int, optional
        Percentile for baseline estimation (default: 10)

    Returns
    -------
    np.ndarray
        dF/F traces (n_cells, n_frames)
    """
    # Subtract neuropil
    F_corrected = F - neuropil_coefficient * Fneu

    # Calculate baseline (F0) as percentile across time
    F0 = np.percentile(F_corrected, percentile, axis=1, keepdims=True)

    # Calculate dF/F
    dff = (F_corrected - F0) / F0

    return dff


def detect_events(
    traces: np.ndarray,
    threshold: float = 2.0,
    min_duration: int = 3,
    method: str = "threshold",
) -> list:
    """
    Detect calcium events (transients) in fluorescence traces.

    Parameters
    ----------
    traces : np.ndarray
        Fluorescence traces (n_cells, n_frames) or dF/F traces
    threshold : float, optional
        Detection threshold in standard deviations (default: 2.0)
    min_duration : int, optional
        Minimum event duration in frames (default: 3)
    method : str, optional
        Detection method: 'threshold' or 'peaks' (default: 'threshold')

    Returns
    -------
    list
        List of event dictionaries for each cell, containing:
        - 'onset': Event onset frames
        - 'offset': Event offset frames
        - 'amplitude': Peak amplitude of each event
    """
    n_cells = traces.shape[0]
    events = []

    for i in range(n_cells):
        trace = traces[i]

        if method == "threshold":
            # Calculate threshold based on noise level
            baseline_std = np.std(trace)
            threshold_value = threshold * baseline_std

            # Find frames above threshold
            above_threshold = trace > threshold_value

            # Find event boundaries
            onsets = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
            offsets = np.where(np.diff(above_threshold.astype(int)) == -1)[0] + 1

            # Handle edge cases
            if above_threshold[0]:
                onsets = np.concatenate(([0], onsets))
            if above_threshold[-1]:
                offsets = np.concatenate((offsets, [len(trace) - 1]))

            # Filter by minimum duration
            durations = offsets - onsets
            valid = durations >= min_duration
            onsets = onsets[valid]
            offsets = offsets[valid]

            # Calculate amplitudes
            amplitudes = [np.max(trace[on:off]) for on, off in zip(onsets, offsets)]

        elif method == "peaks":
            # Find peaks
            peaks, properties = signal.find_peaks(
                trace,
                height=threshold * np.std(trace),
                distance=min_duration,
            )
            onsets = peaks
            offsets = peaks
            amplitudes = properties["peak_heights"]

        else:
            raise ValueError(f"Unknown method: {method}")

        events.append(
            {"onset": onsets, "offset": offsets, "amplitude": np.array(amplitudes)}
        )

    return events


def calculate_correlations(
    traces: np.ndarray, method: str = "pearson"
) -> np.ndarray:
    """
    Calculate pairwise correlations between cells.

    Parameters
    ----------
    traces : np.ndarray
        Fluorescence traces (n_cells, n_frames)
    method : str, optional
        Correlation method: 'pearson' or 'spearman' (default: 'pearson')

    Returns
    -------
    np.ndarray
        Correlation matrix (n_cells, n_cells)
    """
    n_cells = traces.shape[0]
    corr_matrix = np.zeros((n_cells, n_cells))

    if method == "pearson":
        corr_matrix = np.corrcoef(traces)
    elif method == "spearman":
        for i in range(n_cells):
            for j in range(i, n_cells):
                corr, _ = stats.spearmanr(traces[i], traces[j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    else:
        raise ValueError(f"Unknown method: {method}")

    return corr_matrix


def analyze_traces(
    F: np.ndarray,
    Fneu: np.ndarray,
    iscell: np.ndarray,
    neuropil_coefficient: float = 0.7,
) -> Dict[str, np.ndarray]:
    """
    Perform comprehensive analysis of fluorescence traces.

    Parameters
    ----------
    F : np.ndarray
        Raw fluorescence traces
    Fneu : np.ndarray
        Neuropil fluorescence
    iscell : np.ndarray
        Cell classification (n_cells, 2)
    neuropil_coefficient : float, optional
        Neuropil subtraction coefficient (default: 0.7)

    Returns
    -------
    dict
        Dictionary containing:
        - 'dff': dF/F traces for cells only
        - 'correlations': Correlation matrix
        - 'events': Detected events
        - 'cell_indices': Indices of cells (vs non-cells)
    """
    # Filter for cells only
    cell_indices = np.where(iscell[:, 0] == 1)[0]
    F_cells = F[cell_indices]
    Fneu_cells = Fneu[cell_indices]

    # Calculate dF/F
    dff = calculate_dff(F_cells, Fneu_cells, neuropil_coefficient)

    # Calculate correlations
    correlations = calculate_correlations(dff)

    # Detect events
    events = detect_events(dff)

    return {
        "dff": dff,
        "correlations": correlations,
        "events": events,
        "cell_indices": cell_indices,
    }


def calculate_response_properties(
    traces: np.ndarray, frame_rate: float = 30.0
) -> Dict[str, np.ndarray]:
    """
    Calculate response properties for each cell.

    Parameters
    ----------
    traces : np.ndarray
        Fluorescence traces (n_cells, n_frames)
    frame_rate : float, optional
        Imaging frame rate in Hz (default: 30.0)

    Returns
    -------
    dict
        Dictionary containing:
        - 'mean_activity': Mean activity level
        - 'std_activity': Standard deviation of activity
        - 'max_activity': Maximum activity
        - 'event_rate': Events per second
    """
    n_cells = traces.shape[0]

    # Detect events
    events = detect_events(traces)

    # Calculate properties
    mean_activity = np.mean(traces, axis=1)
    std_activity = np.std(traces, axis=1)
    max_activity = np.max(traces, axis=1)

    event_rates = np.array([len(e["onset"]) for e in events]) / (
        traces.shape[1] / frame_rate
    )

    return {
        "mean_activity": mean_activity,
        "std_activity": std_activity,
        "max_activity": max_activity,
        "event_rate": event_rates,
    }
