"""
Utility functions for Suite2p data processing.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal


def filter_cells(
    iscell: np.ndarray,
    min_probability: float = 0.5,
) -> np.ndarray:
    """
    Filter cells based on Suite2p classification probability.

    Parameters
    ----------
    iscell : np.ndarray
        Cell classification array (n_cells, 2)
        Column 0: binary classification (0 or 1)
        Column 1: classification probability
    min_probability : float, optional
        Minimum probability threshold (default: 0.5)

    Returns
    -------
    np.ndarray
        Boolean array indicating which ROIs are cells
    """
    is_classified_as_cell = iscell[:, 0] == 1
    meets_probability = iscell[:, 1] >= min_probability

    return is_classified_as_cell & meets_probability


def normalize_traces(
    traces: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """
    Normalize fluorescence traces.

    Parameters
    ----------
    traces : np.ndarray
        Fluorescence traces (n_cells, n_frames)
    method : str, optional
        Normalization method: 'zscore', 'minmax', or 'robust' (default: 'zscore')

    Returns
    -------
    np.ndarray
        Normalized traces
    """
    if method == "zscore":
        # Z-score normalization
        mean = np.mean(traces, axis=1, keepdims=True)
        std = np.std(traces, axis=1, keepdims=True)
        normalized = (traces - mean) / (std + 1e-10)

    elif method == "minmax":
        # Min-max normalization
        min_val = np.min(traces, axis=1, keepdims=True)
        max_val = np.max(traces, axis=1, keepdims=True)
        normalized = (traces - min_val) / (max_val - min_val + 1e-10)

    elif method == "robust":
        # Robust normalization using median and IQR
        median = np.median(traces, axis=1, keepdims=True)
        q75 = np.percentile(traces, 75, axis=1, keepdims=True)
        q25 = np.percentile(traces, 25, axis=1, keepdims=True)
        iqr = q75 - q25
        normalized = (traces - median) / (iqr + 1e-10)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def baseline_correction(
    traces: np.ndarray,
    window_size: int = 300,
    percentile: int = 10,
) -> np.ndarray:
    """
    Remove slow baseline drift from traces.

    Parameters
    ----------
    traces : np.ndarray
        Fluorescence traces (n_cells, n_frames)
    window_size : int, optional
        Size of sliding window in frames (default: 300)
    percentile : int, optional
        Percentile for baseline estimation (default: 10)

    Returns
    -------
    np.ndarray
        Baseline-corrected traces
    """
    n_cells, n_frames = traces.shape
    corrected = np.zeros_like(traces)

    for i in range(n_cells):
        trace = traces[i]

        # Calculate baseline using sliding window
        baseline = np.zeros(n_frames)
        half_window = window_size // 2

        for j in range(n_frames):
            start = max(0, j - half_window)
            end = min(n_frames, j + half_window)
            baseline[j] = np.percentile(trace[start:end], percentile)

        corrected[i] = trace - baseline

    return corrected


def smooth_traces(
    traces: np.ndarray,
    window_size: int = 5,
    method: str = "gaussian",
) -> np.ndarray:
    """
    Smooth fluorescence traces.

    Parameters
    ----------
    traces : np.ndarray
        Fluorescence traces (n_cells, n_frames)
    window_size : int, optional
        Size of smoothing window (default: 5)
    method : str, optional
        Smoothing method: 'gaussian', 'median', or 'savgol' (default: 'gaussian')

    Returns
    -------
    np.ndarray
        Smoothed traces
    """
    n_cells = traces.shape[0]
    smoothed = np.zeros_like(traces)

    if method == "gaussian":
        # Gaussian smoothing
        sigma = window_size / 6.0  # ~99% of Gaussian within window
        for i in range(n_cells):
            smoothed[i] = signal.gaussian_filter1d(traces[i], sigma=sigma)

    elif method == "median":
        # Median filtering
        for i in range(n_cells):
            smoothed[i] = signal.medfilt(traces[i], kernel_size=window_size)

    elif method == "savgol":
        # Savitzky-Golay filter
        if window_size % 2 == 0:
            window_size += 1  # Must be odd
        polyorder = min(3, window_size - 1)
        for i in range(n_cells):
            smoothed[i] = signal.savgol_filter(
                traces[i], window_length=window_size, polyorder=polyorder
            )

    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return smoothed


def downsample_traces(
    traces: np.ndarray,
    factor: int = 2,
) -> np.ndarray:
    """
    Downsample fluorescence traces.

    Parameters
    ----------
    traces : np.ndarray
        Fluorescence traces (n_cells, n_frames)
    factor : int, optional
        Downsampling factor (default: 2)

    Returns
    -------
    np.ndarray
        Downsampled traces
    """
    return traces[:, ::factor]


def get_active_cells(
    traces: np.ndarray,
    threshold: float = 0.1,
    min_events: int = 5,
) -> np.ndarray:
    """
    Identify active cells based on activity level.

    Parameters
    ----------
    traces : np.ndarray
        Fluorescence traces (n_cells, n_frames)
    threshold : float, optional
        Minimum activity threshold (default: 0.1)
    min_events : int, optional
        Minimum number of events required (default: 5)

    Returns
    -------
    np.ndarray
        Boolean array indicating active cells
    """
    from .analysis import detect_events

    # Calculate activity metrics
    mean_activity = np.mean(traces, axis=1)
    std_activity = np.std(traces, axis=1)

    # Detect events
    events = detect_events(traces)
    n_events = np.array([len(e["onset"]) for e in events])

    # Define active cells
    is_active = (std_activity > threshold) & (n_events >= min_events)

    return is_active


def compute_snr(
    F: np.ndarray,
    Fneu: np.ndarray,
) -> np.ndarray:
    """
    Compute signal-to-noise ratio for each cell.

    Parameters
    ----------
    F : np.ndarray
        Raw fluorescence (n_cells, n_frames)
    Fneu : np.ndarray
        Neuropil fluorescence (n_cells, n_frames)

    Returns
    -------
    np.ndarray
        SNR for each cell
    """
    signal_power = np.var(F, axis=1)
    noise_power = np.var(Fneu, axis=1)

    snr = signal_power / (noise_power + 1e-10)

    return snr
