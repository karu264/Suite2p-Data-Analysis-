"""
Visualization functions for Suite2p data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List


def plot_traces(
    traces: np.ndarray,
    iscell: Optional[np.ndarray] = None,
    n_cells: int = 10,
    time_range: Optional[Tuple[int, int]] = None,
    frame_rate: float = 30.0,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Fluorescence Traces",
) -> plt.Figure:
    """
    Plot fluorescence traces for multiple cells.

    Parameters
    ----------
    traces : np.ndarray
        Fluorescence traces (n_cells, n_frames)
    iscell : np.ndarray, optional
        Cell classification. If provided, only plots cells.
    n_cells : int, optional
        Number of cells to plot (default: 10)
    time_range : tuple, optional
        Time range to plot (start_frame, end_frame)
    frame_rate : float, optional
        Imaging frame rate in Hz (default: 30.0)
    figsize : tuple, optional
        Figure size (default: (12, 8))
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Filter for cells if iscell is provided
    if iscell is not None:
        cell_indices = np.where(iscell[:, 0] == 1)[0]
        traces = traces[cell_indices]

    # Select subset of cells
    n_cells = min(n_cells, traces.shape[0])
    traces_subset = traces[:n_cells]

    # Select time range
    if time_range is not None:
        traces_subset = traces_subset[:, time_range[0] : time_range[1]]
        start_frame = time_range[0]
    else:
        start_frame = 0

    # Create time axis
    n_frames = traces_subset.shape[1]
    time = (np.arange(n_frames) + start_frame) / frame_rate

    # Create plot
    fig, axes = plt.subplots(n_cells, 1, figsize=figsize, sharex=True)
    if n_cells == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, traces_subset[i], linewidth=0.5)
        ax.set_ylabel(f"Cell {i}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(title)

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    correlations: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    title: str = "Correlation Matrix",
) -> plt.Figure:
    """
    Plot correlation matrix as a heatmap.

    Parameters
    ----------
    correlations : np.ndarray
        Correlation matrix (n_cells, n_cells)
    figsize : tuple, optional
        Figure size (default: (10, 8))
    cmap : str, optional
        Colormap (default: 'RdBu_r')
    vmin : float, optional
        Minimum value for colormap (default: -1.0)
    vmax : float, optional
        Maximum value for colormap (default: 1.0)
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        correlations,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )

    ax.set_title(title)
    ax.set_xlabel("Cell Index")
    ax.set_ylabel("Cell Index")

    plt.tight_layout()
    return fig


def plot_cell_map(
    stat: np.ndarray,
    ops: dict,
    iscell: Optional[np.ndarray] = None,
    property_values: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: str = "Cell Map",
) -> plt.Figure:
    """
    Plot spatial distribution of cells.

    Parameters
    ----------
    stat : np.ndarray
        Cell statistics from Suite2p
    ops : dict
        Operations dictionary
    iscell : np.ndarray, optional
        Cell classification. If provided, colors cells differently.
    property_values : np.ndarray, optional
        Values to color cells by (e.g., activity level)
    figsize : tuple, optional
        Figure size (default: (10, 10))
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    Ly, Lx = ops.get("Ly", 512), ops.get("Lx", 512)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each cell
    for i, cell_stat in enumerate(stat):
        ypix = cell_stat["ypix"]
        xpix = cell_stat["xpix"]

        if property_values is not None:
            color = plt.cm.viridis(property_values[i] / np.max(property_values))
        elif iscell is not None:
            color = "green" if iscell[i, 0] == 1 else "red"
        else:
            color = "blue"

        ax.scatter(xpix, ypix, s=1, c=[color], alpha=0.5)

    ax.set_xlim(0, Lx)
    ax.set_ylim(Ly, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(title)

    if iscell is not None and property_values is None:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="green", alpha=0.5, label="Cells"),
            Patch(facecolor="red", alpha=0.5, label="Non-cells"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig


def plot_raster(
    events: List[dict],
    frame_rate: float = 30.0,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Event Raster Plot",
) -> plt.Figure:
    """
    Create raster plot of calcium events.

    Parameters
    ----------
    events : list of dict
        List of event dictionaries from detect_events
    frame_rate : float, optional
        Imaging frame rate in Hz (default: 30.0)
    figsize : tuple, optional
        Figure size (default: (12, 8))
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, cell_events in enumerate(events):
        event_times = cell_events["onset"] / frame_rate
        ax.scatter(
            event_times,
            [i] * len(event_times),
            marker="|",
            s=100,
            c="black",
            linewidths=1,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cell Index")
    ax.set_title(title)
    ax.set_ylim(-1, len(events))

    plt.tight_layout()
    return fig


def plot_mean_image(
    ops: dict,
    image_type: str = "meanImg",
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "gray",
) -> plt.Figure:
    """
    Plot mean or maximum projection image.

    Parameters
    ----------
    ops : dict
        Operations dictionary containing images
    image_type : str, optional
        Type of image: 'meanImg', 'meanImgE', 'max_proj' (default: 'meanImg')
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap (default: 'gray')

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    if image_type not in ops:
        raise ValueError(f"Image type '{image_type}' not found in ops")

    fig, ax = plt.subplots(figsize=figsize)

    img = ops[image_type]
    im = ax.imshow(img, cmap=cmap, aspect="equal")
    ax.set_title(f"{image_type}")
    ax.axis("off")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    return fig
