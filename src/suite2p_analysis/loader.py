"""
Data loading utilities for Suite2p output files.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union


def load_suite2p_data(
    data_path: Union[str, Path],
    load_spks: bool = True,
    load_stat: bool = True,
    load_ops: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load Suite2p output data from a single plane.

    Parameters
    ----------
    data_path : str or Path
        Path to the Suite2p plane folder (e.g., 'suite2p/plane0')
    load_spks : bool, optional
        Whether to load deconvolved spike data (default: True)
    load_stat : bool, optional
        Whether to load cell statistics (default: True)
    load_ops : bool, optional
        Whether to load operations/settings (default: True)

    Returns
    -------
    dict
        Dictionary containing Suite2p data arrays:
        - 'F': Raw fluorescence traces (n_cells, n_frames)
        - 'Fneu': Neuropil fluorescence (n_cells, n_frames)
        - 'iscell': Cell classification (n_cells, 2)
        - 'spks': Deconvolved spikes (if load_spks=True)
        - 'stat': Cell statistics (if load_stat=True)
        - 'ops': Operations dictionary (if load_ops=True)

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist
    ValueError
        If required files are missing
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Path does not exist: {data_path}")

    data = {}

    # Load required files
    required_files = {
        "F": "F.npy",
        "Fneu": "Fneu.npy",
        "iscell": "iscell.npy",
    }

    for key, filename in required_files.items():
        file_path = data_path / filename
        if not file_path.exists():
            raise ValueError(f"Required file not found: {filename}")
        data[key] = np.load(file_path, allow_pickle=True)

    # Load optional files
    if load_spks:
        spks_path = data_path / "spks.npy"
        if spks_path.exists():
            data["spks"] = np.load(spks_path, allow_pickle=True)

    if load_stat:
        stat_path = data_path / "stat.npy"
        if stat_path.exists():
            data["stat"] = np.load(stat_path, allow_pickle=True)

    if load_ops:
        ops_path = data_path / "ops.npy"
        if ops_path.exists():
            data["ops"] = np.load(ops_path, allow_pickle=True).item()

    return data


def load_multiple_planes(
    base_path: Union[str, Path],
    plane_indices: Optional[List[int]] = None,
    **kwargs,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load Suite2p data from multiple imaging planes.

    Parameters
    ----------
    base_path : str or Path
        Path to the Suite2p output folder containing plane subfolders
    plane_indices : list of int, optional
        List of plane indices to load. If None, loads all available planes.
    **kwargs
        Additional arguments passed to load_suite2p_data

    Returns
    -------
    dict
        Dictionary mapping plane indices to their data dictionaries

    Examples
    --------
    >>> data = load_multiple_planes('path/to/suite2p', plane_indices=[0, 1, 2])
    >>> plane0_F = data[0]['F']
    """
    base_path = Path(base_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Path does not exist: {base_path}")

    # Find available planes if not specified
    if plane_indices is None:
        plane_folders = sorted(base_path.glob("plane*"))
        plane_indices = [
            int(p.name.replace("plane", "")) for p in plane_folders if p.is_dir()
        ]

    # Load data from each plane
    all_data = {}
    for plane_idx in plane_indices:
        plane_path = base_path / f"plane{plane_idx}"
        if plane_path.exists():
            try:
                all_data[plane_idx] = load_suite2p_data(plane_path, **kwargs)
            except Exception as e:
                print(f"Warning: Could not load plane {plane_idx}: {e}")
        else:
            print(f"Warning: Plane folder not found: {plane_path}")

    return all_data


def get_cell_masks(stat: np.ndarray, ops: dict) -> List[np.ndarray]:
    """
    Extract spatial footprints (masks) for each cell.

    Parameters
    ----------
    stat : np.ndarray
        Cell statistics array from Suite2p
    ops : dict
        Operations dictionary containing imaging parameters

    Returns
    -------
    list of np.ndarray
        List of 2D arrays representing cell masks
    """
    Ly, Lx = ops.get("Ly", 512), ops.get("Lx", 512)
    masks = []

    for cell_stat in stat:
        mask = np.zeros((Ly, Lx), dtype=np.float32)
        ypix = cell_stat["ypix"]
        xpix = cell_stat["xpix"]
        lam = cell_stat["lam"]

        mask[ypix, xpix] = lam

        masks.append(mask)

    return masks
