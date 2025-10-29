# API Reference

Complete reference for all functions in the Suite2p Data Analysis toolkit.

## Data Loading (`loader.py`)

### load_suite2p_data

```python
load_suite2p_data(data_path, load_spks=True, load_stat=True, load_ops=True)
```

Load Suite2p output data from a single plane.

**Parameters:**
- `data_path` (str or Path): Path to Suite2p plane folder
- `load_spks` (bool): Whether to load deconvolved spike data
- `load_stat` (bool): Whether to load cell statistics
- `load_ops` (bool): Whether to load operations/settings

**Returns:**
- `dict`: Dictionary containing Suite2p data arrays

**Example:**
```python
data = load_suite2p_data('suite2p/plane0')
F = data['F']
```

### load_multiple_planes

```python
load_multiple_planes(base_path, plane_indices=None, **kwargs)
```

Load Suite2p data from multiple imaging planes.

**Parameters:**
- `base_path` (str or Path): Path to Suite2p folder
- `plane_indices` (list of int): Plane indices to load
- `**kwargs`: Additional arguments for load_suite2p_data

**Returns:**
- `dict`: Dictionary mapping plane indices to data

### get_cell_masks

```python
get_cell_masks(stat, ops)
```

Extract spatial footprints (masks) for each cell.

**Parameters:**
- `stat` (np.ndarray): Cell statistics array
- `ops` (dict): Operations dictionary

**Returns:**
- `list`: List of 2D arrays representing cell masks

---

## Analysis (`analysis.py`)

### calculate_dff

```python
calculate_dff(F, Fneu, neuropil_coefficient=0.7, percentile=10)
```

Calculate delta F/F (dF/F) from raw fluorescence traces.

**Parameters:**
- `F` (np.ndarray): Raw fluorescence (n_cells, n_frames)
- `Fneu` (np.ndarray): Neuropil fluorescence
- `neuropil_coefficient` (float): Neuropil subtraction coefficient
- `percentile` (int): Percentile for baseline estimation

**Returns:**
- `np.ndarray`: dF/F traces

**Example:**
```python
dff = calculate_dff(F, Fneu, neuropil_coefficient=0.7)
```

### detect_events

```python
detect_events(traces, threshold=2.0, min_duration=3, method='threshold')
```

Detect calcium events (transients) in fluorescence traces.

**Parameters:**
- `traces` (np.ndarray): Fluorescence traces (n_cells, n_frames)
- `threshold` (float): Detection threshold in standard deviations
- `min_duration` (int): Minimum event duration in frames
- `method` (str): 'threshold' or 'peaks'

**Returns:**
- `list`: List of event dictionaries for each cell

**Example:**
```python
events = detect_events(dff, threshold=2.0, min_duration=3)
```

### calculate_correlations

```python
calculate_correlations(traces, method='pearson')
```

Calculate pairwise correlations between cells.

**Parameters:**
- `traces` (np.ndarray): Fluorescence traces
- `method` (str): 'pearson' or 'spearman'

**Returns:**
- `np.ndarray`: Correlation matrix (n_cells, n_cells)

### analyze_traces

```python
analyze_traces(F, Fneu, iscell, neuropil_coefficient=0.7)
```

Perform comprehensive analysis of fluorescence traces.

**Parameters:**
- `F` (np.ndarray): Raw fluorescence
- `Fneu` (np.ndarray): Neuropil fluorescence
- `iscell` (np.ndarray): Cell classification
- `neuropil_coefficient` (float): Neuropil coefficient

**Returns:**
- `dict`: Contains dff, correlations, events, cell_indices

### calculate_response_properties

```python
calculate_response_properties(traces, frame_rate=30.0)
```

Calculate response properties for each cell.

**Parameters:**
- `traces` (np.ndarray): Fluorescence traces
- `frame_rate` (float): Imaging frame rate in Hz

**Returns:**
- `dict`: Contains mean_activity, std_activity, max_activity, event_rate

---

## Visualization (`visualization.py`)

### plot_traces

```python
plot_traces(traces, iscell=None, n_cells=10, time_range=None,
            frame_rate=30.0, figsize=(12, 8), title='Fluorescence Traces')
```

Plot fluorescence traces for multiple cells.

**Parameters:**
- `traces` (np.ndarray): Fluorescence traces
- `iscell` (np.ndarray): Cell classification
- `n_cells` (int): Number of cells to plot
- `time_range` (tuple): Time range (start_frame, end_frame)
- `frame_rate` (float): Frame rate in Hz
- `figsize` (tuple): Figure size
- `title` (str): Plot title

**Returns:**
- `matplotlib.figure.Figure`: The created figure

### plot_correlation_matrix

```python
plot_correlation_matrix(correlations, figsize=(10, 8), cmap='RdBu_r',
                       vmin=-1.0, vmax=1.0, title='Correlation Matrix')
```

Plot correlation matrix as a heatmap.

**Parameters:**
- `correlations` (np.ndarray): Correlation matrix
- `figsize` (tuple): Figure size
- `cmap` (str): Colormap
- `vmin`, `vmax` (float): Color scale limits
- `title` (str): Plot title

**Returns:**
- `matplotlib.figure.Figure`: The created figure

### plot_cell_map

```python
plot_cell_map(stat, ops, iscell=None, property_values=None,
              figsize=(10, 10), title='Cell Map')
```

Plot spatial distribution of cells.

**Parameters:**
- `stat` (np.ndarray): Cell statistics
- `ops` (dict): Operations dictionary
- `iscell` (np.ndarray): Cell classification
- `property_values` (np.ndarray): Values to color cells by
- `figsize` (tuple): Figure size
- `title` (str): Plot title

**Returns:**
- `matplotlib.figure.Figure`: The created figure

### plot_raster

```python
plot_raster(events, frame_rate=30.0, figsize=(12, 8),
            title='Event Raster Plot')
```

Create raster plot of calcium events.

**Parameters:**
- `events` (list): List of event dictionaries
- `frame_rate` (float): Frame rate in Hz
- `figsize` (tuple): Figure size
- `title` (str): Plot title

**Returns:**
- `matplotlib.figure.Figure`: The created figure

### plot_mean_image

```python
plot_mean_image(ops, image_type='meanImg', figsize=(8, 8), cmap='gray')
```

Plot mean or maximum projection image.

**Parameters:**
- `ops` (dict): Operations dictionary
- `image_type` (str): 'meanImg', 'meanImgE', or 'max_proj'
- `figsize` (tuple): Figure size
- `cmap` (str): Colormap

**Returns:**
- `matplotlib.figure.Figure`: The created figure

---

## Utilities (`utils.py`)

### filter_cells

```python
filter_cells(iscell, min_probability=0.5)
```

Filter cells based on Suite2p classification probability.

**Parameters:**
- `iscell` (np.ndarray): Cell classification (n_cells, 2)
- `min_probability` (float): Minimum probability threshold

**Returns:**
- `np.ndarray`: Boolean array indicating which ROIs are cells

### normalize_traces

```python
normalize_traces(traces, method='zscore')
```

Normalize fluorescence traces.

**Parameters:**
- `traces` (np.ndarray): Fluorescence traces
- `method` (str): 'zscore', 'minmax', or 'robust'

**Returns:**
- `np.ndarray`: Normalized traces

### baseline_correction

```python
baseline_correction(traces, window_size=300, percentile=10)
```

Remove slow baseline drift from traces.

**Parameters:**
- `traces` (np.ndarray): Fluorescence traces
- `window_size` (int): Size of sliding window
- `percentile` (int): Percentile for baseline

**Returns:**
- `np.ndarray`: Baseline-corrected traces

### smooth_traces

```python
smooth_traces(traces, window_size=5, method='gaussian')
```

Smooth fluorescence traces.

**Parameters:**
- `traces` (np.ndarray): Fluorescence traces
- `window_size` (int): Smoothing window size
- `method` (str): 'gaussian', 'median', or 'savgol'

**Returns:**
- `np.ndarray`: Smoothed traces

### downsample_traces

```python
downsample_traces(traces, factor=2)
```

Downsample fluorescence traces.

**Parameters:**
- `traces` (np.ndarray): Fluorescence traces
- `factor` (int): Downsampling factor

**Returns:**
- `np.ndarray`: Downsampled traces

### get_active_cells

```python
get_active_cells(traces, threshold=0.1, min_events=5)
```

Identify active cells based on activity level.

**Parameters:**
- `traces` (np.ndarray): Fluorescence traces
- `threshold` (float): Minimum activity threshold
- `min_events` (int): Minimum number of events

**Returns:**
- `np.ndarray`: Boolean array indicating active cells

### compute_snr

```python
compute_snr(F, Fneu)
```

Compute signal-to-noise ratio for each cell.

**Parameters:**
- `F` (np.ndarray): Raw fluorescence
- `Fneu` (np.ndarray): Neuropil fluorescence

**Returns:**
- `np.ndarray`: SNR for each cell

---

## Data Structures

### Cell Classification Array (iscell)

Shape: `(n_rois, 2)`
- Column 0: Binary classification (0=non-cell, 1=cell)
- Column 1: Classification probability (0.0 to 1.0)

### Events Dictionary

For each cell, contains:
- `'onset'`: np.ndarray of event start frames
- `'offset'`: np.ndarray of event end frames
- `'amplitude'`: np.ndarray of peak amplitudes

### Operations Dictionary (ops)

Contains Suite2p processing parameters:
- `'Ly'`, `'Lx'`: Image dimensions
- `'meanImg'`: Mean fluorescence image
- `'max_proj'`: Maximum projection
- `'frame_rate'`: Imaging frame rate
- And many other Suite2p settings

---

## Common Workflows

### Basic Analysis Pipeline

```python
# 1. Load data
data = load_suite2p_data('suite2p/plane0')

# 2. Filter cells
cell_idx = filter_cells(data['iscell'])
F = data['F'][cell_idx]
Fneu = data['Fneu'][cell_idx]

# 3. Calculate dF/F
dff = calculate_dff(F, Fneu)

# 4. Detect events
events = detect_events(dff)

# 5. Visualize
plot_traces(dff, n_cells=10)
```

### Quality Control

```python
# Compute SNR
snr = compute_snr(F, Fneu)

# Filter high-quality cells
high_snr = snr > 2.0
high_prob = filter_cells(iscell, min_probability=0.7)
quality_cells = high_snr & high_prob

# Use filtered cells
F_quality = F[quality_cells]
```

### Population Analysis

```python
# Calculate correlations
corr = calculate_correlations(dff)

# Find highly correlated pairs
high_corr = np.where(corr > 0.5)

# Calculate population statistics
mean_corr = np.mean(corr[np.triu_indices_from(corr, k=1)])
```
