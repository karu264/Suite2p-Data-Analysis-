# Getting Started with Suite2p Data Analysis

This guide will help you get started with analyzing your Suite2p calcium imaging data.

## Prerequisites

Before you begin, make sure you have:

1. Python 3.7 or higher installed
2. Suite2p output data (from running Suite2p on your imaging data)
3. Basic familiarity with Python and NumPy

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/karu264/Suite2p-Data-Analysis-.git
cd Suite2p-Data-Analysis-
```

### Step 2: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Alternatively, for development:

```bash
pip install -e .
```

## Understanding Suite2p Output

Suite2p produces several output files in the `suite2p/plane0/` directory:

- **F.npy**: Raw fluorescence traces for each ROI
- **Fneu.npy**: Neuropil fluorescence (background signal)
- **iscell.npy**: Cell classification (which ROIs are cells)
- **spks.npy**: Deconvolved spike estimates
- **stat.npy**: Spatial properties of each ROI
- **ops.npy**: Processing parameters and metadata

## Basic Workflow

### 1. Load Your Data

```python
from suite2p_analysis import load_suite2p_data

# Load data from a Suite2p output folder
data = load_suite2p_data('path/to/suite2p/plane0')

# Access different components
F = data['F']          # Raw fluorescence (n_cells, n_frames)
Fneu = data['Fneu']    # Neuropil fluorescence
iscell = data['iscell']  # Cell classification
spks = data['spks']    # Deconvolved spikes
```

### 2. Calculate dF/F

Delta F over F (dF/F) is a standard normalization for calcium imaging data:

```python
from suite2p_analysis import calculate_dff
import numpy as np

# Filter for cells only
cell_indices = np.where(iscell[:, 0] == 1)[0]
F_cells = F[cell_indices]
Fneu_cells = Fneu[cell_indices]

# Calculate dF/F
dff = calculate_dff(F_cells, Fneu_cells, neuropil_coefficient=0.7)
```

### 3. Detect Calcium Events

Identify periods of elevated calcium activity:

```python
from suite2p_analysis import detect_events

# Detect events using a threshold-based method
events = detect_events(dff, threshold=2.0, min_duration=3)

# Each event dictionary contains:
# - 'onset': frame indices where events start
# - 'offset': frame indices where events end
# - 'amplitude': peak amplitude of each event
```

### 4. Visualize Results

Create plots to explore your data:

```python
from suite2p_analysis import plot_traces, plot_correlation_matrix
import matplotlib.pyplot as plt

# Plot fluorescence traces
plot_traces(dff, n_cells=10, frame_rate=30.0)

# Calculate and plot correlations
from suite2p_analysis import calculate_correlations
correlations = calculate_correlations(dff)
plot_correlation_matrix(correlations)

plt.show()
```

## Running Example Scripts

The `examples/` directory contains ready-to-use scripts:

### Basic Analysis

```bash
python examples/basic_analysis.py
```

This script demonstrates:
- Loading Suite2p data
- Calculating dF/F
- Detecting events
- Creating basic visualizations

### Batch Processing

```bash
python examples/batch_processing.py
```

Process multiple experiments and compare results.

### Visualization Examples

```bash
python examples/visualization_examples.py
```

Create various types of plots for Suite2p data.

## Common Tasks

### Filtering Cells

Suite2p classifies ROIs as cells or non-cells. Filter for high-confidence cells:

```python
from suite2p_analysis import filter_cells

# Filter cells with minimum probability threshold
is_cell = filter_cells(iscell, min_probability=0.5)
```

### Baseline Correction

Remove slow baseline drift:

```python
from suite2p_analysis import baseline_correction

# Remove baseline drift using a sliding window
corrected_traces = baseline_correction(dff, window_size=300)
```

### Smoothing Traces

Reduce noise in your traces:

```python
from suite2p_analysis import smooth_traces

# Apply Gaussian smoothing
smoothed = smooth_traces(dff, window_size=5, method='gaussian')
```

### Calculate Response Properties

Characterize cell activity:

```python
from suite2p_analysis import calculate_response_properties

properties = calculate_response_properties(dff, frame_rate=30.0)

# Access properties
mean_activity = properties['mean_activity']
event_rate = properties['event_rate']  # events per second
```

## Working with Multiple Planes

If your data has multiple imaging planes:

```python
from suite2p_analysis import load_multiple_planes

# Load all planes
all_data = load_multiple_planes('path/to/suite2p')

# Access specific plane
plane0_data = all_data[0]
plane1_data = all_data[1]
```

## Tips and Best Practices

1. **Data Organization**: Keep your Suite2p output organized in a consistent folder structure
2. **Parameter Tuning**: Adjust thresholds and parameters based on your data quality
3. **Visualization**: Always visualize your data before and after processing
4. **Documentation**: Keep notes on the parameters you use for each analysis
5. **Reproducibility**: Use scripts and version control for reproducible analyses

## Troubleshooting

### Import Errors

If you get import errors, make sure you've installed the package:

```bash
pip install -e .
```

Or add the src directory to your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

### Memory Issues

For large datasets, consider:
- Processing planes separately
- Downsampling traces if temporal resolution allows
- Using data generators instead of loading everything into memory

### File Not Found

Make sure your Suite2p output path is correct and contains the required files (F.npy, Fneu.npy, iscell.npy).

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed function documentation
- Check out the example scripts in the `examples/` directory
- Customize the analysis pipeline for your specific research questions

## Getting Help

If you encounter issues:

1. Check this documentation
2. Review the example scripts
3. Look at function docstrings (`help(function_name)`)
4. Open an issue on GitHub

## Additional Resources

- [Suite2p Documentation](https://suite2p.readthedocs.io/)
- [Suite2p GitHub](https://github.com/MouseLand/suite2p)
- [Calcium Imaging Analysis Tutorial](https://www.nature.com/articles/s41596-021-00652-9)
