# Quick Start Guide

Get started with Suite2p Data Analysis in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/karu264/Suite2p-Data-Analysis-.git
cd Suite2p-Data-Analysis-

# Install dependencies
pip install -r requirements.txt

# Optional: Install the package
pip install -e .
```

## Basic Usage

### 1. Load Your Data

```python
from suite2p_analysis import load_suite2p_data

# Load Suite2p output
data = load_suite2p_data('path/to/suite2p/plane0')

# Access the data
F = data['F']          # Raw fluorescence
Fneu = data['Fneu']    # Neuropil
iscell = data['iscell']  # Cell classification
```

### 2. Calculate dF/F

```python
from suite2p_analysis import calculate_dff
import numpy as np

# Get cells only
cell_idx = np.where(iscell[:, 0] == 1)[0]
F_cells = F[cell_idx]
Fneu_cells = Fneu[cell_idx]

# Calculate dF/F
dff = calculate_dff(F_cells, Fneu_cells)
```

### 3. Visualize

```python
from suite2p_analysis import plot_traces
import matplotlib.pyplot as plt

# Plot traces
plot_traces(dff, n_cells=10, frame_rate=30.0)
plt.show()
```

### 4. Detect Events

```python
from suite2p_analysis import detect_events

# Find calcium transients
events = detect_events(dff, threshold=2.0)

# Count events per cell
n_events = [len(e['onset']) for e in events]
print(f"Average events per cell: {np.mean(n_events):.1f}")
```

### 5. Analyze Correlations

```python
from suite2p_analysis import calculate_correlations, plot_correlation_matrix

# Calculate pairwise correlations
corr = calculate_correlations(dff)

# Visualize
plot_correlation_matrix(corr)
plt.show()
```

## Running Examples

Try the example scripts:

```bash
# Basic analysis
python examples/basic_analysis.py

# Batch processing
python examples/batch_processing.py

# Visualization examples
python examples/visualization_examples.py
```

## Project Structure

```
Suite2p-Data-Analysis-/
├── src/suite2p_analysis/    # Main package
│   ├── loader.py            # Data loading
│   ├── analysis.py          # Analysis functions
│   ├── visualization.py     # Plotting
│   └── utils.py             # Utilities
├── examples/                # Example scripts
├── docs/                    # Documentation
└── data/                    # Place your data here
```

## Common Tasks

### Filter High-Quality Cells

```python
from suite2p_analysis import filter_cells

# Filter by probability
good_cells = filter_cells(iscell, min_probability=0.7)
F_filtered = F[good_cells]
```

### Smooth Noisy Traces

```python
from suite2p_analysis import smooth_traces

# Apply Gaussian smoothing
smoothed = smooth_traces(dff, window_size=5, method='gaussian')
```

### Process Multiple Experiments

```python
from suite2p_analysis import load_multiple_planes

# Load all planes
all_data = load_multiple_planes('path/to/suite2p')

# Process each plane
for plane_idx, plane_data in all_data.items():
    print(f"Plane {plane_idx}: {plane_data['F'].shape[0]} ROIs")
```

## Getting Help

- 📖 [Full Documentation](docs/getting_started.md)
- 📚 [API Reference](docs/api_reference.md)
- 💡 [Examples](examples/)
- 🐛 [Report Issues](https://github.com/karu264/Suite2p-Data-Analysis-/issues)

## Next Steps

1. Check out the [Getting Started Guide](docs/getting_started.md) for detailed tutorials
2. Explore the [API Reference](docs/api_reference.md) for all available functions
3. Try the example scripts to see the toolkit in action
4. Adapt the examples for your own data and research questions

Happy analyzing! 🔬
