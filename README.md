# Suite2p Data Analysis

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive toolkit for analyzing calcium imaging data processed with Suite2p.

📖 **[Quick Start](QUICKSTART.md)** | **[Documentation](docs/getting_started.md)** | **[API Reference](docs/api_reference.md)** | **[Examples](examples/)**

## Overview

Suite2p is a fast, accurate, and complete two-photon calcium imaging data analysis pipeline. This repository provides tools and scripts to analyze the output from Suite2p, including:

- Loading and preprocessing Suite2p output data
- Visualizing neural activity traces
- Analyzing cell responses and correlations
- Statistical analysis of calcium imaging data
- Batch processing multiple experiments

## Quick Start

```python
from suite2p_analysis import load_suite2p_data, calculate_dff, plot_traces

# Load data
data = load_suite2p_data('path/to/suite2p/plane0')

# Calculate dF/F
dff = calculate_dff(data['F'], data['Fneu'])

# Visualize
plot_traces(dff, n_cells=10)
```

👉 See [QUICKSTART.md](QUICKSTART.md) for a 5-minute tutorial!

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone https://github.com/karu264/Suite2p-Data-Analysis-.git
cd Suite2p-Data-Analysis-
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. (Optional) Install in development mode:
```bash
pip install -e .
```

## Usage

### Loading Suite2p Data

```python
from suite2p_analysis import load_suite2p_data

# Load Suite2p output
data = load_suite2p_data('path/to/suite2p/plane0')

# Access fluorescence traces
F = data['F']  # Raw fluorescence
Fneu = data['Fneu']  # Neuropil fluorescence
spks = data['spks']  # Deconvolved spikes
iscell = data['iscell']  # Cell classification
```

### Basic Analysis

```python
from suite2p_analysis import analyze_traces, plot_traces

# Analyze traces
results = analyze_traces(F, Fneu, iscell)

# Plot example traces
plot_traces(F, iscell, n_cells=10)
```

### Batch Processing

```python
from suite2p_analysis import batch_process

# Process multiple experiments
experiments = ['exp1/suite2p/plane0', 'exp2/suite2p/plane0']
results = batch_process(experiments)
```

## Project Structure

```
Suite2p-Data-Analysis-/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── src/
│   └── suite2p_analysis/
│       ├── __init__.py
│       ├── loader.py
│       ├── analysis.py
│       ├── visualization.py
│       └── utils.py
├── examples/
│   ├── basic_analysis.py
│   ├── batch_processing.py
│   └── visualization_examples.py
├── data/
│   └── .gitkeep
└── docs/
    ├── getting_started.md
    └── api_reference.md
```

## Features

- **Data Loading**: Easy-to-use functions for loading Suite2p output files
- **Preprocessing**: Neuropil subtraction, baseline correction, and normalization
- **Visualization**: Plot neural activity traces, correlation matrices, and more
- **Statistical Analysis**: Calculate response properties, correlations, and population statistics
- **Batch Processing**: Analyze multiple experiments with consistent parameters

## Documentation

For detailed documentation, see the [docs](./docs) folder:
- [Getting Started Guide](./docs/getting_started.md)
- [API Reference](./docs/api_reference.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use Suite2p in your research, please cite:

Pachitariu, M., Stringer, C., Dipoppa, M., Schröder, S., Rossi, L. F., Dalgleish, H., ... & Harris, K. D. (2017). Suite2p: beyond 10,000 neurons with standard two-photon microscopy. bioRxiv, 061507.

## Acknowledgments

- Suite2p development team
- Contributors to this analysis toolkit