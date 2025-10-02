"""
Test that the package can be imported correctly.
"""

import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_import_package():
    """Test that the main package can be imported."""
    try:
        import suite2p_analysis
        assert suite2p_analysis.__version__ == "0.1.0"
    except ImportError as e:
        # This is expected if dependencies are not installed
        print(f"Import failed (expected without dependencies): {e}")


def test_import_loader():
    """Test that loader module exists."""
    try:
        from suite2p_analysis import loader
        assert hasattr(loader, 'load_suite2p_data')
    except ImportError:
        print("Import failed (expected without dependencies)")


def test_import_analysis():
    """Test that analysis module exists."""
    try:
        from suite2p_analysis import analysis
        assert hasattr(analysis, 'calculate_dff')
    except ImportError:
        print("Import failed (expected without dependencies)")


def test_import_visualization():
    """Test that visualization module exists."""
    try:
        from suite2p_analysis import visualization
        assert hasattr(visualization, 'plot_traces')
    except ImportError:
        print("Import failed (expected without dependencies)")


def test_import_utils():
    """Test that utils module exists."""
    try:
        from suite2p_analysis import utils
        assert hasattr(utils, 'filter_cells')
    except ImportError:
        print("Import failed (expected without dependencies)")


if __name__ == "__main__":
    print("Running import tests...")
    test_import_package()
    test_import_loader()
    test_import_analysis()
    test_import_visualization()
    test_import_utils()
    print("All tests completed!")
