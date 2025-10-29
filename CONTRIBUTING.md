# Contributing to Suite2p Data Analysis

Thank you for your interest in contributing to the Suite2p Data Analysis toolkit! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check if the issue already exists in the GitHub issue tracker
2. If not, create a new issue with a clear title and description
3. Include relevant information such as:
   - Steps to reproduce the issue
   - Expected behavior
   - Actual behavior
   - Your environment (Python version, OS, etc.)
   - Example code or data if applicable

### Submitting Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes, following the coding standards below
4. Add or update tests if applicable
5. Update documentation if needed
6. Commit your changes with clear, descriptive commit messages
7. Push to your fork and submit a pull request

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Write docstrings for all functions and classes (Google style)
- Keep functions focused on a single task
- Add type hints where appropriate

### Documentation

- Update docstrings when modifying functions
- Add examples to docstrings for new features
- Update relevant documentation files in `docs/`
- Include comments for complex logic

### Testing

- Write tests for new functionality
- Ensure existing tests pass
- Aim for good test coverage

## Code Review Process

1. All submissions require review before merging
2. Reviewers will check for:
   - Code quality and style
   - Documentation completeness
   - Test coverage
   - Backwards compatibility
3. Address reviewer feedback
4. Once approved, a maintainer will merge your PR

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/karu264/Suite2p-Data-Analysis-.git
   cd Suite2p-Data-Analysis-
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Areas for Contribution

We welcome contributions in these areas:

- **Bug fixes**: Help squash bugs!
- **New analysis methods**: Add new analysis techniques
- **Visualization improvements**: Create new plot types
- **Documentation**: Improve docs and examples
- **Performance**: Optimize slow functions
- **Testing**: Increase test coverage
- **Examples**: Add new example scripts

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Reach out to the maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
