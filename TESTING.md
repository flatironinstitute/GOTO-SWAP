# Testing Guide for GOTO-SWAP

This document explains how to run tests for the GOTO-SWAP backend system.

## Test Structure

The tests are organized in the `tests/` directory:

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_backends.py         # Backend system tests
└── test_placeholder.py      # Placeholder test
```

## Test Categories

Tests are organized by backend and functionality:

- **`TestBackendSystem`**: Core backend system functionality
- **`TestJAXBackend`**: JAX-specific tests (marked with `@pytest.mark.jax`)
- **`TestPyTorchBackend`**: PyTorch-specific tests (marked with `@pytest.mark.pytorch`)
- **`TestBackendCompatibility`**: Backend compatibility and consistency tests
- **`TestBackendErrors`**: Error handling tests

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-cov
```

### Basic Test Commands

#### Run All Available Tests
```bash
# Using pytest directly
pytest tests/

# Using the test runner script
python run_tests.py
```

#### Run Tests for Specific Backend

**JAX Backend Only:**
```bash
# Using pytest directly
pytest -m jax tests/

# Using the test runner script
python run_tests.py --backend jax
```

**PyTorch Backend Only:**
```bash
# Using pytest directly
pytest -m pytorch tests/

# Using the test runner script
python run_tests.py --backend pytorch
```

#### Run with Verbose Output
```bash
pytest -v tests/
python run_tests.py --verbose
```

#### Run with Coverage
```bash
pytest --cov=src/goto_swap --cov-report=term-missing tests/
python run_tests.py --coverage
```

### Check Backend Availability

Before running tests, check which backends are available:
```bash
python run_tests.py --check
```

This will show which backends are installed and ready for testing.

## Test Fixtures

The `conftest.py` file provides several fixtures:

- **`test_data`**: Standard test data (2x32x32 arrays)
- **`test_data_small`**: Smaller test data (1x16x16 arrays) for faster testing

## Test Markers

Tests are marked with pytest markers for easy filtering:

- `@pytest.mark.jax`: Requires JAX backend
- `@pytest.mark.pytorch`: Requires PyTorch backend
- `@pytest.mark.backend`: Backend-specific test
- `@pytest.mark.slow`: Slow-running test

## Example Test Output

```bash
$ python run_tests.py --backend jax
Checking backend availability...
✅ JAX backend available
❌ PyTorch backend not available: No module named 'torch'

Running tests for jax backend...
Command: python -m pytest -m jax tests/
--------------------------------------------------
tests/test_backends.py::TestJAXBackend::test_jax_backend_creation PASSED
tests/test_backends.py::TestJAXBackend::test_jax_wasserstein_loss PASSED
tests/test_backends.py::TestJAXBackend::test_jax_convenience_function PASSED
--------------------------------------------------
✅ All tests passed!
```

## Continuous Integration

For CI/CD pipelines, you can run tests for specific backends:

```yaml
# Example GitHub Actions workflow
- name: Test JAX Backend
  run: |
    pip install goto-swap[jax]
    pytest -m jax tests/

- name: Test PyTorch Backend  
  run: |
    pip install goto-swap[pytorch]
    pytest -m pytorch tests/
```

## Troubleshooting

### Common Issues

1. **ImportError for backend**: Install the required backend dependencies
   ```bash
   pip install goto-swap[jax]  # or goto-swap[pytorch]
   ```

2. **Test failures**: Check that the backend is properly installed and functional
   ```bash
   python run_tests.py --check
   ```

3. **Slow tests**: Use smaller test data or reduce `num_rotations` parameter in tests

### Debug Mode

Run tests with maximum verbosity for debugging:
```bash
pytest -vvv --tb=long tests/
```

## Adding New Tests

When adding new backend tests:

1. Use appropriate pytest markers (`@pytest.mark.jax`, `@pytest.mark.pytorch`)
2. Use the provided fixtures (`test_data`, `test_data_small`)
3. Handle ImportError gracefully with `pytest.skip()`
4. Test both success and error cases
5. Ensure tests are deterministic (use fixed random seeds if needed)
