# GOTO-SWAP Backend Architecture

## Overview

GOTO-SWAP has been restructured to support multiple computational backends (JAX and PyTorch) with optional dependencies. This allows users to choose their preferred framework while keeping installation lightweight.

## Directory Structure

```
src/goto_swap/
├── __init__.py                 # Main package exports
├── base.py                     # Legacy compatibility (deprecated)
├── _command_line.py            # CLI interface
└── backends/                   # Backend system
    ├── __init__.py             # Backend factory and interface
    ├── jax/                    # JAX backend implementation
    │   ├── __init__.py
    │   └── backend.py          # JAXBackend class
    └── pytorch/                # PyTorch backend implementation
        ├── __init__.py
        └── backend.py          # PyTorchBackend class
```

## Architecture Components

### 1. Backend Interface (`backends/__init__.py`)

- **`BackendInterface`**: Abstract base class defining the common API
- **`get_backend(backend_name)`**: Factory function to get backend instances
- **`wasserstein_distance_loss_images()`**: Convenience function with backend selection

### 2. JAX Backend (`backends/jax/backend.py`)

- **`JAXBackend`**: Implements the interface using JAX operations
- Features:
  - JIT compilation with `@jax.jit`
  - Uses `dm_pix` for image operations
  - Uses `flax.nnx` for activation functions
  - Full automatic differentiation support

### 3. PyTorch Backend (`backends/pytorch/backend.py`)

- **`PyTorchBackend`**: Implements the interface using PyTorch operations
- Features:
  - Native PyTorch tensors and operations
  - Custom Gaussian blur implementation
  - Custom rotation implementation using `F.affine_grid`
  - Full automatic differentiation support

### 4. Legacy Compatibility (`base.py`)

- Maintains backward compatibility with existing code
- Issues deprecation warnings
- Redirects to new backend system

## Installation Strategy

### Core Dependencies
```toml
dependencies = [
    "numpy>=1.20.0",
    # Backend-specific dependencies are now optional
]
```

### Optional Dependencies
```toml
[project.optional-dependencies]
jax = ["jax>=0.4.0", "dm_pix>=0.4.0", "flax>=0.7.0"]
pytorch = ["torch>=2.0.0", "torchvision>=0.15.0"]
all = [jax + pytorch dependencies]
```

## Usage Patterns

### 1. Default Usage (JAX)
```python
from goto_swap import wasserstein_distance_loss_images
loss = wasserstein_distance_loss_images(x, x_true)
```

### 2. Backend Selection
```python
from goto_swap import wasserstein_distance_loss_images
loss = wasserstein_distance_loss_images(x, x_true, backend="pytorch")
```

### 3. Direct Backend Access
```python
from goto_swap import get_backend
backend = get_backend("jax")
loss = backend.wasserstein_distance_loss_images(x, x_true)
```

### 4. Legacy Usage (Deprecated)
```python
from goto_swap.base import wasserstein_distance_loss_images  # Shows warning
loss = wasserstein_distance_loss_images(x, x_true)
```

## Error Handling

- **Missing Backend**: Clear error messages with installation instructions
- **Import Errors**: Graceful fallback with helpful error messages
- **Deprecation**: Warnings for legacy usage with migration guidance

## Benefits

1. **Lightweight Installation**: Only install what you need
2. **Framework Flexibility**: Choose JAX or PyTorch based on your workflow
3. **Backward Compatibility**: Existing code continues to work
4. **Extensibility**: Easy to add new backends (e.g., TensorFlow)
5. **Performance**: Each backend optimized for its framework

## Future Extensions

The architecture supports adding new backends by:

1. Creating a new directory under `backends/`
2. Implementing the `BackendInterface`
3. Adding dependencies to `pyproject.toml`
4. Updating the factory function

Example for TensorFlow backend:
```
backends/tensorflow/
├── __init__.py
└── backend.py  # TensorFlowBackend class
```

## Testing

Run the test script to verify functionality:
```bash
python test_backends.py
```

This will test all available backends and show which ones are installed.

