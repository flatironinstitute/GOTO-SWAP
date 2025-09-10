# GOTO-SWAP

**G**radient-based **O**ptimal **T**ransport **O**ptimization -- **S**liced **W**asserstein **A**PI for **P**ipelines

A unified, lightweight API for sliced Wasserstein image targets with multi-backend support (JAX and PyTorch). This package makes it easy to integrate sliced Wasserstein distance computations into your machine learning pipelines.

## Features

- 🚀 **Multi-Backend Support**: Choose between JAX and PyTorch backends
- ⚡ **High Performance**: JIT compilation with JAX, optimized PyTorch operations
- 🔧 **Easy Integration**: Simple API that works with existing ML workflows
- 📦 **Lightweight**: Optional dependencies - only install what you need
- 🧮 **Differentiable**: Full automatic differentiation support
- 🎯 **Specialized**: 1D Earth Mover's Distance (EMD) implementations

## Installation

### Minimal Installation
```bash
pip install goto-swap
```

### Backend-Specific Installation

**JAX Backend:**
```bash
pip install goto-swap[jax]
```

**PyTorch Backend:**
```bash
pip install goto-swap[pytorch]
```

**All Backends:**
```bash
pip install goto-swap[all]
```

## Quick Start

### Basic Usage

```python
import numpy as np
from goto_swap import wasserstein_distance_loss_images, emd_1d_sorted_differentiable

# Prepare your data
x = np.random.randn(2, 32, 32)  # Predicted images
x_true = np.random.randn(2, 32, 32)  # True images

# JAX backend (default)
loss = wasserstein_distance_loss_images(x, x_true)

# PyTorch backend
loss = wasserstein_distance_loss_images(x, x_true, backend="pytorch")
```

### 1D Earth Mover's Distance

```python
import jax.numpy as jnp
from goto_swap import emd_1d_sorted_differentiable

# Sorted positions and weights
u = jnp.array([1.0, 2.0, 3.0])  # Source positions
v = jnp.array([1.5, 2.5, 3.5])  # Target positions
u_weights = jnp.array([0.3, 0.4, 0.3])  # Source weights
v_weights = jnp.array([0.2, 0.5, 0.3])  # Target weights

# Compute EMD
emd_loss = emd_1d_sorted_differentiable(u, v, u_weights, v_weights)
```

### Advanced Usage

```python
from goto_swap import get_backend

# Get specific backend
jax_backend = get_backend("jax")
pytorch_backend = get_backend("pytorch")

# Use backend directly
loss_jax = jax_backend.wasserstein_distance_loss_images(x, x_true)
loss_pytorch = pytorch_backend.wasserstein_distance_loss_images(x, x_true)

# Direct function imports
from goto_swap.backends.jax import emd_1d_sorted_jax_differentiable
from goto_swap.backends.pytorch import emd_1d_sorted_pytorch_differentiable
```

## API Reference

### Main Functions

- `wasserstein_distance_loss_images(x, x_true, num_rotations=100, backend="jax")`: Compute sliced Wasserstein distance between images
- `emd_1d_sorted_differentiable(u, v, u_weights, v_weights, metric='sqeuclidean', tol=1e-12, backend="jax")`: Compute 1D Earth Mover's Distance
- `get_backend(backend_name)`: Get backend instance for advanced usage

### Backend Selection

| Backend | JIT Compilation | GPU Support | TPU Support | Best For |
|---------|----------------|-------------|-------------|----------|
| JAX     | ✅ Native      | ✅          | ✅          | Research, high-performance computing |
| PyTorch | ❌ No          | ✅          | ❌          | Integration with PyTorch workflows |

## Examples

### JAX with JIT Compilation

```python
import jax
import jax.numpy as jnp
from goto_swap import wasserstein_distance_loss_images

# JIT compile for performance
jit_loss = jax.jit(wasserstein_distance_loss_images, static_argnames=['backend'])

# Use compiled function
loss = jit_loss(x, x_true, backend="jax")
```

### PyTorch with Gradients

```python
import torch
from goto_swap import emd_1d_sorted_differentiable

u = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
v = torch.tensor([1.5, 2.5, 3.5])
u_weights = torch.tensor([0.3, 0.4, 0.3])
v_weights = torch.tensor([0.2, 0.5, 0.3])

loss = emd_1d_sorted_differentiable(u, v, u_weights, v_weights, backend="pytorch")
loss.backward()  # Compute gradients
```

### Batch Processing

```python
import jax.numpy as jnp
from goto_swap import wasserstein_distance_loss_images

# Process multiple image pairs
batch_x = jnp.random.randn(10, 32, 32)  # 10 images
batch_x_true = jnp.random.randn(10, 32, 32)

# Vectorized computation
losses = jax.vmap(wasserstein_distance_loss_images)(batch_x, batch_x_true)
```

## Architecture

GOTO-SWAP uses a modular backend system that allows you to choose between JAX and PyTorch implementations:

```
src/goto_swap/
├── __init__.py                 # Main package exports
├── base.py                     # Legacy compatibility (deprecated)
└── backends/                   # Backend system
    ├── __init__.py             # Backend factory and interface
    ├── jax/                    # JAX backend implementation
    │   ├── backend.py          # JAXBackend class
    │   └── emd.py              # JAX EMD implementation
    └── pytorch/                # PyTorch backend implementation
        ├── backend.py          # PyTorchBackend class
        └── emd.py              # PyTorch EMD implementation
```

### Backend Features

| Feature | JAX Backend | PyTorch Backend |
|---------|-------------|-----------------|
| JIT Compilation | ✅ Native | ❌ No |
| Automatic Differentiation | ✅ Native | ✅ Native |
| GPU Support | ✅ | ✅ |
| TPU Support | ✅ | ❌ |
| Memory Efficiency | ✅ | ✅ |
| Ecosystem Integration | JAX/Flax | PyTorch |

## Testing

Run tests for specific backends:

```bash
# Test JAX backend
python run_tests.py --backend jax

# Test PyTorch backend  
python run_tests.py --backend pytorch

# Test all available backends
python run_tests.py
```

### Error Handling

If you try to use a backend that isn't installed:

```python
# This will raise ImportError with helpful message
loss = wasserstein_distance_loss_images(x, x_true, backend="pytorch")
# ImportError: PyTorch backend requested but PyTorch is not installed. 
# Install with: pip install torch torchvision
```

## Migration from Legacy Code

If you're using the legacy `goto_swap.base` module:

```python
# Old (deprecated)
from goto_swap.base import wasserstein_distance_loss_images

# New (recommended)
from goto_swap import wasserstein_distance_loss_images
```

## Related Projects

- **GOTO**: Main project repository with experimental codes - https://github.com/minhuanli/GOTO

## Development

### Adding New Backends

To add a new backend (e.g., TensorFlow):

1. Create a new directory under `backends/`
2. Implement the `BackendInterface` abstract base class
3. Add dependencies to `pyproject.toml`
4. Update the factory function in `backends/__init__.py`

Example structure:
```
backends/tensorflow/
├── __init__.py
├── backend.py      # TensorFlowBackend class
└── emd.py          # TensorFlow EMD implementation
```

### Testing

The test suite includes:
- Backend interface compliance tests
- Backend-specific functionality tests
- Cross-backend consistency tests
- Error handling tests

Run tests with:
```bash
python run_tests.py --backend jax
python run_tests.py --backend pytorch
```

## Contributing

We welcome contributions! Please see our [testing guide](TESTING.md) for development details.

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.

## Citation

If you use GOTO-SWAP in your research, please cite:

```bibtex
@software{goto_swap,
  title={GOTO-SWAP: Gradient-based Optimal Transport Optimization -- Sliced Wasserstein API for Pipelines},
  author={Li, Minhuan and Woollard, Geoffrey and Herreros, David},
  year={2024},
  url={https://github.com/flatironinstitute/GOTO-SWAP}
}
```
