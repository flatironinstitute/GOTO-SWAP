# Backend Migration Guide

GOTO-SWAP now supports multiple computational backends (JAX and PyTorch) with optional dependencies. This allows users to choose their preferred framework while keeping installation lightweight.

## Installation

### Minimal Installation
```bash
pip install goto-swap
```
This installs only the core package with NumPy. You'll need to install backend-specific dependencies separately.

### Backend-Specific Installation

#### JAX Backend
```bash
pip install goto-swap[jax]
```

#### PyTorch Backend  
```bash
pip install goto-swap[pytorch]
```

#### All Backends
```bash
pip install goto-swap[all]
```

## Usage

### New Backend System (Recommended)

```python
from goto_swap import wasserstein_distance_loss_images, emd_1d_sorted_differentiable, get_backend

# Use default JAX backend
loss = wasserstein_distance_loss_images(x, x_true)

# Use specific backend
loss = wasserstein_distance_loss_images(x, x_true, backend="pytorch")

# EMD 1D function
emd_loss = emd_1d_sorted_differentiable(u, v, u_weights, v_weights, backend="jax")

# Get backend instance directly for more control
jax_backend = get_backend("jax")
pytorch_backend = get_backend("pytorch")

loss_jax = jax_backend.wasserstein_distance_loss_images(x, x_true)
loss_pytorch = pytorch_backend.wasserstein_distance_loss_images(x, x_true)
emd_loss_jax = jax_backend.emd_1d_sorted_differentiable(u, v, u_weights, v_weights)

# Direct EMD function imports (for advanced usage)
from goto_swap.backends.jax import emd_1d_sorted_jax_differentiable
from goto_swap.backends.pytorch import emd_1d_sorted_pytorch_differentiable
emd_loss_jax_direct = emd_1d_sorted_jax_differentiable(u, v, u_weights, v_weights)
emd_loss_pytorch_direct = emd_1d_sorted_pytorch_differentiable(u, v, u_weights, v_weights)
```

### Legacy Usage (Deprecated)

```python
# This still works but shows a deprecation warning
from goto_swap.base import wasserstein_distance_loss_images
loss = wasserstein_distance_loss_images(x, x_true)
```

## Backend Comparison

| Feature | JAX Backend | PyTorch Backend |
|---------|-------------|-----------------|
| JIT Compilation | ✅ Native | ❌ No |
| Automatic Differentiation | ✅ Native | ✅ Native |
| GPU Support | ✅ | ✅ |
| TPU Support | ✅ | ❌ |
| Memory Efficiency | ✅ | ✅ |
| Ecosystem Integration | JAX/Flax | PyTorch |

## Migration Steps

1. **Update imports**:
   ```python
   # Old
   from goto_swap.base import wasserstein_distance_loss_images
   
   # New
   from goto_swap import wasserstein_distance_loss_images
   ```

2. **Install backend dependencies**:
   ```bash
   # For JAX users
   pip install goto-swap[jax]
   
   # For PyTorch users  
   pip install goto-swap[pytorch]
   ```

3. **Update function calls** (if using PyTorch):
   ```python
   # Old (JAX only)
   loss = wasserstein_distance_loss_images(x, x_true)
   
   # New (PyTorch)
   loss = wasserstein_distance_loss_images(x, x_true, backend="pytorch")
   ```

## Error Handling

If you try to use a backend that isn't installed:

```python
# This will raise ImportError with helpful message
loss = wasserstein_distance_loss_images(x, x_true, backend="pytorch")
# ImportError: PyTorch backend requested but PyTorch is not installed. 
# Install with: pip install torch torchvision
```

## Performance Notes

- **JAX Backend**: Optimized with JIT compilation, best for research and high-performance computing
- **PyTorch Backend**: Good for integration with existing PyTorch workflows, slightly slower due to no JIT compilation
- Both backends produce numerically equivalent results

## Development

To contribute to backend implementations:

1. **JAX Backend**: `src/goto_swap/backends/jax/backend.py`
2. **PyTorch Backend**: `src/goto_swap/backends/pytorch/backend.py`
3. **Interface**: `src/goto_swap/backends/__init__.py`

All backends must implement the `BackendInterface` abstract base class.

