from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("goto-swap")
except PackageNotFoundError:
    __version__ = "uninstalled"

# Import the new backend system
from .backends import wasserstein_distance_loss_images, get_backend

# For backward compatibility, also import the old function name
# This will use the default JAX backend
__all__ = ["wasserstein_distance_loss_images", "get_backend", "__version__"]