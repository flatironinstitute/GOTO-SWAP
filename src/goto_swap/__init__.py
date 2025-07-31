from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("goto-swap")
except PackageNotFoundError:
    __version__ = "uninstalled"

from .base import wasserstein_distance_loss_images