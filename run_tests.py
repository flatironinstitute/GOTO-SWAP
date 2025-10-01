#!/usr/bin/env python3
"""
Test runner script for GOTO-SWAP backend tests.

This script allows running tests for specific backends or all available backends.
"""

import sys
import subprocess
import argparse


def run_tests(backend=None, verbose=False, coverage=False):
    """
    Run tests for specified backend or all available backends.
    
    Args:
        backend: Backend to test ("jax", "pytorch", or None for all)
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src/goto_swap", "--cov-report=term-missing"])
    
    if backend:
        # Run tests for specific backend
        cmd.extend(["--backend", backend])
        print(f"Running tests for {backend} backend...")
    else:
        # Run all tests
        print("Running all available tests...")
    
    # Add test directory
    cmd.append("tests/")
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 50)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest")
        return False


def check_backend_availability():
    """Check which backends are available."""
    print("Checking backend availability...")
    
    backends = {}
    
    # Check JAX
    try:
        import jax
        import dm_pix
        import flax
        backends["jax"] = True
        print("✅ JAX backend available")
    except ImportError as e:
        backends["jax"] = False
        print(f"❌ JAX backend not available: {e}")
    
    # Check PyTorch
    try:
        import torch
        import torchvision
        backends["pytorch"] = True
        print("✅ PyTorch backend available")
    except ImportError as e:
        backends["pytorch"] = False
        print(f"❌ PyTorch backend not available: {e}")
    
    return backends


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run GOTO-SWAP backend tests")
    parser.add_argument(
        "--backend", 
        choices=["jax", "pytorch"], 
        help="Test specific backend only"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "--coverage", "-c", 
        action="store_true", 
        help="Enable coverage reporting"
    )
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Check backend availability only"
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_backend_availability()
        return
    
    # Check backend availability
    available_backends = check_backend_availability()
    
    if args.backend and not available_backends.get(args.backend, False):
        print(f"❌ {args.backend} backend is not available. Install dependencies first.")
        print(f"   pip install goto-swap[{args.backend}]")
        sys.exit(1)
    
    # Run tests
    success = run_tests(
        backend=args.backend,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
