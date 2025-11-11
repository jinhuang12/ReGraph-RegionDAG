#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workers
-------
Subprocess worker modules for kernel execution and state management.

This package contains:
- state_manager: Incremental state persistence for crash recovery
- triton_runner: Standalone Triton kernel execution
- cuda_runner: Standalone CUDA kernel execution
"""

__version__ = "1.0.0"

__all__ = [
    "state_manager",
    "triton_runner",
    "cuda_runner",
]
