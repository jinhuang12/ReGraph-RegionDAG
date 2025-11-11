"""
IOContract Package - Unified interface for IOContract operations.

This package provides a clean, centralized API for working with IOContracts,
eliminating code duplication across different kernel backends.
"""

from .manager import IOContractManager
from .spec_builders import (
    # Spec creation functions
    create_tensor_spec,
    create_randn_spec,
    create_uniform_spec,
    create_zeros_spec,
    create_ones_spec,
    create_full_spec,
    create_arange_spec,
    # Builder class
    IOContractBuilder
)

# Internal utilities - not exposed in public API
# from .tensor_utils import encode_tensor_to_data, decode_tensor_from_data, materialize_tensor

__all__ = [
    # Main manager
    "IOContractManager",
    # Spec builders
    "create_tensor_spec",
    "create_randn_spec",
    "create_uniform_spec",
    "create_zeros_spec",
    "create_ones_spec",
    "create_full_spec",
    "create_arange_spec",
    "IOContractBuilder",
]