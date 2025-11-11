"""
Helper functions and builder classes for creating IOContract specifications.

Provides convenient methods for creating TensorSpecs and IOContracts.
"""

import json
from typing import Dict, List, Optional, Any, Union
from ..model import (
    KernelCode, TensorSpec, TensorInit, ArgSpec, 
    IOContract, LaunchConfig, LaunchDim
)


# ---------------------------------------------------------------------
# Helper Functions for Creating Tensor Specifications
# ---------------------------------------------------------------------

def create_tensor_spec(
    shape: List[int], 
    dtype: str = "float32",
    init_kind: Optional[str] = None,
    seed: Optional[int] = None,
    **init_kwargs
) -> TensorSpec:
    """
    Create a TensorSpec with server-side generation.
    
    Args:
        shape: Tensor dimensions
        dtype: Data type (float32, float64, int32, etc.)
        init_kind: Generation method (randn, zeros, ones, uniform, full, arange)
        seed: Random seed for reproducibility
        **init_kwargs: Additional init parameters (mean, std, low, high, fill_value, start, step)
    
    Returns:
        TensorSpec configured for server-side generation
    
    Examples:
        # Random normal tensor with mean=5, std=2
        spec = create_tensor_spec([128, 256], "float32", "randn", seed=42, mean=5.0, std=2.0)
        
        # Uniform distribution between -1 and 1
        spec = create_tensor_spec([1024], "float32", "uniform", low=-1.0, high=1.0)
        
        # All zeros
        spec = create_tensor_spec([64, 64], "float32", "zeros")
    """
    if init_kind is None:
        return TensorSpec(shape=shape, dtype=dtype)
    
    init = TensorInit(kind=init_kind, seed=seed, **init_kwargs)
    return TensorSpec(shape=shape, dtype=dtype, init=init)


def create_randn_spec(
    shape: List[int], 
    dtype: str = "float32", 
    seed: Optional[int] = None,
    mean: float = 0.0, 
    std: float = 1.0
) -> TensorSpec:
    """Create a TensorSpec for random normal distribution."""
    return create_tensor_spec(shape, dtype, "randn", seed=seed, mean=mean, std=std)


def create_uniform_spec(
    shape: List[int], 
    dtype: str = "float32", 
    seed: Optional[int] = None,
    low: float = 0.0, 
    high: float = 1.0
) -> TensorSpec:
    """Create a TensorSpec for uniform distribution."""
    return create_tensor_spec(shape, dtype, "uniform", seed=seed, low=low, high=high)


def create_zeros_spec(shape: List[int], dtype: str = "float32") -> TensorSpec:
    """Create a TensorSpec for all zeros."""
    return create_tensor_spec(shape, dtype, "zeros")


def create_ones_spec(shape: List[int], dtype: str = "float32") -> TensorSpec:
    """Create a TensorSpec for all ones."""
    return create_tensor_spec(shape, dtype, "ones")


def create_full_spec(shape: List[int], fill_value: float, dtype: str = "float32") -> TensorSpec:
    """Create a TensorSpec filled with a constant value."""
    return create_tensor_spec(shape, dtype, "full", fill_value=fill_value)


def create_arange_spec(
    shape: List[int], 
    dtype: str = "float32", 
    start: float = 0.0, 
    step: float = 1.0
) -> TensorSpec:
    """Create a TensorSpec for sequential values (arange)."""
    return create_tensor_spec(shape, dtype, "arange", start=start, step=step)


# ---------------------------------------------------------------------
# IOContract Builder Class
# ---------------------------------------------------------------------

class IOContractBuilder:
    """
    Fluent interface for building IOContracts.
    
    Example:
        contract = IOContractBuilder() \\
            .add_input_tensor("x", create_randn_spec([1024, 512], seed=42)) \\
            .add_input_tensor("y", create_ones_spec([512, 256])) \\
            .add_output_tensor("result", [1024, 256], "float32") \\
            .add_scalar("alpha", "float", 2.0) \\
            .add_meta_param("BLOCK_SIZE", 256) \\
            .set_launch_config(launch_config) \\
            .build()
    """
    
    def __init__(self):
        self.args: List[ArgSpec] = []
        self.launch_config: Optional[LaunchConfig] = None
    
    def add_input_tensor(self, name: str, tensor_spec: TensorSpec) -> "IOContractBuilder":
        """Add an input tensor argument."""
        self.args.append(ArgSpec(
            name=name,
            type="tensor",
            tensor_spec=tensor_spec,
            role="input"
        ))
        return self
    
    def add_output_tensor(self, name: str, shape: List[int], dtype: str = "float32") -> "IOContractBuilder":
        """Add an output tensor argument."""
        tensor_spec = TensorSpec(shape=shape, dtype=dtype)
        self.args.append(ArgSpec(
            name=name,
            type="tensor",
            tensor_spec=tensor_spec,
            role="output"
        ))
        return self
    
    def add_inout_tensor(self, name: str, tensor_spec: TensorSpec) -> "IOContractBuilder":
        """Add an input/output tensor argument."""
        self.args.append(ArgSpec(
            name=name,
            type="tensor",
            tensor_spec=tensor_spec,
            role="inout"
        ))
        return self
    
    def add_scalar(self, name: str, scalar_type: str, value: Union[int, float, str, bool]) -> "IOContractBuilder":
        """Add a scalar argument (int, float, str, bool)."""
        self.args.append(ArgSpec(
            name=name,
            type=scalar_type,
            value=value,
            role="input"
        ))
        return self
    
    def add_meta_param(self, name: str, value: int) -> "IOContractBuilder":
        """Add a meta parameter (Triton constexpr)."""
        self.args.append(ArgSpec(
            name=name,
            type="int",
            value=value,
            role="input",
            is_meta=True
        ))
        return self
    
    def set_launch_config(self, launch_config: LaunchConfig) -> "IOContractBuilder":
        """Set the kernel launch configuration."""
        self.launch_config = launch_config
        return self
    
    def build(self) -> IOContract:
        """Build the final IOContract."""
        return IOContract(
            args=self.args,
            launch=self.launch_config
        )
    
# ---------------------------------------------------------------------
# JSON Serialization Helpers
# ---------------------------------------------------------------------

def to_json(obj: Union[IOContract, KernelCode, Dict], pretty: bool = False) -> str:
    """Convert any IOContract object to JSON string."""
    if hasattr(obj, 'to_dict'):
        data = obj.to_dict()
    else:
        data = obj
    
    if pretty:
        return json.dumps(data, indent=2)
    return json.dumps(data)


def save_to_file(obj: Union[IOContract, KernelCode, Dict], filename: str, pretty: bool = True):
    """Save IOContract object to JSON file."""
    with open(filename, 'w') as f:
        f.write(to_json(obj, pretty=pretty))


def load_from_file(filename: str, obj_type: str = "IOContract"):
    """
    Load IOContract object from JSON file.
    
    Args:
        filename: Path to JSON file
        obj_type: Type to deserialize ("IOContract", "KernelCode", or "dict")
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    if obj_type == "IOContract":
        return IOContract.from_dict(data)
    elif obj_type == "KernelCode":
        return KernelCode.from_dict(data)
    else:
        return data    