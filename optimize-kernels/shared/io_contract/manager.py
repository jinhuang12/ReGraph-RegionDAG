"""
IOContract Manager - Unified interface for IOContract operations

Provides kernel-agnostic methods for:
- Converting IOContract to inputs list
- Creating IOContract from inputs
- Extracting outputs based on role specifications
"""

import torch
from typing import List, Any, Optional

from ..model import IOContract, ArgSpec, TensorSpec, LaunchConfig
from .tensor_utils import materialize_tensor


class IOContractManager:
    """
    Unified manager for IOContract operations.
    
    Provides a clean, kernel-agnostic API for converting between
    IOContracts and actual input/output values.
    """
    
    def generate_inputs(self, io_contract: IOContract, device: torch.device) -> List[Any]:
        """
        Generate ALL inputs from IOContract specification.
        
        This includes all arguments defined in the contract, including meta parameters.
        Kernel-specific backends can filter as needed during execution.
        
        Args:
            io_contract: IOContract with argument specifications
            device: Target device for tensor creation
            
        Returns:
            List of inputs in the same order as io_contract.args
            
        Raises:
            ValueError: If required specifications are missing
        """
        if not io_contract or not io_contract.args:
            return []
        
        inputs = []
        
        for arg_spec in io_contract.args:
            if arg_spec.type == "tensor":
                if not arg_spec.tensor_spec:
                    raise ValueError(f"Tensor argument '{arg_spec.name}' requires tensor_spec")
                
                # Materialize tensor from specification
                tensor = materialize_tensor(arg_spec.tensor_spec, default_device=device)
                
                # Ensure tensor is on the correct device
                if tensor.device != device:
                    tensor = tensor.to(device)
                    
                inputs.append(tensor)
                
            elif arg_spec.type in ("int", "float", "bool", "str"):
                if arg_spec.value is None:
                    raise ValueError(f"Scalar argument '{arg_spec.name}' requires value")
                inputs.append(arg_spec.value)
                
            else:
                raise ValueError(f"Unsupported argument type: {arg_spec.type}")
        
        return inputs
    
    def create_io_contract(self,
                          inputs: List[Any],
                          output_indices: Optional[List[int]] = None,
                          inout_indices: Optional[List[int]] = None,
                          launch_config: Optional[LaunchConfig] = None,
                          arg_names: Optional[List[str]] = None) -> IOContract:
        """
        Create an IOContract from actual input values.
        
        Automatically infers tensor properties and creates appropriate ArgSpecs.
        
        Args:
            inputs: List of input values (tensors and scalars)
            output_indices: Indices of arguments that are outputs (default: none)
            inout_indices: Indices of arguments that are input/outputs (default: none)
            launch_config: Optional kernel launch configuration
            arg_names: Optional list of argument names (default: arg_0, arg_1, ...)
            
        Returns:
            IOContract with inferred specifications
            
        Example:
            >>> x = torch.randn(32, 64)
            >>> y = torch.randn(32, 64)
            >>> result = torch.empty(32, 64)
            >>> contract = manager.create_io_contract(
            ...     [x, y, result],
            ...     output_indices=[2]  # result is output
            ... )
        """
        output_indices = output_indices or []
        inout_indices = inout_indices or []
        arg_names = arg_names or [f"arg_{i}" for i in range(len(inputs))]
        
        if len(arg_names) != len(inputs):
            raise ValueError(f"Number of names ({len(arg_names)}) must match number of inputs ({len(inputs)})")
        
        args = []
        
        for i, (inp, name) in enumerate(zip(inputs, arg_names)):
            # Determine role
            if i in output_indices:
                role = "output"
            elif i in inout_indices:
                role = "inout"
            else:
                role = "input"
            
            if torch.is_tensor(inp):
                # Create tensor ArgSpec
                args.append(ArgSpec(
                    name=name,
                    type="tensor",
                    tensor_spec=self._tensor_to_spec(inp),
                    role=role
                ))
            else:
                # Create scalar ArgSpec
                type_name = self._get_scalar_type_name(inp)
                args.append(ArgSpec(
                    name=name,
                    type=type_name,
                    value=inp,
                    role="input"  # Scalars are always inputs
                ))
        
        return IOContract(args=args, launch=launch_config)
    
    def extract_outputs(self, inputs: List[Any], io_contract: IOContract) -> Any:
        """
        Extract output values from inputs based on IOContract role specifications.
        
        Args:
            inputs: List of all input/output values after kernel execution
            io_contract: IOContract specifying roles
            
        Returns:
            Single tensor if one output, tuple of tensors if multiple, None if no outputs
            
        Raises:
            ValueError: If inputs length doesn't match contract
        """
        if not io_contract or not io_contract.args:
            return None
        
        if len(inputs) != len(io_contract.args):
            raise ValueError(
                f"Number of inputs ({len(inputs)}) doesn't match "
                f"number of contract arguments ({len(io_contract.args)})"
            )
        
        outputs = []
        
        for i, arg_spec in enumerate(io_contract.args):
            if arg_spec.role in ("output", "inout"):
                outputs.append(inputs[i])
        
        if not outputs:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)
    
    def _tensor_to_spec(self, tensor: torch.Tensor) -> TensorSpec:
        """
        Create a TensorSpec from an existing tensor with its actual data.
        
        This preserves the exact tensor values by encoding them as TensorData.
        """
        from .tensor_utils import encode_tensor_to_data
        
        # Encode the actual tensor data
        tensor_data = encode_tensor_to_data(tensor, compress="zlib")
        
        return TensorSpec(
            shape=list(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            data=tensor_data  # Include actual tensor data
        )
    
    def _get_scalar_type_name(self, value: Any) -> str:
        """Get the type name for scalar arguments."""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "str"
        else:
            # Fallback to Python type name
            return type(value).__name__