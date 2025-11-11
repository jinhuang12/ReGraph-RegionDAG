"""
Internal tensor utilities for IOContract operations.

Handles tensor encoding/decoding and materialization from specifications.
"""

import base64
import zlib
import numpy as np
import torch
from typing import Optional, Union

from ..model import TensorData, TensorSpec

# Dtype mapping dictionaries
_STR2TORCH = {
    "float64": torch.float64, "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "int64": torch.int64, "int32": torch.int32,
    "int16": torch.int16, "int8": torch.int8, "uint8": torch.uint8, "bool": torch.bool,
}

_STR2NP = {
    "float64": np.float64, "float32": np.float32, "float16": np.float16,
    "int64": np.int64, "int32": np.int32, "int16": np.int16,
    "int8": np.int8, "uint8": np.uint8, "bool": np.bool_
}


def encode_tensor_to_data(t: torch.Tensor, compress: str = "none") -> TensorData:
    """
    Convert a torch tensor to JSON-friendly TensorData (base64).
    
    Args:
        t: Torch tensor to encode
        compress: Compression method ("none" or "zlib")
        
    Returns:
        TensorData with base64-encoded data
    """
    if t.is_cuda:
        t = t.detach().cpu()
    arr = t.detach().contiguous().numpy()
    raw = arr.tobytes(order="C")
    if compress == "zlib":
        raw = zlib.compress(raw)
    b64 = base64.b64encode(raw).decode("ascii")
    return TensorData(
        data_b64=b64, 
        compress=compress, 
        shape=list(arr.shape),
        dtype=str(arr.dtype)
    )


def decode_tensor_from_data(td: TensorData, device: Union[str, torch.device] = "cuda") -> torch.Tensor:
    """
    Decode JSON TensorData back to a torch tensor on device.
    
    Args:
        td: TensorData to decode
        device: Target device for tensor
        
    Returns:
        Decoded torch tensor
    """
    buf = base64.b64decode(td.data_b64)
    if td.compress == "zlib":
        buf = zlib.decompress(buf)
    np_dt = _STR2NP[td.dtype]
    arr = np.frombuffer(buf, dtype=np_dt)
    # Reshape C-contiguously
    arr = arr.reshape(td.shape)
    t = torch.from_numpy(arr.copy())  # own memory
    if isinstance(device, str):
        device = torch.device(device)
    return t.to(device=device)


def materialize_tensor(spec: TensorSpec, default_device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """
    Create a tensor from spec (prefer literal data; else generate via init).
    
    Args:
        spec: TensorSpec defining the tensor
        default_device: Default device if not specified in spec
        
    Returns:
        Materialized tensor
    """
    if spec.data is not None:
        # Use device from spec if available, otherwise use default
        device = getattr(spec, 'device', default_device or 'cuda')
        return decode_tensor_from_data(spec.data, device=device)
    
    # Generate by init (deterministic if seed set)
    dt = _STR2TORCH[spec.dtype]
    # Use device from spec if available, otherwise use default
    device_str = getattr(spec, 'device', default_device or 'cuda')
    if isinstance(device_str, torch.device):
        dev = device_str
    else:
        dev = torch.device(device_str)
    shape = tuple(int(s) for s in spec.shape)
    
    # Check if we're in CUDA graph capture mode
    is_capturing = False
    if dev.type == 'cuda':
        try:
            stream = torch.cuda.current_stream(dev)
            is_capturing = stream.is_capturing()
        except:
            pass
    
    if spec.init is None:
        return torch.empty(shape, dtype=dt, device=dev)
    
    kind = spec.init.kind
    if kind == "randn":
        if is_capturing:
            # During CUDA graph capture, use empty tensor
            t = torch.empty(shape, dtype=dt, device=dev)
        else:
            g = torch.Generator(device=dev)
            if spec.init and spec.init.seed is not None: 
                g.manual_seed(int(spec.init.seed))
            
            # Check if dtype is floating point
            if dt.is_floating_point:
                t = torch.randn(*shape, dtype=dt, device=dev, generator=g)
                if spec.init and spec.init.mean is not None:
                    mean = spec.init.mean or 0.0
                    std = spec.init.std or 1.0
                    t = t * std + mean
            else:
                # For integer dtypes, use randint with appropriate range
                if dt == torch.int8:
                    low, high = -128, 127
                elif dt == torch.uint8:
                    low, high = 0, 255
                elif dt == torch.int16:
                    low, high = -32768, 32767
                elif dt == torch.int32:
                    low, high = -2147483648, 2147483647
                elif dt == torch.int64:
                    low, high = -1000000, 1000000
                elif dt == torch.bool:
                    t = torch.randint(0, 2, shape, dtype=dt, device=dev, generator=g)
                    return t
                else:
                    low, high = -100, 100
                
                t = torch.randint(low, high + 1, shape, dtype=dt, device=dev, generator=g)
        return t
    
    if kind == "zeros": 
        return torch.zeros(shape, dtype=dt, device=dev)
    
    if kind == "ones":  
        return torch.ones(shape, dtype=dt, device=dev)
    
    if kind == "full":
        val = float(spec.init.fill_value or 0.0)
        return torch.full(shape, val, dtype=dt, device=dev)
    
    if kind == "uniform":
        if is_capturing:
            # During CUDA graph capture, use empty tensor
            return torch.empty(shape, dtype=dt, device=dev)
        else:
            g = torch.Generator(device=dev)
            if spec.init.seed is not None: 
                g.manual_seed(int(spec.init.seed))
            
            # Check if dtype is floating point
            if dt.is_floating_point:
                low = float(spec.init.low or 0.0)
                high = float(spec.init.high or 1.0)
                return (low + (high - low) * torch.rand(*shape, dtype=dt, device=dev, generator=g))
            else:
                # For integer dtypes, use randint
                if dt == torch.bool:
                    return torch.randint(0, 2, shape, dtype=dt, device=dev, generator=g)
                else:
                    low = int(spec.init.low or 0)
                    high = int(spec.init.high or 100)
                    return torch.randint(low, high + 1, shape, dtype=dt, device=dev, generator=g)
    
    if kind == "arange":
        # Use appropriate type for start/step based on dtype
        if dt in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            start = int(spec.init.start or 0)
            step = int(spec.init.step or 1)
        else:
            start = float(spec.init.start or 0.0)
            step = float(spec.init.step or 1.0)
        # arange returns 1D; tile to shape if needed
        v = torch.arange(start, start + step * shape[-1], step=step, dtype=dt, device=dev)[:shape[-1]]
        return v.reshape([1] * (len(shape) - 1) + [shape[-1]]).expand(shape).clone()
    
    raise ValueError(f"Unsupported init kind: {spec.init.kind}")

