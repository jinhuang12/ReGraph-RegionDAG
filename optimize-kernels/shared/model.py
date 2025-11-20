import math
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


# ------------------------- Spec & State Models --------------------------
@dataclass
class DeviceProfile:
    gpu_name: str
    arch: str
    sms: int
    regs_per_sm: int
    smem_per_sm: int
    hbm_bw_gbps: float

# Base metadata class that all kernel types can extend
class BaseKernelMetadata(BaseModel):
    """Base metadata for all kernel types"""
    
    class Config:
        """Pydantic config"""
        extra = "allow"  # Allow extra fields for flexibility


class CudaKernelMetadata(BaseKernelMetadata):
    """Metadata specific to CUDA kernels"""
    kernel_name: Optional[str] = None  # Entrypoint kernel to execute
    compiler_options: Optional[List[str]] = None  # e.g., ["-O3", "-use_fast_math"]
    backend: str = "nvrtc"  # "nvcc" or "nvrtc"
    name_expressions: Optional[List[str]] = None  # For template kernels
    jitify: bool = False  # Enable jitify for C++ features
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CudaKernelMetadata":
        """Create from dictionary for backward compatibility"""
        if not d:
            return cls()
        return cls(**d)


class TritonKernelMetadata(BaseKernelMetadata):
    """Metadata specific to Triton kernels"""
    kernel_name: Optional[str] = None  # Entrypoint kernel to execute

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TritonKernelMetadata":
        """Create from dictionary for backward compatibility"""
        if not d:
            return cls()
        return cls(**d)


class MultiKernelMetadata(BaseKernelMetadata):
    """Metadata specific to multi-kernel Python sequences"""
    entry_point: str  # Function name to call (required)
    description: Optional[str] = None  # Optional description of what the sequence does

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultiKernelMetadata":
        """Create from dictionary for backward compatibility"""
        if not d:
            raise ValueError("MultiKernelMetadata requires 'entry_point' field")
        return cls(**d)


# Union type for metadata - can be Pydantic model or dict for flexibility
KernelMetadata = Union[BaseKernelMetadata, Dict[str, Any]]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _enum_val(x):
    return x.value if isinstance(x, Enum) else x

def get_metadata_value(metadata: Optional[Union[BaseKernelMetadata, Dict[str, Any]]], key: str, default: Any = None) -> Any:
    """
    Safely access metadata fields regardless of whether metadata is a Pydantic model or dict.

    Args:
        metadata: Either a BaseKernelMetadata Pydantic model, a dict, or None
        key: The field/key to access
        default: Default value if key is not found

    Returns:
        The value of the field/key, or default if not found or metadata is None
    """
    if metadata is None:
        return default
    if isinstance(metadata, BaseKernelMetadata):
        return getattr(metadata, key, default)
    return metadata.get(key, default)

def _round_float_values(obj: Any, decimal_places: int = 3) -> Any:
    """
    Recursively round all float values in nested dictionaries/lists to specified decimal places.
    Also sanitizes inf, -inf, and nan values to JSON-compliant values.

    Args:
        obj: Object to process (dict, list, float, or other)
        decimal_places: Number of decimal places to round to (default: 3)

    Returns:
        Object with all float values rounded and sanitized
    """
    if isinstance(obj, dict):
        return {k: _round_float_values(v, decimal_places) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_float_values(item, decimal_places) for item in obj]
    elif isinstance(obj, float):
        # Sanitize inf, -inf, and nan to JSON-compliant values
        if math.isnan(obj):
            return 0.0
        elif math.isinf(obj):
            return 1e9 if obj > 0 else -1e9
        else:
            return round(obj, decimal_places)
    else:
        return obj


# Kernel Type Enum
class KernelType(str, Enum):
    """Kernel implementation types"""
    CUDA = "cuda"            # Raw CUDA kernels
    TRITON = "triton"        # Triton kernels
    MULTI_KERNEL = "multi_kernel"  # Python scripts with mixed kernel types


class TensorData(BaseModel):
    """Literal tensor payload sent over JSON (optional)"""
    # Raw storage buffer (row-major unless 'strides' given)
    data_b64: str  # base64 of raw bytes
    # Required to reconstruct
    dtype: str = "float32"  # e.g. "float32"
    shape: List[int] = Field(default_factory=list)
    compress: str = "none"  # "none" | "zlib"

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorData":
        """Backward compatibility wrapper"""
        return cls(**d)
    

class TensorInit(BaseModel):
    """How to generate a tensor deterministically on the server (if no data payload)."""
    kind: str = "randn"  # "randn" | "zeros" | "ones" | "uniform" | "full" | "arange"
    seed: Optional[int] = None
    mean: Optional[float] = None  # randn
    std: Optional[float] = None  # randn
    low: Optional[float] = None  # uniform
    high: Optional[float] = None  # uniform
    fill_value: Optional[float] = None  # full
    start: Optional[float] = None  # arange
    step: Optional[float] = None  # arange

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorInit":
        """Backward compatibility wrapper"""
        return cls(**d)


class TensorSpec(BaseModel):
    """Specification for a tensor argument or output"""
    shape: List[int] = Field(default_factory=list)  # required when init is used; can be omitted if data provided
    dtype: str = "float32"
    init: Optional[TensorInit] = None  # generate on server
    data: Optional[TensorData] = None  # OR literal payload (mutually exclusive with init)

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorSpec":
        """Backward compatibility wrapper"""
        # Handle nested models that might be dicts
        if d.get("init") and isinstance(d["init"], dict):
            d["init"] = TensorInit(**d["init"])
        if d.get("data") and isinstance(d["data"], dict):
            d["data"] = TensorData(**d["data"])
        return cls(**d)

class LaunchDim(BaseModel):
    x: int = 1
    y: int = 1
    z: int = 1

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LaunchDim":
        """Backward compatibility wrapper"""
        return cls(**d)

class LaunchConfig(BaseModel):
    grid: Optional[LaunchDim] = None
    block: Optional[LaunchDim] = None  # CUDA-only
    num_warps: Optional[int] = None  # Triton-only
    num_stages: Optional[int] = None  # Triton-only

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LaunchConfig":
        """Backward compatibility wrapper"""
        # Handle nested models that might be dicts
        if d.get("grid") and isinstance(d["grid"], dict):
            d["grid"] = LaunchDim(**d["grid"])
        if d.get("block") and isinstance(d["block"], dict):
            d["block"] = LaunchDim(**d["block"])
        return cls(**d)

class ArgSpec(BaseModel):
    """Kernel argument specification"""
    name: str
    type: str  # "tensor","int","float","str","bool"
    value: Optional[Union[int, float, str, bool]] = None
    tensor_spec: Optional[TensorSpec] = None
    role: str = "input"  # "input" | "output" | "inout"
    is_meta: bool = False  # Triton constexpr/meta

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArgSpec":
        """Backward compatibility wrapper"""
        # Handle nested model that might be dict
        if d.get("tensor_spec") and isinstance(d["tensor_spec"], dict):
            d["tensor_spec"] = TensorSpec(**d["tensor_spec"])
        return cls(**d)


class IOContract(BaseModel):
    args: List[ArgSpec]
    launch: Optional[LaunchConfig] = None

    model_config = ConfigDict(extra='ignore')

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IOContract":
        """Backward compatibility wrapper"""
        # Handle nested models that might be dicts
        if d.get("args"):
            d["args"] = [ArgSpec(**arg) if isinstance(arg, dict) else arg for arg in d["args"]]
        if d.get("launch") and isinstance(d["launch"], dict):
            d["launch"] = LaunchConfig(**d["launch"])
        return cls(**d)


# Kernel Code Wrapper
class KernelCode(BaseModel):
    """Wrapper for kernel source code with type information"""
    source_code: str
    kernel_type: KernelType
    io: Optional[IOContract] = None
    reference: Optional[Dict[str, Any]] = None
    # Metadata can be typed dataclass or dict for backward compatibility
    metadata: Optional[Union[BaseKernelMetadata, Dict[str, Any]]] = None
    device_profile: Optional[DeviceProfile] = None
    invocation_example: Optional[str] = None
    name: Optional[str] = None

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    @field_validator('metadata', mode='before')
    @classmethod
    def parse_metadata(cls, v, values):
        """Parse metadata if it's a dict and convert to appropriate type if possible"""
        # If already a BaseKernelMetadata instance, keep it
        if isinstance(v, BaseKernelMetadata):
            return v
        # Otherwise keep as dict for backward compatibility
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        result = self.model_dump(exclude_none=True)
        # Handle metadata specially for backward compatibility
        if self.metadata:
            if isinstance(self.metadata, BaseKernelMetadata):
                result["metadata"] = self.metadata.model_dump()
            else:
                result["metadata"] = self.metadata
        if self.reference is not None:
            result["reference"] = self.reference
        # Convert enum to string
        result["kernel_type"] = _enum_val(self.kernel_type)
        # Handle io specially if needed
        if self.io and hasattr(self.io, 'to_dict'):
            result["io"] = self.io.to_dict()
        return result

    def get_typed_metadata(self) -> Optional[BaseKernelMetadata]:
        """Get metadata as typed dataclass if possible"""
        if isinstance(self.metadata, BaseKernelMetadata):
            # Check if it's the correct subtype for this kernel_type
            # If it's a generic BaseKernelMetadata, convert it to the proper type
            if type(self.metadata) == BaseKernelMetadata:
                # Convert to dict and then to proper type
                metadata_dict = self.metadata.model_dump()
                if self.kernel_type == KernelType.CUDA:
                    return CudaKernelMetadata.from_dict(metadata_dict)
                elif self.kernel_type == KernelType.TRITON:
                    return TritonKernelMetadata.from_dict(metadata_dict)
                elif self.kernel_type == KernelType.MULTI_KERNEL:
                    return MultiKernelMetadata.from_dict(metadata_dict)
            # Already correct type, return as-is
            return self.metadata
        elif isinstance(self.metadata, dict):
            if self.kernel_type == KernelType.CUDA:
                return CudaKernelMetadata.from_dict(self.metadata)
            elif self.kernel_type == KernelType.TRITON:
                return TritonKernelMetadata.from_dict(self.metadata)
            elif self.kernel_type == KernelType.MULTI_KERNEL:
                return MultiKernelMetadata.from_dict(self.metadata)
        return None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelCode":
        """Backward compatibility wrapper"""
        kt = d.get("kernel_type")
        kt = KernelType(kt) if not isinstance(kt, KernelType) else kt
        # Handle nested models that might be dicts
        if d.get("io") and isinstance(d["io"], dict):
            d["io"] = IOContract(**d["io"])    
        return cls(
            source_code=d["source_code"],
            kernel_type=kt,
            metadata=d.get("metadata"),
            io=d.get("io"),
            reference=d.get("reference"),
        )
