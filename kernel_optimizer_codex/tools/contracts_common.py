#!/usr/bin/env python3
"""
contracts_common.py

Shared helpers for working with IronFist-style contract JSONs:

  - load_contract(path)
  - import_kernel_module(path)
  - build_entrypoint_and_kwargs(contract_path, module_path, entry_point, device)

We assume:
  contract["kernel"]["metadata"]["kernel_name"]
  contract["kernel"]["metadata"]["entry_point"] (defaults to 'run')
  contract["kernel"]["io"]["args"] is a list of arg specs with:
      { "name": str, "type": "int|float|str|bool|tensor", "value": ..., "tensor_spec": {...} }

Tensor specs roughly follow the style in your eval server:
  {
    "shape": [..],
    "dtype": "torch.float16" | "float16" | ...,
    "init": {
      "kind": "randn|uniform|zeros|ones|full|arange",
      "seed": int?,
      "mean": float?,
      "std": float?,
      "low": float?,
      "high": float?,
      "fill_value": float?,
      "start": float?,
      "step": float?
    }
  }

The exact shape of tensor_spec doesn't have to be perfect — this is easy to tweak later if needed.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import triton.language as tl


def load_contract(path: Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def import_kernel_module(module_path: Path, module_name: Optional[str] = None):
    module_path = Path(module_path).resolve()
    if module_name is None:
        module_name = f"{module_path.stem}_mod"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_entry_point(contract: Dict[str, Any], module, entry_point: Optional[str]):
    kernel = contract.get("kernel", {})
    meta = kernel.get("metadata", {}) or {}
    kernel_type = kernel.get("kernel_type", "multi_kernel")

    if entry_point:
        ep_name = entry_point
    else:
        meta_ep = meta.get("entry_point")
        if kernel_type == "triton":
            ep_name = meta_ep or meta.get("kernel_name") or "run"
        else:
            ep_name = meta_ep or "run"

    func = getattr(module, ep_name, None)
    if func is None or not callable(func):
        raise AttributeError(f"Entry point '{ep_name}' not found or not callable in module")
    return func, ep_name


def _normalize_grid(grid_val: Any) -> Tuple[int, ...]:
    if grid_val is None:
        return (1,)
    if isinstance(grid_val, dict):
        x = int(grid_val.get("x", 1))
        y = int(grid_val.get("y", 1))
        z = int(grid_val.get("z", 1))
        return (x, y, z)
    if isinstance(grid_val, (list, tuple)):
        return tuple(int(v) for v in grid_val)
    try:
        return (int(grid_val),)
    except Exception:
        return (1,)


def _build_triton_call_state(contract: Dict[str, Any], func, device: str):
    """
    Prepare a callable that launches a Triton kernel directly via fn[grid](...),
    separating runtime args from meta args, and propagating launch num_warps/num_stages.
    """
    kernel = contract.get("kernel", {}) or {}
    io = kernel.get("io", {}) or {}
    arg_specs = io.get("args", []) or []
    launch_cfg = io.get("launch", {}) or {}

    target = getattr(func, "fn", func)
    params = inspect.signature(target).parameters

    runtime_kwargs: Dict[str, Any] = {}
    meta_kwargs: Dict[str, Any] = {}
    output_names = []

    for arg in arg_specs:
        name = arg.get("name")
        if not name or name not in params:
            continue
        atype = arg.get("type", "int")
        is_meta = bool(arg.get("is_meta", False))
        if atype == "tensor":
            val = _make_torch_tensor(arg, device=device)
        else:
            val = _parse_scalar_value(arg)
        if is_meta:
            if isinstance(val, str):
                vstr = val
                if vstr.startswith("tl."):
                    attr = vstr.split(".", 1)[1]
                    if hasattr(tl, attr):
                        val = getattr(tl, attr)
                elif hasattr(tl, vstr):
                    val = getattr(tl, vstr)
            if isinstance(val, bool):
                val = int(val)
            meta_kwargs[name] = val
        else:
            runtime_kwargs[name] = val
            if arg.get("role") in ("output", "inout"):
                output_names.append(name)

    grid = _normalize_grid(launch_cfg.get("grid"))
    launch_kwargs: Dict[str, Any] = {}
    if launch_cfg.get("num_warps") is not None:
        launch_kwargs["num_warps"] = int(launch_cfg["num_warps"])
    if launch_cfg.get("num_stages") is not None:
        launch_kwargs["num_stages"] = int(launch_cfg["num_stages"])

    def launcher(**call_kwargs):
        call_runtime = dict(runtime_kwargs)
        call_runtime.update(call_kwargs)
        func[grid](**call_runtime, **meta_kwargs, **launch_kwargs)
        outputs = [call_runtime[n] for n in output_names if n in call_runtime]
        if outputs:
            return outputs[0] if len(outputs) == 1 else tuple(outputs)
        tensor_outputs = [v for v in call_runtime.values() if torch.is_tensor(v)]
        if tensor_outputs:
            return tensor_outputs[0] if len(tensor_outputs) == 1 else tuple(tensor_outputs)
        return None

    return launcher, runtime_kwargs


def _resolve_torch_dtype(dtype_str: str):
    ds = dtype_str
    if isinstance(ds, str) and ds.startswith("torch."):
        ds = ds.split(".", 1)[1]

    ds = str(ds).lower()
    if ds in ("float16", "half", "fp16"):
        return torch.float16
    if ds in ("bfloat16", "bf16"):
        return torch.bfloat16
    if ds in ("float32", "float", "fp32"):
        return torch.float32
    if ds in ("float64", "double", "fp64"):
        return torch.float64
    if ds in ("int8",):
        return torch.int8
    if ds in ("uint8",):
        return torch.uint8
    if ds in ("int16", "short"):
        return torch.int16
    if ds in ("int32", "int"):
        return torch.int32
    if ds in ("int64", "long"):
        return torch.int64
    if ds in ("bool",):
        return torch.bool
    # default
    return torch.float32


def _make_torch_tensor(arg_spec: Dict[str, Any], device: str) -> torch.Tensor:
    ts = arg_spec.get("tensor_spec") or arg_spec.get("spec")
    if not ts:
        raise ValueError(f"tensor_spec missing in arg spec: {arg_spec}")

    shape = tuple(ts.get("shape", []))
    dtype = _resolve_torch_dtype(ts.get("dtype", "float32"))

    dev = torch.device(device)
    init = ts.get("init", {}) or {}
    kind = str(init.get("kind", "randn")).lower()
    seed = init.get("seed", None)

    if seed is not None:
        try:
            seed = int(seed)
            torch.manual_seed(seed)
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    is_float = dtype.is_floating_point

    if kind == "zeros":
        return torch.zeros(shape, dtype=dtype, device=dev)
    if kind == "ones":
        return torch.ones(shape, dtype=dtype, device=dev)
    if kind == "full":
        fill_value = float(init.get("fill_value", 0.0))
        return torch.full(shape, fill_value, dtype=dtype, device=dev)

    if kind == "arange":
        # sequential values with start/step
        start = init.get("start", 0.0)
        step = init.get("step", 1.0)
        if not is_float:
            start = int(start)
            step = int(step)
        length = shape[-1] if shape else 1
        base = torch.arange(start, start + step * length, step=step, dtype=dtype, device=dev)[:length]
        if len(shape) > 1:
            base = base.view(*([1] * (len(shape) - 1)), length)
            base = base.expand(shape).contiguous()
        return base

    if kind == "uniform":
        low = float(init.get("low", 0.0))
        high = float(init.get("high", 1.0))
        if is_float:
            t = torch.empty(shape, dtype=dtype, device=dev)
            t.uniform_(low, high)
            return t
        else:
            # integers / bool
            if dtype == torch.bool:
                t = torch.randint(0, 2, size=shape, dtype=torch.int64, device=dev)
                return t.bool()
            low_i = int(low)
            high_i = int(high)
            return torch.randint(low_i, high_i + 1, size=shape, dtype=dtype, device=dev)

    # default: randn
    if is_float:
        mean = float(init.get("mean", 0.0))
        std = float(init.get("std", 1.0))
        t = torch.randn(shape, dtype=dtype, device=dev)
        if mean != 0.0 or std != 1.0:
            t = t * std + mean
        return t
    else:
        # integer/bool randn -> randint fallback
        if dtype == torch.bool:
            t = torch.randint(0, 2, size=shape, dtype=torch.int64, device=dev)
            return t.bool()
        return torch.randint(-100, 101, size=shape, dtype=dtype, device=dev)


def _parse_scalar_value(arg_spec: Dict[str, Any]) -> Any:
    t = arg_spec.get("type", "int")
    v = arg_spec.get("value")

    if t == "int":
        return int(v)
    if t == "float":
        return float(v)
    if t == "str":
        return str(v)
    if t == "bool":
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(v)
    # Fallback: return as is
    return v


def build_kwargs_for_entrypoint(contract: Dict[str, Any], func, device: str) -> Dict[str, Any]:
    kernel = contract.get("kernel", {})
    io = kernel.get("io", {}) or {}
    arg_specs = io.get("args", []) or []

    sig = inspect.signature(func)
    params = sig.parameters

    kwargs: Dict[str, Any] = {}
    for arg in arg_specs:
        name = arg.get("name")
        if not name or name not in params:
            continue
        atype = arg.get("type", "int")
        if atype == "tensor":
            val = _make_torch_tensor(arg, device=device)
        else:
            val = _parse_scalar_value(arg)
        kwargs[name] = val

    return kwargs


def build_entrypoint_and_kwargs(
    contract_path: Path,
    module_path: Path,
    entry_point: Optional[str],
    device: str,
):
    contract = load_contract(contract_path)
    module = import_kernel_module(module_path)
    kernel_type = (contract.get("kernel", {}) or {}).get("kernel_type", "multi_kernel")

    func, ep_name = _resolve_entry_point(contract, module, entry_point)

    if kernel_type == "triton":
        func, kwargs = _build_triton_call_state(contract, func, device=device)
    else:
        kwargs = build_kwargs_for_entrypoint(contract, func, device=device)

    return contract, module, func, ep_name, kwargs
