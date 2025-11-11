#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuda_runner.py
--------------
Standalone CUDA kernel execution module.

Usage:
    python -m workers.cuda_runner <config_path>

Config JSON format:
{
    "kernel_cubin_path": "/path/to/kernel.cubin",
    "kernel_name": "my_cuda_kernel",
    "io_contract": {...},  // IOContract dict with args and launch config
    "timing": {"warmup": 10, "iters": 100, "repeat": 3},
    "result_path": "/path/to/runner_result.json"
}

Output JSON format (runner_result.json):
{
    "ok": true,
    "mean_ms": 1.234,
    "std_ms": 0.056,
    "error": "...",  // Only on failure
    "traceback": "..."  // Only on failure
}
"""

# Path setup for imports from shared/ (must be before local imports)
import sys
import json
from pathlib import Path

if len(sys.argv) >= 2:
    config_path = sys.argv[1]
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        root_path = config.get("root_path")
        if root_path and root_path not in sys.path:
            sys.path.insert(0, str(Path(root_path).resolve()))
    except Exception:
        pass  # Fallback to PYTHONPATH

import traceback
from typing import Any, Dict, List, Tuple


def load_config(config_path: str) -> Dict[str, Any]:
    """Load runner configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_tensor(arg: Dict[str, Any], cp: Any) -> Any:
    """
    Create a CuPy tensor from ArgSpec.

    Args:
        arg: ArgSpec dict with tensor_spec
        cp: cupy module

    Returns:
        CuPy array
    """
    ts = arg.get("tensor_spec")
    if not ts:
        return None

    shape = tuple(ts["shape"])
    dtype = ts["dtype"]

    # Map dtype string to cupy dtype
    dtype_map = {
        "float32": cp.float32,
        "float16": cp.float16,
        "bfloat16": cp.float16,  # CuPy doesn't have native bfloat16
        "float64": cp.float64,
        "int32": cp.int32,
        "int64": cp.int64,
        "int16": cp.int16,
        "int8": cp.int8,
        "uint8": cp.uint8,
        "bool": cp.bool_
    }
    dt = dtype_map.get(dtype, cp.float32)

    # Get initialization spec
    init_spec = ts.get("init", {})
    if not init_spec:
        # No init spec - return empty array
        return cp.empty(shape, dtype=dt)

    kind = init_spec.get("kind", "randn")
    seed = init_spec.get("seed")

    # Create random state if seed is provided
    rs = None
    if seed is not None:
        rs = cp.random.RandomState(int(seed))

    # Check if dtype is floating point
    is_float = dt in (cp.float16, cp.float32, cp.float64)

    if kind == "zeros":
        arr = cp.zeros(shape, dtype=dt)
    elif kind == "ones":
        arr = cp.ones(shape, dtype=dt)
    elif kind == "full":
        fill_value = init_spec.get("fill_value", 0.0)
        arr = cp.full(shape, fill_value, dtype=dt)
    elif kind == "randn":
        if is_float:
            # Floating point: use standard_normal
            # CuPy random functions only support float32/float64, so generate in float32 and cast if needed
            gen_dtype = cp.float32 if dt in (cp.float16, cp.float32) else cp.float64
            if rs is not None:
                arr = rs.standard_normal(shape, dtype=gen_dtype)
            else:
                arr = cp.random.standard_normal(shape, dtype=gen_dtype)

            # Apply mean/std if provided
            mean = init_spec.get("mean")
            std = init_spec.get("std")
            if mean is not None or std is not None:
                mean = mean if mean is not None else 0.0
                std = std if std is not None else 1.0
                arr = arr * std + mean

            # Cast to target dtype if needed
            if dt != gen_dtype:
                arr = arr.astype(dt)
        else:
            # Integer/bool dtypes: use randint with appropriate range
            if dt == cp.int8:
                low, high = -128, 127
            elif dt == cp.uint8:
                low, high = 0, 255
            elif dt == cp.int16:
                low, high = -32768, 32767
            elif dt == cp.int32:
                low, high = -2147483648, 2147483647
            elif dt == cp.int64:
                low, high = -1000000, 1000000
            elif dt == cp.bool_:
                if rs is not None:
                    arr = rs.randint(0, 2, size=shape).astype(cp.bool_)
                else:
                    arr = cp.random.randint(0, 2, size=shape).astype(cp.bool_)
                return arr
            else:
                low, high = -100, 100

            if rs is not None:
                arr = rs.randint(low, high + 1, size=shape, dtype=dt)
            else:
                arr = cp.random.randint(low, high + 1, size=shape, dtype=dt)
    elif kind == "uniform":
        if is_float:
            # Floating point: use uniform distribution
            # CuPy random functions only support float32/float64, so generate in float32 and cast if needed
            low = float(init_spec.get("low", 0.0))
            high = float(init_spec.get("high", 1.0))
            gen_dtype = cp.float32 if dt in (cp.float16, cp.float32) else cp.float64
            if rs is not None:
                arr = rs.uniform(low, high, size=shape, dtype=gen_dtype)
            else:
                arr = cp.random.uniform(low, high, size=shape, dtype=gen_dtype)

            # Cast to target dtype if needed
            if dt != gen_dtype:
                arr = arr.astype(dt)
        else:
            # Integer dtypes: use randint
            if dt == cp.bool_:
                if rs is not None:
                    arr = rs.randint(0, 2, size=shape).astype(cp.bool_)
                else:
                    arr = cp.random.randint(0, 2, size=shape).astype(cp.bool_)
            else:
                low = int(init_spec.get("low", 0))
                high = int(init_spec.get("high", 100))
                if rs is not None:
                    arr = rs.randint(low, high + 1, size=shape, dtype=dt)
                else:
                    arr = cp.random.randint(low, high + 1, size=shape, dtype=dt)
    elif kind == "arange":
        # Sequential values with start/step
        if dt in (cp.int8, cp.int16, cp.int32, cp.int64, cp.uint8):
            start = int(init_spec.get("start", 0))
            step = int(init_spec.get("step", 1))
        else:
            start = float(init_spec.get("start", 0.0))
            step = float(init_spec.get("step", 1.0))

        # Create 1D arange and reshape/expand to target shape
        length = shape[-1] if shape else 1
        v = cp.arange(start, start + step * length, step=step, dtype=dt)[:length]
        # Reshape to [1, 1, ..., length] and expand to full shape
        if len(shape) > 1:
            v = v.reshape([1] * (len(shape) - 1) + [length])
            arr = cp.broadcast_to(v, shape).copy()
        else:
            arr = v
    else:
        raise ValueError(f"Unsupported init kind: {kind}")

    return arr


def run_cuda_kernel(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute CUDA kernel using CuPy.

    Args:
        config: Configuration dict

    Returns:
        Result dict with timing and status
    """
    try:
        import cupy as cp
        import numpy as np
    except ImportError as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"CuPy not available: {e}",
            "traceback": traceback.format_exc()
        }

    cubin_path = config["kernel_cubin_path"]
    kernel_name = config.get("kernel_name", "kernel")
    io_contract = config["io_contract"]
    timing = config["timing"]

    try:
        # Load CUDA module
        mod = cp.RawModule(path=cubin_path)
        func = mod.get_function(kernel_name)
    except Exception as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"Failed to load CUDA kernel: {e}",
            "traceback": traceback.format_exc()
        }

    # Generate inputs from IOContract
    args = []
    outputs = []

    try:
        for arg_spec in io_contract.get("args", []):
            if arg_spec["type"] == "tensor":
                arr = make_tensor(arg_spec, cp)
                args.append(arr)
                if arg_spec.get("role") in ("output", "inout"):
                    outputs.append(arr)
            else:
                # Scalar argument
                args.append(arg_spec["value"])

        # Get launch configuration
        launch_config = io_contract.get("launch", {})
        grid_spec = launch_config.get("grid", {"x": 1, "y": 1, "z": 1})
        block_spec = launch_config.get("block", {"x": 1, "y": 1, "z": 1})

        grid_t = (int(grid_spec.get("x", 1)), int(grid_spec.get("y", 1)), int(grid_spec.get("z", 1)))
        block_t = (int(block_spec.get("x", 1)), int(block_spec.get("y", 1)), int(block_spec.get("z", 1)))

    except Exception as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"Failed to generate inputs: {e}",
            "traceback": traceback.format_exc()
        }

    # Warmup
    try:
        for _ in range(timing["warmup"]):
            func(grid_t, block_t, tuple(args))
        cp.cuda.Device().synchronize()
    except Exception as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"Kernel execution failed during warmup: {e}",
            "traceback": traceback.format_exc()
        }

    # Timed runs
    times = []
    try:
        for _ in range(timing["repeat"]):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            accum = 0.0

            for _ in range(timing["iters"]):
                start.record()
                func(grid_t, block_t, tuple(args))
                end.record()
                end.synchronize()
                accum += cp.cuda.get_elapsed_time(start, end)

            times.append(accum / timing["iters"])

    except Exception as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"Kernel execution failed during timing: {e}",
            "traceback": traceback.format_exc()
        }

    # Calculate statistics
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))

    # Validate outputs (check for NaN/Inf)
    ok = True
    try:
        for out in outputs:
            if cp.any(cp.isnan(out)) or cp.any(cp.isinf(out)):
                ok = False
                break
    except Exception:
        pass  # Validation failure doesn't crash the runner

    return {
        "ok": bool(ok),
        "mean_ms": mean_ms,
        "std_ms": std_ms
    }


def main() -> int:
    """Main entry point for cuda_runner."""
    if len(sys.argv) < 2:
        print("Usage: python -m workers.cuda_runner <config_path>", file=sys.stderr)
        return 1

    config_path = sys.argv[1]

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # Run kernel
    result = run_cuda_kernel(config)

    # Write result
    result_path = config.get("result_path", "runner_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
