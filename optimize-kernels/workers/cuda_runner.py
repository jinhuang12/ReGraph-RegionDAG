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
import copy
import math
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

import importlib
import math
import traceback
from itertools import accumulate
from typing import Any, Dict, List, Tuple


def load_config(config_path: str) -> Dict[str, Any]:
    """Load runner configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_attr_path(path: str) -> Any:
    """Resolve dotted import/attribute path to a callable/object."""
    parts = path.split(".")
    if not parts:
        raise ValueError("Empty path")
    module = importlib.import_module(parts[0])
    obj = module
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


def _build_flash_attention_inputs(case: Dict[str, Any], torch: Any, device: Any) -> Tuple[List[Any], Dict[str, Any]]:
    """Create packed varlen flash attention inputs for vLLM kernels."""
    seqlens: List[int] = case.get("seqlens") or []
    batch = len(seqlens)
    if batch == 0:
        batch = int(case.get("batch", 1))
        seqlen = int(case.get("seq_len", 1024))
        seqlens = [seqlen for _ in range(batch)]

    num_heads = int(case.get("num_heads", 32))
    head_dim = int(case.get("head_dim", 128))
    dtype_str = case.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    seed = case.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    total_tokens = sum(seqlens)
    qkv = torch.randn((total_tokens, 3, num_heads, head_dim), device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, *accumulate(seqlens)], device=device, dtype=torch.int32)
    max_seqlen = int(max(seqlens))
    scale = float(case.get("softmax_scale", 1.0 / math.sqrt(head_dim)))
    causal = bool(case.get("causal", True))

    # torch.ops.vllm.flash_attn_varlen_qkvpacked signature matches these args
    inputs = [qkv, cu_seqlens, max_seqlen, scale, False, causal]
    return inputs, {}


def _flash_attention_reference(
    inputs: List[Any],
    ref_fn: Any,
    torch: Any,
) -> Any:
    """Compute reference output for flash attention."""
    if ref_fn is not None:
        try:
            return ref_fn(*inputs)
        except Exception:
            pass

    # Fallback: use scaled_dot_product_attention on padded tensors
    qkv, cu_seqlens, max_seqlen, scale, _dropout, causal = inputs
    num_heads = qkv.shape[2]
    head_dim = qkv.shape[3]
    batch = cu_seqlens.numel() - 1
    device = qkv.device
    dtype = qkv.dtype

    qkv_padded = torch.zeros(
        (batch, max_seqlen, 3, num_heads, head_dim), device=device, dtype=dtype
    )
    for b in range(batch):
        start = int(cu_seqlens[b])
        end = int(cu_seqlens[b + 1])
        seqlen = end - start
        qkv_padded[b, :seqlen] = qkv[start:end]

    q = qkv_padded[:, :, 0]
    k = qkv_padded[:, :, 1]
    v = qkv_padded[:, :, 2]
    attn_mask = None
    if causal:
        attn_mask = torch.full(
            (batch, 1, max_seqlen, max_seqlen),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=attn_mask, scale=scale
    )
    out = out.transpose(1, 2)
    flat = torch.zeros((qkv.shape[0], num_heads, head_dim), device=device, dtype=dtype)
    for b in range(batch):
        start = int(cu_seqlens[b])
        end = int(cu_seqlens[b + 1])
        seqlen = end - start
        flat[start:end] = out[b, :seqlen]
    return flat


def _build_rms_norm_inputs(case: Dict[str, Any], torch: Any, device: Any) -> Tuple[List[Any], Dict[str, Any]]:
    shape = case.get("shape") or [case.get("batch", 1), case.get("seq_len", 2048), case.get("hidden", 4096)]
    shape = [int(x) for x in shape]
    dtype_str = case.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    seed = case.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn((shape[-1],), device=device, dtype=dtype)
    eps = float(case.get("eps", 1e-6))
    return [x, weight, eps], {}


def _rms_norm_reference(inputs: List[Any], torch: Any) -> Any:
    x, weight, eps = inputs
    variance = torch.mean(x * x, dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return (x * inv_rms) * weight


def run_vllm_kernel(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute built-in vLLM kernels with reference comparison."""
    try:
        import vllm  # noqa: F401
        import torch
    except Exception as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"vLLM path unavailable: {e}",
            "traceback": traceback.format_exc(),
        }

    if not torch.cuda.is_available():
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": "CUDA device not available for vLLM kernels",
        }

    op_name = config.get("op_name") or ""
    entry_path = config.get("entry_path")
    reference_path = config.get("reference_path")
    cases = config.get("cases", []) or []
    timing = config.get("timing", {"warmup": 1, "iters": 10, "repeat": 1})
    seed = config.get("seed")

    try:
        entry_fn = _resolve_attr_path(entry_path) if entry_path else None
        ref_fn = _resolve_attr_path(reference_path) if reference_path else None
    except Exception as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"Failed to resolve vLLM callables: {e}",
            "traceback": traceback.format_exc(),
        }

    if entry_fn is None:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": "No vLLM entry function provided",
        }

    device = torch.device("cuda")
    dtype_default = config.get("dtype", "float16")
    if seed is not None:
        torch.manual_seed(int(seed))

    times: List[float] = []
    max_diff = 0.0
    ok = True

    for case in cases:
        case = dict(case)
        case.setdefault("dtype", dtype_default)
        if op_name == "flash_attention":
            inputs, kwargs = _build_flash_attention_inputs(case, torch, device)
            reference = lambda: _flash_attention_reference(inputs, ref_fn, torch)
        elif op_name == "rms_norm":
            inputs, kwargs = _build_rms_norm_inputs(case, torch, device)
            reference = lambda: _rms_norm_reference(inputs, torch)
        else:
            return {
                "ok": False,
                "mean_ms": float("inf"),
                "std_ms": 0.0,
                "error": f"Unsupported vLLM op_name {op_name}",
            }

        try:
            for _ in range(timing.get("warmup", 0)):
                entry_fn(*inputs, **kwargs)
            torch.cuda.synchronize()
        except Exception as e:
            return {
                "ok": False,
                "mean_ms": float("inf"),
                "std_ms": 0.0,
                "error": f"vLLM kernel warmup failed: {e}",
                "traceback": traceback.format_exc(),
            }

        try:
            case_times = []
            for _ in range(timing.get("repeat", 1)):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                accum = 0.0
                for _ in range(timing.get("iters", 1)):
                    start.record()
                    out = entry_fn(*inputs, **kwargs)
                    end.record()
                    end.synchronize()
                    accum += float(start.elapsed_time(end))
                case_times.append(accum / max(timing.get("iters", 1), 1))

            times.append(sum(case_times) / len(case_times))

            ref_out = reference()
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(ref_out, tuple):
                ref_out = ref_out[0]

            diff = torch.max(torch.abs(out - ref_out)).item()
            max_diff = max(max_diff, diff)
            if not torch.isfinite(torch.tensor(diff)):
                ok = False
            if diff > float(config.get("tolerance", 1e-3)):
                ok = False
        except Exception as e:
            return {
                "ok": False,
                "mean_ms": float("inf"),
                "std_ms": 0.0,
                "error": f"vLLM kernel execution failed: {e}",
                "traceback": traceback.format_exc(),
            }

    if not times:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": "No vLLM cases executed",
        }

    mean_ms = float(sum(times) / len(times))
    std_ms = float(math.sqrt(sum((t - mean_ms) ** 2 for t in times) / len(times)))
    return {
        "ok": bool(ok),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "max_diff": max_diff,
    }


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


def apply_launch_update_to_io(io_contract: Dict[str, Any], launch_update: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of io_contract with launch updates merged in."""
    updated = copy.deepcopy(io_contract) if io_contract else {"args": []}
    if not launch_update:
        return updated

    launch_cfg = updated.setdefault("launch", {})
    for key in ("grid", "block"):
        if key in launch_update:
            launch_cfg.setdefault(key, {})
            launch_cfg[key].update(launch_update[key] or {})
    return updated


def merge_launch_updates(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    if override:
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                nv = dict(merged[k])
                nv.update(v)
                merged[k] = nv
            else:
                merged[k] = v
    return merged


def aggregate_shape_results(shape_results: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    method = (method or "weighted_mean").lower()
    valid = [sr for sr in shape_results if math.isfinite(sr.get("mean_ms", float("inf")))]
    ok_all = bool(shape_results) and all(sr.get("ok", False) for sr in shape_results)

    if not valid:
        return {"ok": False, "mean_ms": float("inf"), "std_ms": 0.0, "aggregation": {"method": method}}

    weight_sum = sum(max(0.0, float(sr.get("weight", 1.0))) for sr in valid)
    if weight_sum <= 0:
        weight_sum = float(len(valid))

    if method == "worst_case":
        worst = max(valid, key=lambda sr: sr.get("mean_ms", float("inf")))
        mean_ms = float(worst.get("mean_ms", float("inf")))
        std_ms = float(worst.get("std_ms", 0.0))
    else:
        mean_ms = sum(float(sr.get("mean_ms", 0.0)) * max(0.0, float(sr.get("weight", 1.0))) for sr in valid) / weight_sum
        std_ms = sum(float(sr.get("std_ms", 0.0)) * max(0.0, float(sr.get("weight", 1.0))) for sr in valid) / weight_sum

    return {
        "ok": ok_all,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "aggregation": {"method": method, "weights_sum": weight_sum},
    }


def execute_cuda_shape(cp: Any, np: Any, func: Any, io_contract: Dict[str, Any], timing: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single IOContract configuration and measure timings."""
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
                args.append(arg_spec["value"])

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
            "traceback": traceback.format_exc(),
        }

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
            "traceback": traceback.format_exc(),
        }

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
            "traceback": traceback.format_exc(),
        }

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))

    ok = True
    try:
        for out in outputs:
            if cp.any(cp.isnan(out)) or cp.any(cp.isinf(out)):
                ok = False
                break
    except Exception:
        pass

    return {
        "ok": bool(ok),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
    }


def run_cuda_kernel(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute CUDA kernel using CuPy.

    Args:
        config: Configuration dict

    Returns:
        Result dict with timing and status
    """
    if config.get("mode") == "vllm":
        return run_vllm_kernel(config)

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
    base_io_contract = config.get("io_contract", {})
    timing = config["timing"]
    shapes = config.get("representative_shapes") or []
    base_launch_update = config.get("launch_update", {}) or {}

    try:
        mod = cp.RawModule(path=cubin_path)
        func = mod.get_function(kernel_name)
    except Exception as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"Failed to load CUDA kernel: {e}",
            "traceback": traceback.format_exc(),
        }

    def run_for_io(io_contract: Dict[str, Any], launch_update: Dict[str, Any]) -> Dict[str, Any]:
        merged_update = merge_launch_updates(base_launch_update, launch_update or {})
        merged_contract = apply_launch_update_to_io(io_contract, merged_update)
        return execute_cuda_shape(cp, np, func, merged_contract, timing)

    if shapes:
        shape_results: List[Dict[str, Any]] = []
        for idx, shape in enumerate(shapes):
            io_override = shape.get("io") or shape.get("io_contract") or base_io_contract
            shape_res = run_for_io(io_override, shape.get("launch_update"))
            shape_res["shape_name"] = shape.get("name", f"shape_{idx}")
            shape_res["weight"] = float(shape.get("weight", 1.0))
            shape_results.append(shape_res)

        aggregated = aggregate_shape_results(shape_results, config.get("timing_aggregation", "weighted_mean"))
        aggregated["shape_results"] = shape_results
        aggregated["ok"] = aggregated.get("ok", False) and bool(shape_results)
        aggregated["launch_update_applied"] = bool(base_launch_update) or any(
            sr.get("launch_update_applied") for sr in shape_results
        )
        return aggregated

    single_result = run_for_io(base_io_contract, None)
    single_result["launch_update_applied"] = bool(base_launch_update) or single_result.get("launch_update_applied", False)
    return single_result


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
