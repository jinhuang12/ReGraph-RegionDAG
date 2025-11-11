#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
triton_runner.py
----------------
Standalone Triton kernel execution module.

Supports two execution modes:
1. IOContract-based (preferred): Uses metadata.kernel_name with proper module loading
2. Legacy invocation_example: Uses exec() with LAUNCH_UPDATE in locals

Usage:
    python -m workers.triton_runner <config_path>

Config JSON format:
{
    "kernel_module_path": "/path/to/kernel_module.py",
    "kernel_name": "my_kernel",  // Optional, from metadata
    "invocation_example": "...",  // Optional, legacy fallback
    "launch_update": {...},  // Launch configuration overrides
    "io_contract": {...},  // IOContract dict (required for kernel_name mode)
    "timing": {"warmup": 10, "iters": 100, "repeat": 3},
    "result_path": "/path/to/runner_result.json"
}

Output JSON format (runner_result.json):
{
    "ok": true,
    "mean_ms": 1.234,
    "std_ms": 0.056,
    "launch_update_applied": true,
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

import hashlib
import importlib.util
import time
import traceback
from typing import Any, Dict, List, Tuple

# Import from shared module for IOContract handling
from shared.io_contract import IOContractManager
from shared.model import IOContract


def load_config(config_path: str) -> Dict[str, Any]:
    """Load runner configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_kernel_module(kernel_path: Path) -> Tuple[Any, str]:
    """
    Load kernel module with proper sys.modules registration.

    Returns:
        (module, module_name) tuple
    """
    source_code = kernel_path.read_text(encoding="utf-8")
    source_hash = hashlib.md5(source_code.encode()).hexdigest()[:8]
    module_name = f"triton_kernel_{source_hash}"

    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Register for nested imports
    spec.loader.exec_module(module)

    return module, module_name


def find_triton_kernels(module: Any) -> List[Tuple[str, Any]]:
    """
    Find all @triton.jit decorated kernels in module.

    Returns:
        List of (kernel_name, kernel_func) tuples
    """
    try:
        import triton.runtime.jit as trjit
        JIT = trjit.JITFunction
    except Exception:
        JIT = None

    kernels = []
    for name, obj in module.__dict__.items():
        # Check if it's a JITFunction
        if JIT is not None and isinstance(obj, JIT):
            kernels.append((name, obj))
        # Fallback duck-typing for different Triton versions
        elif hasattr(obj, "__getitem__") and hasattr(obj, "__call__") and hasattr(obj, "fn"):
            kernels.append((name, obj))

    return kernels


def generate_inputs_from_io_contract(io_contract: Dict[str, Any], device) -> Tuple[List[Any], Any]:
    """
    Generate tensor inputs from IOContract specification using IOContractManager.

    Returns:
        (inputs, grid) tuple
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch not available")

    if not io_contract:
        raise ValueError("IOContract must be provided")

    # Convert dict to IOContract model
    io_contract_obj = IOContract.from_dict(io_contract)

    # Use IOContractManager to generate inputs
    manager = IOContractManager()
    inputs = manager.generate_inputs(io_contract_obj, device)

    # Extract grid configuration from IOContract
    if io_contract_obj.launch and io_contract_obj.launch.grid:
        grid = (
            io_contract_obj.launch.grid.x,
            io_contract_obj.launch.grid.y,
            io_contract_obj.launch.grid.z
        )
    else:
        grid = (1,)

    return inputs, grid


def apply_launch_update(grid: Tuple, launch_update: Dict[str, Any]) -> Tuple:
    """Apply LAUNCH_UPDATE overrides to grid configuration."""
    if not launch_update:
        return grid

    grid_list = list(grid)
    if "grid_x" in launch_update:
        grid_list[0] = launch_update["grid_x"]
    if "grid_y" in launch_update and len(grid_list) > 1:
        grid_list[1] = launch_update["grid_y"]
    if "grid_z" in launch_update and len(grid_list) > 2:
        grid_list[2] = launch_update["grid_z"]

    return tuple(grid_list)


def run_kernel_io_contract_mode(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute Triton kernel using IOContract-based approach.

    This is the preferred mode using metadata.kernel_name.
    """
    try:
        import torch
        use_events = torch.cuda.is_available()
    except Exception:
        torch = None
        use_events = False

    if not torch:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": "PyTorch not available",
            "launch_update_applied": False
        }

    # Load kernel module
    kernel_path = Path(config["kernel_module_path"])
    kernel_name = config.get("kernel_name")
    timing = config["timing"]
    launch_update = config.get("launch_update", {})
    io_contract = config.get("io_contract", {})

    try:
        module, _ = load_kernel_module(kernel_path)
    except Exception as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"Failed to load kernel module: {e}",
            "traceback": traceback.format_exc(),
            "launch_update_applied": False
        }

    # Find @triton.jit kernels
    kernels = find_triton_kernels(module)

    if not kernels:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": "No @triton.jit kernels found in module",
            "launch_update_applied": False
        }

    # Select kernel by name or use first one
    jit_func = None
    if kernel_name:
        for k_name, func in kernels:
            if k_name == kernel_name:
                jit_func = func
                break

        if jit_func is None:
            available = [k_name for k_name, _ in kernels]
            return {
                "ok": False,
                "mean_ms": float("inf"),
                "std_ms": 0.0,
                "error": f"Kernel '{kernel_name}' not found. Available: {available}",
                "launch_update_applied": False
            }
    else:
        # Use first kernel found
        jit_func = kernels[0][1]

    # Generate inputs from IOContract
    device = torch.device("cuda:0") if use_events else torch.device("cpu")
    try:
        inputs, grid = generate_inputs_from_io_contract(io_contract, device)
    except Exception as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"Failed to generate inputs: {e}",
            "traceback": traceback.format_exc(),
            "launch_update_applied": False
        }

    # Apply launch updates to grid
    grid = apply_launch_update(grid, launch_update)

    # Timing loop
    ok = True
    times = []
    error_msg = None
    error_tb = None

    try:
        # Warmup
        for _ in range(timing["warmup"]):
            jit_func[grid](*inputs)
            if use_events:
                torch.cuda.synchronize()

        # Timed runs
        if use_events:
            for _ in range(timing["repeat"]):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                accum = 0.0

                for _ in range(timing["iters"]):
                    start.record()
                    jit_func[grid](*inputs)
                    end.record()
                    end.synchronize()
                    accum += start.elapsed_time(end)

                times.append(accum / timing["iters"])
        else:
            for _ in range(timing["repeat"]):
                accum = 0.0
                for _ in range(timing["iters"]):
                    t0 = time.time()
                    jit_func[grid](*inputs)
                    t1 = time.time()
                    if torch:
                        torch.cuda.synchronize()
                    accum += (t1 - t0) * 1000.0
                times.append(accum / timing["iters"])

    except Exception as e:
        ok = False
        times = [float("inf")]
        error_msg = str(e)
        error_tb = traceback.format_exc()
        print(f"Error during kernel execution: {error_msg}", file=sys.stderr)
        print(error_tb, file=sys.stderr)

    mean_ms = float(sum(times) / len(times)) if times else float("inf")
    std_ms = float((sum((x - mean_ms) ** 2 for x in times) / len(times)) ** 0.5) if times else 0.0

    result = {
        "ok": bool(ok),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "launch_update_applied": True
    }

    if error_msg:
        result["error"] = error_msg
        result["traceback"] = error_tb

    return result


def run_kernel_legacy_mode(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute Triton kernel using legacy invocation_example approach.

    Fallback mode for backward compatibility.
    """
    try:
        import torch
        use_events = torch.cuda.is_available()
    except Exception:
        torch = None
        use_events = False

    kernel_path = Path(config["kernel_module_path"])
    invocation = config.get("invocation_example", "")
    timing = config["timing"]
    launch_update = config.get("launch_update", {})

    if not invocation:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": "No invocation_example provided for legacy mode",
            "launch_update_applied": False
        }

    # Load kernel module
    try:
        module, _ = load_kernel_module(kernel_path)
    except Exception as e:
        return {
            "ok": False,
            "mean_ms": float("inf"),
            "std_ms": 0.0,
            "error": f"Failed to load kernel module: {e}",
            "traceback": traceback.format_exc(),
            "launch_update_applied": False
        }

    # Check if LAUNCH_UPDATE is used in invocation
    uses_update = "LAUNCH_UPDATE" in invocation

    def do_invoke():
        local_vars = dict(km=module, LAUNCH_UPDATE=launch_update)
        exec(invocation, globals(), local_vars)

    ok = True
    times = []

    try:
        # Warmup
        for _ in range(timing["warmup"]):
            do_invoke()
            if use_events:
                torch.cuda.synchronize()

        # Timed runs
        if use_events:
            for _ in range(timing["repeat"]):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                accum = 0.0

                for _ in range(timing["iters"]):
                    start.record()
                    do_invoke()
                    end.record()
                    end.synchronize()
                    accum += start.elapsed_time(end)

                times.append(accum / timing["iters"])
        else:
            for _ in range(timing["repeat"]):
                accum = 0.0
                for _ in range(timing["iters"]):
                    t0 = time.time()
                    do_invoke()
                    t1 = time.time()
                    if torch:
                        torch.cuda.synchronize()
                    accum += (t1 - t0) * 1000.0
                times.append(accum / timing["iters"])

    except Exception:
        ok = False
        times = [float("inf")]

    mean_ms = float(sum(times) / len(times))
    std_ms = float((sum((x - mean_ms) ** 2 for x in times) / len(times)) ** 0.5)

    return {
        "ok": bool(ok),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "launch_update_applied": bool(uses_update)
    }


def main() -> int:
    """Main entry point for triton_runner."""
    if len(sys.argv) < 2:
        print("Usage: python -m workers.triton_runner <config_path>", file=sys.stderr)
        return 1

    config_path = sys.argv[1]

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        return 1

    # Determine execution mode
    kernel_name = config.get("kernel_name")
    io_contract = config.get("io_contract")

    if kernel_name and io_contract:
        # Preferred: IOContract-based execution
        result = run_kernel_io_contract_mode(config)
    else:
        # Fallback: Legacy invocation_example
        result = run_kernel_legacy_mode(config)

    # Write result
    result_path = config.get("result_path", "runner_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
