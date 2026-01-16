#!/usr/bin/env python3
"""
run_contract_once.py

Helper script: load a contract + kernel module, build inputs, and execute ONE
callable path. Used as the target of Nsight Compute (ncu) profiling.

By default it calls the module entry point (e.g. run()). For IronFist-style
"multi_kernel" contracts, you may prefer profiling benchmark_kernel() instead
so you can swap backends (Triton/CUDA/CUTLASS) without touching contract IO.

Usage (from repo root):

  # Call entrypoint once (default):
  python tools/run_contract_once.py \
      --contract contracts/ironfist/vec_matmul_kernel_small_contract.json \
      --module   kernels/ironfist/vec_matmul_kernel_10ba429b.py \
      --entry-point run \
      --device cuda:0

  # Call benchmark_kernel once (useful for multi_kernel NCU):
  python tools/run_contract_once.py \
      --call benchmark_kernel \
      --warmup 0 --iters 1 --repeat 1 \
      --backend triton \
      --contract contracts/ironfist/gemm_split_k_kernel_large_contract.json \
      --module   kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

# Local imports
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(THIS_DIR))

from contracts_common import build_entrypoint_and_kwargs, import_kernel_module, load_contract  # type: ignore
from contracts_common import _make_torch_tensor, _parse_scalar_value  # type: ignore


def _build_contract_args(contract: dict, device: str, backend: str | None) -> dict:
    kernel = contract.get("kernel", {}) or {}
    io = kernel.get("io", {}) or {}
    arg_specs = io.get("args", []) or []

    args: dict = {}
    for arg in arg_specs:
        name = arg.get("name")
        if not name:
            continue
        atype = arg.get("type", "int")
        if atype == "tensor":
            args[name] = _make_torch_tensor(arg, device=device)
        else:
            args[name] = _parse_scalar_value(arg)
    args["_device"] = device
    if backend:
        args["_backend"] = backend
    return args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True, help="Path to contract JSON")
    ap.add_argument("--module", required=True, help="Path to kernel module .py")
    ap.add_argument("--entry-point", default=None, help="Entry point function name (defaults from contract)")
    ap.add_argument("--device", default="cuda:0", help="Device string (e.g. cuda:0 or cpu)")
    ap.add_argument(
        "--call",
        choices=["entrypoint", "benchmark_kernel"],
        default="entrypoint",
        help="What to execute: module entrypoint (default) or module benchmark_kernel().",
    )
    ap.add_argument("--warmup", type=int, default=0, help="Warmup iterations (benchmark_kernel only)")
    ap.add_argument("--iters", type=int, default=1, help="Inner iterations (benchmark_kernel only)")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat trials (benchmark_kernel only)")
    ap.add_argument("--backend", default=None, help="Optional backend selector string (passed via _backend / KO_BACKEND)")
    args = ap.parse_args()

    contract_path = Path(args.contract)
    module_path = Path(args.module)
    device = args.device

    if args.backend:
        os.environ["KO_BACKEND"] = str(args.backend)

    dev = torch.device(device)
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(0)

    if args.call == "entrypoint":
        contract, module, func, ep_name, kwargs = build_entrypoint_and_kwargs(
            contract_path=contract_path,
            module_path=module_path,
            entry_point=args.entry_point,
            device=device,
        )
        _ = func(**kwargs)
    else:
        contract = load_contract(contract_path)
        module = import_kernel_module(module_path)
        bench_fn = getattr(module, "benchmark_kernel", None)
        if not callable(bench_fn):
            raise SystemExit("Selected --call benchmark_kernel but module has no callable benchmark_kernel()")
        contract_args = _build_contract_args(contract, device=device, backend=args.backend)
        try:
            _ = bench_fn(contract_args=contract_args, warmup=args.warmup, iters=args.iters, repeat=args.repeat)
        except TypeError:
            # Allow positional signature benchmark_kernel(contract_args, warmup, iters, repeat)
            _ = bench_fn(contract_args, args.warmup, args.iters, args.repeat)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
