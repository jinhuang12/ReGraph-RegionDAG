#!/usr/bin/env python3
"""
run_contract_once.py

Helper script: load a contract + kernel module, build inputs, and run the entry
point ONCE. Used as the target of Nsight Compute (ncu) profiling.

Usage (from repo root):

  python tools/run_contract_once.py \
      --contract contracts/vec_matmul_kernel_small_contract.json \
      --module kernels/vec_matmul_kernel.py \
      --entry-point run \
      --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Local imports
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(THIS_DIR))

from contracts_common import build_entrypoint_and_kwargs  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True, help="Path to contract JSON")
    ap.add_argument("--module", required=True, help="Path to kernel module .py")
    ap.add_argument("--entry-point", default=None, help="Entry point function name (defaults from contract)")
    ap.add_argument("--device", default="cuda:0", help="Device string (e.g. cuda:0 or cpu)")
    args = ap.parse_args()

    contract_path = Path(args.contract)
    module_path = Path(args.module)
    device = args.device

    contract, module, func, ep_name, kwargs = build_entrypoint_and_kwargs(
        contract_path=contract_path,
        module_path=module_path,
        entry_point=args.entry_point,
        device=device,
    )

    dev = torch.device(device)
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(0)

    _ = func(**kwargs)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
