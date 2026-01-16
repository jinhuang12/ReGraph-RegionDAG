#!/usr/bin/env python3
"""
Sweep benchmark for LoRA fused kernel: Triton vs CUDA backends.

This script is intentionally "kernel-only": it times the fused shrink+expand
implementation inside `kernels/ironfist/lora_fused_kernel_*.py` via the module's
internal helpers (`_fused_shrink_expand` and `_fused_shrink_expand_cuda`).

It generates a mixture of:
  - suite contract cases (optional),
  - synthetic/random cases within the CUDA backend constraints.

Outputs:
  - JSON summary + per-case results
  - CSV table for quick plotting
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_module(module_path: Path):
    module_path = module_path.resolve()
    name = f"lora_fused_sweep_{module_path.stem}_{abs(hash(str(module_path)))}"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _set_all_seeds(seed: int):
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clear_triton_ptr_caches(mod: Any):
    # The Triton path caches weight tensors in globals (see _get_lora_a_ptr/_get_lora_b_ptr).
    # When generating new random weights repeatedly (as this sweep does), the caches can
    # grow unbounded and retain GPU memory. Clear them between cases.
    if hasattr(mod, "_LORA_A_PTR_DICT"):
        try:
            mod._LORA_A_PTR_DICT.clear()
        except Exception:
            pass
    if hasattr(mod, "_LORA_B_PTR_DICT"):
        try:
            mod._LORA_B_PTR_DICT.clear()
        except Exception:
            pass


@dataclass(frozen=True)
class Case:
    num_tokens: int
    hidden_size: int
    num_loras: int
    num_active_loras: int
    num_slices: int = 1
    # Triton LoRA rank (CUDA backend supports rank=16 only).
    lora_rank: int = 16
    offset_start: int = 0
    add_inputs: bool = False
    dtype: str = "torch.float16"

    # Optional meta (if absent, the script suggests reasonable defaults).
    block_m: int | None = None
    block_n: int | None = None
    block_k: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None

    # Optional tag for provenance (e.g., "suite:lora_fused_kernel_small_contract")
    tag: str | None = None

    def key(self) -> tuple[int, int, int, int, int]:
        return (
            self.num_tokens,
            self.hidden_size,
            self.num_loras,
            self.num_active_loras,
            self.num_slices,
            self.lora_rank,
        )

    def id(self) -> str:
        return (
            f"M{self.num_tokens}_K{self.hidden_size}_L{self.num_loras}_A{self.num_active_loras}_S{self.num_slices}_R{self.lora_rank}"
        )


def _case_from_contract(contract_path: Path) -> Case:
    obj = json.loads(contract_path.read_text())
    args = obj["kernel"]["io"]["args"]
    kv: dict[str, Any] = {}
    for a in args:
        kv[a["name"]] = a["value"]
    return Case(
        num_tokens=int(kv["num_tokens"]),
        hidden_size=int(kv["hidden_size"]),
        num_loras=int(kv["num_loras"]),
        num_active_loras=int(kv["num_active_loras"]),
        num_slices=int(kv.get("num_slices", 1)),
        lora_rank=int(kv.get("lora_rank", 16)),
        offset_start=int(kv.get("offset_start", 0)),
        add_inputs=bool(kv.get("add_inputs", False)),
        dtype=str(kv.get("dtype", "torch.float16")),
        block_m=int(kv.get("block_m")) if "block_m" in kv else None,
        block_n=int(kv.get("block_n")) if "block_n" in kv else None,
        block_k=int(kv.get("block_k")) if "block_k" in kv else None,
        num_warps=int(kv.get("num_warps")) if "num_warps" in kv else None,
        num_stages=int(kv.get("num_stages")) if "num_stages" in kv else None,
        tag=f"suite:{contract_path.stem}",
    )


def _estimate_case_bytes(case: Case, dtype_bytes: int = 2) -> int:
    # Rough allocator footprint estimate for tensors created per case.
    m = int(case.num_tokens)
    k = int(case.hidden_size)
    s = int(case.num_slices)
    l = int(case.num_loras)
    # CUDA always uses rank 16; Triton uses case.lora_rank.
    r = int(max(case.lora_rank, 16))

    inputs = m * k
    output = m * (k * s + int(case.offset_start))
    lora_a = s * l * r * k
    lora_b = s * l * k * r

    elems = inputs + output + lora_a + lora_b
    # Small metadata tensors: token mapping, indices, per-lora counts, etc. Ignore.
    return int(elems) * int(dtype_bytes)


def _validate_case_for_cuda(case: Case) -> tuple[bool, str]:
    if case.dtype != "torch.float16":
        return False, "dtype must be torch.float16 for CUDA backend"
    if case.offset_start != 0:
        return False, "offset_start must be 0 for CUDA backend"
    if case.hidden_size % 64 != 0:
        return False, "hidden_size must be multiple of 64 for CUDA backend"
    if case.num_tokens <= 0 or case.hidden_size <= 0:
        return False, "num_tokens/hidden_size must be > 0"
    if case.num_loras <= 0:
        return False, "num_loras must be > 0"
    if case.num_active_loras <= 0:
        return False, "num_active_loras must be > 0"
    if case.num_active_loras > case.num_loras:
        return False, "num_active_loras must be <= num_loras"
    if case.num_slices <= 0:
        return False, "num_slices must be > 0"
    if case.add_inputs:
        # The benchmark loop reuses the output tensor; add_inputs would accumulate.
        return False, "add_inputs=True not supported in this sweep harness"
    return True, ""


def _suggest_meta(case: Case) -> dict[str, int]:
    # Heuristics derived from the suite contracts (kept intentionally simple).
    m = int(case.num_tokens)
    k = int(case.hidden_size)
    l = int(case.num_loras)
    a = int(case.num_active_loras)
    s = int(case.num_slices)

    if k <= 1024:
        block_n = 64
    elif k <= 2048:
        block_n = 128
    else:
        block_n = 256

    if m <= 128:
        block_m = 64
    elif m <= 256:
        block_m = 32
    elif m <= 4096:
        block_m = 16
    else:
        block_m = 32

    # BLOCK_K is Triton-only; CUDA uses a fixed internal K-block (64).
    if k <= 1024:
        block_k = 256 if s > 1 else 128
        num_stages = 4 if block_k >= 256 else 3
    else:
        # Default to a pipeline-friendly K block. For very large K and single-LoRA,
        # a larger BLOCK_K can sometimes win; we keep it conservative here.
        block_k = 64
        num_stages = 4
        if k >= 4096 and l >= 16:
            num_stages = 5

    num_warps = 4
    if m <= 256 and k >= 2048:
        num_warps = 8

    # High-throughput regime (many LoRAs): match suite-ish defaults.
    if m >= 8192 and k >= 8192 and a >= 256:
        block_m = 32
        block_n = 256
        block_k = 64
        num_warps = 4
        num_stages = 4

    return {
        "block_m": int(block_m),
        "block_n": int(block_n),
        "block_k": int(block_k),
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
    }


def _compute_kernel_config(case: Case, backend: str) -> dict[str, Any]:
    # Start from explicit meta (suite) or heuristic defaults.
    meta = _suggest_meta(case)
    if case.block_m is not None:
        meta["block_m"] = int(case.block_m)
    if case.block_n is not None:
        meta["block_n"] = int(case.block_n)
    if case.block_k is not None:
        meta["block_k"] = int(case.block_k)
    if case.num_warps is not None:
        meta["num_warps"] = int(case.num_warps)
    if case.num_stages is not None:
        meta["num_stages"] = int(case.num_stages)

    # Mirror the in-module benchmark_kernel overrides for apples-to-apples behavior.
    m = int(case.num_tokens)
    k = int(case.hidden_size)
    a = int(case.num_active_loras)

    if m <= 256 and meta["block_m"] > 32:
        meta["block_m"] = 32
    if k >= 2048 and meta["block_k"] < 64:
        meta["block_k"] = 64
        meta["num_stages"] = max(int(meta["num_stages"]), 4)
    if m <= 256 and k >= 2048 and meta["num_warps"] < 8:
        meta["num_warps"] = 8
    if m >= 8192 and k >= 8192 and a >= 256 and meta["num_warps"] < 8:
        meta["num_warps"] = 8
        meta["num_stages"] = max(int(meta["num_stages"]), 5)

    cfg: dict[str, Any] = dict(meta)

    if backend == "cuda":
        # CUDA extension supports 2/4/8 warps only.
        if int(cfg.get("num_warps", 4)) not in (2, 4, 8):
            cfg["num_warps"] = 8 if int(cfg.get("num_warps", 4)) >= 8 else 4

    # High-throughput specialization (matches benchmark_kernel).
    if m >= 8192 and k >= 8192 and a >= 256:
        if backend == "cuda":
            cfg["block_m"] = 16
            cfg["group_n"] = 24
            cfg["num_warps"] = 4
            cfg["split_kernels"] = False
        else:
            cfg["group_n"] = 32

    return cfg


def _prepare_inputs(mod: Any, case: Case, device: str, seed: int):
    import torch

    _set_all_seeds(seed)
    dev = torch.device(device)

    dtype = torch.float16
    inputs = torch.randn(case.num_tokens, case.hidden_size, dtype=dtype, device=dev)

    output_shape = (case.num_tokens, case.hidden_size * case.num_slices + case.offset_start)
    # Single output buffer reused for both backends (add_inputs=False in this sweep).
    output = torch.zeros(*output_shape, dtype=dtype, device=dev)

    token_lora_mapping = torch.randint(-1, case.num_active_loras, (case.num_tokens,), device=dev)
    _, token_indices_sorted = torch.sort(token_lora_mapping, stable=True)
    token_indices_sorted_i64 = token_indices_sorted
    token_indices_sorted_i32 = token_indices_sorted.to(torch.int32).contiguous()

    active_lora_ids, num_tokens_per_lora_raw = torch.unique(token_lora_mapping, sorted=True, return_counts=True)

    max_loras = int(case.num_loras)
    padded_num_tokens_per_lora = torch.zeros(max_loras + 1, dtype=torch.int32, device=dev)
    actual_num_loras = min(int(num_tokens_per_lora_raw.numel()), max_loras + 1)
    if actual_num_loras > 0:
        padded_num_tokens_per_lora[:actual_num_loras] = num_tokens_per_lora_raw[:actual_num_loras].to(torch.int32)

    padded_lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32, device=dev)
    if actual_num_loras > 0:
        padded_lora_ids[:actual_num_loras] = active_lora_ids[:actual_num_loras].to(torch.int32)

    lora_token_start_loc = torch.zeros(max_loras + 2, dtype=torch.int32, device=dev)
    if actual_num_loras > 0:
        lora_token_start_loc[1 : actual_num_loras + 1] = torch.cumsum(
            padded_num_tokens_per_lora[:actual_num_loras], dim=0
        )

    # Reduce overlaunch: cap grid M-dimension to max tokens per active LoRA,
    # and cap the number of LoRAs to the number of unique IDs present.
    grid_loras = int(actual_num_loras)
    grid_m = 0
    if actual_num_loras > 0:
        has_no_lora = int(active_lora_ids[0].item()) == -1
        counts_slice = (
            padded_num_tokens_per_lora[1:actual_num_loras] if has_no_lora else padded_num_tokens_per_lora[:actual_num_loras]
        )
        if counts_slice.numel() > 0:
            grid_m = int(counts_slice.max().item())

    return {
        "inputs": inputs,
        "output": output,
        "token_lora_mapping": token_lora_mapping,
        "token_indices_sorted_i64": token_indices_sorted_i64,
        "token_indices_sorted_i32": token_indices_sorted_i32,
        "num_tokens_per_lora": padded_num_tokens_per_lora,
        "lora_token_start_loc": lora_token_start_loc,
        "lora_ids": padded_lora_ids,
        "grid_m": int(max(grid_m, 0)),
        "grid_loras": int(max(grid_loras, 0)),
    }


def _alloc_weights(case: Case, rank: int, device: str, seed: int):
    import torch

    _set_all_seeds(seed)
    dev = torch.device(device)
    dtype = torch.float16
    lora_a = [torch.randn(case.num_loras, rank, case.hidden_size, dtype=dtype, device=dev) for _ in range(case.num_slices)]
    lora_b = [torch.randn(case.num_loras, case.hidden_size, rank, dtype=dtype, device=dev) for _ in range(case.num_slices)]
    return lora_a, lora_b


def _time_backend(
    mod: Any,
    backend: str,
    case: Case,
    inputs: dict[str, Any],
    kernel_config: dict[str, Any],
    warmup: int,
    iters: int,
    repeat: int,
    device: str,
    seed: int,
) -> tuple[float, float]:
    import torch

    if backend == "cuda":
        fn = mod._fused_shrink_expand_cuda
        token_indices_sorted = inputs["token_indices_sorted_i32"]
        rank = 16
    else:
        fn = mod._fused_shrink_expand
        token_indices_sorted = inputs["token_indices_sorted_i64"]
        rank = int(case.lora_rank)

    output = inputs["output"]
    output.zero_()

    lora_a, lora_b = _alloc_weights(case, rank=rank, device=device, seed=seed)

    # Include grid_m/lora caps (important for fairness).
    cfg = dict(kernel_config)
    cfg["grid_m"] = int(inputs["grid_m"])
    cfg["grid_loras"] = int(inputs["grid_loras"])

    # Warmup (also forces compilation / extension load).
    for _ in range(max(0, warmup)):
        fn(
            inputs=inputs["inputs"],
            lora_a_weights=lora_a,
            lora_b_weights=lora_b,
            output_tensor=output,
            token_lora_mapping=inputs["token_lora_mapping"],
            token_indices_sorted_by_lora_ids=token_indices_sorted,
            num_tokens_per_lora=inputs["num_tokens_per_lora"],
            lora_token_start_loc=inputs["lora_token_start_loc"],
            lora_ids=inputs["lora_ids"],
            scaling=1.0,
            offset_start=case.offset_start,
            add_inputs=case.add_inputs,
            kernel_config=cfg,
        )
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        for _ in range(iters):
            start_evt.record()
            fn(
                inputs=inputs["inputs"],
                lora_a_weights=lora_a,
                lora_b_weights=lora_b,
                output_tensor=output,
                token_lora_mapping=inputs["token_lora_mapping"],
                token_indices_sorted_by_lora_ids=token_indices_sorted,
                num_tokens_per_lora=inputs["num_tokens_per_lora"],
                lora_token_start_loc=inputs["lora_token_start_loc"],
                lora_ids=inputs["lora_ids"],
                scaling=1.0,
                offset_start=case.offset_start,
                add_inputs=case.add_inputs,
                kernel_config=cfg,
            )
            end_evt.record()
            end_evt.synchronize()
            times_ms.append(float(start_evt.elapsed_time(end_evt)))

    mean_ms = float(statistics.mean(times_ms)) if times_ms else float("inf")
    std_ms = float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0
    return mean_ms, std_ms


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0 and math.isfinite(x)]
    if not xs:
        return float("nan")
    return float(math.exp(sum(math.log(x) for x in xs) / len(xs)))


def _parse_int_list(csv_str: str) -> list[int]:
    parts = [p.strip() for p in csv_str.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_case_str(s: str) -> Case:
    # Format: M,K,L,A,S[,R]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) not in (5, 6):
        raise ValueError(f"--case expects 'M,K,L,A,S[,R]' but got: {s!r}")
    m = int(parts[0])
    k = int(parts[1])
    l = int(parts[2])
    a = int(parts[3])
    s_num = int(parts[4])
    r = int(parts[5]) if len(parts) == 6 else 16
    return Case(
        num_tokens=m,
        hidden_size=k,
        num_loras=l,
        num_active_loras=a,
        num_slices=s_num,
        lora_rank=r,
        tag="user:case",
    )


def _default_random_space() -> dict[str, list[int]]:
    return {
        "num_tokens": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10240],
        "hidden_size": [256, 512, 1024, 2048, 2880, 4096, 8192],
        "num_loras": [1, 2, 4, 8, 16, 64, 256, 1024],
        "num_slices": [1],
        "lora_rank": [16],
    }


def _gen_random_cases(n: int, seed: int, space: dict[str, list[int]]) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    for _ in range(n):
        m = rng.choice(space["num_tokens"])
        k = rng.choice(space["hidden_size"])
        l = rng.choice(space["num_loras"])
        s = rng.choice(space["num_slices"])
        r = rng.choice(space["lora_rank"])
        # Active LoRAs: sample a few meaningful ratios.
        candidates = {
            1,
            min(l, 2),
            min(l, 4),
            min(l, 8),
            max(1, l // 2),
            l,
        }
        a = rng.choice(sorted(candidates))
        cases.append(
            Case(
                num_tokens=int(m),
                hidden_size=int(k),
                num_loras=int(l),
                num_active_loras=int(a),
                num_slices=int(s),
                lora_rank=int(r),
                tag="rand",
            )
        )
    return cases


def _meta_specified_count(case: Case) -> int:
    return sum(
        1
        for v in (case.block_m, case.block_n, case.block_k, case.num_warps, case.num_stages)
        if v is not None
    )


def _case_priority(case: Case) -> tuple[int, int]:
    # Prefer suite-provided meta for apples-to-apples comparisons.
    tag = case.tag or ""
    if tag.startswith("user"):
        src = 4
    elif tag.startswith("suite:"):
        src = 3
    elif tag.startswith("anchor:"):
        src = 2
    elif tag == "rand":
        src = 1
    else:
        src = 0
    return (src, _meta_specified_count(case))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        type=str,
        default="kernels/ironfist/lora_fused_kernel_0318783b.py",
        help="Kernel module path (defaults to the optimized LoRA fused module).",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="suites/ironfist/lora_fused_kernel_suite.json",
        help="Suite JSON (used only when --include-suite is set).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-suite", action="store_true", help="Include suite contract cases.")
    parser.add_argument("--random-cases", type=int, default=40, help="Number of random cases to add.")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Add explicit case as 'M,K,L,A,S[,R]' (K=hidden_size, L=num_loras). Can be repeated.",
    )
    parser.add_argument("--cases-json", type=str, default="", help="Optional JSON file with explicit cases.")
    parser.add_argument("--out", type=str, default="runs/lora_fused_sweep_cuda_vs_triton", help="Output directory.")
    parser.add_argument(
        "--max-gb",
        type=float,
        default=0.0,
        help="Skip cases whose estimated allocations exceed this many GB (0 = auto from free memory).",
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.85,
        help="When --max-gb=0, use this fraction of current free memory as cap.",
    )
    parser.add_argument("--tokens", type=str, default="", help="Override random space: comma-separated num_tokens.")
    parser.add_argument("--hidden-sizes", type=str, default="", help="Override random space: comma-separated hidden_size.")
    parser.add_argument("--num-loras", type=str, default="", help="Override random space: comma-separated num_loras.")
    parser.add_argument("--slices", type=str, default="", help="Override random space: comma-separated num_slices.")
    parser.add_argument("--ranks", type=str, default="", help="Override random space: comma-separated lora_rank.")
    parser.add_argument("--grid-m", type=str, default="", help="Cartesian sweep: comma-separated M (num_tokens).")
    parser.add_argument("--grid-k", type=str, default="", help="Cartesian sweep: comma-separated K (hidden_size).")
    parser.add_argument("--grid-max-loras", type=str, default="", help="Cartesian sweep: comma-separated max_loras (num_loras).")
    parser.add_argument("--grid-active-loras", type=str, default="", help="Cartesian sweep: comma-separated num_active_loras (default: == num_loras).")
    parser.add_argument("--grid-slices", type=str, default="", help="Cartesian sweep: comma-separated num_slices.")
    parser.add_argument("--grid-triton-ranks", type=str, default="", help="Cartesian sweep: comma-separated Triton lora_rank values.")
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Only run --case/--cases-json/--grid-* (disables anchors and random cases).",
    )
    parser.add_argument(
        "--compare-rank-mismatch",
        action="store_true",
        help="If Triton lora_rank != 16, still compute speedup vs CUDA rank=16 (NOTE: different math).",
    )
    args = parser.parse_args(argv)

    module_path = Path(args.module)
    suite_path = Path(args.suite)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark (both backends run on GPU).")

    mod = _load_module(module_path)

    # Case sources.
    cases: list[Case] = []
    for s in args.case:
        cases.append(_parse_case_str(str(s)))

    if args.cases_json:
        raw = json.loads(Path(args.cases_json).read_text())
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("--cases-json must be a JSON list of objects")
            item2 = dict(item)
            item2.setdefault("tag", "user")
            cases.append(Case(**item2))

    if args.include_suite:
        suite_obj = json.loads(suite_path.read_text())
        contracts_dir = Path(suite_obj["contracts_dir"])
        for c in suite_obj["contracts"]:
            cases.append(_case_from_contract(contracts_dir / c["filename"]))

    space = _default_random_space()
    if args.tokens:
        space["num_tokens"] = _parse_int_list(args.tokens)
    if args.hidden_sizes:
        space["hidden_size"] = _parse_int_list(args.hidden_sizes)
    if args.num_loras:
        space["num_loras"] = _parse_int_list(args.num_loras)
    if args.slices:
        space["num_slices"] = _parse_int_list(args.slices)
    if args.ranks:
        space["lora_rank"] = _parse_int_list(args.ranks)

    # Optional Cartesian grid sweep (useful for spreadsheet-style tables).
    grid_cases: list[Case] = []
    if args.grid_m or args.grid_k or args.grid_max_loras or args.grid_slices or args.grid_triton_ranks or args.grid_active_loras:
        grid_m = _parse_int_list(args.grid_m) if args.grid_m else []
        grid_k = _parse_int_list(args.grid_k) if args.grid_k else []
        grid_l = _parse_int_list(args.grid_max_loras) if args.grid_max_loras else []
        grid_a = _parse_int_list(args.grid_active_loras) if args.grid_active_loras else []
        grid_s = _parse_int_list(args.grid_slices) if args.grid_slices else [1]
        grid_r = _parse_int_list(args.grid_triton_ranks) if args.grid_triton_ranks else [16]

        if not grid_m or not grid_k or not grid_l:
            raise ValueError("--grid-m, --grid-k, and --grid-max-loras are required when using --grid-* sweep")
        tmp: list[Case] = []
        for k in grid_k:
            for l in grid_l:
                actives = grid_a if grid_a else [l]
                for a in actives:
                    for s_num in grid_s:
                        for r in grid_r:
                            for m in grid_m:
                                tmp.append(
                                    Case(
                                        num_tokens=int(m),
                                        hidden_size=int(k),
                                        num_loras=int(l),
                                        num_active_loras=int(a),
                                        num_slices=int(s_num),
                                        lora_rank=int(r),
                                        tag="grid",
                                    )
                                )
        grid_cases = tmp
        cases.extend(grid_cases)

    if not args.grid_only:
        # Add a couple of explicit "anchor" cases for visibility.
        cases.extend(
            [
                Case(1024, 1024, 1, 1, tag="anchor:small"),
                Case(128, 2048, 4, 2, tag="anchor:base"),
                Case(1024, 8192, 1, 1, tag="anchor:large"),
                Case(10240, 8192, 1024, 512, tag="anchor:high_throughput"),
            ]
        )
        cases.extend(_gen_random_cases(int(args.random_cases), int(args.seed), space))

    # De-dup by primary shape key.
    uniq: dict[tuple[int, int, int, int, int, int], Case] = {}
    for c in cases:
        k = c.key()
        if k not in uniq:
            uniq[k] = c
            continue
        if _case_priority(c) > _case_priority(uniq[k]):
            uniq[k] = c
    cases = list(uniq.values())
    cases.sort(key=lambda c: (c.hidden_size, c.num_tokens, c.num_loras, c.num_active_loras, c.num_slices))

    # Memory cap.
    torch.cuda.set_device(torch.device(args.device))
    free_b, total_b = torch.cuda.mem_get_info()
    if float(args.max_gb) > 0:
        max_bytes = int(float(args.max_gb) * (1024**3))
    else:
        max_bytes = int(float(args.mem_fraction) * float(free_b))

    device_name = torch.cuda.get_device_name(torch.device(args.device))
    print(f"Device: {args.device} ({device_name})")
    print(f"Cases: {len(cases)} (include_suite={bool(args.include_suite)}, random={args.random_cases})")
    print(f"Timing: warmup={args.warmup}, iters={args.iters}, repeat={args.repeat}")
    print(f"Memory cap: {max_bytes / (1024**3):.2f} GiB (free {free_b / (1024**3):.2f} / total {total_b / (1024**3):.2f})")

    results: list[dict[str, Any]] = []
    speedups: list[float] = []
    cuda_wins = 0
    triton_wins = 0

    for idx, case in enumerate(cases):
        ok, reason = _validate_case_for_cuda(case)
        if not ok:
            results.append(
                {
                    "case_id": case.id(),
                    "tag": case.tag,
                    "skipped": True,
                    "skip_reason": reason,
                    "case": case.__dict__,
                }
            )
            continue

        est = _estimate_case_bytes(case, dtype_bytes=2)
        est *= 1.20  # safety factor
        if est > max_bytes:
            results.append(
                {
                    "case_id": case.id(),
                    "tag": case.tag,
                    "skipped": True,
                    "skip_reason": f"estimated_bytes {int(est)} exceeds cap {int(max_bytes)}",
                    "case": case.__dict__,
                    "estimated_bytes": int(est),
                }
            )
            continue

        print(f"[{idx+1:4d}/{len(cases)}] {case.id()} ({case.tag or 'case'})")

        # Clear caches from prior cases to avoid retaining weights across shapes.
        _clear_triton_ptr_caches(mod)

        try:
            case_seed = int(args.seed) + idx
            inputs = _prepare_inputs(mod, case, device=args.device, seed=case_seed)
            triton_cfg = _compute_kernel_config(case, backend="triton")
            cuda_cfg = _compute_kernel_config(case, backend="cuda")
            triton_ms, triton_std = _time_backend(
                mod,
                "triton",
                case,
                inputs,
                triton_cfg,
                warmup=args.warmup,
                iters=args.iters,
                repeat=args.repeat,
                device=args.device,
                seed=case_seed,
            )

            cuda_ms = float("nan")
            cuda_std = float("nan")
            speedup = float("nan")
            cuda_rank = 16
            if int(case.lora_rank) != 16 and not bool(args.compare_rank_mismatch):
                # CUDA cannot run rank != 16; still report Triton-only timing.
                pass
            else:
                cuda_ms, cuda_std = _time_backend(
                    mod,
                    "cuda",
                    case,
                    inputs,
                    cuda_cfg,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeat=args.repeat,
                    device=args.device,
                    seed=case_seed,
                )
                speedup = float(triton_ms) / float(cuda_ms) if cuda_ms > 0 else float("inf")
                if math.isfinite(speedup):
                    speedups.append(speedup)
                if speedup > 1.0:
                    cuda_wins += 1
                elif speedup < 1.0:
                    triton_wins += 1

            results.append(
                {
                    "case_id": case.id(),
                    "tag": case.tag,
                    "skipped": False,
                    "case": case.__dict__,
                    "estimated_bytes": int(est),
                    "grid_m": int(inputs["grid_m"]),
                    "grid_loras": int(inputs["grid_loras"]),
                    "triton": {"mean_ms": triton_ms, "std_ms": triton_std, "kernel_config": triton_cfg},
                    "cuda": {"mean_ms": cuda_ms, "std_ms": cuda_std, "kernel_config": cuda_cfg, "rank": int(cuda_rank)},
                    "triton_rank": int(case.lora_rank),
                    "cuda_rank": int(cuda_rank),
                    "rank_mismatch": bool(int(case.lora_rank) != int(cuda_rank)),
                    "speedup_triton_over_cuda": speedup,
                }
            )
        except Exception as e:
            results.append(
                {
                    "case_id": case.id(),
                    "tag": case.tag,
                    "skipped": False,
                    "error": str(e),
                    "case": case.__dict__,
                }
            )
        finally:
            _clear_triton_ptr_caches(mod)

    geomean_sp = _geomean(speedups)
    median_sp = float(statistics.median(speedups)) if speedups else float("nan")
    p10_sp = float(statistics.quantiles(speedups, n=10)[0]) if len(speedups) >= 10 else float("nan")
    p90_sp = float(statistics.quantiles(speedups, n=10)[-1]) if len(speedups) >= 10 else float("nan")

    summary = {
        "module": str(module_path),
        "device": str(args.device),
        "device_name": device_name,
        "timing": {"warmup": int(args.warmup), "iters": int(args.iters), "repeat": int(args.repeat)},
        "num_cases_total": len(cases),
        "num_cases_ran": sum(1 for r in results if not r.get("skipped") and "error" not in r),
        "num_cases_skipped": sum(1 for r in results if r.get("skipped")),
        "num_cases_error": sum(1 for r in results if "error" in r),
        "cuda_wins": int(cuda_wins),
        "triton_wins": int(triton_wins),
        "geomean_speedup_triton_over_cuda": geomean_sp,
        "median_speedup_triton_over_cuda": median_sp,
        "p10_speedup_triton_over_cuda": p10_sp,
        "p90_speedup_triton_over_cuda": p90_sp,
        "speedups": speedups,
    }

    print("\nSummary:")
    print(f"  Ran: {summary['num_cases_ran']} | Skipped: {summary['num_cases_skipped']} | Error: {summary['num_cases_error']}")
    print(f"  CUDA wins: {cuda_wins} | Triton wins: {triton_wins}")
    print(f"  Geomean speedup (Triton / CUDA): {geomean_sp:.4f}x")
    print(f"  Median speedup  (Triton / CUDA): {median_sp:.4f}x")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # CSV for easy plotting.
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "tag",
                "num_tokens",
                "hidden_size",
                "num_loras",
                "num_active_loras",
                "num_slices",
                "triton_rank",
                "cuda_rank",
                "grid_m",
                "grid_loras",
                "triton_ms",
                "cuda_ms",
                "speedup_triton_over_cuda",
            ]
        )
        for r in results:
            if r.get("skipped") or "error" in r:
                continue
            c = r["case"]
            writer.writerow(
                [
                    r["case_id"],
                    r.get("tag", ""),
                    c["num_tokens"],
                    c["hidden_size"],
                    c["num_loras"],
                    c["num_active_loras"],
                    c["num_slices"],
                    r.get("triton_rank", ""),
                    r.get("cuda_rank", ""),
                    r.get("grid_m", ""),
                    r.get("grid_loras", ""),
                    r["triton"]["mean_ms"],
                    r["cuda"]["mean_ms"],
                    r["speedup_triton_over_cuda"],
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
