#!/usr/bin/env python3
"""
Spreadsheet-style table benchmark for LoRA kernels:

- Triton shrink kernel (separate): kernels/ironfist/lora_shrink_kernel_bdcc2e70.py
- Triton expand kernel (separate): kernels/ironfist/lora_expand_kernel_6fd91708.py
- Triton fused kernel:             kernels/ironfist/lora_fused_kernel_0318783b.py (backend=triton)
- CUDA fused kernel:               kernels/ironfist/lora_fused_kernel_0318783b.py (backend=cuda, rank=16 only)

The output format is designed to match the "M x num_slices" tables from the user's screenshot:
  columns: M, num_slices, Shrink, Expand, fused(Triton), fused(CUDA), Speedup(Triton), Speedup(CUDA)

Where:
  Speedup(Triton) = (Shrink + Expand) / fused(Triton)
  Speedup(CUDA)   = (Shrink + Expand) / fused(CUDA)

Notes:
  - CUDA fused supports rank=16 only; for other Triton ranks, CUDA columns are "n/a".
  - All timings are kernel-only measured with CUDA events.
  - For fair-ish comparisons, the same token->LoRA mapping is used across all paths per row.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load_module(path: Path):
    path = path.resolve()
    name = f"lora_table_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for: {path}")
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


def _clear_ptr_caches(mod: Any):
    for attr in ("_LORA_A_PTR_DICT", "_LORA_B_PTR_DICT"):
        if hasattr(mod, attr):
            try:
                getattr(mod, attr).clear()
            except Exception:
                pass


def _parse_int_list(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


@dataclass(frozen=True)
class RowCfg:
    m: int
    num_slices: int


@dataclass(frozen=True)
class TableCfg:
    k: int
    triton_rank: int
    max_loras: int
    active_loras: int


def _round_pow2_at_least_1(x: int) -> int:
    x = int(max(1, x))
    p = 1
    while p < x:
        p <<= 1
    # Prefer the nearest power-of-two not exceeding x when x is large.
    if p > 1 and p > x:
        p >>= 1
        if p < 1:
            p = 1
    return int(p)


def _shrink_kernel_config(k: int) -> dict[str, int]:
    # Avoid pathological defaults where SPLIT_K is huge for small K.
    block_m = 32
    block_n = 16
    block_k = 256
    split_k = max(1, (k + block_k - 1) // block_k)
    split_k = min(64, _round_pow2_at_least_1(split_k))
    return {
        "block_m": int(block_m),
        "block_n": int(block_n),
        "block_k": int(block_k),
        "split_k": int(split_k),
        "num_warps": 4,
        "num_stages": 2,
    }


def _expand_kernel_config(rank: int) -> dict[str, int]:
    # Expand K dimension == rank; keep it simple.
    block_m = 64
    block_n = 128
    block_k = 16 if rank <= 16 else 32
    return {
        "block_m": int(block_m),
        "block_n": int(block_n),
        "block_k": int(block_k),
        "num_warps": 4,
        "num_stages": 2,
    }


def _fused_kernel_config(num_tokens: int, hidden_size: int, num_active_loras: int, backend: str) -> dict[str, Any]:
    # Mirror the lightweight heuristic used in kernels/ironfist/lora_fused_kernel_0318783b.py:benchmark_kernel.
    block_m = 64
    block_n = 128
    block_k = 16
    num_warps = 4
    num_stages = 2

    if num_tokens <= 256 and block_m > 32:
        block_m = 32
    if hidden_size >= 2048 and block_k < 64:
        block_k = 64
        num_stages = max(num_stages, 4)
    if num_tokens <= 256 and hidden_size >= 2048 and num_warps < 8:
        num_warps = 8
    if num_tokens >= 8192 and hidden_size >= 8192 and num_active_loras >= 256 and num_warps < 8:
        num_warps = 8
        num_stages = max(num_stages, 5)

    cfg: dict[str, Any] = {
        "block_m": int(block_m),
        "block_n": int(block_n),
        "block_k": int(block_k),
        "num_warps": int(num_warps),
        "num_stages": int(num_stages),
    }

    if num_tokens >= 8192 and hidden_size >= 8192 and num_active_loras >= 256:
        if backend == "cuda":
            cfg["block_m"] = 16
            cfg["group_n"] = 24
            cfg["num_warps"] = 4
            cfg["split_kernels"] = False
        else:
            cfg["group_n"] = 32

    if backend == "cuda" and int(cfg.get("num_warps", 4)) not in (2, 4, 8):
        cfg["num_warps"] = 8 if int(cfg.get("num_warps", 4)) >= 8 else 4

    return cfg


def _build_mapping(
    *,
    num_tokens: int,
    num_loras: int,
    num_active_loras: int,
    device: str,
    seed: int,
    allow_no_lora: bool,
) -> dict[str, Any]:
    import torch

    _set_all_seeds(seed)
    dev = torch.device(device)
    if allow_no_lora:
        token_lora_mapping = torch.randint(-1, num_active_loras, (num_tokens,), device=dev)
    else:
        token_lora_mapping = torch.randint(0, num_active_loras, (num_tokens,), device=dev)

    _, token_indices_sorted = torch.sort(token_lora_mapping, stable=True)
    unique_lora_ids, counts = torch.unique(token_lora_mapping, sorted=True, return_counts=True)

    max_loras = int(num_loras)
    padded_num_tokens_per_lora = torch.zeros(max_loras + 1, dtype=torch.int32, device=dev)
    actual_num_loras = min(int(counts.numel()), max_loras + 1)
    if actual_num_loras > 0:
        padded_num_tokens_per_lora[:actual_num_loras] = counts[:actual_num_loras].to(torch.int32)

    padded_lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32, device=dev)
    if actual_num_loras > 0:
        padded_lora_ids[:actual_num_loras] = unique_lora_ids[:actual_num_loras].to(torch.int32)

    lora_token_start_loc = torch.zeros(max_loras + 2, dtype=torch.int32, device=dev)
    if actual_num_loras > 0:
        lora_token_start_loc[1 : actual_num_loras + 1] = torch.cumsum(
            padded_num_tokens_per_lora[:actual_num_loras], dim=0
        )

    grid_loras = int(actual_num_loras)
    grid_m = 0
    if actual_num_loras > 0:
        has_no_lora = int(unique_lora_ids[0].item()) == -1
        counts_slice = (
            padded_num_tokens_per_lora[1:actual_num_loras] if has_no_lora else padded_num_tokens_per_lora[:actual_num_loras]
        )
        if counts_slice.numel() > 0:
            grid_m = int(counts_slice.max().item())

    # Slice arrays to reduce overlaunch; keep them consistent for all kernels.
    num_tokens_per_lora = padded_num_tokens_per_lora[:grid_loras].contiguous()
    lora_ids = padded_lora_ids[:grid_loras].contiguous()
    lora_token_start_loc_s = lora_token_start_loc[: grid_loras + 1].contiguous()

    return {
        "token_lora_mapping": token_lora_mapping,
        "token_indices_sorted": token_indices_sorted.contiguous(),
        "num_tokens_per_lora": num_tokens_per_lora,
        "lora_ids": lora_ids,
        "lora_token_start_loc": lora_token_start_loc_s,
        "grid_m": int(max(grid_m, 0)),
        "grid_loras": int(max(grid_loras, 0)),
    }


def _time_call(fn, warmup: int, iters: int, repeat: int) -> tuple[float, float]:
    import torch

    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        for _ in range(iters):
            start_evt.record()
            fn()
            end_evt.record()
            end_evt.synchronize()
            times_ms.append(float(start_evt.elapsed_time(end_evt)))
    mean_ms = float(statistics.mean(times_ms)) if times_ms else float("inf")
    std_ms = float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0
    return mean_ms, std_ms


def _launch_shrink(
    shrink_mod: Any,
    *,
    x: Any,
    lora_a_weights: list[Any],
    u_out: Any,
    mapping: dict[str, Any],
    scaling: float,
    kernel_config: dict[str, int],
    m_cap: int,
) -> None:
    import triton

    (lora_ptr_tensor, lora_strides_d0, lora_strides_d1, lora_strides_d2) = shrink_mod._get_lora_a_ptr(
        lora_a_weights, x.device
    )
    n = int(lora_a_weights[0].shape[1])  # rank
    k = int(lora_a_weights[0].shape[2])  # hidden_size
    num_slices = int(len(lora_a_weights))
    max_loras = int(mapping["lora_ids"].numel())

    block_m = int(kernel_config["block_m"])
    block_n = int(kernel_config["block_n"])
    block_k = int(kernel_config["block_k"])
    split_k = int(kernel_config["split_k"])
    num_warps = int(kernel_config["num_warps"])
    num_stages = int(kernel_config["num_stages"])

    even_k = (k % (block_k * split_k)) == 0
    grid = (
        split_k * triton.cdiv(int(m_cap), block_m) * triton.cdiv(int(n), block_n),
        num_slices,
        max_loras,
    )

    shrink_mod._lora_shrink_kernel[grid](
        x,
        lora_ptr_tensor,
        u_out,
        int(m_cap),
        int(n),
        int(k),
        mapping["token_indices_sorted"],
        mapping["num_tokens_per_lora"],
        mapping["lora_token_start_loc"],
        mapping["lora_ids"],
        float(scaling),
        x.stride(0),
        x.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        u_out.stride(0),
        u_out.stride(1),
        u_out.stride(2),
        block_m,
        block_n,
        block_k,
        even_k,
        split_k,
        num_slices,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_expand(
    expand_mod: Any,
    *,
    u_in: Any,
    lora_b_weights: list[Any],
    y_out: Any,
    mapping: dict[str, Any],
    offset_start: int,
    add_inputs: bool,
    kernel_config: dict[str, int],
    m_cap: int,
) -> None:
    import triton

    (
        slice_start_tensor,
        lora_ptr_tensor,
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        hidden_sizes_tensor,
        same_stride,
        max_n,
    ) = expand_mod._get_lora_b_ptr(lora_b_weights, int(offset_start), u_in.device)

    # Expand K dimension == rank.
    k = int(lora_b_weights[0].shape[-1])
    num_slices = int(len(lora_b_weights))
    max_loras = int(mapping["lora_ids"].numel())

    block_m = int(kernel_config["block_m"])
    block_n = int(kernel_config["block_n"])
    block_k = int(kernel_config["block_k"])
    num_warps = int(kernel_config["num_warps"])
    num_stages = int(kernel_config["num_stages"])
    even_k = (k % block_k) == 0

    grid = (
        triton.cdiv(int(m_cap), block_m) * triton.cdiv(int(max_n), block_n),
        num_slices,
        max_loras,
    )

    expand_mod._lora_expand_kernel[grid](
        u_in,
        lora_ptr_tensor,
        y_out,
        int(m_cap),
        int(max_n),
        int(k),
        mapping["token_indices_sorted"],
        mapping["num_tokens_per_lora"],
        mapping["lora_token_start_loc"],
        mapping["lora_ids"],
        slice_start_tensor,
        u_in.stride(0),
        u_in.stride(1),
        u_in.stride(2),
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        y_out.stride(0),
        y_out.stride(1),
        hidden_sizes_tensor,
        block_m,
        block_n,
        block_k,
        even_k,
        bool(add_inputs),
        False,  # CAST_TYPE
        num_slices,
        bool(same_stride),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _fmt(v: float | None, unit: str) -> str:
    if v is None or (isinstance(v, float) and (not math.isfinite(v))):
        return "n/a"
    if unit == "us":
        return f"{v * 1000.0:7.1f}"
    return f"{v:7.4f}"


def _fmt_sp(v: float | None) -> str:
    if v is None or (isinstance(v, float) and (not math.isfinite(v))):
        return "n/a"
    return f"{v:8.5f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--unit", type=str, default="us", choices=("us", "ms"))
    parser.add_argument("--allow-no-lora", action="store_true", help="Allow -1 (no-LoRA) bucket in mapping.")
    parser.add_argument(
        "--m",
        type=str,
        required=True,
        help="Comma-separated M (num_tokens) values (rows).",
    )
    parser.add_argument(
        "--num-slices",
        type=str,
        required=True,
        help="Comma-separated num_slices values (rows).",
    )
    parser.add_argument(
        "--k",
        type=str,
        required=True,
        help="Comma-separated K (hidden_size) values (tables).",
    )
    parser.add_argument(
        "--max-loras",
        type=str,
        required=True,
        help="Comma-separated max_loras (num_loras) values (tables).",
    )
    parser.add_argument(
        "--active-loras",
        type=str,
        default="",
        help="Comma-separated active_loras values (tables). Default: == max_loras.",
    )
    parser.add_argument(
        "--triton-ranks",
        type=str,
        default="16",
        help="Comma-separated Triton ranks (tables). CUDA fused is always rank=16.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="runs/lora_table",
        help="Output directory (writes results.csv + results.json).",
    )
    parser.add_argument(
        "--shrink-module",
        type=str,
        default="kernels/ironfist/lora_shrink_kernel_bdcc2e70.py",
    )
    parser.add_argument(
        "--expand-module",
        type=str,
        default="kernels/ironfist/lora_expand_kernel_6fd91708.py",
    )
    parser.add_argument(
        "--fused-module",
        type=str,
        default="kernels/ironfist/lora_fused_kernel_0318783b.py",
    )
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = str(args.device)
    torch.cuda.set_device(torch.device(device))
    device_name = torch.cuda.get_device_name(0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shrink_mod = _load_module(Path(args.shrink_module))
    expand_mod = _load_module(Path(args.expand_module))
    fused_mod = _load_module(Path(args.fused_module))

    m_vals = _parse_int_list(args.m)
    s_vals = _parse_int_list(args.num_slices)
    k_vals = _parse_int_list(args.k)
    l_vals = _parse_int_list(args.max_loras)
    a_vals = _parse_int_list(args.active_loras) if args.active_loras else []
    r_vals = _parse_int_list(args.triton_ranks)

    tables: list[TableCfg] = []
    for k in k_vals:
        if a_vals and len(a_vals) == len(l_vals):
            pairs = list(zip(l_vals, a_vals))
        else:
            pairs = [(l, a) for l in l_vals for a in (a_vals if a_vals else [l])]
        for l, a in pairs:
            if int(a) > int(l):
                # Invalid (would index weights out-of-bounds).
                continue
            for r in r_vals:
                tables.append(TableCfg(k=int(k), triton_rank=int(r), max_loras=int(l), active_loras=int(a)))

    rows: list[RowCfg] = []
    for m in m_vals:
        for s in s_vals:
            rows.append(RowCfg(m=int(m), num_slices=int(s)))

    results: list[dict[str, Any]] = []

    print(f"Device: {device} ({device_name})")
    print(f"Timing: warmup={args.warmup} iters={args.iters} repeat={args.repeat} unit={args.unit}")
    print("")

    for t_idx, tcfg in enumerate(tables):
        print(f"K={tcfg.k}  R(triton)={tcfg.triton_rank}  max_loras={tcfg.max_loras}  active_loras={tcfg.active_loras}")
        print(
            "  "
            + "M".rjust(6)
            + "  "
            + "S".rjust(3)
            + "  "
            + "Shrink".rjust(9)
            + "  "
            + "Expand".rjust(9)
            + "  "
            + "Fused(T)".rjust(9)
            + "  "
            + "Fused(C)".rjust(9)
            + "  "
            + "T/C".rjust(8)
            + "  "
            + "Sp(T)".rjust(8)
            + "  "
            + "Sp(C)".rjust(8)
        )

        for r_idx, rcfg in enumerate(rows):
            # Per-row seed so table is deterministic.
            seed = int(args.seed) + (t_idx * 10000) + (r_idx * 97)
            _set_all_seeds(seed)
            _clear_ptr_caches(shrink_mod)
            _clear_ptr_caches(expand_mod)
            _clear_ptr_caches(fused_mod)

            m = int(rcfg.m)
            k = int(tcfg.k)
            r = int(tcfg.triton_rank)
            l = int(tcfg.max_loras)
            a = int(tcfg.active_loras)
            s = int(rcfg.num_slices)

            # Allocate shared inputs.
            dev = torch.device(device)
            x = torch.randn(m, k, dtype=torch.float16, device=dev)

            lora_a = [torch.randn(l, r, k, dtype=torch.float16, device=dev) for _ in range(s)]
            lora_b = [torch.randn(l, k, r, dtype=torch.float16, device=dev) for _ in range(s)]

            u = torch.zeros(s, m, r, dtype=torch.float16, device=dev)
            y_sep = torch.zeros(m, k * s, dtype=torch.float16, device=dev)
            y_fused_t = torch.zeros(m, k * s, dtype=torch.float16, device=dev)
            y_fused_c = torch.zeros(m, k * s, dtype=torch.float16, device=dev)

            mapping = _build_mapping(
                num_tokens=m,
                num_loras=l,
                num_active_loras=a,
                device=device,
                seed=seed + 1,
                allow_no_lora=bool(args.allow_no_lora),
            )
            token_indices_sorted_i32 = mapping["token_indices_sorted"].to(torch.int32).contiguous()
            if int(mapping["grid_m"]) <= 0 or int(mapping["grid_loras"]) <= 0:
                # No active LoRAs; treat as n/a.
                shrink_ms = float("nan")
                expand_ms = float("nan")
                fused_t_ms = float("nan")
                fused_c_ms = float("nan")
            else:
                m_cap = int(mapping["grid_m"])

                shrink_cfg = _shrink_kernel_config(k)
                expand_cfg = _expand_kernel_config(r)
                fused_t_cfg = _fused_kernel_config(m, k, a, backend="triton")
                fused_c_cfg = _fused_kernel_config(m, k, a, backend="cuda")
                fused_t_cfg["grid_m"] = int(mapping["grid_m"])
                fused_t_cfg["grid_loras"] = int(mapping["grid_loras"])
                fused_c_cfg["grid_m"] = int(mapping["grid_m"])
                fused_c_cfg["grid_loras"] = int(mapping["grid_loras"])

                # Shrink.
                def do_shrink():
                    u.zero_()
                    _launch_shrink(
                        shrink_mod,
                        x=x,
                        lora_a_weights=lora_a,
                        u_out=u,
                        mapping=mapping,
                        scaling=1.0,
                        kernel_config=shrink_cfg,
                        m_cap=m_cap,
                    )

                shrink_ms, _ = _time_call(do_shrink, args.warmup, args.iters, args.repeat)

                # Expand.
                def do_expand():
                    y_sep.zero_()
                    _launch_expand(
                        expand_mod,
                        u_in=u,
                        lora_b_weights=lora_b,
                        y_out=y_sep,
                        mapping=mapping,
                        offset_start=0,
                        add_inputs=False,
                        kernel_config=expand_cfg,
                        m_cap=m_cap,
                    )

                expand_ms, _ = _time_call(do_expand, args.warmup, args.iters, args.repeat)

                # Fused Triton.
                def do_fused_triton():
                    y_fused_t.zero_()
                    fused_mod._fused_shrink_expand(
                        inputs=x,
                        lora_a_weights=lora_a,
                        lora_b_weights=lora_b,
                        output_tensor=y_fused_t,
                        token_lora_mapping=mapping["token_lora_mapping"],
                        token_indices_sorted_by_lora_ids=mapping["token_indices_sorted"],
                        num_tokens_per_lora=mapping["num_tokens_per_lora"],
                        lora_token_start_loc=mapping["lora_token_start_loc"],
                        lora_ids=mapping["lora_ids"],
                        scaling=1.0,
                        offset_start=0,
                        add_inputs=False,
                        kernel_config=fused_t_cfg,
                    )

                fused_t_ms, _ = _time_call(do_fused_triton, args.warmup, args.iters, args.repeat)

                fused_c_ms = float("nan")
                if r == 16 and (k % 64) == 0:
                    def do_fused_cuda():
                        y_fused_c.zero_()
                        fused_mod._fused_shrink_expand_cuda(
                            inputs=x,
                            lora_a_weights=lora_a,
                            lora_b_weights=lora_b,
                            output_tensor=y_fused_c,
                            token_lora_mapping=mapping["token_lora_mapping"],
                            token_indices_sorted_by_lora_ids=token_indices_sorted_i32,
                            num_tokens_per_lora=mapping["num_tokens_per_lora"],
                            lora_token_start_loc=mapping["lora_token_start_loc"],
                            lora_ids=mapping["lora_ids"],
                            scaling=1.0,
                            offset_start=0,
                            add_inputs=False,
                            kernel_config=fused_c_cfg,
                        )

                    fused_c_ms, _ = _time_call(do_fused_cuda, args.warmup, args.iters, args.repeat)

            shrink = float(shrink_ms)
            expand = float(expand_ms)
            fused_t = float(fused_t_ms)
            fused_c = float(fused_c_ms)

            speedup_t = (shrink + expand) / fused_t if math.isfinite(fused_t) and fused_t > 0 else float("nan")
            speedup_c = (shrink + expand) / fused_c if math.isfinite(fused_c) and fused_c > 0 else float("nan")
            triton_over_cuda = fused_t / fused_c if math.isfinite(fused_t) and math.isfinite(fused_c) and fused_c > 0 else float("nan")

            print(
                "  "
                + str(m).rjust(6)
                + "  "
                + str(s).rjust(3)
                + "  "
                + _fmt(shrink, args.unit)
                + "  "
                + _fmt(expand, args.unit)
                + "  "
                + _fmt(fused_t, args.unit)
                + "  "
                + _fmt(fused_c, args.unit)
                + "  "
                + _fmt_sp(triton_over_cuda)
                + "  "
                + _fmt_sp(speedup_t)
                + "  "
                + _fmt_sp(speedup_c)
            )

            results.append(
                {
                    "k": k,
                    "triton_rank": r,
                    "max_loras": l,
                    "active_loras": a,
                    "m": m,
                    "num_slices": s,
                    "unit": args.unit,
                    "shrink_ms": shrink,
                    "expand_ms": expand,
                    "fused_triton_ms": fused_t,
                    "fused_cuda_ms": fused_c,
                    "speedup_triton_over_cuda": triton_over_cuda,
                    "speedup_sep_over_triton": speedup_t,
                    "speedup_sep_over_cuda": speedup_c,
                }
            )
        print("")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "k",
                "triton_rank",
                "max_loras",
                "active_loras",
                "m",
                "num_slices",
                "shrink_ms",
                "expand_ms",
                "fused_triton_ms",
                "fused_cuda_ms",
                "speedup_triton_over_cuda",
                "speedup_sep_over_triton",
                "speedup_sep_over_cuda",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r["k"],
                    r["triton_rank"],
                    r["max_loras"],
                    r["active_loras"],
                    r["m"],
                    r["num_slices"],
                    r["shrink_ms"],
                    r["expand_ms"],
                    r["fused_triton_ms"],
                    r["fused_cuda_ms"],
                    r.get("speedup_triton_over_cuda", float("nan")),
                    r["speedup_sep_over_triton"],
                    r["speedup_sep_over_cuda"],
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
