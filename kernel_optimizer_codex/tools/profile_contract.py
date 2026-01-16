#!/usr/bin/env python3
"""
profile_contract.py

Profile a single IronFist-style contract against a given kernel module & entry point.

It:
  - Loads the contract JSON.
  - Imports the kernel module (from kernels/<kernel_name>.py or custom path).
  - Builds inputs per contract.io.args, restricted to the function's signature.
  - If the module exposes benchmark_kernel(...), delegates timing to it for kernel-only
    measurement; otherwise times the contract entrypoint.
  - Runs warmup + timed loops using torch.cuda Events when on GPU.
  - Optionally invokes Nsight Compute (ncu) once via run_contract_once.py
    to produce an .ncu-rep file.

Outputs:
  out_dir/result.json with fields:
    {
      "ok": true,
      "mean_ms": ...,
      "std_ms": ...,
      "contract_path": ...,
      "module_path": ...,
      "entry_point": "run",
      "kernel_name": "...",
      "device": "cuda:0",
      "timing": {...},
      "output_summary": {...},
      "ncu_report": "path/to/ncu_report.ncu-rep" | null,
      ...
    }

Usage (from repo root):

  python tools/profile_contract.py \
      --contract contracts/vec_matmul_kernel_small_contract.json \
      --module kernels/vec_matmul_kernel.py \
      --entry-point run \
      --device cuda:0 \
      --warmup 10 --iters 100 --repeat 5 \
      --out runs/vec_matmul_kernel/baseline/small \
      [--with-ncu] [--ncu-bin ncu]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import shutil

import numpy as np
import torch

# Local imports
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from contracts_common import build_entrypoint_and_kwargs, _parse_scalar_value  # type: ignore
from contracts_common import _make_torch_tensor  # type: ignore


def _clone_output(obj: Any) -> Any:
    """Clone output structure for stable correctness comparison."""
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_output(o) for o in obj)
    return obj


def summarize_output(obj: Any) -> Dict[str, Any]:
    """
    Produce a small, cheap summary of the output for correctness sanity checks.
    This is not a full correctness oracle, just a quick fingerprint.
    """
    summary: Dict[str, Any] = {"kind": None}

    if obj is None:
        summary["kind"] = "none"
        return summary

    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        summary.update(
            {
                "kind": "tensor",
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "mean": float(t.float().mean().item()),
                "std": float(t.float().std().item()),
                "norm": float(t.float().norm().item()),
            }
        )
        return summary

    if isinstance(obj, (list, tuple)):
        summary["kind"] = "sequence"
        summary["length"] = len(obj)
        # if it's a sequence of tensors, summarize first one
        if len(obj) > 0 and isinstance(obj[0], torch.Tensor):
            t = obj[0].detach()
            summary["elem0_shape"] = list(t.shape)
            summary["elem0_dtype"] = str(t.dtype)
            summary["elem0_norm"] = float(t.float().norm().item())
        return summary

    try:
        summary["kind"] = type(obj).__name__
        # best-effort scalar conversion
        if isinstance(obj, (int, float)):
            summary["value"] = float(obj)
    except Exception:
        pass
    return summary


def build_benchmark_args(contract: Dict[str, Any], device: str) -> Dict[str, Any]:
    """
    Convert contract.io.args into a light dict for benchmark_kernel fast-paths.
    Tensor args are materialized as torch tensors on the requested device; scalar
    args are parsed to Python scalars. Device is included for convenience.
    """
    kernel = contract.get("kernel", {}) or {}
    io = kernel.get("io", {}) or {}
    arg_specs = io.get("args", []) or []

    bench_args: Dict[str, Any] = {}
    for arg in arg_specs:
        name = arg.get("name")
        if not name:
            continue
        atype = arg.get("type", "int")
        if atype == "tensor":
            bench_args[name] = _make_torch_tensor(arg, device=device)
        else:
            bench_args[name] = _parse_scalar_value(arg)
    bench_args["_device"] = device
    return bench_args


def _set_torch_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def run_ncu_profile(
    contract_path: Path,
    module_path: Path,
    entry_point: str,
    device: str,
    out_dir: Path,
    ncu_bin: str,
    ncu_args: Optional[str] = None,
    backend: Optional[str] = None,
    ncu_call: str = "entrypoint",
) -> Optional[Path]:
    """
    Launch Nsight Compute (ncu) once on the contract entrypoint, writing a .ncu-rep.
    We run:

      ncu -f --set full --target-processes all --export <out_base> \
          python tools/run_contract_once.py --contract ... --module ... --entry-point ... --device ...

    This follows typical Nsight Compute CLI usage for batch profiling. 
    """
    # Resolve paths so NCU can reliably emit reports and find cached Triton artifacts.
    contract_path = contract_path.resolve()
    module_path = module_path.resolve()
    out_dir = out_dir.resolve()
    out_base = out_dir / "ncu_report"
    rep_path = out_base.with_suffix(".ncu-rep")
    script_path = THIS_DIR / "run_contract_once.py"

    cmd = [
        ncu_bin,
        "-f",
        "--set",
        "full",
        "--target-processes",
        "all",
        "--import-source",
        "yes",
        "--export",
        str(out_base),
    ]
    if ncu_args:
        cmd.extend(shlex.split(ncu_args))
    cmd.extend([sys.executable, str(script_path), "--contract", str(contract_path), "--module", str(module_path)])
    cmd.extend(["--device", device])
    if ncu_call == "benchmark_kernel":
        cmd.extend(["--call", "benchmark_kernel", "--warmup", "0", "--iters", "1", "--repeat", "1"])
    else:
        cmd.extend(["--call", "entrypoint", "--entry-point", entry_point])
    if backend:
        cmd.extend(["--backend", backend])

    env = os.environ.copy()
    if backend:
        env["KO_BACKEND"] = backend
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    (out_dir / "ncu_cmd.txt").write_text(cmd_str + "\n", encoding="utf-8")
    print(f"[INFO] Running Nsight Compute:\n  {cmd_str}")
    try:
        proc = subprocess.run(cmd, env=env, cwd=str(out_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print(f"[WARN] ncu binary '{ncu_bin}' not found on PATH; skipping NCU profile")
        return None

    (out_dir / "ncu_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "ncu_stderr.log").write_text(proc.stderr, encoding="utf-8")

    if proc.returncode != 0:
        print(f"[WARN] ncu exited with code {proc.returncode}; see ncu_stderr.log")
        return None

    if not rep_path.exists():
        print(f"[WARN] Expected NCU report {rep_path} not found; check ncu logs")
        return None

    print(f"[INFO] NCU report written to {rep_path}")
    return rep_path


def _compare_outputs(ref: Any, cand: Any, rtol: float, atol: float) -> tuple[bool, Optional[str]]:
    """Lightweight structural + numeric comparison."""
    if ref is None and cand is None:
        return True, None
    if isinstance(ref, torch.Tensor) and isinstance(cand, torch.Tensor):
        ok = torch.allclose(ref, cand, rtol=rtol, atol=atol)
        return ok, None if ok else "tensor mismatch"
    if isinstance(ref, (list, tuple)) and isinstance(cand, (list, tuple)):
        if len(ref) != len(cand):
            return False, "sequence length differs"
        for i, (r, c) in enumerate(zip(ref, cand)):
            ok, err = _compare_outputs(r, c, rtol, atol)
            if not ok:
                return False, f"element {i}: {err}"
        return True, None
    if isinstance(ref, (int, float)) and isinstance(cand, (int, float)):
        ok = math.isclose(ref, cand, rel_tol=rtol, abs_tol=atol)
        return ok, None if ok else "scalar mismatch"
    return False, f"type mismatch ({type(ref)} vs {type(cand)})"


def profile_one_contract(
    contract_path: Path,
    module_path: Path,
    baseline_module_path: Optional[Path],
    entry_point: Optional[str],
    device: str,
    warmup: int,
    iters: int,
    repeat: int,
    out_dir: Path,
    backend: Optional[str] = None,
    timing_target: str = "auto",
    correctness_mode: str = "auto",
    ncu_call: str = "auto",
    with_ncu: bool = False,
    ncu_bin: str = "ncu",
    ncu_args: Optional[str] = None,
    rtol: float = 1e-4,
    atol: float = 1e-3,
    timing_mode: str = "events",
) -> Dict[str, Any]:
    """
    Programmatic entrypoint used by profile_suite.py and the CLI.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if baseline_module_path is None:
        baseline_module_path = module_path
    baseline_module_path = baseline_module_path.resolve()
    module_path = module_path.resolve()

    # Persist copies of candidate and baseline modules inside this run dir for reproducibility.
    cand_copy = out_dir / "candidate_module.py"
    base_copy = out_dir / "baseline_module.py"
    shutil.copy2(module_path, cand_copy)
    shutil.copy2(baseline_module_path, base_copy)

    dev = torch.device(device)
    prev_backend_env = os.environ.get("KO_BACKEND")
    if backend is not None:
        os.environ["KO_BACKEND"] = backend

    try:
        # Seed before building any inputs so baseline/candidate comparisons are stable.
        _set_torch_seed(0, dev)
        contract, module, func, ep_name, kwargs = build_entrypoint_and_kwargs(
            contract_path=contract_path,
            module_path=cand_copy,
            entry_point=entry_point,
            device=device,
        )
        _set_torch_seed(0, dev)
        _, baseline_module, baseline_func, _, _ = build_entrypoint_and_kwargs(
            contract_path=contract_path,
            module_path=base_copy,
            entry_point=entry_point,
            device=device,
        )

        kernel = contract.get("kernel", {}) or {}
        meta = kernel.get("metadata", {}) or {}
        kernel_name = meta.get("kernel_name") or module_path.stem
        kernel_type = str(kernel.get("kernel_type", "multi_kernel"))

        benchmark_fn = getattr(module, "benchmark_kernel", None)
        baseline_benchmark_fn = getattr(baseline_module, "benchmark_kernel", None)
        benchmark_available = callable(benchmark_fn)
        baseline_benchmark_available = callable(baseline_benchmark_fn)

        if timing_target not in ("auto", "benchmark_kernel", "entrypoint"):
            raise ValueError(f"Unknown timing_target: {timing_target!r}")
        if correctness_mode not in ("auto", "entrypoint", "seeded_entrypoint", "benchmark_kernel", "skip"):
            raise ValueError(f"Unknown correctness_mode: {correctness_mode!r}")
        if ncu_call not in ("auto", "entrypoint", "benchmark_kernel"):
            raise ValueError(f"Unknown ncu_call: {ncu_call!r}")

        # Decide correctness mode.
        effective_correctness_mode = correctness_mode
        if correctness_mode == "auto":
            if kernel_type != "triton" and benchmark_available and baseline_benchmark_available:
                effective_correctness_mode = "benchmark_kernel"
            elif kernel_type != "triton":
                effective_correctness_mode = "seeded_entrypoint"
            else:
                effective_correctness_mode = "entrypoint"

        # Decide whether to time benchmark_kernel or the entrypoint.
        if timing_target == "benchmark_kernel":
            if not benchmark_available:
                raise RuntimeError("timing_target=benchmark_kernel requested but module has no benchmark_kernel()")
            benchmark_used = True
        elif timing_target == "entrypoint":
            benchmark_used = False
        else:  # auto
            benchmark_used = benchmark_available

        outputs: Any = None
        output_summary: Dict[str, Any] = {}

        # Correctness check (always before timing).
        correct = True
        correctness_error: Optional[str] = None

        if effective_correctness_mode == "skip":
            correct, correctness_error = True, None
        elif effective_correctness_mode == "benchmark_kernel":
            if not (benchmark_available and baseline_benchmark_available):
                raise RuntimeError(
                    "correctness_mode=benchmark_kernel requested but benchmark_kernel missing on baseline or candidate module"
                )
            bench_args = build_benchmark_args(contract, device=device)
            if backend is not None:
                bench_args["_backend"] = backend

            _set_torch_seed(0, dev)
            try:
                base_res = baseline_benchmark_fn(contract_args=bench_args, warmup=0, iters=1, repeat=1)  # type: ignore[misc]
            except TypeError:
                base_res = baseline_benchmark_fn(bench_args, 0, 1, 1)  # type: ignore[misc]
            _set_torch_seed(0, dev)
            try:
                cand_res = benchmark_fn(contract_args=bench_args, warmup=0, iters=1, repeat=1)  # type: ignore[misc]
            except TypeError:
                cand_res = benchmark_fn(bench_args, 0, 1, 1)  # type: ignore[misc]

            base_out = base_res.get("output") if isinstance(base_res, dict) else None
            cand_out = cand_res.get("output") if isinstance(cand_res, dict) else None
            if base_out is None or cand_out is None:
                correct, correctness_error = False, "benchmark_kernel did not return an 'output' for correctness"
            else:
                correct, correctness_error = _compare_outputs(_clone_output(base_out), cand_out, rtol=rtol, atol=atol)
        else:
            # entrypoint correctness; optionally reseed before each call for shape-only multi_kernel contracts.
            arg_specs = (contract.get("kernel", {}) or {}).get("io", {}).get("args", []) or []
            output_names = {a.get("name") for a in arg_specs if a.get("role") in ("output", "inout")}
            with torch.no_grad():
                if effective_correctness_mode == "seeded_entrypoint":
                    _set_torch_seed(0, dev)
                baseline_out = baseline_func(**kwargs)
                baseline_out_frozen = _clone_output(baseline_out)
                for name in output_names:
                    if name and name in kwargs and torch.is_tensor(kwargs[name]):
                        kwargs[name].zero_()
                if effective_correctness_mode == "seeded_entrypoint":
                    _set_torch_seed(0, dev)
                candidate_out = func(**kwargs)
            correct, correctness_error = _compare_outputs(baseline_out_frozen, candidate_out, rtol=rtol, atol=atol)

        if not correct:
            print(f"[WARN] Correctness failed for contract {contract_path.name}: {correctness_error}")

        # Reuse the same kwargs for timing; zero outputs to start clean.
        arg_specs = (contract.get("kernel", {}) or {}).get("io", {}).get("args", []) or []
        output_names = {a.get("name") for a in arg_specs if a.get("role") in ("output", "inout")}
        for name in output_names:
            if name and name in kwargs and torch.is_tensor(kwargs[name]):
                kwargs[name].zero_()
        timing_kwargs = kwargs
        benchmark_result: Optional[Dict[str, Any]] = None
        timing_path = "events"

        # Timing
        if benchmark_used:
            bench_args = build_benchmark_args(contract, device=device)
            if backend is not None:
                bench_args["_backend"] = backend
            try:
                benchmark_result = benchmark_fn(  # type: ignore[misc]
                    contract_args=bench_args,
                    warmup=warmup,
                    iters=iters,
                    repeat=repeat,
                )
            except TypeError:
                # Allow positional signature benchmark_kernel(contract_args, warmup, iters, repeat)
                benchmark_result = benchmark_fn(bench_args, warmup, iters, repeat)  # type: ignore[misc]
            except Exception as e:
                if timing_target == "benchmark_kernel":
                    raise
                print(f"[WARN] benchmark_kernel raised {e}; falling back to entry point timing")
                benchmark_used = False

            if benchmark_used and not isinstance(benchmark_result, dict):
                if timing_target == "benchmark_kernel":
                    raise RuntimeError("benchmark_kernel returned non-dict result")
                print("[WARN] benchmark_kernel returned non-dict result; falling back to entry point timing")
                benchmark_used = False
                benchmark_result = None

        if benchmark_used and benchmark_result is not None:
            mean_ms = float(benchmark_result.get("mean_ms", math.inf))
            std_ms = float(benchmark_result.get("std_ms", 0.0))
            outputs = benchmark_result.get("output")
            output_summary = benchmark_result.get("output_summary") or summarize_output(outputs)
            ok_flag = bool(benchmark_result.get("ok", math.isfinite(mean_ms))) and correct
            timing_path = "benchmark_kernel"
        else:
            times: list[float] = []

            # Warmup (outside graphs/events timing)
            for _ in range(max(0, warmup)):
                outputs = func(**timing_kwargs)
            if dev.type == "cuda":
                torch.cuda.synchronize()

            def _time_with_events() -> Tuple[list[float], str]:
                evt_times: list[float] = []
                for _ in range(repeat):
                    if dev.type == "cuda" and torch.cuda.is_available():
                        start_evt = torch.cuda.Event(enable_timing=True)
                        end_evt = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        start_evt.record()
                        for _ in range(iters):
                            outputs = func(**timing_kwargs)
                        end_evt.record()
                        end_evt.synchronize()
                        elapsed_ms = float(start_evt.elapsed_time(end_evt)) / max(1, iters)
                    else:
                        t0 = time.perf_counter()
                        for _ in range(iters):
                            outputs = func(**timing_kwargs)
                        t1 = time.perf_counter()
                        elapsed_ms = (t1 - t0) * 1000.0 / max(1, iters)
                    evt_times.append(elapsed_ms)
                return evt_times, "events"

            def _time_with_cuda_graph() -> Tuple[list[float], str]:
                if dev.type != "cuda":
                    raise RuntimeError("CUDA graphs requested but device is not CUDA")

                # Build static args for capture (clone tensor inputs)
                static_args: Dict[str, Any] = {}
                orig_args: Dict[str, Any] = {}
                for k, v in timing_kwargs.items():
                    if torch.is_tensor(v):
                        buf = torch.empty_like(v, device=dev)
                        buf.copy_(v)
                        static_args[k] = buf
                        orig_args[k] = v
                    else:
                        static_args[k] = v
                        orig_args[k] = v

                capture_stream = torch.cuda.Stream(device=dev)
                with torch.cuda.stream(capture_stream), torch.no_grad():
                    for _ in range(3):
                        _ = func(**static_args)
                capture_stream.synchronize()
                with torch.cuda.stream(capture_stream), torch.no_grad():
                    _ = func(**static_args)
                capture_stream.synchronize()

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=capture_stream), torch.no_grad():
                    _ = func(**static_args)
                torch.cuda.synchronize()

                evt_times: list[float] = []
                for _ in range(repeat):
                    # refresh inputs
                    for k, buf in static_args.items():
                        if torch.is_tensor(buf) and torch.is_tensor(orig_args[k]):
                            buf.copy_(orig_args[k])
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_evt.record()
                    for _ in range(iters):
                        g.replay()
                    end_evt.record()
                    end_evt.synchronize()
                    evt_times.append(float(start_evt.elapsed_time(end_evt)) / max(1, iters))
                return evt_times, "cuda_graphs"

            if timing_mode == "cuda_graphs":
                try:
                    times, timing_path = _time_with_cuda_graph()
                except Exception as e:
                    print(f"[WARN] CUDA graphs timing failed ({e}); falling back to events")
                    times, timing_path = _time_with_events()
            else:
                times, timing_path = _time_with_events()

            times_arr = np.array(times, dtype=float)
            mean_ms = float(times_arr.mean()) if times_arr.size else math.inf
            std_ms = float(times_arr.std(ddof=0)) if times_arr.size else 0.0
            output_summary = summarize_output(outputs)
            ok_flag = math.isfinite(mean_ms) and correct

        effective_ncu_call = ncu_call
        if effective_ncu_call == "auto":
            effective_ncu_call = "benchmark_kernel" if benchmark_used else "entrypoint"

        ncu_report_path: Optional[Path] = None
        if with_ncu:
            ncu_report_path = run_ncu_profile(
                contract_path=contract_path,
                module_path=cand_copy,
                entry_point=ep_name,
                device=device,
                out_dir=out_dir,
                ncu_bin=ncu_bin,
                ncu_args=ncu_args,
                backend=backend,
                ncu_call=effective_ncu_call,
            )

        result: Dict[str, Any] = {
            "ok": ok_flag,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "num_trials": repeat,
            "iters_per_trial": iters,
            "device": device,
            "contract_path": str(contract_path),
            "module_path": str(module_path),
            "baseline_module_path": str(baseline_module_path),
            "entry_point": ep_name,
            "kernel_name": kernel_name,
            "kernel_type": kernel_type,
            "backend": backend,
            "timing_target": timing_target,
            "correctness_mode": effective_correctness_mode,
            "ncu_call": effective_ncu_call,
            "ncu_args": ncu_args,
            "timing": {
                "warmup": warmup,
                "iters": iters,
                "repeat": repeat,
            },
            "timing_mode": timing_path,
            "correct": correct,
            "correctness_error": correctness_error,
            "rtol": rtol,
            "atol": atol,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "output_summary": output_summary,
            "ncu_report": str(ncu_report_path) if ncu_report_path is not None else None,
            "benchmark_kernel_used": benchmark_used,
            "benchmark_kernel_available": benchmark_available,
        }

        (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        if prev_backend_env is None:
            os.environ.pop("KO_BACKEND", None)
        else:
            os.environ["KO_BACKEND"] = prev_backend_env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True, help="Path to contract JSON")
    ap.add_argument("--module", required=True, help="Path to kernel module .py")
    ap.add_argument("--baseline-module", help="Path to known-good baseline kernel module .py (defaults to --module)")
    ap.add_argument("--entry-point", default=None, help="Entry point function name (defaults from contract)")
    ap.add_argument("--device", default="cuda:0", help="Device string")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    ap.add_argument("--iters", type=int, default=100, help="Inner iterations per trial")
    ap.add_argument("--repeat", type=int, default=5, help="Number of timed trials")
    ap.add_argument("--out", required=True, help="Output directory for profiling results")
    ap.add_argument(
        "--backend",
        default=None,
        help="Optional backend selector string (passed via _backend / KO_BACKEND to kernels that support it).",
    )
    ap.add_argument(
        "--timing-target",
        choices=["auto", "benchmark_kernel", "entrypoint"],
        default="auto",
        help="What to time: benchmark_kernel (if available) or the entrypoint (default: auto).",
    )
    ap.add_argument(
        "--correctness-mode",
        choices=["auto", "entrypoint", "seeded_entrypoint", "benchmark_kernel", "skip"],
        default="auto",
        help="Correctness strategy (default: auto). seeded_entrypoint is useful for multi_kernel contracts.",
    )
    ap.add_argument(
        "--ncu-call",
        choices=["auto", "entrypoint", "benchmark_kernel"],
        default="auto",
        help="What to run under ncu when using --with-ncu (default: auto).",
    )
    ap.add_argument("--with-ncu", action="store_true", help="Also run Nsight Compute once")
    ap.add_argument("--ncu-bin", default="ncu", help="Nsight Compute CLI binary (default: 'ncu')")
    ap.add_argument(
        "--ncu-args",
        default=None,
        help="Optional raw Nsight Compute args to append (e.g. \"--kernel-name <substr> --metrics sm__throughput_pct\").",
    )
    ap.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for correctness")
    ap.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for correctness")
    ap.add_argument("--timing-mode", choices=["events", "cuda_graphs"], default="cuda_graphs", help="Timing mode to use (default: cuda_graphs)")
    args = ap.parse_args()

    contract_path = Path(args.contract)
    module_path = Path(args.module)
    baseline_module_path = Path(args.baseline_module) if args.baseline_module else None
    out_dir = Path(args.out)

    res = profile_one_contract(
        contract_path=contract_path,
        module_path=module_path,
        baseline_module_path=baseline_module_path,
        entry_point=args.entry_point,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        repeat=args.repeat,
        out_dir=out_dir,
        backend=args.backend,
        timing_target=args.timing_target,
        correctness_mode=args.correctness_mode,
        ncu_call=args.ncu_call,
        with_ncu=args.with_ncu,
        ncu_bin=args.ncu_bin,
        ncu_args=args.ncu_args,
        rtol=args.rtol,
        atol=args.atol,
        timing_mode=args.timing_mode,
    )

    print(json.dumps(res, indent=2))
    return 0 if res.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
