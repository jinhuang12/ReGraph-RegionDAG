#!/usr/bin/env python3
"""
profile_suite.py

Profile an entire kernel suite (multiple contracts / shapes) against a given
kernel module, and compute suite-level aggregate metrics (geomean runtime, and
geomean speedup vs baseline if available).

Usage (from repo root):

  # Baseline:
  python tools/profile_suite.py \
      --suite suites/vec_matmul_kernel_suite.json \
      --module kernels/vec_matmul_kernel.py \
      --device cuda:0 \
      --tag baseline \
      --warmup 10 --iters 100 --repeat 5

  # Candidate after code changes:
  python tools/profile_suite.py \
      --suite suites/vec_matmul_kernel_suite.json \
      --module kernels/vec_matmul_kernel.py \
      --device cuda:0 \
      --tag struct_001

Outputs:
  runs/<kernel_name>/<tag>/summary.json
  plus per-contract results in runs/<kernel_name>/<tag>/<contract_name>/result.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Local imports
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from profile_contract import profile_one_contract  # type: ignore


def geomean(values: List[float], weights: List[float]) -> float:
    """Weighted geometric mean."""
    if not values:
        return math.inf
    total_w = sum(weights) if weights else float(len(values))
    if total_w <= 0.0:
        total_w = float(len(values))
    acc = 0.0
    for v, w in zip(values, weights):
        if v <= 0.0:
            continue
        acc += (w / total_w) * math.log(v)
    return math.exp(acc) if acc != 0.0 else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="Path to suite JSON")
    ap.add_argument(
        "--module",
        default=None,
        help="Path to kernel module .py (default: kernels/<kernel_name>.py, or kernels/<subdir>/<kernel_name>.py when suite is under suites/<subdir>/)",
    )
    ap.add_argument("--baseline-module", help="Path to known-good baseline kernel module .py (defaults vary; see below)")
    ap.add_argument("--device", default="cuda:0", help="Device string")
    ap.add_argument("--tag", required=True, help="Tag name for this run (e.g. baseline, meta_001, struct_001)")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    ap.add_argument("--iters", type=int, default=100, help="Inner iterations per trial")
    ap.add_argument("--repeat", type=int, default=5, help="Number of timed trials")
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
    ap.add_argument("--with-ncu", action="store_true", help="Also run Nsight Compute per contract")
    ap.add_argument("--ncu-bin", default="ncu", help="Nsight Compute CLI binary (default: 'ncu')")
    ap.add_argument(
        "--ncu-args",
        default=None,
        help="Optional raw Nsight Compute args to append when using --with-ncu (e.g. \"--kernel-name <substr>\").",
    )
    ap.add_argument(
        "--timing-mode",
        choices=["events", "cuda_graphs"],
        default="cuda_graphs",
        help="Timing mode to use (default: cuda_graphs)",
    )
    args = ap.parse_args()

    suite_path = Path(args.suite).resolve()
    suite_dir = suite_path.parent

    suites_root: Path | None = None
    repo_root = suite_dir.parent
    # Handle nested suites/<subdir>/ paths by finding the ancestor named "suites"
    for parent in suite_path.parents:
        if parent.name == "suites":
            suites_root = parent
            repo_root = parent.parent
            break
    suite_subdir = ""
    if suites_root is not None:
        try:
            rel = suite_dir.relative_to(suites_root)
            suite_subdir = "" if str(rel) == "." else rel.as_posix()
        except ValueError:
            suite_subdir = ""

    with suite_path.open("r", encoding="utf-8") as f:
        suite = json.load(f)

    kernel_name = suite["kernel_name"]
    entry_point = suite.get("entry_point", "run")
    contracts_dir_rel = suite.get("contracts_dir", "contracts")
    # Back-compat: old suites sometimes used "contracts" even when living under suites/<subdir>/.
    if contracts_dir_rel == "contracts" and suite_subdir:
        alt_contracts = f"contracts/{suite_subdir}"
        if (repo_root / alt_contracts).is_dir():
            contracts_dir_rel = alt_contracts
    objective = suite.get("objective", "geomean_speedup")

    contracts_dir = (repo_root / contracts_dir_rel).resolve()
    if args.module:
        module_path = Path(args.module).resolve()
    else:
        module_rel = Path("kernels") / suite_subdir if suite_subdir else Path("kernels")
        module_path = (repo_root / module_rel / f"{kernel_name}.py").resolve()

    baseline_arg = Path(args.baseline_module).resolve() if args.baseline_module else None

    # Snapshot baseline module for reproducibility. For the baseline tag, default to the candidate module
    # if no baseline path is provided. For other tags, if no baseline arg is provided, fall back to the stored
    # snapshot; error if neither exists.
    baseline_snapshot = repo_root / "runs" / kernel_name / "baseline" / "baseline_module.py"
    if args.tag == "baseline":
        baseline_module_path = baseline_arg or module_path
        baseline_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline_module_path, baseline_snapshot)
    else:
        if baseline_arg is not None:
            baseline_module_path = baseline_arg
        elif baseline_snapshot.exists():
            baseline_module_path = baseline_snapshot
        else:
            raise SystemExit("Baseline module not provided and no baseline snapshot found; rerun baseline or supply --baseline-module.")

    tag = args.tag
    runs_root = repo_root / "runs" / kernel_name / tag

    effective_timing_mode = args.timing_mode
    baseline_summary_path = repo_root / "runs" / kernel_name / "baseline" / "summary.json"
    if tag != "baseline" and args.timing_mode == "cuda_graphs" and baseline_summary_path.exists():
        try:
            base_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
            base_mode = base_summary.get("timing_mode")
            if base_mode and base_mode != "cuda_graphs":
                print("[WARN] Baseline was not captured with cuda_graphs; forcing timing-mode=events for fairness")
                effective_timing_mode = "events"
        except Exception as e:
            print(f"[WARN] Failed to read baseline timing mode ({e}); defaulting to events")
            effective_timing_mode = "events"

    cases_config = suite.get("contracts", []) or []
    if not cases_config:
        raise SystemExit(f"No contracts listed in suite {suite_path}")

    print(f"[INFO] Profiling suite '{suite['suite_name']}' for kernel '{kernel_name}' with tag '{tag}'")
    print(f"[INFO] contracts_dir = {contracts_dir}, module = {module_path}")

    case_results: List[Dict[str, Any]] = []

    for c in cases_config:
        case_name = c.get("name") or c.get("filename")
        filename = c.get("filename") or c.get("contract_path")
        weight = float(c.get("weight", 1.0))

        if not filename:
            print(f"[WARN] Skipping case {case_name}: no filename/contract_path")
            continue

        contract_path = (contracts_dir / filename).resolve()
        case_out_dir = runs_root / case_name
        print(f"\n[INFO] Case '{case_name}' ({filename})")
        print(f"       contract = {contract_path}")
        res = profile_one_contract(
            contract_path=contract_path,
            module_path=module_path,
            baseline_module_path=baseline_module_path,
            entry_point=entry_point,
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
            repeat=args.repeat,
            out_dir=case_out_dir,
            backend=args.backend,
            timing_target=args.timing_target,
            correctness_mode=args.correctness_mode,
            ncu_call=args.ncu_call,
            with_ncu=args.with_ncu,
            ncu_bin=args.ncu_bin,
            ncu_args=args.ncu_args,
            timing_mode=effective_timing_mode,
        )

        case_results.append(
            {
                "name": case_name,
                "filename": filename,
                "weight": weight,
                "mean_ms": res.get("mean_ms"),
                "std_ms": res.get("std_ms"),
                "ok": res.get("ok", False),
                "correct": res.get("correct", False),
                "correctness_error": res.get("correctness_error"),
                "timing_mode": res.get("timing_mode"),
            }
        )

    # Aggregate metrics
    filtered_cases = [
        c
        for c in case_results
        if c.get("ok", False)
        and c.get("correct", False)
        and c.get("mean_ms") is not None
        and math.isfinite(float(c["mean_ms"]))
        and float(c["mean_ms"]) > 0.0
    ]
    times = [float(c["mean_ms"]) for c in filtered_cases]
    weights = [float(c["weight"]) for c in filtered_cases]

    geomean_ms = geomean(times, weights) if times else float("inf")
    invalid_cases = len(case_results) - len(filtered_cases)

    # Compare vs baseline if available
    baseline_summary_path = repo_root / "runs" / kernel_name / "baseline" / "summary.json"
    geomean_speedup_vs_baseline: Optional[float] = None

    if tag != "baseline" and baseline_summary_path.exists():
        try:
            base = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
            base_cases = base.get("cases", []) or []
            base_map = {c["filename"]: c for c in base_cases if "filename" in c and c.get("mean_ms")}

            spd_values: List[float] = []
            spd_weights: List[float] = []
            for c in case_results:
                fname = c.get("filename")
                if fname not in base_map:
                    continue
                t_new = c.get("mean_ms")
                t_old = base_map[fname].get("mean_ms")
                if not (t_new and t_old) or t_new <= 0.0 or t_old <= 0.0:
                    continue
                s = float(t_old) / float(t_new)
                spd_values.append(s)
                spd_weights.append(float(base_map[fname].get("weight", 1.0)))
            if spd_values:
                geomean_speedup_vs_baseline = geomean(spd_values, spd_weights)
        except Exception as e:
            print(f"[WARN] Failed to compute speedup vs baseline: {e}")

    summary = {
        "suite_name": suite["suite_name"],
        "kernel_name": kernel_name,
        "entry_point": entry_point,
        "contracts_dir": contracts_dir_rel,
        "tag": tag,
        "device": args.device,
        "backend": args.backend,
        "timing_target": args.timing_target,
        "correctness_mode": args.correctness_mode,
        "ncu_call": args.ncu_call,
        "ncu_args": args.ncu_args,
        "timing_mode": effective_timing_mode,
        "timing": {
            "warmup": args.warmup,
            "iters": args.iters,
            "repeat": args.repeat,
        },
        "all_correct": all(c.get("correct", False) for c in case_results) if case_results else False,
        "objective": objective,
        "cases": case_results,
        "aggregate": {
            "geomean_ms": geomean_ms,
            "geomean_speedup_vs_baseline": geomean_speedup_vs_baseline,
        },
        "baseline_summary_path": str(baseline_summary_path) if baseline_summary_path.exists() else None,
    }

    runs_root.mkdir(parents=True, exist_ok=True)
    summary_path = runs_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[INFO] Summary written to {summary_path}")
    if math.isfinite(geomean_ms):
        if geomean_speedup_vs_baseline is not None:
            print(
                f"[RESULT] tag={tag}  geomean_ms={geomean_ms:.3f}  "
                f"geomean_speedup_vs_baseline={geomean_speedup_vs_baseline:.3f}x"
            )
        else:
            print(f"[RESULT] tag={tag}  geomean_ms={geomean_ms:.3f} (no baseline yet)")
    else:
        print(f"[RESULT] tag={tag}  geomean_ms=inf (some cases failed)")
    if invalid_cases > 0 or not math.isfinite(geomean_ms):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
