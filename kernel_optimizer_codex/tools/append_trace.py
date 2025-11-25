#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/append_trace.py
---------------------

Append a structured JSON record to a per-kernel trace file.

Supports both:
  - single-run summaries (mean_ms / baseline_mean_ms / speedup_vs_baseline),
  - suite summaries under aggregate.* (geomean_ms / geomean_speedup_vs_baseline).

Typical usage (suite-level):

    python tools/append_trace.py \
      --kernel-name vec_matmul_kernel \
      --summary runs/vec_matmul_kernel/struct_001/summary.json \
      --baseline runs/vec_matmul_kernel/baseline/summary.json \
      --tag struct_001 \
      --edit-kind structural \
      --description "Introduced 2-stage async pipeline over K and retuned BLOCK sizes."

This appends one JSON object to:

    runs/<kernel_name>/trace.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _repo_relative(path: Path) -> str:
    """
    Best-effort conversion to a path relative to repo root; falls back to absolute.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _extract_metrics(summary: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Try to extract (mean_ms, baseline_mean_ms, speedup_vs_baseline) from either:
      - flat single-run summary, or
      - suite-level summary under 'aggregate'.
    """
    # flat style
    if "mean_ms" in summary:
        mean_ms = float(summary.get("mean_ms", 0.0) or 0.0)
        baseline_mean_ms = summary.get("baseline_mean_ms")
        if baseline_mean_ms is not None:
            baseline_mean_ms = float(baseline_mean_ms)
        speedup = summary.get("speedup_vs_baseline")
        if speedup is not None:
            speedup = float(speedup)
        return mean_ms, baseline_mean_ms, speedup

    # suite-style: look under 'aggregate'
    agg = summary.get("aggregate") or {}
    mean_ms = agg.get("geomean_ms")
    if mean_ms is not None:
        mean_ms = float(mean_ms)
    baseline_mean_ms = agg.get("baseline_geomean_ms")
    if baseline_mean_ms is not None:
        baseline_mean_ms = float(baseline_mean_ms)
    speedup = agg.get("geomean_speedup_vs_baseline")
    if speedup is not None:
        speedup = float(speedup)
    return mean_ms, baseline_mean_ms, speedup


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Append a structured trace record for a kernel optimization attempt.",
    )
    p.add_argument(
        "--kernel-name",
        required=True,
        help="Logical kernel name (e.g. vec_matmul_kernel). Used for trace path.",
    )
    p.add_argument(
        "--summary",
        required=True,
        help="Path to candidate summary.json (suite or single-run).",
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="Optional path to baseline summary.json.",
    )
    p.add_argument(
        "--tag",
        required=True,
        help="Candidate tag (e.g. meta_001, struct_003).",
    )
    p.add_argument(
        "--edit-kind",
        choices=["meta_params", "structural", "mixed"],
        required=True,
        help="What kind of edit this candidate represents.",
    )
    p.add_argument(
        "--description",
        required=True,
        help="Short human-readable description of the change.",
    )
    p.add_argument(
        "--files-modified",
        nargs="*",
        default=None,
        help="Optional list of modified files (paths relative to repo root).",
    )
    p.add_argument(
        "--contract",
        default=None,
        help="Optional contract name or contract JSON path (for per-contract notes).",
    )
    p.add_argument(
        "--ncu-summary",
        default=None,
        help="Optional path to ncu_summary.json for this candidate.",
    )
    args = p.parse_args(argv)

    kernel_name = args.kernel_name
    summary_path = Path(args.summary).expanduser()
    if not summary_path.is_absolute():
        summary_path = (REPO_ROOT / summary_path).resolve()
    else:
        summary_path = summary_path.resolve()
    if not summary_path.exists():
        print(f"[append_trace] summary not found: {summary_path}", file=sys.stderr)
        return 1

    baseline_path: Optional[Path] = None
    baseline_summary: Optional[Dict[str, Any]] = None
    if args.baseline:
        candidate_baseline = Path(args.baseline).expanduser()
        if not candidate_baseline.is_absolute():
            candidate_baseline = (REPO_ROOT / candidate_baseline).resolve()
        else:
            candidate_baseline = candidate_baseline.resolve()
        baseline_path = candidate_baseline
        if not baseline_path.exists():
            print(f"[append_trace] baseline not found: {baseline_path}", file=sys.stderr)
            baseline_path = None
        else:
            try:
                baseline_summary = _load_json(baseline_path)
            except Exception as e:
                print(f"[append_trace] failed to load baseline: {e}", file=sys.stderr)
                baseline_path = None
                baseline_summary = None

    try:
        summary = _load_json(summary_path)
    except Exception as e:
        print(f"[append_trace] failed to load summary: {e}", file=sys.stderr)
        return 1

    mean_ms, baseline_mean_ms, speedup = _extract_metrics(summary)
    if baseline_summary:
        base_mean, _, _ = _extract_metrics(baseline_summary)
        if baseline_mean_ms is None:
            baseline_mean_ms = base_mean
    if (
        speedup is None
        and mean_ms is not None
        and baseline_mean_ms is not None
        and baseline_mean_ms > 0.0
        and mean_ms > 0.0
    ):
        try:
            speedup = float(baseline_mean_ms) / float(mean_ms)
        except Exception:
            speedup = None

    record: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "kernel_name": kernel_name,
        "candidate_tag": args.tag,
        "edit_kind": args.edit_kind,
        "description": args.description,
        "summary_path": _repo_relative(summary_path),
        "baseline_summary": _repo_relative(baseline_path) if baseline_path else None,
        "contract": args.contract,
        "ncu_summary": args.ncu_summary,
        "mean_ms": mean_ms,
        "baseline_mean_ms": baseline_mean_ms,
        "speedup_vs_baseline": speedup,
        "files_modified": args.files_modified or [],
    }

    trace_dir = REPO_ROOT / "runs" / kernel_name
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / "trace.jsonl"

    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"[append_trace] appended record to {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
