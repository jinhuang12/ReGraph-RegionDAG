#!/usr/bin/env python3
"""
tools/ptx_source_summary.py
---------------------------

CLI wrapper around PtxSourceCorrelator (from ptx_source_correlator.py) to pull
PTX/SASS correlated to source lines from an Nsight Compute report (.ncu-rep).

Why useful?
  - Quickly see which source lines correspond to hot PCs when tuning kernels.
  - Attach per-PC metric values (e.g., inst_executed or stall samples) for a
    given source span.

Typical usage (snippet for a source span):

  python tools/ptx_source_summary.py \
      --report runs/gemm_split_k_kernel/ncu_import_source_test3/ncu_report.ncu-rep \
      --kernel gemm_split_k_kernel \
      --source-file kernels/gemm_split_k_kernel.py \
      --start-line 90 --end-line 130 \
      --include-sass \
      --include-metric \
      --extra-metric smsp__pcsamp_sample_count \
      > runs/gemm_split_k_kernel/ncu_import_source_test3/source_span.json

Full mapping filtered to one file:

  python tools/ptx_source_summary.py \
      --report runs/gemm_split_k_kernel/ncu_import_source_test3/ncu_report.ncu-rep \
      --kernel gemm_split_k_kernel \
      --source-file gemm_split_k_kernel.py \
      --mode mapping \
      > runs/gemm_split_k_kernel/ncu_import_source_test3/source_mapping.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ptx_source_correlator import PtxSourceCorrelator  # type: ignore


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Correlate PTX/SASS to source lines from an Nsight Compute .ncu-rep.",
    )
    p.add_argument("--report", required=True, help="Path to .ncu-rep report.")
    p.add_argument("--kernel", default=None, help="Substring to select kernel (action) name.")
    p.add_argument("--nvtx", default=None, help="Substring to select NVTX range name.")
    p.add_argument(
        "--source-file",
        required=True,
        help="Source file path or substring to filter correlation results (substring accepted).",
    )
    p.add_argument(
        "--mode",
        choices=["snippet", "mapping"],
        default="snippet",
        help="snippet: return lines between start/end; mapping: full mapping for the filtered file.",
    )
    p.add_argument("--start-line", type=int, default=None, help="Start line (required for snippet).")
    p.add_argument("--end-line", type=int, default=None, help="End line (required for snippet).")
    p.add_argument("--base-metric", default=None, help="Base metric to enumerate PCs (default: heuristics).")
    p.add_argument(
        "--include-sass",
        action="store_true",
        help="Include SASS alongside PTX.",
    )
    p.add_argument(
        "--include-metric",
        action="store_true",
        help="Attach per-PC metric values (base + extras).",
    )
    p.add_argument(
        "--extra-metric",
        action="append",
        default=[],
        help="Extra source-correlated metrics to attach per PC (can be used multiple times).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional path to write JSON output (otherwise printed to stdout).",
    )
    args = p.parse_args(argv)

    report_path = Path(args.report).resolve()
    if not report_path.exists():
        print(json.dumps({"error": f"Report not found: {report_path}"}), file=sys.stderr)
        return 1

    correlator = PtxSourceCorrelator(report_path=report_path)
    if not getattr(correlator, "available", False):
        print(
            json.dumps(
                {"error": "PtxSourceCorrelator unavailable (ncu_report missing or report not found)."}
            ),
            file=sys.stderr,
        )
        return 1

    source_filter = args.source_file

    if args.mode == "snippet":
        if args.start_line is None or args.end_line is None:
            print(json.dumps({"error": "snippet mode requires --start-line and --end-line"}), file=sys.stderr)
            return 1
        data: Dict[str, Any] = correlator.get_ptx_snippet_for_source_span(
            nvtx_range=args.nvtx,
            kernel_name=args.kernel,
            source_file=source_filter,
            start_line=int(args.start_line),
            end_line=int(args.end_line),
            max_insts=256,
            include_sass=args.include_sass,
            include_metric_value=args.include_metric,
            extra_metric_names=args.extra_metric,
        )
    else:
        data = correlator.correlate_ptx_to_source(
            nvtx_range=args.nvtx,
            kernel_name=args.kernel,
            base_metric_name=args.base_metric,
            include_sass=args.include_sass,
            include_metric_value=args.include_metric,
            source_file_filter=source_filter,
            extra_metric_names=args.extra_metric,
        )

    out_json = json.dumps(data, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json, encoding="utf-8")
    else:
        print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
