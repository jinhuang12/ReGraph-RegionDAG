#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/summarize_ncu.py
----------------------

CLI wrapper around NcuMetricsContext (from kernel_opt_tooling.py).

Typical usage (contract spec example):

    python tools/profile_kernel.py \
        --spec contracts/vec_matmul_kernel_small_contract.json \
        --out  runs/vec_matmul_kernel_small_contract/candidate_async_v2 \
        --baseline-summary runs/vec_matmul_kernel_small_contract/base/summary.json \
        --with-ncu

    python tools/summarize_ncu.py \
        --report runs/vec_matmul_kernel_small_contract/candidate_async_v2/profile.ncu-rep \
        --mode summary \
        > runs/vec_matmul_kernel_small_contract/candidate_async_v2/ncu_summary.json

Modes:
    - summary       : compact SoL/occupancy/memory summary (default).
    - get_values    : query specific metrics by canonical or raw name.
    - search_names  : substring search over metric names/descriptions.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernel_opt_tooling import NcuMetricsContext  # type: ignore


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Summarize an Nsight Compute .ncu-rep report into JSON.",
    )
    p.add_argument(
        "--report",
        required=True,
        help="Path to .ncu-rep file produced by Nsight Compute.",
    )
    p.add_argument(
        "--mode",
        choices=["summary", "get_values", "search_names"],
        default="summary",
        help="Summary mode (default: summary).",
    )
    p.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Metric names for get_values() (canonical or raw).",
    )
    p.add_argument(
        "--name-kind",
        choices=["canonical", "ncu", "auto"],
        default="auto",
        help="How to interpret names for get_values().",
    )
    p.add_argument(
        "--query",
        default=None,
        help="Substring for search_names().",
    )
    p.add_argument(
        "--max-results",
        type=int,
        default=32,
        help="Max results for search_names().",
    )
    p.add_argument(
        "--range-idx",
        type=int,
        default=0,
        help="Nsight range index to inspect (default: 0).",
    )
    p.add_argument(
        "--action-idx",
        type=int,
        default=0,
        help="Nsight action index to inspect (default: 0).",
    )
    args = p.parse_args(argv)

    report_path = Path(args.report).resolve()
    if not report_path.exists():
        print(f"[summarize_ncu] Report not found: {report_path}", file=sys.stderr)
        return 1

    try:
        ctx = NcuMetricsContext(
            report_path=str(report_path),
            range_idx=args.range_idx,
            action_idx=args.action_idx,
        )
    except Exception as e:
        print(
            json.dumps(
                {"error": f"Failed to create NcuMetricsContext: {e}"},
                indent=2,
            )
        )
        return 1

    if args.mode == "summary":
        data = ctx.summary()
    elif args.mode == "get_values":
        names = args.names or []
        data = ctx.get_values(names, name_kind=args.name_kind)
    else:  # search_names
        if not args.query:
            print(
                json.dumps(
                    {"error": "search_names mode requires --query"},
                    indent=2,
                )
            )
            return 1
        data = ctx.search_names(args.query, max_results=args.max_results)

    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
