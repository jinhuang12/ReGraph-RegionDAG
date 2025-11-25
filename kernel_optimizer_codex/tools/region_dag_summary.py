#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/region_dag_summary.py
---------------------------

CLI wrapper around RegionDagContext (from kernel_opt_tooling.py).

Typical usage:

    # If you have a PTX file for a candidate (e.g. compiled CUDA kernel),
    # you can generate a Region-DAG overview:

    python tools/region_dag_summary.py \
        --ptx runs/vec_matmul_kernel_small_contract/candidate_async_v2/kernel.ptx \
        --mode overview \
        > runs/vec_matmul_kernel_small_contract/candidate_async_v2/region_dag_overview.json

Modes:
    - overview      : global summary + hottest stages/regions.
    - stage_detail  : details for a single stage_id (requires --stage-id).
    - region_detail : PTX snippet + work profile for a single region_id (requires --region-id).
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kernel_opt_tooling import RegionDagContext  # type: ignore


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Summarize a PTX file into Region-DAG JSON.",
    )
    p.add_argument(
        "--ptx",
        required=True,
        help="Path to PTX file (e.g., kernel.ptx).",
    )
    p.add_argument(
        "--mode",
        choices=["overview", "stage_detail", "region_detail"],
        default="overview",
        help="Summary mode (default: overview).",
    )
    p.add_argument(
        "--stage-id",
        type=int,
        default=None,
        help="Stage ID (required for stage_detail).",
    )
    p.add_argument(
        "--region-id",
        type=int,
        default=None,
        help="Region ID (required for region_detail).",
    )
    p.add_argument(
        "--max-hot-regions",
        type=int,
        default=8,
        help="Max number of hot regions (overview).",
    )
    p.add_argument(
        "--max-hot-stages",
        type=int,
        default=4,
        help="Max number of hot stages (overview).",
    )
    p.add_argument(
        "--max-insts",
        type=int,
        default=16,
        help="Max PTX lines for region_detail snippet.",
    )
    args = p.parse_args(argv)

    ptx_path = Path(args.ptx).resolve()
    if not ptx_path.exists():
        print(f"[region_dag_summary] PTX not found: {ptx_path}", file=sys.stderr)
        return 1

    ptx_text = ptx_path.read_text(encoding="utf-8")
    try:
        ctx = RegionDagContext(ptx_text, kernel_name=ptx_path.name)
    except Exception as e:
        print(
            json.dumps(
                {"error": f"Failed to build RegionDagContext: {e}"},
                indent=2,
            )
        )
        return 1

    if args.mode == "overview":
        data = ctx.overview(
            max_hot_regions=args.max_hot_regions,
            max_hot_stages=args.max_hot_stages,
        )
    elif args.mode == "stage_detail":
        if args.stage_id is None:
            print(
                json.dumps(
                    {"error": "stage_detail mode requires --stage-id"},
                    indent=2,
                )
            )
            return 1
        data = ctx.stage_detail(args.stage_id)
        if data is None:
            data = {"error": f"Stage {args.stage_id} not found"}
    else:  # region_detail
        if args.region_id is None:
            print(
                json.dumps(
                    {"error": "region_detail mode requires --region-id"},
                    indent=2,
                )
            )
            return 1
        data = ctx.region_detail(args.region_id, max_insts=args.max_insts)
        if data is None:
            data = {"error": f"Region {args.region_id} not found"}

    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
