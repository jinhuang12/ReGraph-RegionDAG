#!/usr/bin/env python3
"""
kernel_opt_tooling.py

All-in-one support script for GPU kernel optimization with an LLM + Nsight Compute:

  * RegionDagContext:
      - Wraps your PTX→Region-DAG builder (ptx_dag_tool_v2.build_all).
      - Exposes compact summaries and drilldown views (overview / stage / region).
      - Uses a *static work profile* (flops, bytes, atomics, insts) — no static timing.

  * NcuMetricsContext:
      - Wraps Nsight Compute's Python Report Interface (ncu_report).
      - Exposes a canonical, LLM-friendly device metrics schema.
      - Supports summary(), get_values() and search_names() for metrics.
      - Uses a central canonical→NCU metric mapping.

  * PtxSourceCorrelator:
      - Wraps Nsight Compute correlation APIs to map PCs → source lines → PTX/SASS.
      - Provides a high-level "snippet" view for a given source span.
      - Suitable as the implementation of a get_ptx_by_source tool.

  * ToolRegistry + LLMCandidateGenerator:
      - Define function-tools for the OpenAI Responses API:
           region_dag_inspect
           ncu_metrics_inspect
           get_ptx_by_source
      - Run a standard tool loop:
           call → detect tool calls → execute → send tool_result → continue
      - Enforce STRICT JSON output with a fixed candidate schema.

This file is intentionally verbose and heavily commented so future you (and an LLM)
can understand and extend it without needing to re-derive the design.

Dependencies:
    - ptx_dag_tool_v2.py (your PTX→Region-DAG builder)
    - Nsight Compute + ncu_report (Nsight Compute Python Report Interface)
    - openai (Python SDK)
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import subprocess
import tempfile
import shutil
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging setup (simple default)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

# ---------------------------------------------------------------------------
# Region-DAG wrappers (using your existing ptx_dag_tool_v2)
# ---------------------------------------------------------------------------

from ptx_dag_tool_v2 import build_all, Region, Stage  # type: ignore
from ptx_source_correlator import PtxSourceCorrelator  # type: ignore
from shared.model import DeviceProfile


class RegionDagContext:
    """
    RegionDagContext encapsulates a single kernel's Region-DAG built from PTX.

    Responsibilities:
      - Build Region-DAG via ptx_dag_tool_v2.build_all(ptx_text).
      - Maintain indexes by region id and by stage id.
      - Expose the static work profile and control-flow data (loops/divergence).
    """

    def __init__(self, ptx_text: str, kernel_name: str = "kernel"):
        if build_all is None:
            raise RuntimeError("ptx_dag_tool_v2.build_all is not available.")
        self.kernel_name = kernel_name
        self.ptx_text = ptx_text

        # Call the builder
        (
            self.insts,
            self.regions,
            self.edges,
            self.stages,
            self.loops,   # NEW: loop overlay
        ) = build_all(self.ptx_text)

        # Build lookup maps
        self.region_by_id = {r.id: r for r in self.regions}
        self.stage_by_id = {s.id: s for s in self.stages}
        self.stage_for_region = {}
        for stage in self.stages:
            for region_id in stage.region_ids:
                self.stage_for_region[region_id] = stage.id

        # No stage time calculation — work profile only
        logger.info("RegionDagContext initialized (Static Work Profile).")

    def overview(self, max_hot_regions=8, max_hot_stages=4) -> Dict[str, Any]:
        """
        Returns a compact kernel summary based on static work profile.

        - No 'times', 'bottleneck_resource', or 'peaks_used'.
        - Adds 'loops' and 'stages_with_potential_divergence'.
        - Each stage returns a 'work_profile'.
        """

        # Sort stages by a simple "work" score
        def work_score(stage: Stage):
            return stage.global_bytes + stage.shared_read_bytes + \
                   stage.shared_write_bytes + stage.flops

        sorted_stages = sorted(self.stages, key=work_score, reverse=True)
        hot_stages = sorted_stages[:max_hot_stages]

        # Sort regions by raw work
        def region_work_score(region: Region):
            return region.global_read + region.global_write + \
                   region.shared_read + region.shared_write + region.flops

        sorted_regions = sorted(self.regions, key=region_work_score, reverse=True)
        hot_regions = sorted_regions[:max_hot_regions]

        # Stages with divergence
        stages_with_divergence = [
            s.id for s in self.stages if getattr(s, "has_potential_divergence", False)
        ]

        # Build a phase mix histogram per stage (phase → total instruction_count)
        def stage_phase_mix(s: Stage) -> Dict[str, int]:
            hist: Dict[str, int] = {}
            for rid in s.region_ids:
                r = self.region_by_id.get(rid)
                if not r:
                    continue
                hist[r.phase] = hist.get(r.phase, 0) + r.instruction_count
            return hist

        return {
            "metadata": {
                "kernel_name": self.kernel_name,
                "counts": {
                    "num_instructions": len(self.insts),
                    "num_regions": len(self.regions),
                    "num_edges": len(self.edges),
                    "num_stages": len(self.stages),
                    "num_loops": len(self.loops),
                },
            },
            "pipeline_stages": [
                {
                    "stage_id": s.id,
                    "region_ids": s.region_ids,
                    "phase_mix": stage_phase_mix(s),
                    "work_profile": {
                        "global_bytes": s.global_bytes,
                        "shared_read_bytes": s.shared_read_bytes,
                        "shared_write_bytes": s.shared_write_bytes,
                        "flops": s.flops,
                        "instruction_count": s.instruction_count,
                        "atomic_ops": s.atomic_ops,
                    },
                    "has_potential_divergence": getattr(s, "has_potential_divergence", False),
                } for s in hot_stages
            ],
            "hot_regions": [
                {
                    "region_id": r.id,
                    "stage_id": self.stage_for_region.get(r.id),
                    "phase": r.phase,
                    "global_bytes": r.global_read + r.global_write,
                    "shared_bytes": r.shared_read + r.shared_write,
                    "flops": r.flops,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                } for r in hot_regions
            ],
            "loops": [loop.__dict__ for loop in self.loops],
            "stages_with_potential_divergence": stages_with_divergence,
        }

    def stage_detail(self, stage_id: int) -> Optional[Dict[str, Any]]:
        """
        Detailed view of a single stage's work profile.
        """
        stage = self.stage_by_id.get(stage_id)
        if not stage:
            return None

        return {
            "stage_id": stage.id,
            "region_ids": stage.region_ids,
            "work_profile": {
                "global_bytes": stage.global_bytes,
                "shared_read_bytes": stage.shared_read_bytes,
                "shared_write_bytes": stage.shared_write_bytes,
                "flops": stage.flops,
                "instruction_count": stage.instruction_count,
                "atomic_ops": stage.atomic_ops,
            },
            "has_potential_divergence": getattr(stage, "has_potential_divergence", False),
            "regions": [
                {
                    "region_id": r.id,
                    "phase": r.phase,
                    "global_bytes": r.global_read + r.global_write,
                    "shared_bytes": r.shared_read + r.shared_write,
                    "flops": r.flops,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                } for r in (self.region_by_id.get(rid) for rid in stage.region_ids) if r
            ],
        }

    def region_detail(self, region_id: int, max_insts: int = 16) -> Optional[Dict[str, Any]]:
        """
        View of a single region, including a PTX snippet.
        """
        region = self.region_by_id.get(region_id)
        if not region:
            return None

        # Get PTX snippet (physical lines)
        ptx_lines = self.ptx_text.splitlines()
        snippet_lines = ptx_lines[region.start_line - 1 : region.end_line]
        if len(snippet_lines) > max_insts:
            snippet_lines = snippet_lines[:max_insts] + ["... (truncated)"]

        return {
            "region_id": region.id,
            "stage_id": self.stage_for_region.get(region.id),
            "phase": region.phase,
            "work_profile": {
                "global_bytes": region.global_read + region.global_write,
                "shared_bytes": region.shared_read + region.shared_write,
                "flops": region.flops,
                "instruction_count": region.instruction_count,
            },
            "ptx_source_line_range": [region.start_line, region.end_line],
            "ptx_snippet": [
                f"// PTX L{region.start_line + i}: {line}"
                for i, line in enumerate(snippet_lines)
            ],
        }

# ---------------------------------------------------------------------------
# Nsight Compute metrics wrappers + canonical mapping
# ---------------------------------------------------------------------------

# Try importing ncu_report; tool functions will simply fail gracefully if unavailable.
NCU_REPORT_AVAILABLE = False
try:
    import glob
    ncu_paths = [
        "/opt/nvidia/nsight-compute/*/extras/python",
        "/usr/local/cuda/nsight-compute*/extras/python",
        "/usr/local/NVIDIA-Nsight-Compute/extras/python",
    ]
    _ncu_report_path = None
    for pattern in ncu_paths:
        matches = glob.glob(pattern)
        if matches:
            _ncu_report_path = sorted(matches)[-1]
            break
    if _ncu_report_path and _ncu_report_path not in sys.path:
        sys.path.insert(0, _ncu_report_path)
    import ncu_report  # type: ignore
    NCU_REPORT_AVAILABLE = True
    logger.info("Loaded ncu_report from: %s", _ncu_report_path)
except Exception as e:
    logger.warning("ncu_report not available: %s", e)
    ncu_report = None  # type: ignore


def _metric_value(action, name: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Small helper: return (value, unit) if metric exists and is numeric; else (None, None).
    """
    try:
        if name in action:
            m = action[name]
            v = m.value()
            if isinstance(v, (int, float)):
                try:
                    unit = m.unit()
                except Exception:
                    unit = None
                return float(v), unit
    except Exception as e:
        logger.debug("Could not extract metric %s: %s", name, e)
    return None, None


def _to_seconds(val: float, unit: Optional[str]) -> Optional[float]:
    """
    Convert a duration to seconds if the unit is known; otherwise return None.
    """
    if val is None:
        return None
    if not unit:
        return None
    u = unit.lower()
    if u in ("s", "sec", "second", "seconds"):
        return float(val)
    if u in ("ms", "millisecond", "milliseconds"):
        return float(val) / 1e3
    if u in ("us", "µs", "microsecond", "microseconds"):
        return float(val) / 1e6
    if u in ("ns", "nanosecond", "nanoseconds"):
        return float(val) / 1e9
    return None


def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    """
    Safe division: returns None if n or d is None, or if d == 0.
    """
    try:
        if n is None or d is None or d == 0:
            return None
        return float(n) / float(d)
    except Exception:
        return None


# -------------------------------
# Canonical → NCU metric mappings
# -------------------------------
FLOP_WEIGHTS: Dict[str, float] = {
    # Floating-point scalar ops
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 2.0,
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 1.0,
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 1.0,
    "smsp__sass_thread_inst_executed_op_fmad_pred_on.sum": 2.0,
    # Tensor core ops (approximate MMA FLOPs; adjust per arch as needed)
    "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum": 256.0,
    "smsp__sass_thread_inst_executed_op_dmma_pred_on.sum": 512.0,
}

def compute_arithmetic_intensity(raw_metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute arithmetic intensity (FLOPs / DRAM byte) from raw Nsight metrics.
    Uses dram__bytes_read.sum + dram__bytes_write.sum and weighted FLOP counters.
    """
    bytes_read = float(raw_metrics.get("dram__bytes_read.sum", 0.0) or 0.0)
    bytes_write = float(raw_metrics.get("dram__bytes_write.sum", 0.0) or 0.0)
    dram_bytes = bytes_read + bytes_write

    flops = 0.0
    for name, weight in FLOP_WEIGHTS.items():
        try:
            flops += float(raw_metrics.get(name, 0.0) or 0.0) * weight
        except Exception:
            continue

    if dram_bytes <= 0.0:
        ai = math.inf if flops > 0 else 0.0
    else:
        ai = flops / dram_bytes

    return {
        "arith_intensity_flop_per_byte": ai,
        "arith_intensity_flops": flops,
        "arith_intensity_dram_bytes": dram_bytes,
    }

def compute_roofline_bound(ai_flop_per_byte: float, device_profile: Optional[DeviceProfile]) -> Dict[str, Any]:
    """
    Classify bound type using a simple roofline balance point if device peak FLOPs
    and memory bandwidth are known.
    """
    if not device_profile:
        return {"roofline_bound": "unknown"}

    peak_flops_tflops = device_profile.peak_flops_tflops
    mem_bw_gbps = device_profile.mem_bandwidth_gbps or device_profile.hbm_bw_gbps

    if peak_flops_tflops is None or mem_bw_gbps is None or mem_bw_gbps <= 0:
        return {"roofline_bound": "unknown"}

    I_crit = (peak_flops_tflops * 1e12) / (mem_bw_gbps * 1e9)  # FLOP/byte
    bound = "unknown"
    if math.isfinite(ai_flop_per_byte):
        if ai_flop_per_byte < 0.5 * I_crit:
            bound = "memory_bound"
        elif ai_flop_per_byte > 2.0 * I_crit:
            bound = "compute_bound"
        else:
            bound = "balanced"
    return {"roofline_bound": bound, "roofline_I_crit": I_crit}

CANONICAL_METRIC_SECTIONS: Dict[str, Dict[str, str]] = {
    # High-level Speed-of-Light / throughput
    "speed_of_light": {
        "compute_memory_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "compute_throughput_pct":        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm_throughput_pct":             "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu_dram_throughput_pct":       "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram_throughput_pct":           "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "memory_throughput_pct":         "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    },

    # Compact "device summary" section
    "detailed_metrics": {
        "l1_hit_rate_pct":        "l1tex__t_sector_hit_rate.pct",
        "l2_hit_rate_pct":        "lts__t_sector_hit_rate.pct",
        "warp_occupancy_pct":     "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm_active_cycles_pct":   "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "instructions_per_cycle": "sm__inst_executed.avg.per_cycle_active",  # IPC
        "waves_per_sm":           "launch__waves_per_multiprocessor",
    },

    # Memory hierarchy / bandwidth
    "memory_metrics": {
        "dram_avg_bandwidth_gb_s":   "dram__bytes.avg.per_second",    # will be converted to GB/s
        "dram_total_bandwidth_gb_s": "dram__bytes.sum.per_second",    # will be converted to GB/s
        "dram_active_cycles_pct":    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "l1_writeback_active_pct":   "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_elapsed",
        "l1_read_sectors_pct":       "l1tex__m_xbar2l1tex_read_sectors.avg.pct_of_peak_sustained_elapsed",
        "l2_throughput_pct":         "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    },

    # Compute utilization & occupancy limiters
    "compute_metrics": {
        "fma_pipe_utilization_pct":    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
        "fp64_pipe_utilization_pct":   "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
        "alu_pipe_utilization_pct":    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
        "xu_pipe_utilization_pct":     "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
        "tensor_pipe_utilization_pct": "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
        "instructions_per_cycle":      "sm__inst_executed.avg.per_cycle_active",
        "occupancy_limit_blocks":      "launch__occupancy_limit_blocks",
        "occupancy_limit_registers":   "launch__occupancy_limit_registers",
        "occupancy_limit_shared_mem":  "launch__occupancy_limit_shared_mem",
        "occupancy_limit_warps":       "launch__occupancy_limit_warps",
        "registers_per_thread":        "launch__registers_per_thread",
    },

    # Pipe-level active fractions
    "pipeline_metrics": {
        "fma_pipe_active_pct":    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "alu_pipe_active_pct":    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "tensor_pipe_active_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "shared_pipe_active_pct": "sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "fp64_pipe_active_pct":   "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm_issue_active_pct":    "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    },

    # Launch/occupancy description
    "occupancy_metrics": {
        "occupancy_limit_registers": "launch__occupancy_limit_registers",
        "occupancy_limit_shared_mem": "launch__occupancy_limit_shared_mem",
        "occupancy_limit_warps":      "launch__occupancy_limit_warps",
        "occupancy_limit_blocks":     "launch__occupancy_limit_blocks",
        "waves_per_sm":               "launch__waves_per_multiprocessor",
        "block_size":                 "launch__block_size",
        "grid_size":                  "launch__grid_size",
        "shared_mem_per_block":       "launch__shared_mem_per_block",
    },

    # Warp stall reasons
    "stall_metrics": {
        "stall_long_scoreboard_pct":  "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "stall_short_scoreboard_pct": "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "stall_barrier_pct":          "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "stall_not_selected_pct":     "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    },

    # Scheduler behavior
    "scheduler_metrics": {
        "warps_eligible_per_cycle": "smsp__warps_eligible.avg.per_cycle_active",
        "inst_issued_per_cycle":    "smsp__inst_issued.avg.per_cycle_active",
        "issue_active_pct":         "smsp__issue_active.avg.pct_of_peak_sustained_active",
    },

    # Access pattern diagnostics (raw counters only).
    "access_pattern_metrics": {
        "l2_theoretical_sectors_global":           "memory_l2_theoretical_sectors_global",
        "l2_theoretical_sectors_global_ideal":     "memory_l2_theoretical_sectors_global_ideal",
        "l2_theoretical_sectors_global_excessive": "derived__memory_l2_theoretical_sectors_global_excessive",
        "shared_bank_conflicts_load_sum":          "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "shared_bank_conflicts_store_sum":         "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    },

    # Roofline inputs
    "roofline_raw": {
        "flop_count_sp":     "flop_count_sp",
        "flop_count_hp":     "flop_count_hp",
        "flop_count_dp":     "flop_count_dp",
        "flop_count_tensor": "flop_count_tensor",
        "dram_bytes_sum":    "dram__bytes.sum",
    },

    # Timing context
    "timing_metrics": {
        "gpu_time_duration_sum":  "gpu__time_duration.sum",
        "gpc_cycles_elapsed_max": "gpc__cycles_elapsed.max",
    },
}

# Flatten canonical mapping
CANONICAL_TO_NCU_METRICS: Dict[str, str] = {}
for section, mapping in CANONICAL_METRIC_SECTIONS.items():
    for canonical_name, ncu_name in mapping.items():
        if canonical_name in CANONICAL_TO_NCU_METRICS:
            if CANONICAL_TO_NCU_METRICS[canonical_name] != ncu_name:
                logger.warning(
                    "Canonical metric %s mapped to multiple NCU names (%s, %s); keeping the first.",
                    canonical_name,
                    CANONICAL_TO_NCU_METRICS[canonical_name],
                    ncu_name,
                )
            continue
        CANONICAL_TO_NCU_METRICS[canonical_name] = ncu_name


@dataclass
class MetricInfo:
    name: str
    value: Any
    unit: str
    description: str


class NcuMetricsContext:
    """
    NcuMetricsContext wraps a single Nsight Compute IAction (kernel profile) and
    exposes:

      - summary(): a curated device metrics JSON for the LLM prompt.
      - get_values(): values for requested canonical or raw metrics.
      - search_names(): substring search over metric names/descriptions.
    """

    def __init__(self, report_path: str, range_idx: int = 0, action_idx: int = 0, device_profile: Optional[DeviceProfile] = None):
        if not NCU_REPORT_AVAILABLE:
            raise RuntimeError("ncu_report is not available; Nsight Compute must be installed.")

        self.report_path = report_path
        self.report = ncu_report.load_report(report_path)  # type: ignore
        self.range = self.report.range_by_idx(range_idx)
        self.action = self.range.action_by_idx(action_idx)
        self.kernel_name = self.action.name()
        self.device_profile = device_profile

        # Cache all metrics.
        self._metrics: Dict[str, MetricInfo] = self._collect_metrics()
        self.raw_metrics: Dict[str, Any] = {name: mi.value for name, mi in self._metrics.items()}

    def _collect_metrics(self) -> Dict[str, MetricInfo]:
        metrics: Dict[str, MetricInfo] = {}
        for name in self.action.metric_names():
            m = self.action[name]
            try:
                value = m.value()
                unit = ""
                try:
                    unit = m.unit()
                except Exception:
                    unit = ""
                desc = ""
                try:
                    desc = m.description()
                except Exception:
                    desc = ""
            except Exception:
                continue

            metrics[name] = MetricInfo(
                name=name,
                value=value,
                unit=unit,
                description=desc,
            )
        return metrics

    # ------------------------- Summary (mode = "summary") -------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact, canonical device summary.

        Structure:
            {
              "kernel_name": str,
              "device_name": str | None,
              "speed_of_light": {...},
              "scheduling": {...},
              "memory": {...},
              "arith_intensity_flop_per_byte": float,
              "arith_intensity_flops": float,
              "arith_intensity_dram_bytes": float,
              "roofline_bound": str,
              "roofline_I_crit": float | None
            }
        """
        # Helper: read by canonical name
        def get_canon(key: str, default=None):
            res = self.get_values([key], name_kind="canonical")
            if res["results"]:
                return res["results"][0]["value"]
            return default

        # Helper: read by canonical and convert to GB/s if unit denotes bytes/sec
        def get_canon_gbps(key: str) -> Optional[float]:
            raw = CANONICAL_TO_NCU_METRICS.get(key)
            if not raw:
                return None
            val, unit = _metric_value(self.action, raw)
            if val is None:
                return None
            u = (unit or "").lower()
            if "byte" in u and "second" in u or u in ("b/s", "bytes/second"):
                return float(val) / 1e9
            return float(val)

        # Device name is often available via device__attribute_display_name
        device_name_val = None
        mi = self._metrics.get("device__attribute_display_name")
        if mi:
            device_name_val = mi.value

        summary = {
            "kernel_name": self.kernel_name,
            "device_name": device_name_val,
            "speed_of_light": {
                "sm_throughput_pct": get_canon("sm_throughput_pct"),
                "gpu_dram_throughput_pct": get_canon("gpu_dram_throughput_pct"),
                "compute_throughput_pct": get_canon("compute_throughput_pct"),
                "memory_throughput_pct": get_canon("memory_throughput_pct"),
            },
            "scheduling": {
                "warp_occupancy_pct": get_canon("warp_occupancy_pct"),
                "sm_active_cycles_pct": get_canon("sm_active_cycles_pct"),
                "instructions_per_cycle": get_canon("instructions_per_cycle"),
                "waves_per_sm": get_canon("waves_per_sm"),
            },
            "memory": {
                "dram_bw_gb_s": get_canon_gbps("dram_avg_bandwidth_gb_s"),
                "dram_active_cycles_pct": get_canon("dram_active_cycles_pct"),
                "l1_hit_rate_pct": get_canon("l1_hit_rate_pct"),
                "l2_hit_rate_pct": get_canon("l2_hit_rate_pct"),
                "l2_throughput_pct": get_canon("l2_throughput_pct"),
            },
        }

        # Arithmetic intensity + roofline
        try:
            ai_info = compute_arithmetic_intensity(self.raw_metrics)
            summary.update(ai_info)
            summary.update(compute_roofline_bound(ai_info["arith_intensity_flop_per_byte"], self.device_profile))
        except Exception:
            summary.update({
                "arith_intensity_flop_per_byte": None,
                "arith_intensity_flops": None,
                "arith_intensity_dram_bytes": None,
                "roofline_bound": "unknown",
                "roofline_I_crit": None,
            })
        return summary

    # ------------------------- Get values (mode = "get_values") -------------------

    def get_values(self, names: List[str], name_kind: Literal["canonical", "ncu", "auto"] = "auto") -> Dict[str, Any]:
        """
        Get metric values for requested metric names.
        """
        results: List[Dict[str, Any]] = []
        unresolved: List[str] = []

        for requested in names:
            resolved_name = None
            resolution = None

            if name_kind in ("canonical", "auto") and requested in CANONICAL_TO_NCU_METRICS:
                resolved_name = CANONICAL_TO_NCU_METRICS[requested]
                resolution = "canonical"

            if resolved_name is None and name_kind in ("ncu", "auto"):
                if requested in self._metrics:
                    resolved_name = requested
                    resolution = "ncu_exact"

            # Small fallback: substring search.
            if resolved_name is None and name_kind == "auto":
                matches = [
                    name for name in self._metrics.keys()
                    if requested.lower() in name.lower()
                ]
                if len(matches) == 1:
                    resolved_name = matches[0]
                    resolution = "ncu_substring"

            if resolved_name is None or resolved_name not in self._metrics:
                unresolved.append(requested)
                continue

            mi = self._metrics[resolved_name]
            results.append({
                "requested": requested,
                "resolution": resolution,
                "ncu_name": mi.name,
                "value": mi.value,
                "unit": mi.unit,
                "description": mi.description,
            })

        return {
            "results": results,
            "unresolved": unresolved,
        }

    # ------------------------- Search (mode = "search_names") ---------------------

    def search_names(self, query: str, max_results: int = 32) -> Dict[str, Any]:
        """
        Search metric names and descriptions for a substring.
        """
        q = query.lower()
        matches: List[Dict[str, Any]] = []
        for mi in self._metrics.values():
            if q in mi.name.lower() or q in mi.description.lower():
                matches.append({
                    "name": mi.name,
                    "unit": mi.unit,
                    "description": mi.description,
                })
                if len(matches) >= max_results:
                    break
        return {"query": query, "matches": matches}


# ---------------------------------------------------------------------------
# Tool registry + OpenAI Responses API integration
# ---------------------------------------------------------------------------

# ------------------------ OpenAI tools schemas ------------------------------

REGION_DAG_TOOL_SCHEMA = {
    "type": "function",
    "name": "region_dag_inspect",
    "description": (
        "Inspect the Region-DAG for the current kernel. "
        "Use this to understand stages, regions, and per-region/stage work profiles (FLOPs and bytes)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["overview", "stage_detail", "region_detail"],
                "description": (
                    "'overview' → global summary + hottest stages/regions; "
                    "'stage_detail' → details for one stage; "
                    "'region_detail' → details + PTX snippet for one region."
                ),
            },
            "stage_id": {
                "type": "integer",
                "description": "Required when mode == 'stage_detail'.",
            },
            "region_id": {
                "type": "integer",
                "description": "Required when mode == 'region_detail'.",
            },
            "max_hot_regions": {
                "type": "integer",
                "minimum": 1,
                "maximum": 64,
                "default": 8,
                "description": "Max number of hottest regions to include (overview).",
            },
            "max_hot_stages": {
                "type": "integer",
                "minimum": 1,
                "maximum": 16,
                "default": 4,
                "description": "Max number of hottest stages to include (overview).",
            },
            "max_insts": {
                "type": "integer",
                "minimum": 1,
                "maximum": 64,
                "default": 16,
                "description": "Max PTX lines in region_detail snippet.",
            },
        },
        "required": ["mode"],
    },
}

NCU_METRICS_TOOL_SCHEMA = {
    "type": "function",
    "name": "ncu_metrics_inspect",
    "description": (
        "Inspect Nsight Compute device metrics for the current kernel. "
        "Use 'summary' for canonical SoL/occupancy, 'get_values' for specific metrics, "
        "and 'search_names' to discover metric IDs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["summary", "get_values", "search_names"],
                "description": (
                    "'summary' → small canonical device summary; "
                    "'get_values' → query specific metrics; "
                    "'search_names' → substring search over metric names/descriptions."
                ),
            },
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Metric names to query when mode == 'get_values'. Can be canonical "
                    "names (e.g. 'warp_occupancy_pct') or raw Nsight Compute metric IDs."
                ),
            },
            "name_kind": {
                "type": "string",
                "enum": ["canonical", "ncu", "auto"],
                "default": "auto",
                "description": (
                    "How to interpret 'names' when mode == 'get_values': "
                    "'canonical' -> custom aliases; 'ncu' -> raw Nsight IDs; "
                    "'auto' -> canonical, then exact NCU, then substring."
                ),
            },
            "query": {
                "type": "string",
                "description": "Substring for mode == 'search_names'.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 128,
                "default": 32,
                "description": "Max results from 'search_names'.",
            },
        },
        "required": ["mode"],
    },
}

PTX_SOURCE_TOOL_SCHEMA = {
    "type": "function",
    "name": "get_ptx_by_source",
    "description": (
        "Return PTX (and optionally SASS) instructions for a given source span, "
        "correlated via an Nsight Compute report."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source_file": {
                "type": "string",
                "description": "Absolute path or path relative to working directory of the source file.",
            },
            "start_line": {
                "type": "integer",
                "description": "First (1-based) source line to include.",
            },
            "end_line": {
                "type": "integer",
                "description": "Last (1-based) source line to include.",
            },
            "max_insts": {
                "type": "integer",
                "minimum": 1,
                "maximum": 256,
                "default": 64,
                "description": "Maximum number of PTX instructions to return.",
            },
            "include_sass": {
                "type": "boolean",
                "default": False,
                "description": "Whether to include SASS alongside PTX.",
            },
            "include_metric_value": {
                "type": "boolean",
                "default": False,
                "description": "Whether to include per-PC metric values if available.",
            },
        },
        "required": ["source_file", "start_line", "end_line"],
    },
}

APPLY_PATCH_TOOL = {"type": "apply_patch"}

TOOLS = [
    REGION_DAG_TOOL_SCHEMA,
    NCU_METRICS_TOOL_SCHEMA,
    PTX_SOURCE_TOOL_SCHEMA,
    APPLY_PATCH_TOOL,
]

# ------------------------ Tool registry + dispatcher -------------------------

ToolHandler = Callable[[Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    """
    Simple mapping from tool name → Python handler.
    """

    def __init__(self):
        self._handlers: Dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler):
        self._handlers[name] = handler

    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._handlers:
            return {"error": f"unknown tool: {name}"}
        return self._handlers[name](args or {})


# ----------------------- Global contexts (to be set by caller) ---------------

CURRENT_REGION_DAG_CTX: Optional[RegionDagContext] = None
CURRENT_NCU_CTX: Optional[NcuMetricsContext] = None
CURRENT_PTX_CORRELATOR: Optional[PtxSourceCorrelator] = None


# ----------------------- Tool handlers for our three tools -------------------

def region_dag_inspect_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implementation of the 'region_dag_inspect' tool.
    """
    ctx = CURRENT_REGION_DAG_CTX
    if ctx is None:
        return {"error": "RegionDagContext is not initialized (CURRENT_REGION_DAG_CTX is None)."}

    mode = args.get("mode")
    if mode == "overview":
        return ctx.overview(
            max_hot_regions=int(args.get("max_hot_regions", 8)),
            max_hot_stages=int(args.get("max_hot_stages", 4)),
        )
    elif mode == "stage_detail":
        stage_id = args.get("stage_id")
        if stage_id is None:
            return {"error": "stage_id is required for mode 'stage_detail'."}
        return ctx.stage_detail(int(stage_id))
    elif mode == "region_detail":
        region_id = args.get("region_id")
        if region_id is None:
            return {"error": "region_id is required for mode 'region_detail'."}
        max_insts = int(args.get("max_insts", 16))
        return ctx.region_detail(int(region_id), max_insts=max_insts)
    else:
        return {"error": f"unknown mode for region_dag_inspect: {mode}"}


def ncu_metrics_inspect_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implementation of the 'ncu_metrics_inspect' tool.
    """
    ctx = CURRENT_NCU_CTX
    if ctx is None:
        return {"error": "NcuMetricsContext is not initialized (CURRENT_NCU_CTX is None)."}

    mode = args.get("mode")
    if mode == "summary":
        return ctx.summary()
    elif mode == "get_values":
        names = args.get("names") or []
        name_kind = args.get("name_kind", "auto")
        return ctx.get_values(names, name_kind=name_kind)
    elif mode == "search_names":
        query = args.get("query")
        if not query:
            return {"error": "query is required for mode 'search_names'."}
        max_results = int(args.get("max_results", 32))
        return ctx.search_names(query, max_results=max_results)
    else:
        return {"error": f"unknown mode for ncu_metrics_inspect: {mode}"}


def get_ptx_by_source_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implementation of the 'get_ptx_by_source' tool.
    """
    ctx = CURRENT_PTX_CORRELATOR
    if ctx is None:
        return {"error": "PtxSourceCorrelator is not initialized (CURRENT_PTX_CORRELATOR is None)."}

    source_file = args.get("source_file")
    start_line = args.get("start_line")
    end_line = args.get("end_line")
    if source_file is None or start_line is None or end_line is None:
        return {"error": "source_file, start_line, end_line are required."}

    max_insts = int(args.get("max_insts", 64))
    include_sass = bool(args.get("include_sass", False))
    include_metric_value = bool(args.get("include_metric_value", False))

    return ctx.get_ptx_snippet_for_source_span(
        nvtx_range=None,
        kernel_name=None,
        source_file=source_file,
        start_line=int(start_line),
        end_line=int(end_line),
        max_insts=max_insts,
        include_sass=include_sass,
        include_metric_value=include_metric_value,
        extra_metric_names=None,
    )


# Register handlers into a default registry
DEFAULT_TOOL_REGISTRY = ToolRegistry()
DEFAULT_TOOL_REGISTRY.register("region_dag_inspect", region_dag_inspect_tool)
DEFAULT_TOOL_REGISTRY.register("ncu_metrics_inspect", ncu_metrics_inspect_tool)
DEFAULT_TOOL_REGISTRY.register("get_ptx_by_source", get_ptx_by_source_tool)


# ----------------------- OpenAI Responses API integration --------------------

LLM_SYSTEM_PROMPT = """
You are an expert GPU performance engineer optimizing Triton/CUDA kernels in a real git repository.

== Context you receive ==
1) ncu_signals: Nsight Compute metrics for THIS kernel.
   • These are the ground truth for bottlenecks: SoL throughput, stalls, occupancy, etc.

2) region_summary: a compact summary of a Region-DAG built from PTX.
   • Regions are contiguous PTX segments with a phase label
     (mem_param_load, mem_global_load, mem_shared_load, mem_async_copy, async_wait,
      barrier, compute, global_store, shared_store, atomic, control, addr_arith, other).
   • Stages group regions between hard synchronization points (barriers / async waits).
   • Each region and stage has a static work profile: FLOPs, global/shared bytes,
     instruction counts, and atomics.

3) source_code: the CURRENT version of the kernel from the working copy on disk,
   plus an example host invocation if available.

== Tools available ==
You have these tools via the Responses API:

1) region_dag_inspect(mode=..., ...)
   • Inspect pipeline stages and regions, and their static work profiles.

2) ncu_metrics_inspect(mode=..., ...)
   • Inspect Nsight Compute metrics (summary, specific metrics, or metric-name search).

3) get_ptx_by_source(source_file, start_line, end_line, ...)
   • Inspect PTX/SASS around specific source lines.

4) apply_patch  (BUILT-IN)
   • This is your ONLY way to create, update, move, or delete files in the repository.
   • When you need to change code, emit apply_patch_call operations with small, coherent diffs
     targeting the real source files in the working tree.
   • Never assume code has changed unless your apply_patch_call_output status is "completed".

== Workflow ==
1) Read ncu_signals, region_summary, and source_code.
2) Form a hypothesis about the bottleneck:
   • memory-bound vs compute-bound vs latency/scoreboard vs occupancy vs launch-config issues.
3) Use tools as needed:
   • region_dag_inspect to localize hot stages/regions.
   • ncu_metrics_inspect for detailed stall/utilization metrics.
   • get_ptx_by_source to inspect PTX/SASS around hot loops or suspicious address arithmetic.
4) Once you have a clear hypothesis, use apply_patch to modify the kernel (and, if needed,
   closely related launch/config code) in the working tree. Keep each patch surgical:
   • Prefer retuning block sizes, tiling, async pipelines, prefetching, or minor layout changes
     over sweeping rewrites.
   • Focus patches on the diagnosed hot path.
5) After all tool calls (including all apply_patch operations) are finished, respond with STRICT JSON
   describing the best candidate(s) you actually implemented via apply_patch:

{
  "candidates": [{
    "think": "Short reasoning about what you changed and why it should help (concise).",
    "method": "Short name for the optimization method (e.g., 'retune_block_sizes', 'deepen_async_pipeline').",
    "detail": "Precise description of the code changes you made via apply_patch: which files, which kernel
               parameters or loops you edited, and how that affects performance.",
    "code": "A SHORT illustrative code excerpt (not a diff) showing the key changed loop or kernel,
             consistent with the edits you applied."
  }]
}

If you decide you cannot safely improve the kernel, return:
{
  "candidates": []
}

== Constraints ==
• Do NOT print diffs in the JSON. Real edits must be done via apply_patch; the JSON is a human-readable summary only.
• Do NOT rely on the JSON alone to change code; only files modified via apply_patch are real.
• Output ONLY valid JSON as described (no markdown, no backticks, no extra commentary).
"""

def _collect_text_outputs(resp) -> str:
    """
    Collect any plain text content from response.output items.
    """
    segs: List[str] = []
    for item in getattr(resp, "output", []) or []:
        content_list = getattr(item, "content", None)
        if not content_list:
            continue
        for c in content_list:
            text = getattr(c, "text", None)
            if isinstance(text, str):
                segs.append(text)
    return "\n".join(segs).strip()


def _maybe_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to parse a JSON object from the given text, ignoring common ```json fences.
    """
    try:
        s = text.strip()
        s = re.sub(r"^```(?:json)?", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
        return json.loads(s)
    except Exception:
        return None


def extract_function_and_patch_calls(resp) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract function-tool calls (region_dag_inspect / ncu_metrics_inspect / get_ptx_by_source)
    and built-in apply_patch calls from a Responses API response.

    Returns:
        (function_calls, patch_calls)

        function_calls: [{ "name": str, "arguments": dict, "call_id": str }]
        patch_calls:    [{ "operation": dict, "call_id": str }]
    """
    fn_calls: List[Dict[str, Any]] = []
    patch_calls: List[Dict[str, Any]] = []

    def _as_dict(obj: Any) -> Any:
        try:
            return obj.model_dump()  # type: ignore
        except Exception:
            try:
                return obj.to_dict()  # type: ignore
            except Exception:
                return obj

    for item in getattr(resp, "output", []) or []:
        t = getattr(item, "type", None)

        if t == "function_call":
            name = getattr(item, "name", None)
            args = getattr(item, "arguments", None)
            call_id = getattr(item, "id", None) or getattr(item, "call_id", None)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if name:
                fn_calls.append({"name": name, "arguments": args or {}, "call_id": call_id})
            continue

        if t == "apply_patch_call":
            op = _as_dict(getattr(item, "operation", None))
            call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
            if op is not None and call_id is not None:
                patch_calls.append({"operation": op, "call_id": call_id})
            continue

        for c in getattr(item, "content", []) or []:
            ct = getattr(c, "type", None)
            if ct == "function_call":
                name = getattr(c, "name", None)
                args = getattr(c, "arguments", None)
                call_id = getattr(c, "id", None) or getattr(c, "call_id", None)
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                if name:
                    fn_calls.append({"name": name, "arguments": args or {}, "call_id": call_id})
                continue

            if ct == "apply_patch_call":
                op = _as_dict(getattr(c, "operation", None))
                call_id = getattr(c, "call_id", None) or getattr(c, "id", None)
                if op is not None and call_id is not None:
                    patch_calls.append({"operation": op, "call_id": call_id})

    return fn_calls, patch_calls


class LLMCandidateGenerator:
    """
    Wraps the OpenAI Responses API to:
      - Provide the system prompt and initial kernel context.
      - Provide the tool schemas (TOOLS).
      - Run a tool loop until the model returns final STRICT JSON.
    """

    def __init__(self, model: str, api_base: Optional[str] = None, tool_registry: Optional[ToolRegistry] = None):
        self.model = model
        self.api_base = api_base
        self.client = None
        self.prev_id_by_kernel: Dict[str, str] = {}
        base_tools = TOOLS.copy()
        self._supports_apply_patch = model.startswith("gpt-5.1")
        self.tools = base_tools
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY
        # Assumes kernel_opt_tooling.py lives at repo root. Update if moved.
        self.repo_root = Path(__file__).resolve().parent

    def _lazy_client(self):
        if self.client is None:
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError("OpenAI SDK not installed. `pip install openai`") from e
            self.client = OpenAI(base_url=self.api_base) if self.api_base else OpenAI()

    def _mk_messages(self, kernel: Any, region_summary: Optional[Dict[str, Any]] = None, ncu_summary: Optional[Dict[str, Any]] = None, workdir_path: Optional[str] = None, source_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Build the initial "input" messages for the Responses API.

        Expected kernel object shape (adapt as needed):
            kernel.name: str
            kernel.source_code: str
            kernel.invocation_example: str (or "")
        """
        sys_block = {
            "role": "system",
            "content": [
                {"type": "input_text", "text": LLM_SYSTEM_PROMPT},
            ],
        }

        content_blocks: List[Dict[str, Any]] = [
            {"type": "input_text", "text": "KERNEL NAME:\n" + str(getattr(kernel, "name", "unknown_kernel"))},
            {"type": "input_text", "text": "KERNEL SOURCE:\n" + (getattr(kernel, "source_code", "") or "")},
            {"type": "input_text", "text": "INVOCATION_EXAMPLE:\n" + (getattr(kernel, "invocation_example", "") or "")},
            {"type": "input_text", "text": "NCU SUMMARY (GROUND TRUTH):\n" + json.dumps(ncu_summary or {}, indent=2)},
            {"type": "input_text", "text": "REGION SUMMARY (STATIC STRUCTURE & WORK PROFILE):\n" + json.dumps(region_summary or {}, indent=2)},
            {"type": "input_text", "text": "WORKDIR (PATCH ROOT):\n" + str(workdir_path or self.repo_root)},
            {"type": "input_text", "text": "Output STRICT JSON only per spec. No markdown."},
        ]
        if source_path:
            content_blocks.insert(
                2,
                {"type": "input_text", "text": "KERNEL SOURCE PATH:\n" + str(source_path)},
            )

        user_payload = {
            "role": "user",
            "content": content_blocks,
        }

        return [sys_block, user_payload]

    def _send(self, *, input_payload: Optional[List[Dict[str, Any]]] = None, previous_response_id: Optional[str] = None, tool_outputs: Optional[List[Dict[str, Any]]] = None):
        """
        Thin wrapper around client.responses.create, supporting tool_outputs.
        """
        kwargs: Dict[str, Any] = dict(model=self.model, tools=self.tools, input=(input_payload or []))
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        if tool_outputs:
            kwargs["tool_outputs"] = tool_outputs
        return self.client.responses.create(**kwargs)

    def _send_tool_result(self, *, prev_id: str, name: str, call_id: Optional[str], result_obj: Dict[str, Any]):
        """
        Send function-tool results back as a `function_call_output` event.
        """
        if not call_id:
            call_id = f"{name}_no_id"

        event = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result_obj),
        }
        return self._send(input_payload=[event], previous_response_id=prev_id)

    def _send_apply_patch_result(self, *, prev_id: str, call_id: str, status: str, log_output: str):
        """
        Send apply_patch execution result back as an `apply_patch_call_output` event.
        """
        event = {
            "type": "apply_patch_call_output",
            "call_id": call_id,
            "status": status,
            "output": log_output,
        }
        return self._send(input_payload=[event], previous_response_id=prev_id)

    def _run_apply_patch_operation(self, workdir_path: str, operation: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Apply a single apply_patch operation to the working tree.

        operation: { "type": "update_file"|"create_file"|"delete_file", "diff": "...", "path": "..." }
        Returns (success, log_output).
        """
        op = operation or {}
        diff_raw = op.get("diff", "")
        repo = Path(workdir_path) if workdir_path else self.repo_root
        if not diff_raw:
            return False, "No diff provided"

        # Normalize path for patching; prefer relative to workdir when possible.
        path_raw = op.get("path") or ""
        rel_path = ""
        if path_raw:
            p = Path(path_raw)
            try:
                rel_path = str(p.relative_to(repo))
            except Exception:
                rel_path = p.name if p.name else str(p)

        def _wrap_if_needed(text: str) -> str:
            """
            If the diff is missing headers (common with apply_patch_call),
            wrap it in the cookbook *** Begin Patch / *** Update File format
            so process_patch can apply it.
            """
            stripped = text.lstrip()
            if stripped.startswith("*** Begin Patch") or stripped.startswith("--- ") or stripped.startswith("diff --git"):
                return text
            if rel_path and "@@" in text:
                return f"*** Begin Patch\n*** Update File: {rel_path}\n{text.rstrip()}\n*** End Patch\n"
            return text

        diff = _wrap_if_needed(diff_raw)

        if diff.strip().startswith("*** Begin Patch"):
            try:
                def open_file(path: str) -> str:
                    with open(repo / path, "rt", encoding="utf-8") as f:
                        return f.read()

                def write_file(path: str, content: str) -> None:
                    tgt = repo / path
                    tgt.parent.mkdir(parents=True, exist_ok=True)
                    with open(tgt, "wt", encoding="utf-8") as f:
                        f.write(content)

                def remove_file(path: str) -> None:
                    tgt = repo / path
                    if tgt.exists():
                        tgt.unlink()

                result = process_patch(diff, open_file, write_file, remove_file)
                return True, result
            except DiffError as e:
                return False, f"Patch failed: {e}"
            except Exception as e:
                snippet = diff[:200].replace("\n", "\\n")
                return False, f"Patch exception: {e}; diff_head={snippet}"

        patch_bin = shutil.which("patch")
        if not patch_bin:
            return False, "patch binary not found on PATH"
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf:
                tf.write(diff if diff.endswith("\n") else diff + "\n")
                tf_path = tf.name
            proc = subprocess.run(
                [patch_bin, "-p0", "-i", tf_path],
                cwd=str(repo),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            log = (proc.stdout or "") + (proc.stderr or "")
            os.unlink(tf_path)
            if proc.returncode != 0:
                return False, f"patch failed rc={proc.returncode}: {log}"
            return True, f"patch applied. {log.strip()}"
        except Exception as e:
            snippet = diff[:200].replace("\n", "\\n")
            return False, f"Patch exception: {e}; diff_head={snippet}"

    # -------- NEW: relabel_methods (Algorithm 1 “relabel” step) ----------
    def relabel_methods(self, methods_catalog: List[str], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map each candidate['method'] to a canonical name in methods_catalog (O) if it’s a synonym;
        otherwise, propose a new short canonical name. Returns a list with:
            { "canonical": str, "existed": bool }
        """
        self._lazy_client()
        catalog = methods_catalog or []
        items = [{"method": (c.get("method") or "").strip(), "detail": (c.get("detail") or "").strip()} for c in (candidates or [])]

        relabel_system = (
            "You are a strict normalizer of optimization method names.\n"
            "Given a catalog O of canonical method names and a list of new steps, "
            "map each step to an EXISTING name in O if it is a synonym (same operation), "
            "otherwise propose a NEW short canonical name (lowercase snake_case, 2-6 words). "
            "Respond STRICT JSON ONLY as a list with objects: {canonical, existed}."
        )
        relabel_user = {
            "catalog_O": catalog,
            "steps": items
        }
        msgs = [
            {"role": "system", "content": [{"type":"input_text","text": relabel_system}]},
            {"role": "user", "content": [{"type":"input_text","text": json.dumps(relabel_user)}]}
        ]
        try:
            resp = self.client.responses.create(model=self.model, input=msgs)
            text = _collect_text_outputs(resp)
            out = _maybe_json(text)
            if isinstance(out, list):
                # validate items (canonical string)
                cleaned: List[Dict[str, Any]] = []
                for obj in out:
                    can = (obj.get("canonical") or "").strip()
                    ex  = bool(obj.get("existed", False))
                    if not can:
                        can = ""
                    cleaned.append({"canonical": can, "existed": ex})
                if len(cleaned) == len(items):
                    return cleaned
        except Exception:
            pass
        # Fallback: identity mapping
        return [{"canonical": (c.get("method") or "opt").strip(), "existed": ((c.get("method") or "").strip() in catalog)} for c in (candidates or [])]

    def propose(
        self,
        kernel: Any,
        region_summary: Optional[Dict[str, Any]] = None,
        ncu_summary: Optional[Dict[str, Any]] = None,
        workdir_path: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Main entrypoint.

        Returns parsed JSON dict (or None on failure).
        """
        self._lazy_client()

        messages = self._mk_messages(kernel, region_summary, ncu_summary, workdir_path, source_path)
        kernel_key = getattr(kernel, "name", "kernel")
        prev_id: Optional[str] = self.prev_id_by_kernel.get(kernel_key)

        logger.debug(f"[LLM] propose start kernel={kernel_key} model={self.model} prev_id={prev_id}")

        resp = self._send(
            input_payload=messages if not prev_id else [],
            previous_response_id=prev_id,
        )
        self.prev_id_by_kernel[kernel_key] = resp.id

        MAX_TURNS = 8
        for _ in range(MAX_TURNS):
            fn_calls, patch_calls = extract_function_and_patch_calls(resp)

            if fn_calls:
                logger.debug(f"[LLM] function_calls detected (count={len(fn_calls)}) for kernel={kernel_key}")
                for tc in fn_calls:
                    out = self.tool_registry.dispatch(tc["name"], tc.get("arguments") or {})
                    resp = self._send_tool_result(
                        prev_id=resp.id,
                        name=tc["name"],
                        call_id=tc.get("call_id"),
                        result_obj=out,
                    )
                    self.prev_id_by_kernel[kernel_key] = resp.id
                continue

            if patch_calls and self._supports_apply_patch:
                if not workdir_path:
                    raise RuntimeError("apply_patch_call received but workdir_path is None")

                for pc in patch_calls:
                    op = pc["operation"]
                    call_id = pc["call_id"]
                    ok, logmsg = self._run_apply_patch_operation(workdir_path=workdir_path, operation=op)
                    resp = self._send_apply_patch_result(
                        prev_id=resp.id,
                        call_id=call_id,
                        status="completed" if ok else "failed",
                        log_output=logmsg,
                    )
                    self.prev_id_by_kernel[kernel_key] = resp.id
                continue

            raw_text = _collect_text_outputs(resp)
            logger.debug(f"[LLM] raw_text len={len(raw_text)} for kernel={kernel_key}")
            parsed = _maybe_json(raw_text)
            if isinstance(parsed, dict):
                logger.debug(f"[LLM] parsed JSON with keys={list(parsed.keys())} for kernel={kernel_key}")
                return parsed

            reminder = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Please resend STRICT JSON per the output schema.",
                        }
                    ],
                }
            ]
            resp = self._send(input_payload=reminder, previous_response_id=resp.id)
            self.prev_id_by_kernel[kernel_key] = resp.id

        logger.warning(f"[LLM] propose failed to return JSON for kernel={kernel_key}")
        return None


# ----------------------- apply_patch cookbook parser -----------------------

class ActionType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


class FileChange(BaseModel):
    type: ActionType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    move_path: Optional[str] = None


class Commit(BaseModel):
    changes: dict[str, FileChange] = Field(default_factory=dict)


class Chunk(BaseModel):
    orig_index: int = -1
    del_lines: list[str] = Field(default_factory=list)
    ins_lines: list[str] = Field(default_factory=list)


class PatchAction(BaseModel):
    type: ActionType
    new_file: Optional[str] = None
    chunks: list[Chunk] = Field(default_factory=list)
    move_path: Optional[str] = None


class Patch(BaseModel):
    actions: dict[str, PatchAction] = Field(default_factory=dict)


class DiffError(ValueError):
    pass


class Parser(BaseModel):
    current_files: dict[str, str] = Field(default_factory=dict)
    lines: list[str] = Field(default_factory=list)
    index: int = 0
    patch: Patch = Field(default_factory=Patch)
    fuzz: int = 0

    def is_done(self, prefixes: Optional[tuple[str, ...]] = None) -> bool:
        if self.index >= len(self.lines):
            return True
        if prefixes and self.lines[self.index].startswith(prefixes):
            return True
        return False

    def startswith(self, prefix: Optional[tuple[str, ...]]) -> bool:
        assert self.index < len(self.lines)
        return self.lines[self.index].startswith(prefix)

    def read_str(self, prefix: str = "", return_everything: bool = False) -> str:
        assert self.index < len(self.lines)
        if self.lines[self.index].startswith(prefix):
            text = self.lines[self.index] if return_everything else self.lines[self.index][len(prefix):]
            self.index += 1
            return text
        return ""

    def parse(self):
        while not self.is_done(("*** End Patch",)):
            path = self.read_str("*** Update File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Update File Error: Duplicate Path: {path}")
                move_to = self.read_str("*** Move to: ")
                if path not in self.current_files:
                    raise DiffError(f"Update File Error: Missing File: {path}")
                text = self.current_files[path]
                action = self.parse_update_file(text)
                action.move_path = move_to
                self.patch.actions[path] = action
                continue
            path = self.read_str("*** Delete File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Delete File Error: Duplicate Path: {path}")
                if path not in self.current_files:
                    raise DiffError(f"Delete File Error: Missing File: {path}")
                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue
            path = self.read_str("*** Add File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Add File Error: Duplicate Path: {path}")
                self.patch.actions[path] = self.parse_add_file()
                continue
            raise DiffError(f"Unknown Line: {self.lines[self.index]}")
        if not self.startswith(("*** End Patch",)):
            raise DiffError("Missing End Patch")
        self.index += 1

    def parse_update_file(self, text: str) -> PatchAction:
        action = PatchAction(type=ActionType.UPDATE)
        lines = text.split("\n")
        index = 0
        while not self.is_done((
            "*** End Patch",
            "*** Update File:",
            "*** Delete File:",
            "*** Add File:",
            "*** End of File",
        )):
            def_str = self.read_str("@@ ")
            section_str = ""
            if not def_str and self.lines[self.index] == "@@":
                section_str = self.lines[self.index]
                self.index += 1
            if not (def_str or section_str or index == 0):
                raise DiffError(f"Invalid Line:\n{self.lines[self.index]}")
            if def_str.strip():
                found = False
                if not [s for s in lines[:index] if s == def_str]:
                    for i, s in enumerate(lines[index:], index):
                        if s == def_str:
                            index = i + 1
                            found = True
                            break
                if not found and not [s for s in lines[:index] if s.strip() == def_str.strip()]:
                    for i, s in enumerate(lines[index:], index):
                        if s.strip() == def_str.strip():
                            index = i + 1
                            self.fuzz += 1
                            found = True
                            break
            next_chunk_context, chunks, end_patch_index, eof = peek_next_section(self.lines, self.index)
            new_index, fuzz = find_context(lines, next_chunk_context, index, eof)
            if new_index == -1:
                raise DiffError(f"Invalid Context {index}:\n" + "\n".join(next_chunk_context))
            self.fuzz += fuzz
            for ch in chunks:
                ch.orig_index += new_index
                action.chunks.append(ch)
            index = new_index + len(next_chunk_context)
            self.index = end_patch_index
        return action

    def parse_add_file(self) -> PatchAction:
        lines = []
        while not self.is_done((
            "*** End Patch",
            "*** Update File:",
            "*** Delete File:",
            "*** Add File:",
        )):
            s = self.read_str()
            if not s.startswith("+"):
                raise DiffError(f"Invalid Add File Line: {s}")
            s = s[1:]
            lines.append(s)
        return PatchAction(type=ActionType.ADD, new_file="\n".join(lines))


def find_context_core(lines: list[str], context: list[str], start: int) -> tuple[int, int]:
    if not context:
        return start, 0
    for i in range(start, len(lines)):
        if lines[i : i + len(context)] == context:
            return i, 0
    for i in range(start, len(lines)):
        if [s.rstrip() for s in lines[i : i + len(context)]] == [s.rstrip() for s in context]:
            return i, 1
    for i in range(start, len(lines)):
        if [s.strip() for s in lines[i : i + len(context)]] == [s.strip() for s in context]:
            return i, 100
    return -1, 0


def find_context(lines: list[str], context: list[str], start: int, eof: bool) -> tuple[int, int]:
    if eof:
        new_index, fuzz = find_context_core(lines, context, len(lines) - len(context))
        if new_index != -1:
            return new_index, fuzz
        new_index, fuzz = find_context_core(lines, context, start)
        return new_index, fuzz + 10000
    return find_context_core(lines, context, start)


def peek_next_section(lines: list[str], index: int) -> tuple[list[str], list[Chunk], int, bool]:
    old: list[str] = []
    del_lines: list[str] = []
    ins_lines: list[str] = []
    chunks: list[Chunk] = []
    mode = "keep"
    orig_index = index
    while index < len(lines):
        s = lines[index]
        if s.startswith(("@@", "*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:", "*** End of File")):
            break
        if s == "***":
            break
        elif s.startswith("***"):
            raise DiffError(f"Invalid Line: {s}")
        index += 1
        last_mode = mode
        if s == "":
            s = " "
        if s[0] == "+":
            mode = "add"
        elif s[0] == "-":
            mode = "delete"
        elif s[0] == " ":
            mode = "keep"
        else:
            raise DiffError(f"Invalid Line: {s}")
        s = s[1:]
        if mode == "keep" and last_mode != mode:
            if ins_lines or del_lines:
                chunks.append(Chunk(orig_index=len(old) - len(del_lines), del_lines=del_lines, ins_lines=ins_lines))
            del_lines = []
            ins_lines = []
        if mode == "delete":
            del_lines.append(s)
            old.append(s)
        elif mode == "add":
            ins_lines.append(s)
        elif mode == "keep":
            old.append(s)
    if ins_lines or del_lines:
        chunks.append(Chunk(orig_index=len(old) - len(del_lines), del_lines=del_lines, ins_lines=ins_lines))
    if index < len(lines) and lines[index] == "*** End of File":
        index += 1
        return old, chunks, index, True
    if index == orig_index:
        raise DiffError(f"Nothing in this section - {index=} {lines[index]}")
    return old, chunks, index, False


def text_to_patch(text: str, orig: dict[str, str]) -> tuple[Patch, int]:
    lines = text.strip().split("\n")
    if len(lines) < 2 or not lines[0].startswith("*** Begin Patch") or lines[-1] != "*** End Patch":
        raise DiffError("Invalid patch text")
    parser = Parser(current_files=orig, lines=lines, index=1)
    parser.parse()
    return parser.patch, parser.fuzz


def identify_files_needed(text: str) -> list[str]:
    lines = text.strip().split("\n")
    result = set()
    for line in lines:
        if line.startswith("*** Update File: "):
            result.add(line[len("*** Update File: ") :])
        if line.startswith("*** Delete File: "):
            result.add(line[len("*** Delete File: ") :])
    return list(result)


def _get_updated_file(text: str, action: PatchAction, path: str) -> str:
    assert action.type == ActionType.UPDATE
    orig_lines = text.split("\n")
    dest_lines: list[str] = []
    orig_index = 0
    dest_index = 0
    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(f"_get_updated_file: {path}: chunk.orig_index {chunk.orig_index} > len(lines) {len(orig_lines)}")
        if orig_index > chunk.orig_index:
            raise DiffError(f"_get_updated_file: {path}: orig_index {orig_index} > chunk.orig_index {chunk.orig_index}")
        dest_lines.extend(orig_lines[orig_index : chunk.orig_index])
        delta = chunk.orig_index - orig_index
        orig_index += delta
        dest_index += delta
        if chunk.ins_lines:
            dest_lines.extend(chunk.ins_lines)
            dest_index += len(chunk.ins_lines)
        orig_index += len(chunk.del_lines)
    dest_lines.extend(orig_lines[orig_index:])
    delta = len(orig_lines) - orig_index
    orig_index += delta
    dest_index += delta
    assert orig_index == len(orig_lines)
    assert dest_index == len(dest_lines)
    return "\n".join(dest_lines)


def patch_to_commit(patch: Patch, orig: dict[str, str]) -> Commit:
    commit = Commit()
    for path, action in patch.actions.items():
        if action.type == ActionType.DELETE:
            commit.changes[path] = FileChange(type=ActionType.DELETE, old_content=orig[path])
        elif action.type == ActionType.ADD:
            commit.changes[path] = FileChange(type=ActionType.ADD, new_content=action.new_file)
        elif action.type == ActionType.UPDATE:
            new_content = _get_updated_file(text=orig[path], action=action, path=path)
            commit.changes[path] = FileChange(
                type=ActionType.UPDATE,
                old_content=orig[path],
                new_content=new_content,
                move_path=action.move_path,
            )
    return commit


def load_files(paths: list[str], open_fn: Callable) -> dict[str, str]:
    orig = {}
    for path in paths:
        orig[path] = open_fn(path)
    return orig


def apply_commit(commit: Commit, write_fn: Callable, remove_fn: Callable) -> None:
    for path, change in commit.changes.items():
        if change.type == ActionType.DELETE:
            remove_fn(path)
        elif change.type == ActionType.ADD:
            write_fn(path, change.new_content)
        elif change.type == ActionType.UPDATE:
            if change.move_path:
                write_fn(change.move_path, change.new_content)
                remove_fn(path)
            else:
                write_fn(path, change.new_content)


def process_patch(text: str, open_fn: Callable, write_fn: Callable, remove_fn: Callable) -> str:
    assert text.startswith("*** Begin Patch")
    paths = identify_files_needed(text)
    orig = load_files(paths, open_fn)
    patch, _ = text_to_patch(text, orig)
    commit = patch_to_commit(patch, orig)
    apply_commit(commit, write_fn, remove_fn)
    return "apply_patch completed"

# ---------------------------------------------------------------------------
# Example usage (skeleton)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("This module is meant to be imported into your optimization pipeline.")
    logger.info("See docstring and comments for integration details.")
