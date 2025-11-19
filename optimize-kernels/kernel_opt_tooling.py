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

import dataclasses
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal

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

try:
    from ptx_dag_tool_v2 import build_all, Region, Stage  # type: ignore
except Exception as e:
    logger.warning("Could not import ptx_dag_tool_v2: %s", e)
    build_all = None  # type: ignore
    Region = None     # type: ignore
    Stage = None      # type: ignore


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

    def __init__(self, report_path: str, range_idx: int = 0, action_idx: int = 0):
        if not NCU_REPORT_AVAILABLE:
            raise RuntimeError("ncu_report is not available; Nsight Compute must be installed.")

        self.report_path = report_path
        self.report = ncu_report.load_report(report_path)  # type: ignore
        self.range = self.report.range_by_idx(range_idx)
        self.action = self.range.action_by_idx(action_idx)
        self.kernel_name = self.action.name()

        # Cache all metrics.
        self._metrics: Dict[str, MetricInfo] = self._collect_metrics()

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
              "memory": {...}
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
# PTX↔source correlator + snippet API
# ---------------------------------------------------------------------------

class PtxSourceCorrelator:
    """
    Correlate PTX (and optionally SASS) to source lines using a .ncu-rep file.

    The core API we care about for tool usage is:
        get_ptx_snippet_for_source_span(...)
    """

    PREFERRED_BASE_METRICS: List[str] = [
        "inst_executed",
        "thread_inst_executed",
        "thread_inst_executed_true",
        "derived__avg_thread_executed",
        "smsp__pcsamp_sample_count",
    ]

    EXCLUDED_BASE_PREFIXES: Tuple[str, ...] = (
        "launch__",
        "device__",
        "numa__",
        "nvlink__",
        "profiler__",
        "pmsampling:dramc__",
    )

    def __init__(self, report_path: Path):
        self.report_path = Path(report_path)
        self._report = None

        self.available = bool(
            NCU_REPORT_AVAILABLE and self.report_path.exists()
        )

        if not NCU_REPORT_AVAILABLE:
            logger.debug("ncu_report API not available in PtxSourceCorrelator.")
        elif not self.report_path.exists():
            logger.warning("NCU report file not found for PtxSourceCorrelator: %s", self.report_path)

    # --------------------- Public: snippet for LLM tool -------------------------

    def get_ptx_snippet_for_source_span(
        self,
        nvtx_range: Optional[str],
        kernel_name: Optional[str],
        source_file: str,
        start_line: int,
        end_line: int,
        max_insts: int = 64,
        include_sass: bool = False,
        include_metric_value: bool = False,
        extra_metric_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        High-level, LLM-friendly API that returns a correlated PTX/SASS snippet.
        """
        if not self.available:
            return {
                "source_file": source_file,
                "start_line": start_line,
                "end_line": end_line,
                "total_instructions": 0,
                "truncated": False,
                "lines": [],
                "error": "PtxSourceCorrelator not available (no report or ncu_report).",
            }

        mapping = self.correlate_ptx_to_source(
            nvtx_range=nvtx_range,
            kernel_name=kernel_name,
            base_metric_name=None,
            include_sass=include_sass,
            include_metric_value=include_metric_value,
            source_file_filter=source_file,
            extra_metric_names=extra_metric_names,
        )

        if source_file not in mapping:
            return {
                "source_file": source_file,
                "start_line": start_line,
                "end_line": end_line,
                "total_instructions": 0,
                "truncated": False,
                "lines": [],
            }

        file_map = mapping[source_file]

        # Load source text.
        try:
            with open(source_file, "r", encoding="utf-8", errors="replace") as f:
                src_lines = f.read().splitlines()
        except Exception:
            src_lines = []

        def get_src_line(ln: int) -> str:
            if 1 <= ln <= len(src_lines):
                return src_lines[ln - 1]
            return ""

        lines_out: List[Dict[str, Any]] = []
        total_insts = 0
        remaining = max_insts
        truncated = False

        for line_no in sorted(file_map.keys()):
            if line_no < start_line or line_no > end_line:
                continue
            if remaining <= 0:
                truncated = True
                break

            entries = file_map[line_no]
            entries_sorted = sorted(entries, key=lambda e: e.get("pc", 0))

            line_entries_out: List[Dict[str, Any]] = []
            for entry in entries_sorted:
                if remaining <= 0:
                    truncated = True
                    break

                pc = entry.get("pc")
                ptx = (entry.get("ptx") or "").rstrip()
                sass = (entry.get("sass") or "").rstrip() if include_sass else None
                metrics = entry.get("metrics") or {} if include_metric_value else {}

                line_entries_out.append(
                    {
                        "pc": f"0x{pc:x}" if isinstance(pc, int) else None,
                        "ptx": ptx,
                        "sass": sass,
                        "metrics": metrics,
                    }
                )
                total_insts += 1
                remaining -= 1

            if line_entries_out:
                lines_out.append(
                    {
                        "line": line_no,
                        "source": get_src_line(line_no),
                        "entries": line_entries_out,
                    }
                )

        return {
            "source_file": source_file,
            "start_line": start_line,
            "end_line": end_line,
            "total_instructions": total_insts,
            "truncated": truncated,
            "lines": lines_out,
        }

    # --------------------- Original mapping API (streamlined) -------------------

    def correlate_ptx_to_source(
        self,
        nvtx_range: Optional[str] = None,
        kernel_name: Optional[str] = None,
        base_metric_name: Optional[str] = None,
        include_sass: bool = False,
        include_metric_value: bool = False,
        source_file_filter: Optional[Any] = None,
        extra_metric_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        """
        Core correlation logic (streamlined from your script).

        Returns nested mapping:
            {
              "source_file_path": {
                  line_number: [
                      {"pc": int, "ptx": str, "sass": str?, "metrics": {...}?},
                      ...
                  ],
                  ...
              },
              ...
            }
        """
        if not self.available:
            return {}

        report = self._load_report()
        if report is None:
            return {}

        action = self._select_action(report, nvtx_range=nvtx_range, kernel_name=kernel_name)
        if action is None:
            logger.warning("No matching kernel action found in report.")
            return {}

        # Pick base metric.
        base_name, base_metric, base_cid = self._pick_base_metric(action, explicit_base=base_metric_name)
        if base_metric is None or base_cid is None:
            logger.warning("Could not find a suitable base metric with correlation IDs.")
            return {}

        logger.info(
            "Using base metric '%s' with %d correlation IDs for PTX/source mapping.",
            base_name,
            base_cid.num_instances(),
        )

        # Build address→value maps for extra metrics.
        extra_metric_names = extra_metric_names or []
        extra_metric_maps = self._build_extra_metric_maps(action, extra_metric_names)

        file_pred = self._make_file_predicate(source_file_filter)
        result: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

        num_ids = base_cid.num_instances()
        for idx in range(num_ids):
            try:
                pc = base_cid.as_uint64(idx)
            except Exception as e:
                logger.debug("Failed to read correlation ID at %d: %s", idx, e)
                continue
            if not pc:
                continue

            src_info = action.source_info(pc)
            if src_info is None:
                continue

            file_name = src_info.file_name()
            line_no = src_info.line()
            if not file_name or not isinstance(line_no, int):
                continue

            if file_pred is not None and not file_pred(file_name):
                continue

            ptx = action.ptx_by_pc(pc) or ""
            if not ptx.strip():
                continue

            entry: Dict[str, Any] = {"pc": pc, "ptx": ptx}

            if include_sass:
                sass = action.sass_by_pc(pc) or ""
                if sass.strip():
                    entry["sass"] = sass

            if include_metric_value:
                metrics_for_pc: Dict[str, Any] = {}
                try:
                    metrics_for_pc[base_name] = base_metric.value(idx)
                except Exception:
                    pass

                for mname, addr_map in extra_metric_maps.items():
                    if pc in addr_map:
                        metrics_for_pc[mname] = addr_map[pc]

                if metrics_for_pc:
                    entry["metrics"] = metrics_for_pc

            file_bucket = result.setdefault(file_name, {})
            line_bucket = file_bucket.setdefault(line_no, [])
            line_bucket.append(entry)

        return result

    def _load_report(self):
        if self._report is None:
            try:
                self._report = ncu_report.load_report(str(self.report_path))  # type: ignore
            except Exception as e:
                logger.error("Failed to load report %s: %s", self.report_path, e)
                self._report = None
        return self._report

    def _select_action(self, report, nvtx_range: Optional[str], kernel_name: Optional[str]):
        ctx = report
        num_ranges = getattr(ctx, "num_ranges", lambda: 0)()
        if num_ranges == 0:
            return None

        chosen_range = None
        for ridx in range(num_ranges):
            rng = ctx.range_by_idx(ridx)
            try:
                rname = rng.name()
            except Exception:
                rname = ""
            if nvtx_range is None or (rname and nvtx_range in rname):
                chosen_range = rng
                break

        if chosen_range is None:
            chosen_range = ctx.range_by_idx(0)

        num_actions = getattr(chosen_range, "num_actions", lambda: 0)()
        if num_actions == 0:
            return None

        if kernel_name is None:
            return chosen_range.action_by_idx(0)

        for aidx in range(num_actions):
            action = chosen_range.action_by_idx(aidx)
            try:
                aname = action.name()
            except Exception:
                aname = ""
            if kernel_name in aname:
                return action

        return chosen_range.action_by_idx(0)

    def _pick_base_metric(self, action, explicit_base: Optional[str] = None):
        metric_names = list(action.metric_names())
        metrics: Dict[str, Any] = {}
        for name in metric_names:
            try:
                m = action.metric_by_name(name)
            except Exception:
                m = None
            if m is not None:
                metrics[name] = m

        def usable(name: str, m) -> Optional[Any]:
            try:
                if not m.has_correlation_ids():
                    return None
                cid = m.correlation_ids()
                if cid is None or cid.num_instances() == 0:
                    return None
                return cid
            except Exception:
                return None

        if explicit_base is not None:
            m = metrics.get(explicit_base)
            if m is not None:
                cid = usable(explicit_base, m)
                if cid is not None:
                    return explicit_base, m, cid
                else:
                    logger.warning("Requested base metric '%s' has no usable correlation IDs.", explicit_base)

        for pname in self.PREFERRED_BASE_METRICS:
            m = metrics.get(pname)
            if m is None:
                continue
            cid = usable(pname, m)
            if cid is not None:
                return pname, m, cid

        best_name = None
        best_metric = None
        best_cid = None
        best_count = -1
        for name, m in metrics.items():
            if name.startswith(self.EXCLUDED_BASE_PREFIXES):
                continue
            cid = usable(name, m)
            if cid is None:
                continue
            count = cid.num_instances()
            if count > best_count:
                best_name = name
                best_metric = m
                best_cid = cid
                best_count = count

        return best_name, best_metric, best_cid

    def _build_extra_metric_maps(self, action, extra_metric_names: List[str]) -> Dict[str, Dict[int, Any]]:
        maps: Dict[str, Dict[int, Any]] = {}

        for mname in extra_metric_names:
            try:
                m = action.metric_by_name(mname)
            except Exception:
                m = None

            if not m:
                logger.warning("Extra metric '%s' not found on this action.", mname)
                continue

            try:
                if not m.has_correlation_ids():
                    logger.warning("Extra metric '%s' has no correlation IDs; skipping.", mname)
                    continue
                cid = m.correlation_ids()
                if cid is None or cid.num_instances() == 0:
                    logger.warning("Extra metric '%s' has 0 correlation IDs; skipping.", mname)
                    continue

                addr_to_val: Dict[int, Any] = {}
                n = cid.num_instances()
                for idx in range(n):
                    addr = cid.as_uint64(idx)
                    val = m.value(idx)
                    addr_to_val[addr] = val
                maps[mname] = addr_to_val

            except Exception as e:
                logger.warning("Failed to process extra metric '%s': %s", mname, e)

        return maps

    def _make_file_predicate(self, filt: Optional[Any]):
        if filt is None:
            return None
        if isinstance(filt, str):
            needle = filt

            def pred(path: str) -> bool:
                return needle in path

            return pred
        if callable(filt):
            return filt
        return None


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

TOOLS = [REGION_DAG_TOOL_SCHEMA, NCU_METRICS_TOOL_SCHEMA, PTX_SOURCE_TOOL_SCHEMA]

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
You are an expert GPU performance engineer optimizing a Triton/CUDA kernel.

== Objects you will receive ==
1) ncu_signals: Nsight Compute metrics for the same kernel:
   • examples: sm__throughput.avg.pct_of_peak_sustained_elapsed,
               sm__warps_active.avg.pct_of_peak_sustained_active,
               dram__bytes.avg.per_second,
               smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,
               smsp__warp_issue_stalled_barrier_per_warp_active.pct.
   • These are ground truth for diagnosing stalls vs overlap.

2) region_summary: a compact summary of a Region-DAG built from PTX:
   • Each node (region) is a contiguous run of PTX with the same phase:
     {mem_param_load | mem_global_load | mem_shared_load | mem_async_copy | async_wait |
      barrier | compute | global_store | shared_store | atomic | control | addr_arith | other}
   • Edges:
     - flow: linear program order between regions (may overlap at runtime)
     - barrier: CTA barrier (e.g., bar.sync) → hard stage cut (no overlap across it)
     - async_dep: cp.async group(s) that feed the first shared-load consumer after a wait
   • Per-region metrics: bytes_global_r/w, bytes_shared_r/w, flops.
   • Stages: contiguous region groups split at barrier/async_wait; each stage has FLOPs and bytes.

3) source_code: original Triton/CUDA kernel (or a diffable excerpt). Optionally: PTX/source map.

== Your Core Workflow ==
• Analyze the ncu_signals to understand the behavior of the kernel (ground truth).
• Use the region_summary to find the *location* of the bottleneck (SoL, stalls, occupancy, pipe utilizations).
• Form hypothesis and drill down
  - "Hypothesis: The kernel is memory-bound (from NCU), and Stage 1 seems
      to be the main loading stage (from DAG). I will inspect its source code."
  - Call 'region_dag_inspect(mode="stage_detail", stage_id=1)' for details.
  - Call 'get_ptx_by_source(...)' for the source-correlated PTX.
  - Call 'ncu_metrics_inspect(mode="get_values", ...)' for more stall data.
• Once you have a confirmed hypothesis, propose optimizations in the STRICT JSON format below.  

== What “optimize” means ==
• Propose concrete code or launch changes that reduce kernel latency:
  - more overlap (async pipelines, double buffering),
  - fewer stalls (barrier stalls, long scoreboard, uncoalesced accesses),
  - higher effective DRAM / L2 BW,
  - better tensor-core or ALU utilization,
  - less epilogue traffic and redundant moves.
• Prefer surgical changes that target the dominant stage(s).

== What to output ==
STRICT JSON only:
{
  "candidates": [{
    "think": "short reasoning about why this helps (concise).",
    "method": "the optimization method (e.g., 'deepen async pipeline to 4 groups')",
    "detail": "precise, minimal steps to apply in this codebase (Triton/CUDA params, memory layout, etc.).",
    "code": "a unified diff or a full snippet that compiles (whichever is smaller)."
  }]
}

== Constraints ==
• Output only valid JSON. No markdown, no prose outside JSON fields.
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


def extract_tool_calls(resp) -> List[Dict[str, Any]]:
    """
    Extract tool calls from a Responses API response.

    Returns a list of: {"name": <str>, "arguments": <dict>, "call_id": <optional str>}
    """
    calls: List[Dict[str, Any]] = []

    for item in getattr(resp, "output", []) or []:
        t = getattr(item, "type", None)
        if t in ("tool_call", "function_call"):
            tool_obj = getattr(item, "tool_call", None) or getattr(item, "tool", None) \
                       or getattr(item, "function_call", None) or getattr(item, "function", None)
            if tool_obj:
                name = getattr(tool_obj, "name", None)
                args = getattr(tool_obj, "arguments", None)
                call_id = getattr(tool_obj, "id", None) or getattr(item, "id", None)
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                if name:
                    calls.append({"name": name, "arguments": args or {}, "call_id": call_id})
            continue

        for c in getattr(item, "content", []) or []:
            ct = getattr(c, "type", None)
            if ct in ("tool_call", "function_call"):
                tool_obj = getattr(c, "tool_call", None) or getattr(c, "tool", None) \
                           or getattr(c, "function_call", None) or getattr(c, "function", None)
                if tool_obj:
                    name = getattr(tool_obj, "name", None)
                    args = getattr(tool_obj, "arguments", None)
                    call_id = getattr(tool_obj, "id", None) or getattr(c, "id", None)
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    if name:
                        calls.append({"name": name, "arguments": args or {}, "call_id": call_id})

    return calls


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
        self.tools = TOOLS
        self.tool_registry = tool_registry or DEFAULT_TOOL_REGISTRY

    def _lazy_client(self):
        if self.client is None:
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError("OpenAI SDK not installed. `pip install openai`") from e
            self.client = OpenAI(base_url=self.api_base) if self.api_base else OpenAI()

    def _mk_messages(self, kernel: Any, region_summary: Optional[Dict[str, Any]] = None, ncu_summary: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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

        user_payload = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "KERNEL NAME:\n" + str(getattr(kernel, "name", "unknown_kernel"))},
                {"type": "input_text", "text": "KERNEL SOURCE:\n" + (getattr(kernel, "source_code", "") or "")},
                {"type": "input_text", "text": "INVOCATION_EXAMPLE:\n" + (getattr(kernel, "invocation_example", "") or "")},
                {"type": "input_text", "text": "NCU SUMMARY (GROUND TRUTH):\n" + json.dumps(ncu_summary or {}, indent=2)},
                {"type": "input_text", "text": "REGION SUMMARY (STATIC STRUCTURE & WORK PROFILE):\n" + json.dumps(region_summary or {}, indent=2)},
                {"type": "input_text", "text": "Output STRICT JSON only per spec. No markdown."},
            ],
        }

        return [sys_block, user_payload]

    def _send(self, *, input_payload: List[Dict[str, Any]], previous_response_id: Optional[str] = None):
        """
        Thin wrapper around client.responses.create.
        """
        kwargs: Dict[str, Any] = dict(model=self.model, input=input_payload, tools=self.tools)
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        return self.client.responses.create(**kwargs)

    def _send_tool_result(self, *, prev_id: str, name: str, call_id: Optional[str], result_obj: Dict[str, Any]):
        """
        Send tool results back to the model with a 'tool' role message.
        """
        tool_content = [
            {
                "type": "tool_result",
                "tool_name": name,
                **({"tool_call_id": call_id} if call_id else {}),
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(result_obj),
                    }
                ],
            }
        ]
        tool_msg = {"role": "tool", "content": tool_content}
        return self._send(input_payload=[tool_msg], previous_response_id=prev_id)
    
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

    def propose(self, kernel: Any, region_summary: Optional[Dict[str, Any]] = None, ncu_summary: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Main entrypoint:

            result = generator.propose(kernel, region_summary)

        Returns parsed JSON dict (or None on failure).
        """
        self._lazy_client()

        messages = self._mk_messages(kernel, region_summary, ncu_summary)
        kernel_key = getattr(kernel, "name", "kernel")
        prev_id: Optional[str] = self.prev_id_by_kernel.get(kernel_key)

        # First call: if there's no previous id, send full context; otherwise continue.
        resp = self._send(
            input_payload=messages if not prev_id else [],
            previous_response_id=prev_id,
        )
        self.prev_id_by_kernel[kernel_key] = resp.id

        MAX_TURNS = 8
        for _ in range(MAX_TURNS):
            tool_calls = extract_tool_calls(resp)
            if tool_calls:
                # Execute locally and return tool_result.
                for tc in tool_calls:
                    out = self.tool_registry.dispatch(tc["name"], tc.get("arguments") or {})
                    resp = self._send_tool_result(
                        prev_id=resp.id,
                        name=tc["name"],
                        call_id=tc.get("call_id"),
                        result_obj=out,
                    )
                    self.prev_id_by_kernel[kernel_key] = resp.id
                continue

            # No tool calls: attempt to parse final JSON.
            raw_text = _collect_text_outputs(resp)
            parsed = _maybe_json(raw_text)
            if isinstance(parsed, dict):
                return parsed

            # Nudge the model once.
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

        # If we exit the loop without valid JSON, return None.
        return None


# ---------------------------------------------------------------------------
# Example usage (skeleton)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("This module is meant to be imported into your optimization pipeline.")
    logger.info("See docstring and comments for integration details.")
