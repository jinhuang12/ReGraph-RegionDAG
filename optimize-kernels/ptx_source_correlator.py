#!/usr/bin/env python3
"""
ptx_source_correlator.py

Correlate PTX (and SASS) instructions to source code lines from an Nsight Compute
report (.ncu-rep), and attach per-instruction metrics (e.g. inst_executed,
warp stall sampling metrics).

This script uses the Nsight Compute Python Report Interface (`ncu_report`) to:
  1. Load a .ncu-rep file.
  2. Select a kernel action (optionally by NVTX range name and/or kernel name).
  3. Pick a *source-correlated* base metric (default: inst_executed).
  4. For each instance of that metric:
       addr = correlation_ids.as_uint64(i)
       src  = action.source_info(addr) -> (file, line)
       ptx  = action.ptx_by_pc(addr)
       sass = action.sass_by_pc(addr) (optional)
  5. Optionally attach values from additional (extra) source-correlated metrics
     at the same address (e.g. warp stall metrics).

Requirements
------------
- Nsight Compute installed on the machine.
- Access to Nsight Compute's Python report module (`ncu_report`), typically
  located under <NsightComputeInstall>/extras/python.
- The .ncu-rep must have:
    * source <-> instruction correlation (compile with -lineinfo or equivalent),
    * at least one source-correlated metric (e.g. SourceCounters section).

Typical Nsight Compute CLI collection command
---------------------------------------------
Example for a Python/Triton kernel with NVTX markers:

  ncu -f -o ncu_lora_shrink \
      --set full \
      --section SourceCounters \
      --import-source yes \
      --target-processes all \
      --nvtx --nvtx-include "lora_shrink_kernel/" \
      python /path/to/your_script.py

Usage
-----
Basic:

  python ptx_source_correlator.py -i ncu_lora_shrink.ncu-rep \
    --only-file kernel_current.py

With SASS, base metric, and warp stall metrics:

  python ptx_source_correlator.py -i ncu_lora_shrink.ncu-rep \
    --only-file kernel_current.py \
    --include-sass \
    --include-metric \
    --base-metric inst_executed \
    --extra-metric smsp__pcsamp_sample_count \
    --extra-metric smsp__pcsamp_warps_issue_stalled_long_scoreboard \
    --extra-metric smsp__pcsamp_warps_issue_stalled_not_selected \
    --extra-metric smsp__pcsamp_warps_issue_stalled_selected \
    --csv ptx_source_with_stalls.csv

The console output will show, per source line:

  <line_no> | <source code>
            [pc=0xADDR] PTX: ...
                         SASS: ...
                         inst_executed=...
                         smsp__pcsamp_sample_count=...
                         smsp__pcsamp_warps_issue_stalled_long_scoreboard=...
                         ...

The CSV will contain one row per (source_file, line, pc) with PTX, SASS,
and flattened metric columns (`metric:<name>`).
"""

import argparse
import csv
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Dynamic import of Nsight Compute Python Report Interface (ncu_report)
# -----------------------------------------------------------------------------

NCU_REPORT_AVAILABLE = False
try:
    # Common install paths for Nsight Compute extras/python
    NCU_PYTHON_CANDIDATES = [
        "/opt/nvidia/nsight-compute/*/extras/python",
        "/usr/local/cuda/nsight-compute*/extras/python",
        "/usr/local/NVIDIA-Nsight-Compute/extras/python",
    ]

    ncu_report_path = None
    for pattern in NCU_PYTHON_CANDIDATES:
        matches = glob.glob(pattern)
        if matches:
            # Use the latest version if multiple found
            ncu_report_path = sorted(matches)[-1]
            break

    if ncu_report_path:
        sys.path.insert(0, ncu_report_path)
        import ncu_report  # type: ignore

        NCU_REPORT_AVAILABLE = True
        logger.info("Loaded ncu_report from: %s", ncu_report_path)
    else:
        logger.warning(
            "Nsight Compute 'extras/python' not found. "
            "ncu_report API will be unavailable."
        )
except ImportError as e:
    logger.warning("Could not import ncu_report: %s", e)
    ncu_report = None  # type: ignore


# -----------------------------------------------------------------------------
# PtxSourceCorrelator
# -----------------------------------------------------------------------------

class PtxSourceCorrelator:
    """
    Correlate PTX instructions to source lines for a single kernel in an
    Nsight Compute report (.ncu-rep).

    High-level algorithm
    --------------------
    1. Load report via ncu_report.load_report().
    2. Select one IAction (kernel) via NVTX range name and/or kernel name.
    3. Pick a *base metric* which:
         - has instance values, and
         - has correlation IDs (IMetric.has_correlation_ids() == True).
       Common choice: "inst_executed". 
    4. Let cid = metric.correlation_ids() (another IMetric object).
       Then for i in [0, cid.num_instances()):
           addr = cid.as_uint64(i)  # absolute PC address
           src_info = action.source_info(addr)  # -> ISourceInfo
           ptx = action.ptx_by_pc(addr)
           sass = action.sass_by_pc(addr)       # optional
    5. Group entries by (file_name, line_no), optionally attach extra metrics.
    """

    #: Base metrics we prefer to use for enumerating PCs
    PREFERRED_BASE_METRICS: List[str] = [
        "inst_executed",
        "thread_inst_executed",
        "thread_inst_executed_true",
        "derived__avg_thread_executed",
        "smsp__pcsamp_sample_count",
    ]

    #: Metrics that have correlation IDs but are clearly not per-instruction PCs
    #: (launch configuration, NUMA, NVLink, profiler bookkeeping, etc.).
    EXCLUDED_BASE_PREFIXES: Tuple[str, ...] = (
        "launch__",
        "device__",
        "numa__",
        "nvlink__",
        "profiler__",
        "pmsampling:dramc__",  # device-level DRAM sampling
    )

    def __init__(self, report_path: Path):
        """
        Args:
            report_path: Path to a .ncu-rep file generated by Nsight Compute.
        """
        self.report_path = Path(report_path)
        self._report = None

        self.available = bool(
            NCU_REPORT_AVAILABLE and self.report_path.exists()
        )

        if not NCU_REPORT_AVAILABLE:
            logger.debug("ncu_report API not available.")
        elif not self.report_path.exists():
            logger.warning("NCU report file not found: %s", self.report_path)

    # -------------------------------------------------------------------------
    # Core public API
    # -------------------------------------------------------------------------
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

        `source_file` may be a full path or a substring; we'll pick the first
        mapping key that contains the substring if there's no exact match.
        """

        mapping = self.correlate_ptx_to_source(
            nvtx_range=nvtx_range,
            kernel_name=kernel_name,
            base_metric_name=None,
            include_sass=include_sass,
            include_metric_value=include_metric_value,
            source_file_filter=source_file,
            extra_metric_names=extra_metric_names,
        )

        resolved_file = source_file
        if source_file not in mapping:
            candidates = [k for k in mapping.keys() if source_file in k]
            if len(candidates) == 1:
                resolved_file = candidates[0]
            else:
                return {
                    "source_file": source_file,
                    "start_line": start_line,
                    "end_line": end_line,
                    "total_instructions": 0,
                    "truncated": False,
                    "lines": [],
                }

        if resolved_file not in mapping:
            return {
                "source_file": source_file,
                "start_line": start_line,
                "end_line": end_line,
                "total_instructions": 0,
                "truncated": False,
                "lines": [],
            }

        file_map = mapping[resolved_file]

        # Load source text.
        source_path_to_read = resolved_file
        try:
            with open(source_path_to_read, "r", encoding="utf-8", errors="replace") as f:
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
    
    def correlate_ptx_to_source(
        self,
        nvtx_range: Optional[str] = None,
        kernel_name: Optional[str] = None,
        base_metric_name: Optional[str] = None,
        include_sass: bool = False,
        include_metric_value: bool = False,
        source_file_filter: Optional[Union[str, Callable[[str], bool]]] = None,
        extra_metric_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
        """
        Build mapping:
            {
              "source_file_path": {
                  line_number: [
                      {
                          "pc": <int>,
                          "ptx": <str>,
                          "sass": <str>,        # if include_sass == True
                          "metrics": {          # if include_metric_value == True
                              "<metric_name>": value,
                              ...
                          }
                      },
                      ...
                  ],
                  ...
              },
              ...
            }

        Args:
            nvtx_range:
                Optional substring to match the NVTX range name (e.g.
                "lora_shrink_kernel/"). If None, the first range is used.
            kernel_name:
                Optional substring to match the kernel (action) name. If None,
                the first action in the chosen range is used.
            base_metric_name:
                Name of metric to use as the *base* for enumerating PCs.
                If None, use PREFERRED_BASE_METRICS in order, then any
                suitable metric with correlation IDs.
            include_sass:
                If True, also fetch disassembled SASS for each PC via
                IAction.sass_by_pc(address). 
            include_metric_value:
                If True, attach metric values to each entry under "metrics".
            source_file_filter:
                If a str, only keep records whose source file path contains
                that substring. If a callable, it must be `Callable[[str], bool]`
                and decides which source files to keep.
            extra_metric_names:
                List of additional metric names to pull in by correlation IDs
                and attach per PC. This is how you get warp stall metrics like:
                    - smsp__pcsamp_sample_count
                    - smsp__pcsamp_warps_issue_stalled_long_scoreboard
                    - smsp__pcsamp_warps_issue_stalled_not_selected
                    - smsp__pcsamp_warps_issue_stalled_selected
                  etc.

        Returns:
            Nested mapping as described above.
        """
        if not self.available:
            logger.warning(
                "ncu_report API or report file unavailable; returning empty mapping."
            )
            return {}

        report = self._load_report()
        if report is None:
            return {}

        action = self._select_action(
            report, nvtx_range=nvtx_range, kernel_name=kernel_name
        )
        if action is None:
            logger.warning("No matching kernel action found in report.")
            return {}

        file_pred = self._make_file_predicate(source_file_filter)
        extra_metric_names = extra_metric_names or []

        def correlate_for_action(act) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
            # Pick base metric
            base_name, base_metric, base_cid = self._pick_base_metric(
                act,
                explicit_base=base_metric_name,
            )
            if base_metric is None or base_cid is None:
                logger.warning(
                    "Could not find a source-correlated base metric with correlation IDs."
                )
                return {}

            logger.info(
                "Using base metric '%s' with %d correlation IDs for PTX/source mapping.",
                base_name,
                base_cid.num_instances(),
            )

            # Build addr -> value maps for extra metrics
            extra_metric_maps = self._build_extra_metric_maps(
                act, extra_metric_names
            )

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

                src_info = act.source_info(pc)
                if src_info is None:
                    continue

                file_name = src_info.file_name()
                line_no = src_info.line()
                if not file_name or not isinstance(line_no, int):
                    continue

                if file_pred is not None and not file_pred(file_name):
                    continue

                ptx = act.ptx_by_pc(pc) or ""
                if not ptx.strip():
                    continue

                entry: Dict[str, Any] = {"pc": pc, "ptx": ptx}

                if include_sass:
                    sass = act.sass_by_pc(pc) or ""
                    if sass.strip():
                        entry["sass"] = sass

                if include_metric_value:
                    metrics_for_pc: Dict[str, Any] = {}
                    try:
                        metrics_for_pc[base_name] = base_metric.value(idx)
                    except Exception as e:
                        logger.debug(
                            "Could not read base metric '%s' at instance %d: %s",
                            base_name,
                            idx,
                            e,
                        )

                    for mname, addr_map in extra_metric_maps.items():
                        if pc in addr_map:
                            metrics_for_pc[mname] = addr_map[pc]

                    if metrics_for_pc:
                        entry["metrics"] = metrics_for_pc

                file_bucket = result.setdefault(file_name, {})
                line_bucket = file_bucket.setdefault(line_no, [])
                line_bucket.append(entry)

            return result

        result = correlate_for_action(action)

        # If no kernel filter was given and a source filter is present, try other
        # actions until one yields matches. This helps when the default action
        # (first in the report) isn't the desired kernel.
        if (
            not result
            and kernel_name is None
            and source_file_filter is not None
        ):
            num_ranges = getattr(report, "num_ranges", lambda: 0)()
            for ridx in range(num_ranges):
                rng = report.range_by_idx(ridx)
                num_actions = getattr(rng, "num_actions", lambda: 0)()
                for aidx in range(num_actions):
                    act = rng.action_by_idx(aidx)
                    if act is action:
                        continue
                    alt_result = correlate_for_action(act)
                    if alt_result:
                        try:
                            aname = act.name()
                        except Exception:
                            aname = "<unknown>"
                        logger.info(
                            "Fallback selected action '%s' after filtering by source file.",
                            aname,
                        )
                        return alt_result

        return result

    def to_rows(
        self,
        mapping: Dict[str, Dict[int, List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        """
        Flatten nested mapping into a list of row dicts.

        Each row corresponds to a single (source_file, line, pc) entry and has:
            {
              "source_file": str,
              "line": int,
              "pc": int,
              "ptx": str,
              "sass": str,
              "metric:<name>": value,  # one column per metric name
              ...
            }
        """
        rows: List[Dict[str, Any]] = []

        for src, lines_map in mapping.items():
            for line_no, entries in lines_map.items():
                for entry in entries:
                    row: Dict[str, Any] = {
                        "source_file": src,
                        "line": line_no,
                        "pc": entry.get("pc"),
                        "ptx": entry.get("ptx", ""),
                        "sass": entry.get("sass", ""),
                    }
                    metrics = entry.get("metrics") or {}
                    for mname, val in metrics.items():
                        col = f"metric:{mname}"
                        row[col] = val
                    rows.append(row)

        # Stable ordering: by file, then line, then pc
        rows.sort(key=lambda r: (r["source_file"], r["line"], r.get("pc", 0)))
        return rows

    def export_csv(
        self,
        mapping: Dict[str, Dict[int, List[Dict[str, Any]]]],
        out_path: Union[str, Path],
    ) -> Path:
        """
        Export the flattened mapping to CSV.

        The CSV header always contains:
          - source_file
          - line
          - pc
          - ptx
          - sass

        And one additional column for each encountered metric name:
          - metric:<metric_name>
        """
        rows = self.to_rows(mapping)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Base columns
        base_cols = ["source_file", "line", "pc", "ptx", "sass"]
        metric_cols: List[str] = []

        if rows:
            # Collect all metric:<name> columns
            col_set = set()
            for r in rows:
                for k in r.keys():
                    if k.startswith("metric:"):
                        col_set.add(k)
            metric_cols = sorted(col_set)

        fieldnames = base_cols + metric_cols

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                # Ensure all keys exist
                for c in fieldnames:
                    r.setdefault(c, "")
                writer.writerow(r)

        return out_path

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _load_report(self):
        if self._report is None:
            try:
                self._report = ncu_report.load_report(str(self.report_path))  # type: ignore
            except Exception as e:
                logger.error("Failed to load report %s: %s", self.report_path, e)
                self._report = None
        return self._report

    def _select_action(
        self,
        report,
        nvtx_range: Optional[str],
        kernel_name: Optional[str],
    ):
        """
        Select a single IAction (kernel) from the report.

        Strategy:
          1. Iterate ranges via IContext.num_ranges() / IContext.range_by_idx().
          2. If nvtx_range is given, pick the first range whose name() contains it.
          3. Within that range, if kernel_name is given, pick the first action
             whose name() contains it; else, use the first action.
          4. Fallback: first action of the first range. 
        """
        ctx = report  # IContext
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
            # Fallback: first range
            chosen_range = ctx.range_by_idx(0)

        # Select action
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

        # Fallback: first action
        return chosen_range.action_by_idx(0)

    def _pick_base_metric(
        self,
        action,
        explicit_base: Optional[str] = None,
    ) -> Tuple[Optional[str], Any, Any]:
        """
        Choose a base metric whose correlation IDs represent instruction PCs.

        Returns:
            (base_metric_name, base_metric_object, base_correlation_ids_metric)
        """
        # All metric objects for this action
        metric_names = list(action.metric_names())
        metrics: Dict[str, Any] = {}
        for name in metric_names:
            try:
                m = action.metric_by_name(name)
            except Exception:
                m = None
            if m is not None:
                metrics[name] = m

        # Helper to test if a metric is usable as base
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

        # 1) If explicit base specified, try that first
        if explicit_base is not None:
            m = metrics.get(explicit_base)
            if m is not None:
                cid = usable(explicit_base, m)
                if cid is not None:
                    return explicit_base, m, cid
                else:
                    logger.warning(
                        "Requested base metric '%s' has no usable correlation IDs; "
                        "falling back to default heuristics.",
                        explicit_base,
                    )

        # 2) Try preferred base metric names in order
        for pname in self.PREFERRED_BASE_METRICS:
            m = metrics.get(pname)
            if m is None:
                continue
            cid = usable(pname, m)
            if cid is not None:
                return pname, m, cid

        # 3) Fallback: any metric with correlation IDs that is not obviously
        #    launch/NUMA/etc.; pick the one with the most instances.
        best_name: Optional[str] = None
        best_metric: Any = None
        best_cid: Any = None
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

    def _build_extra_metric_maps(
        self,
        action,
        extra_metric_names: List[str],
    ) -> Dict[str, Dict[int, Any]]:
        """
        For each extra metric name, build a mapping:

            extra_metric_maps[metric_name] = { address (pc): metric_value }

        Used to attach additional values (e.g. warp stall metrics) to each
        instruction address in the base loop.
        """
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
                    logger.warning(
                        "Extra metric '%s' has no correlation IDs; skipping.", mname
                    )
                    continue
                cid = m.correlation_ids()
                if cid is None or cid.num_instances() == 0:
                    logger.warning(
                        "Extra metric '%s' has 0 correlation ID instances; skipping.",
                        mname,
                    )
                    continue

                addr_to_val: Dict[int, Any] = {}
                n = cid.num_instances()
                for idx in range(n):
                    addr = cid.as_uint64(idx)
                    val = m.value(idx)
                    addr_to_val[addr] = val

                maps[mname] = addr_to_val
                logger.info(
                    "Extra metric '%s' has %d address->value entries.",
                    mname,
                    len(addr_to_val),
                )
            except Exception as e:
                logger.warning(
                    "Failed to process extra metric '%s': %s", mname, e
                )

        return maps

    def _make_file_predicate(
        self,
        filt: Optional[Union[str, Callable[[str], bool]]],
    ) -> Optional[Callable[[str], bool]]:
        """
        Turn a string or callable into a file path predicate.

        - If filt is None: return None (no filtering).
        - If filt is a string: return lambda path: filt in path.
        - If filt is callable: return it as is.
        """
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


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Correlate PTX (and SASS) instructions to source code lines from an "
            "Nsight Compute .ncu-rep file, optionally attaching metric values "
            "per instruction address (PC)."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="report",
        type=Path,
        required=True,
        help="Path to .ncu-rep report file generated by Nsight Compute.",
    )
    parser.add_argument(
        "--nvtx",
        dest="nvtx_range",
        default=None,
        help="Substring to select NVTX range name (optional).",
    )
    parser.add_argument(
        "--kernel",
        dest="kernel_name",
        default=None,
        help="Substring to select kernel (action) name (optional).",
    )
    parser.add_argument(
        "--only-file",
        dest="only_file",
        default=None,
        help=(
            "If set, only show source files whose path contains this substring "
            "(e.g. 'kernel_current.py')."
        ),
    )
    parser.add_argument(
        "--base-metric",
        dest="base_metric",
        default=None,
        help=(
            "Base metric used to enumerate PCs (default: first available in "
            "['inst_executed', 'thread_inst_executed', ...])."
        ),
    )
    parser.add_argument(
        "--extra-metric",
        dest="extra_metrics",
        action="append",
        default=[],
        help=(
            "Additional source-correlated metrics to attach per PC (can be "
            "specified multiple times). "
            "Example: --extra-metric smsp__pcsamp_warps_issue_stalled_long_scoreboard"
        ),
    )
    parser.add_argument(
        "--include-sass",
        action="store_true",
        help="Also print SASS per PC (via sass_by_pc).",
    )
    parser.add_argument(
        "--include-metric",
        action="store_true",
        help="Attach metric values (base + extra) to each PC and print them.",
    )
    parser.add_argument(
        "--csv",
        dest="csv",
        default=None,
        help=(
            "Optional path to write a CSV with one row per "
            "(source_file, line, pc)."
        ),
    )

    args = parser.parse_args()

    correlator = PtxSourceCorrelator(report_path=args.report)
    if not correlator.available:
        print(
            "ERROR: ncu_report API is not available or report file not found.\n"
            "  - Ensure Nsight Compute is installed.\n"
            "  - Ensure its 'extras/python' directory is discoverable, or edit\n"
            "    NCU_PYTHON_CANDIDATES in this script to match your installation."
        )
        sys.exit(2)

    mapping = correlator.correlate_ptx_to_source(
        nvtx_range=args.nvtx_range,
        kernel_name=args.kernel_name,
        base_metric_name=args.base_metric,
        include_sass=args.include_sass,
        include_metric_value=args.include_metric,
        source_file_filter=args.only_file,
        extra_metric_names=args.extra_metrics,
    )

    if not mapping:
        print(
            "No correlation found.\n"
            "  - Confirm your kernel was built with line info (e.g. -lineinfo).\n"
            "  - Confirm the report includes source-correlated metrics "
            "    (e.g. SourceCounters section).\n"
            "  - Confirm your NVTX/kernal filters are correct."
        )
        sys.exit(1)

    # Helper to read a specific line from a source file on disk.
    def get_source_line_text(path: str, line_no: int) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            if 1 <= line_no <= len(lines):
                return lines[line_no - 1]
        except Exception:
            pass
        return ""

    # Pretty print to stdout
    for src_path in sorted(mapping.keys()):
        print(f"\n=== {src_path} ===")
        lines_map = mapping[src_path]
        for line_no in sorted(lines_map.keys()):
            src_line = get_source_line_text(src_path, line_no)
            header = f"{line_no:6d} | {src_line}" if src_line else f"{line_no:6d} |"
            print(header)

            # Deduplicate by PC for this line
            seen_pcs = set()
            for entry in sorted(
                lines_map[line_no], key=lambda e: e.get("pc", 0)
            ):
                pc = entry.get("pc")
                if pc in seen_pcs:
                    continue
                seen_pcs.add(pc)

                ptx = (entry.get("ptx") or "").rstrip()
                if args.include_sass and entry.get("sass"):
                    print(f"          [pc=0x{pc:x}] PTX: {ptx}")
                    print(f"                       SASS: {entry['sass'].rstrip()}")
                else:
                    print(f"          [pc=0x{pc:x}] {ptx}")

                if args.include_metric:
                    metrics = entry.get("metrics") or {}
                    for mname in sorted(metrics.keys()):
                        print(f"                       {mname}={metrics[mname]}")

    # Optional CSV export
    if args.csv:
        out_path = correlator.export_csv(mapping, args.csv)
        print(f"\n[wrote CSV] {out_path}")

if __name__ == "__main__":
    main()
