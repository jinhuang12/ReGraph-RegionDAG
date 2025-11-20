#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_kernels.py
-------------------
True Monte-Carlo Graph Search (MCGS) for optimizing Triton or CUDA kernels,
now with OFFLINE ReGraph collection and reuse.

Additions in this revision:
- Dataset logger (regraph_dataset.jsonl) capturing (from_method → to_method, reward, ok, etc.)
- Offline ReGraph aggregator (regraph.json) with node/edge stats, CLI to build it
- Online use of the offline ReGraph to constrain/guide expansion at unvisited nodes
- Relabel step maps free-form LLM methods to canonical CUDA method set every step

MCGS properties (aligned with the paper):
- Selection: P-UCB at each visited node (graph-aware), progressive widening
- Expansion: first visit -> expand *all* successors (LLM proposals), then pick one to evaluate
- Rollout: ε-greedy argmax_a [ Q - λ N ] with step cap and early stop on failure
- Reward: Eq.(3) piecewise; backprop the *maximum* reward observed in the rollout
- Graph: transpositions enabled (child id keyed by worker-reported state hash if USE_HASH_FOR_NODE_ID)

Author: (Your team)
"""

import argparse
import dataclasses
from enum import Enum
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import workers modules
from workers.state_manager import save_partial_state, load_partial_state, get_latest_phase_data
from shared.model import DeviceProfile, KernelCode, _round_float_values, get_metadata_value
# --- Tools (LLM + Region-DAG + NCU) ---
import kernel_opt_tooling
from kernel_opt_tooling import LLMCandidateGenerator, RegionDagContext, NcuMetricsContext, PtxSourceCorrelator

# --------------------------- Constants & Config --------------------------
# --- ReGraphT-style knobs ---
ANTI_CYCLE_LAMBDA = 0.08    # discourage revisiting same (state,action) within THIS selection path
EPSILON_SELECT    = 0.05    # tiny exploration in selection
# --- Rollout policy (Eq.2) ---
EPSILON_ROLLOUT   = 0.10    # ε for ε-greedy rollout
LAMBDA_REG        = 0.50    # λ for Q(s,a) - λ N(s,a)
ROLLOUT_MAX_STEPS = 4       # rollout horizon cap

# Node identity: use worker's state_hash as node id (graph with transpositions)
USE_HASH_FOR_NODE_ID = True

MAX_DEPTH = 8
PW_K0, PW_K1, PW_ALPHA = 2, 3, 0.5
DEFAULT_TIMING = {"warmup": 10, "iters": 100, "repeat": 3}
COMPILE_CACHE_DIR = Path(os.environ.get("COMPILE_CACHE_DIR", "/tmp/kernel_compile_cache"))

# --- NEW: optional LLM relabeler (G.4-style) ---
RELABLER_STRICT_JSON_INSTR = (
    "Return STRICT JSON only as an array of objects. Each object must be:\n"
    '{"index": <int>, "existed": "yes"|"no", "method": "<canonical_or_original>"}\n'
)

# ------------------------- Utility / Filesystem -------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_str(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))

def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def indent(s: str, n: int) -> str:
    pad = " " * n
    return "\n".join((pad + line) for line in s.splitlines())

def arch_sm_to_int(arch: str) -> int:
    m = re.search(r"sm_(\d+)", arch or "")
    return int(m.group(1)) if m else 0

# --------------------------- Subprocess Helper ---------------------------
def run_subprocess(cmd: List[str], cwd: Optional[Path]=None, env: Optional[Dict[str,str]]=None, timeout: Optional[int]=None) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return 124, out, err


def find_ncu_cli() -> Optional[str]:
    """Return the first available Nsight Compute CLI binary."""
    for cand in ("ncu", "nv-nsight-cu-cli"):
        path = shutil.which(cand)
        if path:
            return path
    return None


def find_latest_ncu_report(workdir: Path) -> Optional[Path]:
    """Find the most recent Nsight Compute report generated in workdir."""
    candidates: List[Path] = []
    for ext in (".ncu-rep", ".nsight-cuprof-report"):
        candidates.extend(workdir.glob(f"*{ext}"))
    if not candidates:
        return None
    try:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None


def run_runner_with_optional_ncu(
    base_cmd: List[str],
    workdir: Path,
    env: Dict[str, str],
    kernel_name: Optional[str],
    ncu_control: Dict[str, Any],
    timeout: int = 900,
) -> Tuple[int, str, str, Optional[str]]:
    """Run the runner command, optionally wrapped in Nsight Compute."""
    ncu_report_path: Optional[str] = None
    if ncu_control.get("enabled"):
        ncu_bin = find_ncu_cli()
        if ncu_bin:
            ncu_cmd = [ncu_bin, "-f", "-o", "ncu_report"]
            if kernel_name:
                ncu_cmd.extend(["-k", kernel_name])
            ncu_cmd.extend(base_cmd)
            rc, out, err = run_subprocess(ncu_cmd, cwd=workdir, env=env, timeout=timeout)
            report = find_latest_ncu_report(workdir)
            if report:
                ncu_report_path = str(report)
            return rc, out, err, ncu_report_path

    rc, out, err = run_subprocess(base_cmd, cwd=workdir, env=env, timeout=timeout)
    return rc, out, err, ncu_report_path

# --------------------------- Patch Application ---------------------------
def apply_unified_diff(workdir: Path, files: List[Dict[str, str]]) -> bool:
    """Apply a list of unified diffs. Requires `patch` on PATH; returns True on success."""
    if not files: return True
    patch_bin = shutil.which("patch")
    if not patch_bin: return False
    diff_path = workdir / "candidate.diff"
    with diff_path.open("w", encoding="utf-8") as f:
        for fobj in files:
            diff = fobj.get("diff", "")
            if diff:
                f.write(diff)
                if not diff.endswith("\n"): f.write("\n")
    rc, out, err = run_subprocess([patch_bin, "-p0", "-r", "rejects.txt", "-i", str(diff_path)], cwd=workdir)
    return rc == 0

# --------------------------- Dataset & ReGraph ---------------------------
class ReGraphDataset:
    """Append-only dataset for transitions; builds/loads aggregated ReGraph."""
    def __init__(self, outdir: Path):
        self.outdir = Path(outdir)
        self.dataset_path = self.outdir / "regraph_dataset.jsonl"
        self.regraph_path = self.outdir / "regraph.json"

    def append_transition(self, rec: Dict[str, Any]) -> None:
        """Append a single transition row."""
        append_jsonl(self.dataset_path, rec)

    def build_regraph(self) -> Dict[str, Any]:
        """Aggregate dataset.jsonl into a compact ReGraph (nodes + edges)."""
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if not self.dataset_path.exists():
            rg = {"nodes": {}, "edges": {}, "built_at": now_iso()}
            json_dump(rg, self.regraph_path)
            return rg

        def _inc_node(m: str, ok: bool, reward: float):
            st = nodes.setdefault(m, {"count": 0, "success": 0, "sum_reward": 0.0})
            st["count"] += 1
            st["success"] += 1 if (reward or 0.0) > 0.0 and ok else 0
            st["sum_reward"] += float(reward or 0.0)

        def _inc_edge(u: str, v: str, ok: bool, reward: float):
            bucket = edges.setdefault(u, {})
            st = bucket.setdefault(v, {"count": 0, "success": 0, "sum_reward": 0.0})
            st["count"] += 1
            st["success"] += 1 if (reward or 0.0) > 0.0 and ok else 0
            st["sum_reward"] += float(reward or 0.0)

        with self.dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                u = row.get("from_method") or "START"
                v = row.get("to_method")
                ok = bool(row.get("ok", False))
                reward = float(row.get("reward", 0.0))
                if not v:  # ignore malformed
                    continue
                _inc_node(u, ok, reward)
                _inc_node(v, ok, reward)
                _inc_edge(u, v, ok, reward)

        # finalize means
        for m, st in nodes.items():
            cnt = max(1, st["count"])
            st["mean_reward"] = st["sum_reward"] / float(cnt)
        for u, adj in edges.items():
            for v, st in adj.items():
                cnt = max(1, st["count"])
                st["mean_reward"] = st["sum_reward"] / float(cnt)

        rg = {"nodes": nodes, "edges": edges, "built_at": now_iso()}
        json_dump(rg, self.regraph_path)
        return rg

    def load_regraph(self) -> Dict[str, Any]:
        if self.regraph_path.exists():
            try:
                return json_load(self.regraph_path)
            except Exception:
                pass
        return {"nodes": {}, "edges": {}, "built_at": None}

# --------------------------- Graph Structures ---------------------------
@dataclass
class EdgeStats:
    Qsum: float = 0.0
    Nsa:  float = 0.0
    Qbar: float = 0.0
    child: Optional[str] = None      # child state id if expanded

@dataclass
class NodeStats:
    visits: int = 0
    source_code: str = ""
    ncu: Dict[str, Any] = field(default_factory=dict)
    edges: Dict[str, EdgeStats] = field(default_factory=dict)   # action_id -> EdgeStats
    actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # action_id -> candidate payload
    applied_actions: List[str] = field(default_factory=list)    # edge ids so far
    applied_methods: List[str] = field(default_factory=list)    # canonical methods so far
    last_method: Optional[str] = "START"
    impl_hash: str = ""                                         # worker-reported state hash
    ptx_path: Optional[str] = None
    ncu_report_path: Optional[str] = None

# --------------------------- Relabel (canonical methods) ------------------
class LLMRelabeler:
    def __init__(self, model: str):
        self.model = model
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:
                raise RuntimeError("OpenAI SDK not installed for relabel.") from e
            self._client = OpenAI()

    def relabel(self, existing_methods: list[str], steps: list[dict]) -> list[dict]:
        """
        steps: list of {'method': str, 'detail': str?, 'code': str?} (code optionally omitted)
        returns: [{'index': i, 'existed': 'yes'|'no', 'method': '<canonical_or_original>'}, ...]
        """
        self._lazy_client()

        compact_steps = [
            {"index": i, "method": (s.get("method") or "").strip(), "detail": (s.get("detail") or "")[:512]}
            for i, s in enumerate(steps)
        ]

        sys_msg = {
            "role": "system",
            "content": [{"type": "input_text", "text":
                "You are normalizing CUDA optimization method names. "
                "If a step's 'method' semantically matches one of the EXISTING METHODS, "
                "output that exact canonical name. Otherwise, keep the original method name unchanged."
            }]
        }
        user_msg = {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "EXISTING METHODS:\n" + json.dumps(existing_methods, ensure_ascii=False)},
                {"type": "input_text", "text": "STEPS:\n" + json.dumps(compact_steps, ensure_ascii=False)},
                {"type": "input_text", "text": RELABLER_STRICT_JSON_INSTR},
            ]
        }

        resp = self._client.responses.create(
            model=self.model,
            input=[sys_msg, user_msg]
        )

        text_blobs = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    text_blobs.append(t)
        raw = "\n".join(text_blobs).strip()

        s = raw.strip()
        if s.startswith("```"):
            s = s.split("```", 2)[-1].strip()
        relabeled = json.loads(s)
        if not isinstance(relabeled, list):
            raise ValueError("Relabeler did not return a JSON array.")
        return relabeled

# >>> CHANGED: add a simple fallback and a helper used by both expansion and rollout
def relabel_fallback(existing_methods: list[str], steps: list[dict]) -> list[dict]:
    out = []
    existing_lower = [m.lower() for m in existing_methods]
    for i, s in enumerate(steps):
        method_raw = (s.get("method") or "").strip()
        mlow = method_raw.lower()
        match = None
        for j, canon in enumerate(existing_lower):
            if mlow == canon or canon in mlow or mlow in canon:
                match = existing_methods[j]; break
        if match:
            out.append({"index": i, "existed": "yes", "method": match})
        else:
            out.append({"index": i, "existed": "no", "method": method_raw})
    return out

def relabel_methods(orch: "Orchestrator", search_state: Dict[str, Any], steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Canonicalize step['method'] -> step['method_canonical'] against evolving O in search_state['method_catalog'].
    Grows O when relabel returns existed == 'no'.
    """
    O = list(search_state.get("method_catalog", []))
    if not steps:
        return steps
    try:
        relabeled = orch.relabeler.relabel(existing_methods=O, steps=steps)
    except Exception:
        relabeled = relabel_fallback(existing_methods=O, steps=steps)

    for item in relabeled:
        i = int(item["index"])
        canon = (item.get("method") or "").strip()
        steps[i]["method_canonical"] = canon
        if item.get("existed") == "no":
            if canon and canon not in O:
                O.append(canon)
    search_state["method_catalog"] = O
    return steps

# --------------------------- Orchestrator Core ---------------------------
class Orchestrator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.outdir = Path(args.outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.gpus = [g.strip() for g in args.gpus.split(",") if g.strip()!=""]
        if not self.gpus:
            raise ValueError("--gpus must specify at least one device index")
        self.llm = LLMCandidateGenerator(model=args.llm_model)
        self.relabeler = LLMRelabeler(model=args.llm_model)
        # Offline ReGraph dataset & graph
        self.rgds = ReGraphDataset(self.outdir)
        if args.regraph_autobuild:
            self.regraph = self.rgds.build_regraph()
        else:
            self.regraph = self.rgds.load_regraph()

    def load_kernel_specs(self) -> List[KernelCode]:
        p = Path(self.args.input)
        specs: List[KernelCode] = []
        if p.is_dir():
            for f in sorted(p.glob("*.json")):
                spec = json_load(f)
                specs.append(KernelCode(**spec["kernel"], name=spec.get("name"), device_profile=DeviceProfile(**spec["device_profile"])))
        else:
            spec = json_load(p)
            specs.append(KernelCode(**spec["kernel"], name=spec.get("name"), device_profile=DeviceProfile(**spec["device_profile"])))
        return specs

    def kernel_dir(self, k: KernelCode) -> Path:
        if not k.name:
            raise ValueError(f"KernelCode object missing required 'name' field")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", k.name)
        return self.outdir / "kernels" / safe

    # ---------- Baseline ----------
    def baseline_or_resume(self, k: KernelCode) -> Tuple[Dict[str,Any], Dict[str,Any]]:
        kdir = self.kernel_dir(k)
        state_path = kdir / "search_state.json"
        if self.args.resume and state_path.exists():
            search_state = json_load(state_path)
            run = json_load(kdir / "baseline" / "run.json")
            root_hash = search_state.get("root_state_hash")
            root_node = search_state.get("nodes", {}).get(root_hash, {}) if root_hash else {}
            updated = False
            if root_node is not None:
                if not root_node.get("ptx_path") and run.get("ptx_path"):
                    root_node["ptx_path"] = run.get("ptx_path")
                    updated = True
                if not root_node.get("ncu_report_path") and run.get("ncu_report_path"):
                    root_node["ncu_report_path"] = run.get("ncu_report_path")
                    updated = True
                if updated and root_hash:
                    search_state.setdefault("nodes", {})[root_hash] = root_node
                    json_dump(search_state, state_path)
            return search_state, run

        kdir.mkdir(parents=True, exist_ok=True)
        json_dump(k.to_dict(), kdir / "input_spec.json")

        # Baseline build/run on GPU 0
        bdir = kdir / "baseline"
        bdir.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpus[0]

        control = {
            "kernel": k.to_dict(),
            "variant": {"full_source_code": k.source_code, "patch": {"files": []}, "launch_update": {}},
            "workdir": str(bdir),
            "ncu": {"enabled": True, "collect": True},
            "timing": DEFAULT_TIMING,
        }
        json_dump(control, bdir / "control.json")

        rc, out, err = run_subprocess([sys.executable, __file__, "--worker", str(bdir / "control.json")], env=env)
        write_text(bdir / "stdout.log", out)
        write_text(bdir / "stderr.log", err)
        if rc != 0:
            raise RuntimeError(f"Baseline worker failed (rc={rc}). See {bdir}/stderr.log")
        run = json_load(bdir / "result.json")
        json_dump(run, bdir / "run.json")

        # Materialized baseline
        source_path = bdir / ("kernel_module.py" if k.kernel_type == "triton" else "kernel.cu")
        baseline_source = source_path.read_text(encoding="utf-8") if source_path.exists() else k.source_code
        worker_root_hash = run.get("state_hash", "")
        root_hash = worker_root_hash if USE_HASH_FOR_NODE_ID else str(uuid.uuid4())

        # Graph init
        search_state = {
            "baseline_ms": run.get("mean_ms", 1e8),
            "best_ms": run.get("mean_ms", 1e8),
            "best_state_hash": root_hash,
            "best_variant_dir": str(bdir),
            "root_state_hash": root_hash,
            "method_catalog": [],  # evolving canonical set O
            "nodes": { root_hash: dataclasses.asdict(NodeStats(
                visits=0,
                source_code=baseline_source,
                ncu=run.get("ncu_metrics", {"kernel_time_ms": run.get("mean_ms", 0.0)}),
                edges={}, actions={}, applied_actions=[], applied_methods=[],
                last_method="START",
                impl_hash=worker_root_hash,
                ptx_path=run.get("ptx_path"),
                ncu_report_path=run.get("ncu_report_path")
            ))},
            "events_path": str(self.kernel_dir(k) / "events.jsonl"),
            "trace_path": str(self.kernel_dir(k) / "trace.md"),
        }

        # Trace header
        write_text(Path(search_state["trace_path"]),
                   f"# Optimization trace for `{k.name}`\n\n- Baseline mean: **{run.get('mean_ms', 0.0):.3f} ms** (std {run.get('std_ms',0.0):.3f})\n- Device: {k.device_profile.gpu_name if k.device_profile else 'unknown'} / {k.device_profile.arch if k.device_profile else ''}\n\n")

        # Persist state
        root_ns = NodeStats(**search_state["nodes"][root_hash])
        root_ns.edges = {k: v if isinstance(v, EdgeStats) else EdgeStats(**v) for k, v in root_ns.edges.items()}
        search_state["nodes"][root_hash] = dataclasses.asdict(root_ns)
        json_dump(search_state, state_path)
        return search_state, run

    # ---------- MCGS Loop ----------
    def run_mcgs_for_kernel(self, k: KernelCode):
        kdir = self.kernel_dir(k)
        search_state_path = kdir / "search_state.json"
        search_state = json_load(search_state_path)
        baseline_ms = float(search_state["baseline_ms"])
        best_ms = float(search_state["best_ms"])
        root_hash = search_state["root_state_hash"]

        if math.isinf(baseline_ms):
            print(f"[MCGS] Skipping kernel '{k.name}': baseline execution failed (mean_ms = inf)")
            return

        budget = int(self.args.budget_builds)
        gpu_rr = 0

        def _prepare_contexts(ns: NodeStats) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            region_summary: Dict[str, Any] = {}
            ncu_summary: Dict[str, Any] = {}

            # Clear existing globals before populating for this node
            kernel_opt_tooling.CURRENT_REGION_DAG_CTX = None
            kernel_opt_tooling.CURRENT_NCU_CTX = None
            kernel_opt_tooling.CURRENT_PTX_CORRELATOR = None

            rctx: Optional[RegionDagContext] = None
            nctx: Optional[NcuMetricsContext] = None
            ptx_corr: Optional[PtxSourceCorrelator] = None
            try:
                if ns.ptx_path and Path(ns.ptx_path).exists():
                    rctx = RegionDagContext(Path(ns.ptx_path).read_text(encoding="utf-8"), kernel_name=k.name)
                    region_summary = rctx.overview()
                if ns.ncu_report_path and Path(ns.ncu_report_path).exists():
                    nctx = NcuMetricsContext(ns.ncu_report_path, range_idx=0, action_idx=0)
                    ncu_summary = nctx.summary()
                    ptx_corr = PtxSourceCorrelator(ns.ncu_report_path)
                kernel_opt_tooling.CURRENT_REGION_DAG_CTX = rctx
                kernel_opt_tooling.CURRENT_NCU_CTX = nctx
                kernel_opt_tooling.CURRENT_PTX_CORRELATOR = ptx_corr
            except Exception:
                kernel_opt_tooling.CURRENT_REGION_DAG_CTX = None
                kernel_opt_tooling.CURRENT_NCU_CTX = None
                kernel_opt_tooling.CURRENT_PTX_CORRELATOR = None

            return region_summary, ncu_summary

        while budget > 0:
            path: List[Tuple[str, str]] = []  # (state_hash, action_id)
            s = root_hash
            depth = 0

            local_visits: Dict[Tuple[str, str], int] = {}
            seen_states_in_path: set[str] = set([s])

            # ------ Selection ------
            while depth < MAX_DEPTH:
                ns = NodeStats(**search_state["nodes"][s])
                ns.edges = {k: v if isinstance(v, EdgeStats) else EdgeStats(**v) for k, v in ns.edges.items()}
                search_state["nodes"][s] = dataclasses.asdict(ns)

                # Progressive widening limit
                limit = max(1, PW_K0 + int(PW_K1 * (ns.visits ** PW_ALPHA)))

                # If no actions yet -> expand here
                if not ns.edges:
                    break

                # P-UCB(s): score = Qbar + sqrt(2 ln N / n)
                cand: List[Tuple[float, str]] = []
                for a, es in ns.edges.items():
                    Qbar, Nsa = es.Qbar, es.Nsa
                    U = Qbar + math.sqrt(max(0.0, 2.0 * math.log(max(1.0, float(ns.visits)) + 1.0)) / (Nsa + 1.0))
                    U -= ANTI_CYCLE_LAMBDA * float(local_visits.get((s, a), 0))  # >>> CHANGED: single penalty
                    if es.child and es.child in seen_states_in_path:
                        U -= 1e9  # hard discourage cycles within path
                    cand.append((U, a))

                if not cand:
                    break
                cand.sort(reverse=True)
                cand_actions = [a for _, a in cand[:limit]]

                # ε-greedy selection
                a_star = random.choice(cand_actions) if random.random() < EPSILON_SELECT else cand_actions[0]
                local_visits[(s, a_star)] = 1 + local_visits.get((s, a_star), 0)
                path.append((s, a_star))
                es = ns.edges[a_star]

                # Descend if child exists; else expand here
                if es.child:
                    s = es.child
                    seen_states_in_path.add(s)
                    depth += 1
                    continue
                else:
                    break

            if not path:
                break

            leaf_state, leaf_action = path[-1]
            leaf_ns = NodeStats(**search_state["nodes"][leaf_state])
            leaf_ns.edges = {k: v if isinstance(v, EdgeStats) else EdgeStats(**v) for k, v in leaf_ns.edges.items()}

            # ------ Expansion on first visit ------
            region_summary, ncu_summary = _prepare_contexts(leaf_ns)

            if not leaf_ns.edges:
                # Ask LLM for proposals (region/NCU summaries optional)
                proposals_obj: Optional[Dict[str, Any]] = None
                try:
                    proposals_obj = self.llm.propose(k, region_summary=region_summary, ncu_summary=ncu_summary)  # if signature supports it
                except TypeError:
                    proposals_obj = self.llm.propose(k, region_summary=region_summary)  # backward-compat
                if not proposals_obj or not proposals_obj.get("candidates"):
                    # Count a visit and try another iteration
                    leaf_ns.visits += 1
                    search_state["nodes"][leaf_state] = dataclasses.asdict(leaf_ns)
                    json_dump(search_state, search_state_path)
                    continue

                # === ReGraphT-style relabel(LLM, τ, O) ===
                steps = proposals_obj["candidates"]
                steps = relabel_methods(self, search_state, steps)  # >>> CHANGED: unified relabel

                # Offline ReGraph constraint at first expansion
                last_m = leaf_ns.last_method or "START"
                allowed = set(self.regraph.get("edges", {}).get(last_m, {}).keys())
                if allowed:
                    filtered = [c for c in steps if c.get("method_canonical") in allowed]
                    if filtered:
                        steps = filtered

                # Register ALL proposals as actions with canonicalized methods
                def _aid(c: Dict[str, Any]) -> str:
                    key = (c.get("method_canonical","") + "\n" + (c.get("code","") or "")).encode("utf-8")
                    return sha256_bytes(key)[:16]

                for c in steps:
                    aid = _aid(c)
                    if aid not in leaf_ns.edges:
                        leaf_ns.edges[aid] = EdgeStats()
                    leaf_ns.actions[aid] = c

                # Save node after registering actions
                search_state["nodes"][leaf_state] = dataclasses.asdict(leaf_ns)
                json_dump(search_state, search_state_path)

            # Choose one action to materialize now (greedy on Q; ties arbitrary)
            leaf_ns = NodeStats(**search_state["nodes"][leaf_state])
            if leaf_ns.edges:
                scored = [((es.Qbar, a)) for a, es in leaf_ns.edges.items()]
                scored.sort(key=lambda t: t[0], reverse=True)
                leaf_action = scored[0][1]
                path[-1] = (leaf_state, leaf_action)

            chosen = leaf_ns.actions.get(leaf_action, {})
            def _candidate_to_variant(code_text: str, launch_update: Optional[Dict[str,Any]] = None) -> Dict[str, Any]:
                code_text = code_text or ""
                is_diff = code_text.startswith("--- ") or "+++" in code_text or code_text.startswith("diff --git")
                if is_diff:
                    return {"patch": {"files": [{"diff": code_text}]}, "launch_update": (launch_update or {})}
                else:
                    return {"full_source_code": code_text, "launch_update": (launch_update or {})}
            prop = _candidate_to_variant(chosen.get("code",""), chosen.get("launch_update"))

            # ------ Materialize chosen action ------
            gpu = self.gpus[gpu_rr % len(self.gpus)]; gpu_rr += 1
            job_id = str(uuid.uuid4())
            vdir = self.kernel_dir(k) / "variants" / job_id
            vdir.mkdir(parents=True, exist_ok=True)

            control = {
                "kernel": k.to_dict(),
                "variant": prop,
                "workdir": str(vdir),
                "ncu": {"enabled": True, "collect": True},
                "timing": DEFAULT_TIMING,
                "base_source_code": leaf_ns.source_code,
            }
            json_dump(control, vdir / "control.json")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            p = subprocess.Popen([sys.executable, __file__, "--worker", str(vdir / "control.json")],
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = p.communicate()
            write_text(vdir / "stdout.log", out)
            write_text(vdir / "stderr.log", err)

            ok = False; result = {}
            if (vdir / "result.json").exists():
                result = json_load(vdir / "result.json")
                ok = bool(result.get("ok", False))

            ms = float(result.get("mean_ms", 1e9))
            worker_child_hash = result.get("state_hash")
            child_id = worker_child_hash if USE_HASH_FOR_NODE_ID else str(uuid.uuid4())

            # Reward (Eq.3)
            if not ok:
                reward = -1.0
            else:
                speedup = (baseline_ms / ms) if ms > 0 else 0.0
                reward = speedup if speedup >= 1.0 else (speedup - 1.0)

            # If success, add/merge child node and edge pointer
            if ok and child_id:
                materialized_source = result.get("materialized_source", "") or leaf_ns.source_code
                if child_id not in search_state["nodes"]:
                    search_state["nodes"][child_id] = dataclasses.asdict(NodeStats(
                        visits=0,
                        source_code=materialized_source,
                        ncu=result.get("ncu_metrics", {"kernel_time_ms": ms}),
                        edges={}, actions={}, applied_actions=(leaf_ns.applied_actions or []) + [leaf_action],
                        applied_methods=(leaf_ns.applied_methods or []) + [chosen.get("method_canonical","")],
                        last_method=chosen.get("method_canonical",""),
                        impl_hash=worker_child_hash,
                        ptx_path=result.get("ptx_path"),
                        ncu_report_path=result.get("ncu_report_path")
                    ))
                # attach child pointer
                leaf_ns.edges[leaf_action].child = child_id

                # Incumbent best
                if ms < best_ms:
                    best_ms = ms
                    search_state["best_ms"] = ms
                    search_state["best_state_hash"] = child_id
                    search_state["best_variant_dir"] = str(vdir)
                    with open(search_state["trace_path"], "a", encoding="utf-8") as f:
                        f.write(f"- {now_iso()}: **new best** {ms:.3f} ms via `{chosen.get('method_canonical','?')}` (reward {reward:.3f}) → {vdir}\n")
                    bdir = self.kernel_dir(k) / "best"
                    try:
                        if bdir.exists():
                            if bdir.is_symlink() or bdir.is_file():
                                bdir.unlink()
                            else:
                                shutil.rmtree(bdir)
                        os.symlink(vdir, bdir, target_is_directory=True)
                    except OSError:
                        if bdir.exists(): shutil.rmtree(bdir)
                        shutil.copytree(vdir, bdir)
                    json_dump({"best_ms": best_ms, "best_variant_dir": str(bdir), "action": leaf_action},
                              self.kernel_dir(k) / "best_summary.json")

            # Dataset logging (expansion step)
            self.rgds.append_transition({
                "t": now_iso(), "kernel": k.name, "arch": (k.device_profile.arch if k.device_profile else None),
                "from_method": leaf_ns.last_method or "START",
                "to_method": chosen.get("method_canonical"),
                "ok": ok, "mean_ms": ms, "baseline_ms": baseline_ms,
                "speedup": (baseline_ms / ms) if (ok and ms>0) else 0.0,
                "reward": reward, "candidate_id": leaf_action, "variant_dir": str(vdir)
            })

            # ------ Rollout (regularized) ------
            rollout_best = reward
            cur_state = leaf_ns.edges[leaf_action].child if ok and leaf_ns.edges.get(leaf_action) else None
            steps_taken = 0
            while ok and cur_state and steps_taken < ROLLOUT_MAX_STEPS:
                cur_ns = NodeStats(**search_state["nodes"][cur_state])
                cur_ns.edges = {k: v if isinstance(v, EdgeStats) else EdgeStats(**v) for k, v in cur_ns.edges.items()}

                # First visit? expand all successors
                if not cur_ns.edges:
                    # Ask LLM and relabel
                    region_summary, ncu_summary = _prepare_contexts(cur_ns)
                    props_obj = None
                    try:
                        props_obj = self.llm.propose(k, region_summary=region_summary, ncu_summary=ncu_summary)
                    except TypeError:
                        props_obj = self.llm.propose(k, region_summary=region_summary)
                    candlist = relabel_methods(self, search_state, props_obj.get("candidates", []) if props_obj else [])

                    # ReGraph constraint for rollout node
                    last_m = cur_ns.last_method or "START"
                    allowed = set(self.regraph.get("edges", {}).get(last_m, {}).keys())
                    if allowed:
                        filt = [c for c in candlist if c.get("method_canonical") in allowed]
                        if filt: candlist = filt

                    for c in candlist:
                        aid = sha256_bytes((c.get("method_canonical","")+"\n"+(c.get("code","") or "")).encode("utf-8"))[:16]
                        if aid not in cur_ns.edges:
                            cur_ns.edges[aid] = EdgeStats()
                        cur_ns.actions[aid] = c

                    search_state["nodes"][cur_state] = dataclasses.asdict(cur_ns)
                    json_dump(search_state, search_state_path)
                    if not cur_ns.edges:
                        break

                # π(a|s): argmax_a [Q - λ N] w.p. 1-ε; else random
                scored = [((es.Qbar - LAMBDA_REG * es.Nsa), a) for a, es in cur_ns.edges.items()]
                a_roll = random.choice(list(cur_ns.edges.keys())) if random.random() < EPSILON_ROLLOUT else sorted(scored, reverse=True)[0][1]
                cand = cur_ns.actions.get(a_roll, {})

                # Materialize rollout step
                vdir2 = self.kernel_dir(k) / "variants" / str(uuid.uuid4())
                vdir2.mkdir(parents=True, exist_ok=True)
                def _to_variant(c):
                    code = c.get("code","") or ""
                    is_diff = code.startswith("--- ") or "+++" in code or code.startswith("diff --git")
                    return {"patch":{"files":[{"diff":code}]}} if is_diff else {"full_source_code":code}
                control2 = {"kernel": k.to_dict(), "variant": _to_variant(cand), "workdir": str(vdir2),
                            "ncu":{"enabled": True, "collect": True}, "timing": DEFAULT_TIMING,
                            "base_source_code": cur_ns.source_code}
                json_dump(control2, vdir2 / "control.json")
                env2 = os.environ.copy(); env2["CUDA_VISIBLE_DEVICES"] = self.gpus[(steps_taken + gpu_rr) % len(self.gpus)]
                p2 = subprocess.Popen([sys.executable, __file__, "--worker", str(vdir2 / "control.json")],
                                      env=env2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out2, err2 = p2.communicate()
                write_text(vdir2 / "stdout.log", out2); write_text(vdir2 / "stderr.log", err2)

                ok2 = False; res2 = {}
                if (vdir2 / "result.json").exists():
                    res2 = json_load(vdir2 / "result.json"); ok2 = bool(res2.get("ok", False))

                parent_state = cur_state  # >>> CHANGED: remember parent before changing cur_state

                if not ok2:
                    r2 = -1.0
                    ok = False
                else:
                    ms2 = float(res2.get("mean_ms", 1e9))
                    sp2 = (baseline_ms / ms2) if ms2 > 0 else 0.0
                    r2 = sp2 if sp2 >= 1.0 else (sp2 - 1.0)
                    # attach next child
                    child2 = res2.get("state_hash")
                    child2 = child2 if USE_HASH_FOR_NODE_ID else str(uuid.uuid4())
                    cur_ns.edges[a_roll].child = child2
                    if child2 not in search_state["nodes"]:
                        search_state["nodes"][child2] = dataclasses.asdict(NodeStats(
                            visits=0, source_code=res2.get("materialized_source","") or cur_ns.source_code,
                            ncu=res2.get("ncu_metrics",{}), edges={}, actions={}, applied_actions=(cur_ns.applied_actions or [])+[a_roll],
                            applied_methods=(cur_ns.applied_methods or [])+[cand.get("method_canonical","")],
                            last_method=cand.get("method_canonical",""),
                            impl_hash=res2.get("state_hash"), ptx_path=res2.get("ptx_path"), ncu_report_path=res2.get("ncu_report_path")
                        ))
                    cur_state = child2

                # update edge stats for rollout edge (persist to PARENT)  >>> CHANGED
                cur_ns.visits += 1
                esr = cur_ns.edges[a_roll]
                esr.Nsa += 1.0; esr.Qsum += r2; esr.Qbar = esr.Qsum / max(1.0, esr.Nsa)
                search_state["nodes"][parent_state] = dataclasses.asdict(cur_ns)  # write back to parent
                json_dump(search_state, search_state_path)

                rollout_best = max(rollout_best, r2)
                steps_taken += 1

                # dataset logging (rollout transitions)
                self.rgds.append_transition({
                    "t": now_iso(), "kernel": k.name, "arch": (k.device_profile.arch if k.device_profile else None),
                    "from_method": cur_ns.last_method or "START",
                    "to_method": cand.get("method_canonical"),
                    "ok": ok2, "mean_ms": ms2 if ok2 else None, "baseline_ms": baseline_ms,
                    "speedup": (baseline_ms / ms2) if (ok2 and ms2>0) else 0.0,
                    "reward": r2, "candidate_id": a_roll, "variant_dir": str(vdir2)
                })

            # ------ Backpropagate max rollout reward ------
            search_state["nodes"][leaf_state] = dataclasses.asdict(leaf_ns)
            for (state_h, action_h) in path:
                nsi = NodeStats(**search_state["nodes"][state_h])
                nsi.edges = {k: v if isinstance(v, EdgeStats) else EdgeStats(**v) for k, v in nsi.edges.items()}
                esi = nsi.edges[action_h]
                nsi.visits += 1
                esi.Nsa += 1.0
                esi.Qsum += rollout_best
                esi.Qbar = esi.Qsum / max(1.0, esi.Nsa)
                search_state["nodes"][state_h] = dataclasses.asdict(nsi)

            # Event log (human trace)
            evt = {
                "t": now_iso(), "kernel": k.name, "job_id": str(vdir.name), "gpu": self.gpus[(gpu_rr-1) % len(self.gpus)],
                "leaf_state": leaf_state, "action": leaf_action, "candidate_id": leaf_action,
                "mean_ms": ms, "reward": reward, "ok": ok,
                "child_state": child_id if ok else None,
                "variant_dir": str(vdir), "worker_child_hash": worker_child_hash,
                "canonical_method": chosen.get("method_canonical")
            }
            append_jsonl(Path(search_state["events_path"]), evt)

            # Persist
            json_dump(search_state, search_state_path)
            budget -= 1
            if budget <= 0:
                break

        print(f"[{k.name}] Finished. Best: {best_ms:.3f} ms. Artifacts under {self.kernel_dir(k)}")

# ----------------------------- Worker Mode -------------------------------
def worker_main(control_path: str) -> int:
    """
    Worker:
      - Materializes source: full_source_code > base_source_code+patch > base_source_code.
      - Triton: uses triton_runner; CUDA: nvcc -> cubin/ptx + cuda_runner.
      - Writes result.json with (ok, mean_ms, std_ms, state_hash, ncu_metrics, materialized_source, ptx_path, ncu_report_path).
    """
    try:
        control = json_load(Path(control_path))
        k = KernelCode.model_validate(control["kernel"])
        variant = control.get("variant", {}) or {}
        workdir = Path(control["workdir"])
        timing = control.get("timing", DEFAULT_TIMING)
        ncu_control = control.get("ncu", {}) or {}
        base_source_code = control.get("base_source_code", k.source_code)
        launch_update = variant.get("launch_update", {}) or {}

        if k.kernel_type == "triton":
            target_rel = "kernel_module.py"
        elif k.kernel_type == "cuda":
            target_rel = "kernel.cu"
        else:
            json_dump({"ok": False, "error": f"unknown kernel_type {k.kernel_type}"}, workdir / "result.json")
            return 1

        target_path = workdir / target_rel
        workdir.mkdir(parents=True, exist_ok=True)

        # Materialize source (full override > base + patch > base)
        if variant.get("full_source_code"):
            write_text(target_path, variant["full_source_code"])
        else:
            write_text(target_path, base_source_code)
            patch_obj = variant.get("patch", {})
            if patch_obj and patch_obj.get("files"):
                if not apply_unified_diff(workdir, patch_obj["files"]):
                    json_dump({"ok": False, "error": "patch apply failed"}, workdir / "result.json")
                    return 1

        # Save state after materialization
        materialized_source = target_path.read_text(encoding="utf-8")
        state_hash = sha256_str(materialized_source + json.dumps(launch_update, sort_keys=True))
        save_partial_state(workdir, "materialized", {
            "source_hash": state_hash,
            "source_path": str(target_path),
            "kernel_type": k.kernel_type
        })

        if k.kernel_type == "triton":
            runner_config = {
                "root_path": str(Path(__file__).parent.resolve()),
                "kernel_module_path": str(target_path.resolve()),
                "kernel_name": get_metadata_value(k.metadata, "kernel_name", None),
                "invocation_example": k.invocation_example or "",
                "launch_update": launch_update,
                "io_contract": k.io.to_dict() if k.io else {},
                "timing": timing,
                "result_path": str((workdir / "runner_result.json").resolve())
            }
            json_dump(runner_config, workdir / "runner_config.json")

            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent.resolve())
            base_cmd = [sys.executable, "-m", "workers.triton_runner", str((workdir / "runner_config.json").resolve())]
            rc, out, err, ncu_report_path = run_runner_with_optional_ncu(
                base_cmd, workdir, env, runner_config.get("kernel_name"), ncu_control
            )
            write_text(workdir / "runner_stdout.log", out)
            write_text(workdir / "runner_stderr.log", err)

            if rc != 0 or not (workdir / "runner_result.json").exists():
                json_dump({"ok": False, "error": "runner failed", "rc": rc}, workdir / "result.json")
                return 1

            r = json_load(workdir / "runner_result.json")

            save_partial_state(workdir, "timed", {
                "mean_ms": r.get("mean_ms", 1e9),
                "std_ms": r.get("std_ms", 0.0),
                "ok": r.get("ok", False),
                "launch_update_applied": r.get("launch_update_applied", False)
            })

            # Placeholders for PTX/SASS until you wire Triton → PTX extraction
            ptx_path = workdir / "kernel.ptx"
            write_text(ptx_path, "// Triton PTX placeholder")
            write_text(workdir / "kernel.sass", "// SASS placeholder")
            outj = {
                "ok": bool(r.get("ok", False)),
                "mean_ms": float(r.get("mean_ms", 1e9)),
                "std_ms": float(r.get("std_ms", 0.0)),
                "state_hash": state_hash,
                "ncu_metrics": {"kernel_time_ms": float(r.get("mean_ms", 0.0))},
                "materialized_source": materialized_source,
                "launch_update_applied": bool(r.get("launch_update_applied", False)),
                "ptx_path": str(ptx_path),
                "ncu_report_path": ncu_report_path
            }
            json_dump(outj, workdir / "result.json")
            return 0

        else:  # CUDA
            runner_mode = get_metadata_value(k.metadata, "runner_mode", None)
            vllm_meta = get_metadata_value(k.metadata, "vllm", {}) or {}

            if runner_mode == "vllm":
                runner_config = {
                    "root_path": str(Path(__file__).parent.resolve()),
                    "mode": "vllm",
                    "op_name": vllm_meta.get("op_name", get_metadata_value(k.metadata, "kernel_name", "")),
                    "entry_path": vllm_meta.get("entry_path") or vllm_meta.get("entry"),
                    "reference_path": vllm_meta.get("reference_path"),
                    "cases": vllm_meta.get("shape_distribution", []),
                    "dtype": vllm_meta.get("dtype", "float16"),
                    "seed": vllm_meta.get("seed"),
                    "timing": timing,
                    "result_path": str((workdir / "runner_result.json").resolve())
                }
                json_dump(runner_config, workdir / "runner_config.json")

                env = os.environ.copy()
                env['PYTHONPATH'] = str(Path(__file__).parent.resolve())
                base_cmd = [sys.executable, "-m", "workers.cuda_runner", str((workdir / "runner_config.json").resolve())]
                rc, out, err, ncu_report_path = run_runner_with_optional_ncu(
                    base_cmd, workdir, env, None, ncu_control
                )
                write_text(workdir / "runner_stdout.log", out)
                write_text(workdir / "runner_stderr.log", err)

                if rc != 0 or not (workdir / "runner_result.json").exists():
                    json_dump({"ok": False, "error": "runner failed", "rc": rc}, workdir / "result.json")
                    return 1

                r = json_load(workdir / "runner_result.json")

                save_partial_state(workdir, "timed", {
                    "mean_ms": r.get("mean_ms", 1e9),
                    "std_ms": r.get("std_ms", 0.0),
                    "ok": r.get("ok", False)
                })

                final_state_hash = sha256_str(materialized_source + json.dumps(runner_config, sort_keys=True))
                outj = {
                    "ok": bool(r.get("ok", False)),
                    "mean_ms": float(r.get("mean_ms", 1e9)),
                    "std_ms": float(r.get("std_ms", 0.0)),
                    "state_hash": final_state_hash,
                    "ncu_metrics": {"kernel_time_ms": float(r.get("mean_ms", 0.0))},
                    "materialized_source": materialized_source,
                    "ptx_path": "",
                    "ncu_report_path": ncu_report_path
                }
                if "max_diff" in r:
                    outj["max_diff"] = r["max_diff"]
                json_dump(outj, workdir / "result.json")
                return 0

            nvcc = shutil.which("nvcc")
            nvdisasm = shutil.which("nvdisasm") or shutil.which("cuobjdump")
            if not nvcc:
                json_dump({"ok": False, "error": "nvcc not on PATH"}, workdir / "result.json")
                return 1

            arch = (k.device_profile.arch if k.device_profile else "sm_90")
            src_text = target_path.read_text(encoding="utf-8")
            src_hash = sha256_str(arch + "::" + src_text)
            COMPILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached_ptx  = COMPILE_CACHE_DIR / f"{src_hash}.ptx"
            cached_cubin= COMPILE_CACHE_DIR / f"{src_hash}.cubin"

            if cached_ptx.exists() and cached_cubin.exists():
                shutil.copy(cached_ptx, workdir / "kernel.ptx")
                shutil.copy(cached_cubin, workdir / "kernel.cubin")
            else:
                rc, out, err = run_subprocess([nvcc, "-O3", f"-arch={arch}", "-Xptxas", "-v", "-ptx", target_rel, "-o", "kernel.ptx"], cwd=workdir)
                write_text(workdir / "nvcc_ptx_stdout.log", out)
                write_text(workdir / "nvcc_ptx_stderr.log", err)
                if rc != 0:
                    json_dump({"ok": False, "error": "nvcc -ptx failed"}, workdir / "result.json")
                    return 1
                rc, out, err = run_subprocess([nvcc, "-O3", f"-arch={arch}", "-Xptxas", "-v", "-cubin", target_rel, "-o", "kernel.cubin"], cwd=workdir)
                write_text(workdir / "nvcc_cubin_stdout.log", out)
                write_text(workdir / "nvcc_cubin_stderr.log", err)
                if rc != 0:
                    json_dump({"ok": False, "error": "nvcc -cubin failed"}, workdir / "result.json")
                    return 1
                try:
                    shutil.copy(workdir / "kernel.ptx", cached_ptx)
                    shutil.copy(workdir / "kernel.cubin", cached_cubin)
                except Exception:
                    pass

            save_partial_state(workdir, "compiled", {
                "ptx_path": str(workdir / "kernel.ptx"),
                "cubin_path": str(workdir / "kernel.cubin"),
                "arch": arch
            })

            if nvdisasm:
                if "nvdisasm" in nvdisasm:
                    rc, out, err = run_subprocess([nvdisasm, "kernel.cubin"], cwd=workdir)
                else:
                    rc, out, err = run_subprocess([nvdisasm, "--dump-sass", "kernel.cubin"], cwd=workdir)
                write_text(workdir / "kernel.sass", out if rc == 0 else f"disasm failed: {err}")

            effective_io = k.io.to_dict() if k.io else None
            if effective_io and "launch" in effective_io:
                if "grid" in launch_update:
                    effective_io["launch"]["grid"].update(launch_update["grid"])
                if "block" in launch_update:
                    effective_io["launch"]["block"].update(launch_update["block"])

            runner_config = {
                "root_path": str(Path(__file__).parent.resolve()),
                "kernel_cubin_path": str((workdir / "kernel.cubin").resolve()),
                "kernel_name": get_metadata_value(k.metadata, "kernel_name", "kernel"),
                "io_contract": effective_io or {},
                "timing": timing,
                "result_path": str((workdir / "runner_result.json").resolve())
            }
            json_dump(runner_config, workdir / "runner_config.json")

            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent.resolve())
            base_cmd = [sys.executable, "-m", "workers.cuda_runner", str((workdir / "runner_config.json").resolve())]
            rc, out, err, ncu_report_path = run_runner_with_optional_ncu(
                base_cmd, workdir, env, runner_config.get("kernel_name"), ncu_control
            )
            write_text(workdir / "runner_stdout.log", out)
            write_text(workdir / "runner_stderr.log", err)

            if rc != 0 or not (workdir / "runner_result.json").exists():
                json_dump({"ok": False, "error": "runner failed", "rc": rc}, workdir / "result.json")
                return 1

            r = json_load(workdir / "runner_result.json")

            save_partial_state(workdir, "timed", {
                "mean_ms": r.get("mean_ms", 1e9),
                "std_ms": r.get("std_ms", 0.0),
                "ok": r.get("ok", False)
            })

            final_state_hash = sha256_str(materialized_source + json.dumps(effective_io or {}, sort_keys=True))
            outj = {
                "ok": bool(r.get("ok", False)),
                "mean_ms": float(r.get("mean_ms", 1e9)),
                "std_ms": float(r.get("std_ms", 0.0)),
                "state_hash": final_state_hash,
                "ncu_metrics": {"kernel_time_ms": float(r.get("mean_ms", 0.0))},
                "materialized_source": materialized_source,
                "ptx_path": str(workdir / "kernel.ptx"),
                "ncu_report_path": ncu_report_path
            }
            json_dump(outj, workdir / "result.json")
            return 0

    except Exception as e:
        tb = traceback.format_exc()
        out = {"ok": False, "error": str(e), "traceback": tb}
        try:
            workdir = Path(control["workdir"]) if "control" in locals() else Path(control_path).parent
            partial_state = get_latest_phase_data(workdir)
            if partial_state:
                out["partial_state"] = partial_state
        except Exception:
            pass
        json_dump(out, Path(control_path).parent / "result.json")
        return 1

# ------------------------------- CLI -------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCGS + Region-DAG prior + Offline ReGraph for GPU kernel optimization")
    p.add_argument("--input", default="kernels.json", help="JSON spec file or directory with .json specs")
    p.add_argument("--outdir", default="out", help="Output directory")
    p.add_argument("--gpus", default="0", help="Comma-separated GPU indices (e.g., 0,1,2)")
    p.add_argument("--budget-builds", type=int, default=20, help="Max rollout builds per kernel")
    p.add_argument("--llm-model", type=str, default="gpt-5-mini", help="OpenAI Responses model")
    p.add_argument("--resume", action="store_true", help="Resume from existing outdir state")
    p.add_argument("--worker", nargs="?", help="(internal) Worker mode: path to control.json")
    # Offline ReGraph controls
    p.add_argument("--build-regraph", action="store_true", help="Scan dataset and rebuild offline ReGraph, then exit")
    p.add_argument("--regraph-autobuild", action="store_true", help="Rebuild ReGraph before running search")
    return p.parse_args(argv)

def main():
    args = parse_args()
    # Worker mode
    if args.worker:
        sys.exit(worker_main(args.worker))

    orch = Orchestrator(args)

    # Offline build-only mode
    if args.build_regraph:
        rg = orch.rgds.build_regraph()
        print(f"ReGraph built with {len(rg.get('nodes',{}))} nodes.")
        return

    kernels = orch.load_kernel_specs()
    for k in kernels:
        orch.baseline_or_resume(k)
        orch.run_mcgs_for_kernel(k)
    print("Done.")

if __name__ == "__main__":
    main()
