#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_kernels.py
-------------------
True Monte-Carlo Graph Search (MCGS) with a Region-DAG prior for optimizing Triton or CUDA kernels.

Key properties of this revision:
- TRUE MCGS:
  * Selection: multi-depth path from the root with PUCT at each visited node (progressive widening).
  * Expansion: exactly one new child created at the leaf via LLM or rule-based proposal.
  * Backpropagation: measured reward is propagated along the entire selected path (all edges).
  * Transpositions: graph, not tree. States keyed by 'state_hash' are re-used, edges point to those nodes.
  * NO greedy "promote child to root" behavior; the root stays the baseline. We maintain an 'incumbent best'.
- Dirichlet noise: applied (once) to root priors to encourage root exploration.
- Priors (Region-DAG): recomputed/merged for any node when fresh NCU metrics arrive, not only once.
- Hashing: worker is the single source-of-truth for state hashes; orchestrator never precomputes them.
- NCU cadence: '--ncu-every' triggers collection at selected nodes by visit cadence; results update priors.
- Patch application: uses system 'patch' for robust diff application; failures abort the variant to avoid no-op builds.
- CUDA compile cache: artifacts cached by (arch, sha256(source)) to skip recompiles.
- Worker exits nonzero on hard failure; orchestrator always reads result.json.
- Triton fixes:
  * Runner records 'launch_update_applied' (True/False). Warns when non-empty LAUNCH_UPDATE is unused.
  * If 'metadata.kernel_name' is present, runner invokes that function and passes LAUNCH_UPDATE explicitly.

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
 
# --------------------------- Constants & Config --------------------------
GRAPH_VERSION = "2025-11-11.mcgsv1"
MAX_DEPTH = 8
DEFAULT_RULE_CANDIDATE_LIMIT = 6
MIN_PRIOR_GAIN = 0.02
PUCT_C = 1.0
ROOT_NOISE_EPS = 0.25
ROOT_NOISE_ALPHA = 0.30
PW_K0, PW_K1, PW_ALPHA = 2, 3, 0.5
DEFAULT_TIMING = {"warmup": 10, "iters": 100, "repeat": 3}
COMPILE_CACHE_DIR = Path(os.environ.get("COMPILE_CACHE_DIR", "/tmp/kernel_compile_cache"))

LLM_SYSTEM_PROMPT = """You are generating safe, minimal optimization patches for GPU kernels.
You must output STRICT JSON with a single top-level object matching the provided schema.
You may only propose moves from the allowed vocabulary. Do NOT change algorithmic semantics.
If a unified diff cannot be produced reliably, include a 'full_source_code' override instead.
"""

# Canonical move vocabulary (extend over time and bump GRAPH_VERSION).
CANONICAL_MOVES = [
    {"name": "enable_async_pipeline", "brief": "Overlap gmem with compute via cp.async/TMA", "touches": ["gmem","overlap"], "params": {"stages":{"enum":[2,3]}}},
    {"name": "vectorize_global_loads", "brief": "Use ld/st.v2/v4/v8 when aligned", "touches": ["gmem"], "params": {"width":{"enum":[2,4,8]},"targets":{"enum":["LoadQ","LoadKV","LoadA","LoadB","X","Y","C"],"list":True}}},
    {"name": "switch_to_mma_or_wgmma", "brief": "Promote dot/FMA to MMA/WGMMA", "touches": ["compute"], "params": {"mma_kind":{"enum":["mma.sync","wgmma"]}}},
    {"name": "pad_tail_and_mask", "brief": "Pad tails for contiguous vectorization", "touches": ["gmem","control"], "params": {"dim":{"enum":["K","N"]},"multiple_of":{"enum":[8,16,32]}}},
    {"name": "change_block_sizes", "brief": "Retune BLOCK sizes or launch to shift occupancy/overlap", "touches": ["compute","gmem","overlap"], "params": {"BLOCK_M":"int","BLOCK_N":"int","BLOCK_K":"int","num_warps":"int","num_stages":"int"}},
    {"name": "cache_policy_cg", "brief": "Use .cg to reduce L1 pollution for streaming", "touches": ["gmem"], "params": {}},
    {"name": "avoid_atomics_reduce", "brief": "Restructure to avoid atomics", "touches": ["atomics","writeback"], "params": {}},
    {"name": "grouped_gemm_colmajor", "brief": "Grouped scheduling/col-major to improve reuse", "touches": ["compute","gmem"], "params": {}},
    {"name": "softmax_math_fusion", "brief": "Fuse/approx math to reduce SFU", "touches": ["sfu"], "params": {}},
]

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


# ------------------------ Region-DAG (prior model) ----------------------
@dataclass
class RegionCaps:
    compute_peak: float; bw_global: float; bw_shared: float; atomic_tp: float; sfu_tp: float

@dataclass
class RegionDagSummary:
    total_time_pred_ms: float
    stage_max: str
    caps: RegionCaps

def infer_region_caps_from_ncu(metrics: Dict[str, Any], profile: DeviceProfile) -> RegionCaps:
    dram_util = float(metrics.get("dram_util", 0.6))
    tensor_util = float(metrics.get("tensor_util", 0.4))
    sfu_util = float(metrics.get("sfu_util", 0.2))
    bw_global = (profile.hbm_bw_gbps * 1e9) / 8.0 * max(0.05, min(dram_util, 1.0))
    base_tflops = 120e12 if arch_sm_to_int(profile.arch) >= 90 else 80e12
    compute_peak = base_tflops * max(0.05, min(tensor_util, 1.0))
    bw_shared = 3.0e12
    atomic_tp = 1.0e12
    sfu_tp = 200e9 * max(0.05, min(sfu_util, 1.0))
    return RegionCaps(compute_peak, bw_global, bw_shared, atomic_tp, sfu_tp)

def region_dag_prior_from_metrics(metrics: Dict[str, Any], profile: DeviceProfile) -> Tuple[RegionDagSummary, Dict[str, Dict[str, float]]]:
    caps = infer_region_caps_from_ncu(metrics, profile)
    util = {
        "compute": float(metrics.get("tensor_util", 0.4)),
        "gmem": float(metrics.get("dram_util", 0.6)),
        "sfu": float(metrics.get("sfu_util", 0.2)),
        "atomics": float(metrics.get("atom_util", 0.1)),
        "smem": float(metrics.get("smem_util", 0.1)),
    }
    stage_max = max(util.items(), key=lambda kv: kv[1])[0]
    total_ms = float(metrics.get("kernel_time_ms", 1.0))
    summary = RegionDagSummary(total_ms, stage_max, caps)

    move_effects: Dict[str, Dict[str, float]] = {}
    def touches(name: str) -> List[str]:
        for m in CANONICAL_MOVES:
            if m["name"] == name: return m["touches"]
        return []

    for mv in CANONICAL_MOVES:
        name = mv["name"]; t = set(touches(name))
        affects_bottleneck = (stage_max in t) or ("overlap" in t and stage_max in ["gmem","compute"])
        base_q0 = 0.02
        if name == "enable_async_pipeline" and affects_bottleneck: base_q0 = 0.18
        elif name == "switch_to_mma_or_wgmma" and stage_max == "compute": base_q0 = 0.12
        elif name == "vectorize_global_loads" and stage_max == "gmem": base_q0 = 0.10
        elif name == "softmax_math_fusion" and stage_max == "sfu": base_q0 = 0.08
        elif name == "grouped_gemm_colmajor" and stage_max in ["compute","gmem"]: base_q0 = 0.06
        elif name == "avoid_atomics_reduce" and stage_max == "atomics": base_q0 = 0.10
        elif name == "cache_policy_cg" and stage_max == "gmem": base_q0 = 0.04
        elif name == "change_block_sizes": base_q0 = 0.04
        move_effects[name] = {"q0": base_q0, "p0": max(1e-4, base_q0)}

    z = sum(e["p0"] for e in move_effects.values()) or 1.0
    for e in move_effects.values(): e["p0"] /= z
    return summary, move_effects

# -------------------- OpenAI LLM-based candidate generation -------------
class LLMCandidateGenerator:
    def __init__(self, model: str, api_base: Optional[str] = None):
        self.model = model; self.api_base = api_base; self.client = None
        self.prev_id_by_kernel: Dict[str, str] = {}

    def _lazy_client(self):
        if self.client is None:
            try:
                from openai import OpenAI
            except Exception as e:
                raise RuntimeError("OpenAI SDK not installed. `pip install openai`") from e
            self.client = OpenAI(base_url=self.api_base) if self.api_base else OpenAI()

    def propose(self, kernel: KernelCode, region_summary: RegionDagSummary, allowed_moves: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        self._lazy_client()
        km = {
            "kernel_type": kernel.kernel_type,
            "name": kernel.name,
            "device_profile": _round_float_values(dataclasses.asdict(kernel.device_profile) if kernel.device_profile else {}),
            "region_summary": _round_float_values(dataclasses.asdict(region_summary)),
            "allowed_moves": allowed_moves,
            "schema": {
                "candidate_id": "uuid4 string",
                "moves": [{"name": "enum", "params": "object with small integers/enums only"}],
                "preconditions_asserted": ["string..."],
                "predicted_effect": {"timelines_touched": ["gmem|compute|sfu|atomics|overlap|control"], "gmem_scale": "float?", "comp_scale": "float?", "overlap": "bool?"},
                "patch": {"format":"unified-diff", "files":[{"path":"<file>", "diff":"<udiff>"}]},
                "full_source_code": "optional string",
                "launch_update": "optional Triton/CUDA launch knobs",
                "risk_notes": "string",
                "justification": "2-3 lines tying to bottleneck",
            },
        }
        src_preview = kernel.source_code
        inv_preview = (kernel.invocation_example or "")
        km_json = json.dumps(km, indent=2)
        user_content = [
            {"type": "input_text", "text": "KERNEL SOURCE:\n" + src_preview},
            {"type": "input_text", "text": "INVOCATION_EXAMPLE (may be empty):\n" + inv_preview},
            {"type": "input_text", "text": "SCHEMA (JSON):\n" + km_json},
            {"type": "input_text", "text": "Output STRICT JSON only. No prose. No markdown."},
        ]
        messages = [
            {"role": "system", "content": [{"type": "input_text", "text": LLM_SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]
        kwargs = dict(model=self.model, input=messages)
        prev_id = self.prev_id_by_kernel.get(kernel.name)
        if prev_id: kwargs["previous_response_id"] = prev_id
        try:
            resp = self.client.responses.create(**kwargs)
            self.prev_id_by_kernel[kernel.name] = resp.id
            raw_segments: List[str] = []
            if getattr(resp, "output", None):
                for output in resp.output:
                    for content_item in getattr(output, "content", []) or []:
                        text_val = getattr(content_item, "text", None)
                        if isinstance(text_val, str):
                            raw_segments.append(text_val)
            raw = "\n".join(raw_segments)
            raw_str = str(raw).strip()
            raw_str = re.sub(r"^```(json)?", "", raw_str).strip()
            raw_str = re.sub(r"```$", "", raw_str).strip()
            return json.loads(raw_str)
        except Exception as e:
            print(f"[LLM] generation failed: {e}", file=sys.stderr)
            return None

# ------------------------ Rule-based fallback candidates -----------------
def CANONICAL_MOVE_TOUCHES(name: str) -> List[str]:
    for m in CANONICAL_MOVES:
        if m["name"] == name:
            return m["touches"]
    return []

def rule_based_candidates(kernel: KernelCode, region_summary: RegionDagSummary,
                          allowed_names: Optional[List[str]]=None, limit: int = DEFAULT_RULE_CANDIDATE_LIMIT) -> List[Dict[str, Any]]:
    """Basic safe candidates when LLM unavailable."""
    allowed_names = set(allowed_names or [m["name"] for m in CANONICAL_MOVES])
    out = []
    if kernel.kernel_type == "cuda":
        if "change_block_sizes" in allowed_names:
            for bx in [64, 128, 256, 512]:
                out.append({
                    "candidate_id": str(uuid.uuid4()),
                    "moves": [{"name":"change_block_sizes","params":{}}],
                    "preconditions_asserted": [],
                    "predicted_effect": {"timelines_touched": ["compute","gmem"]},
                    "patch": {"format":"unified-diff","files":[]},
                    "full_source_code": None,
                    "launch_update": {"block": {"x": bx, "y": 1, "z": 1}},
                    "risk_notes": "",
                    "justification": f"Rule: block.x={bx} to shift occupancy",
                })
                if len(out) >= limit: break
    else:
        if "change_block_sizes" in allowed_names:
            for nw in [2, 4, 8]:
                out.append({
                    "candidate_id": str(uuid.uuid4()),
                    "moves": [{"name":"change_block_sizes","params":{}}],
                    "preconditions_asserted": [],
                    "predicted_effect": {"timelines_touched": ["compute","overlap"]},
                    "patch": {"format":"unified-diff","files":[]},
                    "full_source_code": None,
                    "launch_update": {"num_warps": nw},
                    "risk_notes": "",
                    "justification": f"Rule: num_warps={nw} to trade occupancy/throughput",
                })
                if len(out) >= limit: break
    return out

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

# --------------------------- Graph Structures ---------------------------
@dataclass
class EdgeStats:
    Qsum: float = 0.0
    Nsa:  float = 0.0
    Qbar: float = 0.0
    P0:   float = 0.0
    Q0:   float = 0.0
    N0:   float = 2.0
    child: Optional[str] = None      # child state hash if expanded

@dataclass
class NodeStats:
    visits: int = 0
    priors_built: bool = False
    source_code: str = ""
    ncu: Dict[str, Any] = field(default_factory=dict)
    edges: Dict[str, EdgeStats] = field(default_factory=dict)   # action -> EdgeStats
    applied_actions: List[str] = field(default_factory=list)    # descriptive: sequence so far

# --------------------------- Orchestrator Core ---------------------------
class Orchestrator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.outdir = Path(args.outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.gpus = [g.strip() for g in args.gpus.split(",") if g.strip()!=""]
        if not self.gpus:
            raise ValueError("--gpus must specify at least one device index")
        self.ncu_every = int(args.ncu_every)
        self.min_pred_gain = float(args.min_pred_gain)
        self.llm = None
        if not args.no_llm:
            self.llm = LLMCandidateGenerator(model=args.llm_model, api_base=os.environ.get("OPENAI_BASE_URL"))
        json_dump({"graph_version": GRAPH_VERSION, "canonical_moves": CANONICAL_MOVES}, self.outdir / "global_tactic_graph.json")

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

    # ---------- Priors ----------
    def _build_or_merge_priors(self, k: KernelCode, ns: NodeStats) -> RegionDagSummary:
        metrics = ns.ncu or {"kernel_time_ms": 1.0}
        profile = k.device_profile or DeviceProfile("unknown", "sm_90", 1, 65536, 228000, 1500.0)
        summary, move_effects = region_dag_prior_from_metrics(metrics, profile)
        for mv in CANONICAL_MOVES:
            a = mv["name"]
            eff = move_effects.get(a, {"p0": 0.0, "q0": 0.0})
            es = ns.edges.get(a)
            if es is None:
                ns.edges[a] = EdgeStats(P0=eff["p0"], Q0=eff["q0"], N0=2.0, child=None)
            else:
                if es.Nsa < 3:
                    es.P0 = eff["p0"]; es.Q0 = eff["q0"]
                else:
                    es.P0 = es.P0 or eff["p0"]; es.Q0 = es.Q0 or eff["q0"]
        ns.priors_built = True
        return summary

    def _mix_root_dirichlet(self, ns: NodeStats):
        # Apply once to root priors
        names = [m["name"] for m in CANONICAL_MOVES]
        noise = [random.gammavariate(ROOT_NOISE_ALPHA, 1.0) for _ in names]
        z = sum(noise) or 1.0
        noise = [x/z for x in noise]
        for name, n in zip(names, noise):
            if name not in ns.edges: continue
            ns.edges[name].P0 = (1.0 - ROOT_NOISE_EPS) * ns.edges[name].P0 + ROOT_NOISE_EPS * n

    # ---------- Baseline ----------
    def baseline_or_resume(self, k: KernelCode) -> Tuple[Dict[str,Any], Dict[str,Any]]:
        kdir = self.kernel_dir(k)
        state_path = kdir / "search_state.json"
        if self.args.resume and state_path.exists():
            return json_load(state_path), json_load(kdir / "baseline" / "run.json")

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
            "ncu": {"enabled": self.ncu_every > 0, "collect": True},
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
        root_hash = run["state_hash"]

        # Graph init
        search_state = {
            "graph_version": GRAPH_VERSION,
            "baseline_ms": run.get("mean_ms", 1e8),
            "best_ms": run.get("mean_ms", 1e8),
            "best_state_hash": root_hash,
            "best_variant_dir": str(bdir),
            "root_state_hash": root_hash,
            "nodes": { root_hash: dataclasses.asdict(NodeStats(
                visits=0, priors_built=False,
                source_code=baseline_source,
                ncu=run.get("ncu_metrics", {"kernel_time_ms": run.get("mean_ms", 0.0)}),
                edges={}, applied_actions=[]
            ))},
            "events_path": str(self.kernel_dir(k) / "events.jsonl"),
            "trace_path": str(self.kernel_dir(k) / "trace.md"),
        }

        # Write trace header
        write_text(Path(search_state["trace_path"]),
                   f"# Optimization trace for `{k.name}`\n\n- Baseline mean: **{run.get('mean_ms', 0.0):.3f} ms** (std {run.get('std_ms',0.0):.3f})\n- Device: {k.device_profile.gpu_name if k.device_profile else 'unknown'} / {k.device_profile.arch if k.device_profile else ''}\n- Graph version: `{GRAPH_VERSION}`\n\n")

        # Build priors at root + apply Dirichlet noise once
        root_ns = NodeStats(**search_state["nodes"][root_hash])
        root_ns.edges = {k: v if isinstance(v, EdgeStats) else EdgeStats(**v) for k, v in root_ns.edges.items()}
        _ = self._build_or_merge_priors(k, root_ns)
        self._mix_root_dirichlet(root_ns)
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

        # Early failure detection: stop MCGS if baseline has inf (baseline execution failed)
        if math.isinf(baseline_ms):
            print(f"[MCGS] Skipping kernel '{k.name}': baseline execution failed (mean_ms = inf)")
            return

        budget = int(self.args.budget_builds)
        gpu_rr = 0

        while budget > 0:
            path: List[Tuple[str, str]] = []  # (state_hash, action)
            s = root_hash
            depth = 0

            # ------ Selection: walk from root using PUCT ------
            while depth < MAX_DEPTH:
                ns = NodeStats(**search_state["nodes"][s])
                ns.edges = {k: v if isinstance(v, EdgeStats) else EdgeStats(**v) for k, v in ns.edges.items()}
                summary = self._build_or_merge_priors(k, ns) if not ns.priors_built else \
                          self._build_or_merge_priors(k, ns)  # merge priors if ncu updated
                # Progressive widening limit
                limit = max(1, PW_K0 + int(PW_K1 * (ns.visits ** PW_ALPHA)))

                # Candidate actions: exclude those that don't touch bottleneck
                cand: List[Tuple[float, str]] = []
                for a, es in ns.edges.items():
                    touches = set(CANONICAL_MOVE_TOUCHES(a))
                    touches_bottleneck = summary.stage_max in touches or ("overlap" in touches and summary.stage_max in ["gmem","compute"])
                    if not touches_bottleneck:
                        continue
                    # Proper pruning rules
                    Qbar = es.Qbar if es.Nsa > 0 else es.Q0
                    Nsa = es.Nsa if es.Nsa > 0 else es.N0
                    if es.Nsa == 0 and es.Q0 < self.min_pred_gain:
                        continue
                    if es.Nsa > 5 and Qbar < self.min_pred_gain:
                        continue
                    U = Qbar + PUCT_C * es.P0 * (math.sqrt(max(1.0, float(ns.visits))) / (1.0 + Nsa))
                    cand.append((U, a))

                if not cand:
                    # Nothing promising; stop selection here
                    break

                cand.sort(reverse=True)
                cand_actions = [a for _, a in cand[:limit]]

                # Choose best action
                a_star = cand_actions[0]
                path.append((s, a_star))
                es = ns.edges[a_star]

                # Descend if child exists; otherwise expand here
                if es.child:
                    s = es.child
                    depth += 1
                    continue
                else:
                    # Expand at this leaf
                    break

            # If path ended with all expanded actions at deepest node, consume a visit and continue
            if not path:
                # Root had nothing actionable (unlikely); stop
                break

            leaf_state, leaf_action = path[-1]
            leaf_ns = NodeStats(**search_state["nodes"][leaf_state])
            leaf_ns.edges = {k: v if isinstance(v, EdgeStats) else EdgeStats(**v) for k, v in leaf_ns.edges.items()}

            # ------ Expansion: create child for (leaf_state, leaf_action) ------
            # Prepare a proposal (LLM constrained to this action or rule-based)
            summary_leaf = self._build_or_merge_priors(k, leaf_ns)
            proposals: List[Dict[str,Any]] = []
            if self.llm:
                allowed = [m for m in CANONICAL_MOVES if m["name"] == leaf_action]
                cand = self.llm.propose(k, summary_leaf, allowed)
                if cand: proposals.append(cand)
            if not proposals:
                proposals.extend(rule_based_candidates(k, summary_leaf, allowed_names=[leaf_action], limit=1))
            if not proposals:
                # Mark a visit and continue
                leaf_ns.visits += 1
                search_state["nodes"][leaf_state] = dataclasses.asdict(leaf_ns)
                json_dump(search_state, search_state_path)
                continue

            prop = proposals[0]
            if not prop.get("moves") or prop["moves"][0]["name"] != leaf_action:
                # Skip bad proposal; account visit
                leaf_ns.visits += 1
                search_state["nodes"][leaf_state] = dataclasses.asdict(leaf_ns)
                json_dump(search_state, search_state_path)
                continue

            # Launch worker
            gpu = self.gpus[gpu_rr % len(self.gpus)]; gpu_rr += 1
            job_id = str(uuid.uuid4())
            vdir = self.kernel_dir(k) / "variants" / job_id
            vdir.mkdir(parents=True, exist_ok=True)
            control = {
                "kernel": k.to_dict(),
                "variant": prop,
                "workdir": str(vdir),
                "ncu": {"enabled": self.ncu_every > 0, "collect": (leaf_ns.visits % self.ncu_every == 0) if self.ncu_every>0 else False},
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
            child_hash = result.get("state_hash")
            reward = max(0.0, baseline_ms / ms - 1.0) if ok else 0.0

            # If success, add/merge child node and edge
            if ok and child_hash:
                materialized_source = result.get("materialized_source", "") or leaf_ns.source_code
                child_entry = search_state["nodes"].get(child_hash)
                if not child_entry:
                    search_state["nodes"][child_hash] = dataclasses.asdict(NodeStats(
                        visits=0, priors_built=False,
                        source_code=materialized_source,
                        ncu=result.get("ncu_metrics", {"kernel_time_ms": ms}),
                        edges={}, applied_actions=(leaf_ns.applied_actions or []) + [leaf_action]
                    ))
                    # Build priors for the new child
                    child_ns = NodeStats(**search_state["nodes"][child_hash])
                    child_ns.edges = {k: v if isinstance(v, EdgeStats) else EdgeStats(**v) for k, v in child_ns.edges.items()}
                    _ = self._build_or_merge_priors(k, child_ns)
                    search_state["nodes"][child_hash] = dataclasses.asdict(child_ns)

                # Attach edge child pointer
                leaf_ns.edges[leaf_action].child = child_hash

                # Update incumbent best (for reporting) but DO NOT promote root
                if ms < best_ms:
                    best_ms = ms
                    search_state["best_ms"] = ms
                    search_state["best_state_hash"] = child_hash
                    search_state["best_variant_dir"] = str(vdir)
                    with open(search_state["trace_path"], "a", encoding="utf-8") as f:
                        f.write(f"- {now_iso()}: **new best** {ms:.3f} ms via `{leaf_action}` (reward {reward:.3f}) → {vdir}\n")
                    bdir = self.kernel_dir(k) / "best"
                    try:
                        if bdir.exists():
                            if bdir.is_symlink() or bdir.is_file():
                                bdir.unlink()
                            else:
                                shutil.rmtree(bdir)
                        os.symlink(vdir, bdir, target_is_directory=True)
                    except OSError as e:
                        print(f"[WARN] Symlink failed: {e}. Copying instead.", file=sys.stderr)
                        if bdir.exists(): shutil.rmtree(bdir)
                        shutil.copytree(vdir, bdir)
                    json_dump({"best_ms": best_ms, "best_variant_dir": str(bdir), "action": leaf_action},
                              self.kernel_dir(k) / "best_summary.json")

            # ------ Backpropagation: update all edges on path ------
            # Materialize node objects (may have been updated)
            search_state["nodes"][leaf_state] = dataclasses.asdict(leaf_ns)
            # Add leaf edge stats if missing
            if leaf_action not in leaf_ns.edges:
                leaf_ns.edges[leaf_action] = EdgeStats(P0=0.0, Q0=0.0, N0=1.0, child=child_hash)

            # Iterate over the path edges and update values
            for (state_h, action_h) in path:
                nsi = NodeStats(**search_state["nodes"][state_h])
                nsi.edges = {k: v if isinstance(v, EdgeStats) else EdgeStats(**v) for k, v in nsi.edges.items()}
                esi = nsi.edges[action_h]
                nsi.visits += 1
                if esi.Nsa == 0:
                    # seed with prior virtual visits
                    esi.Nsa = max(1.0, esi.N0)
                    esi.Qsum = esi.Q0 * esi.N0
                    esi.Qbar = esi.Q0
                esi.Nsa += 1.0
                esi.Qsum += reward
                esi.Qbar = esi.Qsum / max(1.0, esi.Nsa)
                search_state["nodes"][state_h] = dataclasses.asdict(nsi)

            # Event log
            evt = {
                "t": now_iso(), "kernel": k.name, "job_id": str(vdir.name), "gpu": self.gpus[(gpu_rr-1) % len(self.gpus)],
                "leaf_state": leaf_state, "action": leaf_action, "candidate_id": prop.get("candidate_id"),
                "mean_ms": ms, "reward": reward, "ok": ok, "child_state": child_hash, "variant_dir": str(vdir),
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
      - For Triton: either 'metadata.kernel_name' (preferred) or 'invocation_example'.
        * Records 'launch_update_applied' (True/False) based on static check (LAUNCH_UPDATE token) and kernel_name path.
      - For CUDA: uses compile cache; times with CUDA events; NaN/Inf check on outputs.
      - Writes result.json with (ok, mean_ms, std_ms, state_hash, ncu_metrics, materialized_source).
      - Returns 1 on hard failures.
    """
    try:
        control = json_load(Path(control_path))
        k = KernelCode.model_validate(control["kernel"])
        variant = control.get("variant", {}) or {}
        workdir = Path(control["workdir"])
        timing = control.get("timing", DEFAULT_TIMING)
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
            # Create runner configuration
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

            # Execute triton_runner module
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent.resolve())
            rc, out, err = run_subprocess(
                [sys.executable, "-m", "workers.triton_runner", str((workdir / "runner_config.json").resolve())],
                cwd=workdir, env=env, timeout=900
            )
            write_text(workdir / "runner_stdout.log", out)
            write_text(workdir / "runner_stderr.log", err)

            if rc != 0 or not (workdir / "runner_result.json").exists():
                json_dump({"ok": False, "error": "runner failed", "rc": rc}, workdir / "result.json")
                return 1

            r = json_load(workdir / "runner_result.json")

            # Save state after timing
            save_partial_state(workdir, "timed", {
                "mean_ms": r.get("mean_ms", 1e9),
                "std_ms": r.get("std_ms", 0.0),
                "ok": r.get("ok", False),
                "launch_update_applied": r.get("launch_update_applied", False)
            })

            write_text(workdir / "kernel.ptx", "// Triton PTX placeholder")
            write_text(workdir / "kernel.sass", "// SASS placeholder")
            outj = {
                "ok": bool(r.get("ok", False)),
                "mean_ms": float(r.get("mean_ms", 1e9)),
                "std_ms": float(r.get("std_ms", 0.0)),
                "state_hash": state_hash,
                "ncu_metrics": {"kernel_time_ms": float(r.get("mean_ms", 0.0))},
                "materialized_source": materialized_source,
                "launch_update_applied": bool(r.get("launch_update_applied", False)),
            }
            json_dump(outj, workdir / "result.json")
            return 0

        else:  # CUDA
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

            # Save state after compilation
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

            # Create runner configuration
            runner_config = {
                "root_path": str(Path(__file__).parent.resolve()),
                "kernel_cubin_path": str((workdir / "kernel.cubin").resolve()),
                "kernel_name": get_metadata_value(k.metadata, "kernel_name", "kernel"),
                "io_contract": effective_io or {},
                "timing": timing,
                "result_path": str((workdir / "runner_result.json").resolve())
            }
            json_dump(runner_config, workdir / "runner_config.json")

            # Execute cuda_runner module
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent.resolve())
            rc, out, err = run_subprocess(
                [sys.executable, "-m", "workers.cuda_runner", str((workdir / "runner_config.json").resolve())],
                cwd=workdir, env=env, timeout=900
            )
            write_text(workdir / "runner_stdout.log", out)
            write_text(workdir / "runner_stderr.log", err)

            if rc != 0 or not (workdir / "runner_result.json").exists():
                json_dump({"ok": False, "error": "runner failed", "rc": rc}, workdir / "result.json")
                return 1

            r = json_load(workdir / "runner_result.json")

            # Save state after timing
            save_partial_state(workdir, "timed", {
                "mean_ms": r.get("mean_ms", 1e9),
                "std_ms": r.get("std_ms", 0.0),
                "ok": r.get("ok", False)
            })

            # Compute final state hash (for consistency with Triton path)
            final_state_hash = sha256_str(materialized_source + json.dumps(effective_io or {}, sort_keys=True))
            outj = {
                "ok": bool(r.get("ok", False)),
                "mean_ms": float(r.get("mean_ms", 1e9)),
                "std_ms": float(r.get("std_ms", 0.0)),
                "state_hash": final_state_hash,
                "ncu_metrics": {"kernel_time_ms": float(r.get("mean_ms", 0.0))},
                "materialized_source": materialized_source,
            }
            json_dump(outj, workdir / "result.json")
            return 0

    except Exception as e:
        tb = traceback.format_exc()
        out = {"ok": False, "error": str(e), "traceback": tb}

        # Try to include partial state for debugging
        try:
            workdir = Path(control["workdir"]) if "control" in locals() else Path(control_path).parent
            partial_state = get_latest_phase_data(workdir)
            if partial_state:
                out["partial_state"] = partial_state
        except Exception:
            pass  # Don't fail if we can't load partial state

        json_dump(out, Path(control_path).parent / "result.json")
        return 1

# ------------------------------- CLI -------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCGS + Region-DAG prior for GPU kernel optimization")
    p.add_argument("--input", default="kernels.json", help="JSON spec file or directory with .json specs")
    p.add_argument("--outdir", default="out", help="Output directory")
    p.add_argument("--gpus", default="0", help="Comma-separated GPU indices (e.g., 0,1,2)")
    p.add_argument("--budget-builds", type=int, default=20, help="Max rollout builds per kernel")
    p.add_argument("--min-pred-gain", type=float, default=MIN_PRIOR_GAIN, help="Prune threshold for weak priors")
    p.add_argument("--llm-model", type=str, default="gpt-5-mini", help="OpenAI Responses model")
    p.add_argument("--no-llm", action="store_true", help="Disable LLM; use rule-based moves only")
    p.add_argument("--resume", action="store_true", help="Resume from existing outdir state")
    p.add_argument("--ncu-every", type=int, default=0, help="Collect metrics every N visits at the selected node (0=disable)")
    p.add_argument("--worker", nargs="?", help="(internal) Worker mode: path to control.json")
    return p.parse_args(argv)

def main():
    args = parse_args()
    if args.worker:
        sys.exit(worker_main(args.worker))

    orch = Orchestrator(args)
    kernels = orch.load_kernel_specs()
    for k in kernels:
        orch.baseline_or_resume(k)
        orch.run_mcgs_for_kernel(k)
    print("Done.")

if __name__ == "__main__":
    main()
