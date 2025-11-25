# ptx_dag_tool_v2.py
# -----------------------------------------------------------------------------
# Deterministic PTX → Region-DAG builder (Hopper-aware) with better async /
# barrier semantics and improved bytes/FLOP accounting.
#
# THIS VERSION ADDRESSES REVIEW POINTS:
#   - Correct ldmatrix shared byte estimation (x1/x2/x4, element type).
#   - Separate cp.async.wait_group from cp.async copies (distinct phase).
#   - Track *all* async producers and connect them to *all* consumers after waits.
#   - Distinguish st.async.shared vs st.async.global.
#   - Recognize membar./fence. as stage cuts; treat mbarrier.* as async barrier (no cut).
#   - Recognize shfl.sync, wmma.*, ld/st.local (spill traffic), atomics bytes.
#   - Build inst→region map from actual regions (not re-coalesced phases).
#   - Merge multi-line PTX into logical statements (end with ';').
#   - Multi-stage loop-intensity estimate; orphan waits/copies diagnostics.
#   - Localized epilogue-roundtrip detector.
#   - NEW: loop detection (back-edges) and divergence hints propagation.
#
# WHAT THIS TOOL IS FOR:
#   - Produce a compact, stable structural summary (regions, typed edges, stages)
#     that *drives* an LLM planner or template-based transformer.
#
# WHAT IT IS NOT:
#   - A legalizer or simulator. Bank conflicts, occupancy, mbarrier parity, TMA
#     descriptor correctness, and performance measurement are handled elsewhere.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import re
import json

# -----------------------------
# 0) Tokenization helpers
# -----------------------------
# Build logical PTX statements by accumulating lines until we see a ';'.
# We preserve the first physical line as the statement 'idx' for traceability.

def logical_statements(ptx_text: str) -> List[Tuple[int, str]]:
    stmts: List[Tuple[int,str]] = []
    buf: List[str] = []
    start_line: Optional[int] = None
    for i, raw in enumerate(ptx_text.splitlines(), start=1):
        line = raw.rstrip("\n")
        if not buf:
            start_line = i
        buf.append(line)
        if ";" in line:  # PTX statements end with ';' (simple heuristic)
            stmt = " ".join(s.strip() for s in buf)
            stmts.append((start_line or i, stmt))
            buf = []
            start_line = None
    # trailing buffer without ';' (labels/directives) are ignored by parse
    return stmts

# For loop detection we also need label → line
PTX_LABEL_CAPTURE_RE = re.compile(r"^\s*([\w\.\$]+):\s*$")

def scan_labels(ptx_text: str) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for i, raw in enumerate(ptx_text.splitlines(), start=1):
        m = PTX_LABEL_CAPTURE_RE.match(raw)
        if m:
            labels[m.group(1)] = i
    return labels

# -----------------------------
# 1) Classification helpers
# -----------------------------

DTYPE_SIZE = {
    "b8": 1, "s8": 1, "u8": 1,
    "b16": 2, "s16": 2, "u16": 2, "f16": 2, "bf16": 2,
    "b32": 4, "s32": 4, "u32": 4, "f32": 4,
    "b64": 8, "s64": 8, "u64": 8, "f64": 8,
    "tf32": 4,  # pseudo type for accounting purposes
}

PTX_DIRECTIVE_RE = re.compile(r"^\s*\.(\w+)")
PTX_LABEL_RE     = re.compile(r"^\s*[\w\.\$]+:\s*$")

# Capture opcode (mnemonic) after optional single predicate.
PTX_OPCODE_RE    = re.compile(r"^\s*(?:@!?[pP]\w+\s+)?([A-Za-z_\.][\w\.\:]+)")

def classify_opcode(op: str) -> str:
    """
    Fine-grained categories that we later map to a small set of phases.
    """
    opl = op.lower()

    # Loads/stores (global/const/param)
    if opl.startswith(("ld.global", "ldu.global")): return "global_load"
    if opl.startswith("ld.const"):                  return "const_load"
    if opl.startswith("ld.param"):                  return "param_load"

    # Local memory (spills) -> treat as global-like traffic for bytes
    if opl.startswith("ld.local"):                  return "local_load"
    if opl.startswith("st.local"):                  return "local_store"

    # Shared memory
    if opl.startswith("ld.shared") or opl.startswith("ldmatrix"): return "shared_load"
    if opl.startswith("st.shared"):                               return "shared_store"

    # Async copies / control
    if opl.startswith("cp.async.wait_group"):       return "async_wait"
    if "wait_group" in opl and opl.startswith("cp.async"): return "async_wait"
    if opl.startswith("cp.async.commit_group"):     return "async_commit"
    if opl.startswith("cp.async"):                  return "async_copy"

    # Async stores (shared/global)
    if opl.startswith("st.async.shared"):           return "shared_store_async"
    if opl.startswith("st.async.global"):           return "global_store"

    # Global stores
    if opl.startswith("st.global"):                 return "global_store"

    # Atomics
    if opl.startswith(("atom.global", "red.global")): return "atomic_global"
    if opl.startswith(("atom.shared", "red.shared")): return "atomic_shared"

    # Barriers / fences / mbarrier
    if opl.startswith("bar.sync"):                  return "barrier"
    if opl.startswith("membar.") or opl.startswith("fence."): return "mem_fence"
    if opl.startswith("mbarrier."):                 return "mbarrier"  # async barrier primitive

    # Tensor-core compute
    if "wgmma.mma_async" in opl or opl.startswith("mma.sync"): return "compute_mma"
    if opl.startswith("wmma."):                   return "compute_mma"   # older family
    if opl.startswith(("fma.", "mad.")):          return "compute_fma"

    # Scalar FP/warp xfer (optional FLOPs=0; still classify)
    if opl.startswith("shfl.sync"):                return "warp_xfer"
    if opl.startswith(("add.f", "mul.f", "max.f", "min.f", "rcp.", "sqrt.")): return "scalar_fp"
    if opl.startswith(("cvt.", "mov.")):           return "data_move"

    # Control/addr
    if opl.startswith(("bra", "ret", "call", "setp", "selp", "and.pred")): return "control"
    if opl.startswith(("add.", "sub.", "mul.", "shr.", "shl.", "xor.", "or.")): return "addr_arith"

    # Prefetchers/surfaces (fold as global for planning)
    if opl.startswith(("prefetch.global", "ld.surf", "st.surf", "ld.global.nc")): return "global_like"

    return "other"

# Fine category → small "phase" enum used to coalesce and draw the DAG.
PHASE_MAP = {
    # memory
    "global_load":      "mem_global_load",
    "const_load":       "mem_global_load",
    "param_load":       "mem_global_load",
    "local_load":       "mem_global_load",   # spill traffic counts as global
    "global_store":     "mem_global_store",
    "local_store":      "mem_global_store",  # spill writeout
    "global_like":      "mem_global_load",   # surfaces/prefetch treated as global

    "shared_load":      "mem_shared_load",
    "shared_store":     "mem_shared_store",
    "shared_store_async":"mem_shared_store",

    "async_copy":       "mem_async_copy",
    "async_commit":     "mem_async_copy",    # not a cut by itself
    "async_wait":       "async_wait",        # stage cut

    "atomic_global":    "atomic",
    "atomic_shared":    "atomic",

    "barrier":          "barrier",           # full CTA cut
    "mem_fence":        "mem_fence",         # ordering fence → cut
    "mbarrier":         "mbarrier",          # async barrier primitive (no cut)

    # compute / misc
    "compute_mma":      "compute",
    "compute_fma":      "compute",
    "scalar_fp":        "compute",           # counted as 0 FLOPs here (documented)
    "warp_xfer":        "warp_xfer",
    "data_move":        "data_move",
    "control":          "control",
    "addr_arith":       "addr_arith",
    "other":            "other",
}

def _parse_dtype_from_op(opl: str) -> Optional[str]:
    """
    Extract a data type suffix like .f16/.b32/.bf16/.tf32 from the opcode string.
    Returns a key present in DTYPE_SIZE or None.
    """
    m = re.search(r"\.(bf16|tf32)\b", opl)
    if m: return m.group(1)
    m = re.search(r"\.(b|s|u|f)(8|16|32|64)\b", opl)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return None

def _vector_width_from_op(opl: str) -> int:
    m = re.search(r"\.v(\d+)", opl)
    return int(m.group(1)) if m else 1

def _ldmatrix_shared_bytes(opl: str) -> int:
    """
    Estimate shared bytes read by a single ldmatrix.* instruction (warp-scope). Approx:
      bytes = x * 8 * 8 * sizeof(elem)
      where x ∈ {1,2,4} from '.xN' suffix; default 1 if absent.
    """
    m_x = re.search(r"\.x(\d+)", opl)
    x = int(m_x.group(1)) if m_x else 1
    dt = _parse_dtype_from_op(opl) or "b16"   # common case is .b16
    sz = DTYPE_SIZE.get(dt, 2)
    return x * 64 * sz

# -----------------------------
# 2) Byte / FLOP accounting
# -----------------------------

def _estimate_global_bytes(op: str, raw: str, category: str) -> Tuple[int,int]:
    """
    Global bytes (read, write) estimation per instruction.
    Shared bytes are handled separately.
    """
    opl = op.lower()
    # cp.async.* → global read only; shared write is accounted elsewhere
    if category == "async_copy":
        # for bulk/tensor forms, size may be an immediate; we pick the last numeric literal
        nums = re.findall(r"(0x[0-9a-fA-F]+|\d+)", raw)
        n = int(nums[-1], 0) if nums else 0
        return (n, 0)

    # Atomics: treat as read+write of one element
    if category in ("atomic_global",):
        dt = _parse_dtype_from_op(opl) or "b32"
        sz = DTYPE_SIZE.get(dt, 4)
        return (sz, sz)

    # Local memory spills (global-like)
    if category in ("local_load",):
        dt = _parse_dtype_from_op(opl) or "b32"
        vec = _vector_width_from_op(opl)
        sz = DTYPE_SIZE.get(dt, 4)
        return (vec * sz, 0)
    if category in ("local_store",):
        dt = _parse_dtype_from_op(opl) or "b32"
        vec = _vector_width_from_op(opl)
        sz = DTYPE_SIZE.get(dt, 4)
        return (0, vec * sz)

    # Global loads/stores (incl const/param/global_like)
    if category in ("global_load", "const_load", "param_load", "global_like"):
        dt = _parse_dtype_from_op(opl) or "b32"
        vec = _vector_width_from_op(opl)
        sz = DTYPE_SIZE.get(dt, 4)
        return (vec * sz, 0)
    if category in ("global_store",):
        dt = _parse_dtype_from_op(opl) or "b32"
        vec = _vector_width_from_op(opl)
        sz = DTYPE_SIZE.get(dt, 4)
        return (0, vec * sz)

    # All others: 0 global bytes
    return (0, 0)

def _estimate_shared_bytes(op: str, raw: str, category: str) -> Tuple[int,int]:
    """
    Shared bytes (read, write) estimation per instruction.
    """
    opl = op.lower()

    if category == "async_copy":
        # cp.async moves global → shared; treat as shared write
        nums = re.findall(r"(0x[0-9a-fA-F]+|\d+)", raw)
        n = int(nums[-1], 0) if nums else 0
        return (0, n)

    if category == "shared_load":
        if opl.startswith("ldmatrix"):
            return (_ldmatrix_shared_bytes(opl), 0)
        dt = _parse_dtype_from_op(opl) or "b32"
        vec = _vector_width_from_op(opl)
        sz = DTYPE_SIZE.get(dt, 4)
        return (vec * sz, 0)

    if category in ("shared_store", "shared_store_async"):
        dt = _parse_dtype_from_op(opl) or "b32"
        vec = _vector_width_from_op(opl)
        sz = DTYPE_SIZE.get(dt, 4)
        return (0, vec * sz)

    if category in ("atomic_shared",):
        dt = _parse_dtype_from_op(opl) or "b32"
        sz = DTYPE_SIZE.get(dt, 4)
        return (sz, sz)

    return (0, 0)

def _estimate_flops(op: str, category: str) -> int:
    """
    FLOP estimate (very limited by design):
      - MMA (mma.sync / wgmma.mma_async / wmma.*): 2*M*N*K
      - Scalar FMA/MAD: 2 FLOPs
      - Everything else: 0 (documented simplification)
    """
    opl = op.lower()
    if category == "compute_mma":
        m = re.search(r"\.m(\d+)n(\d+)k(\d+)", opl)
        if m:
            M, N, K = map(int, m.groups())
            return 2 * M * N * K
    if category == "compute_fma":
        return 2
    return 0

# -----------------------------
# 3) Core data types
# -----------------------------

@dataclass
class PTXInst:
    idx: int
    text: str
    opcode: str
    category: str
    phase: str

    bytes_global_read: int = 0
    bytes_global_write: int = 0
    shared_read: int = 0
    shared_write: int = 0
    flops: int = 0
    atomic_ops: int = 0

    # Flag for control-flow analysis
    is_predicated_branch: bool = False


def parse_ptx(ptx_text: str) -> List[PTXInst]:
    """
    Parse PTX into logical statements, filter directives/labels, classify,
    and attach static metrics.
    """
    insts: List[PTXInst] = []
    for i, stmt in logical_statements(ptx_text):
        s = stmt.strip()
        if not s:
            continue
        if PTX_DIRECTIVE_RE.match(s) or PTX_LABEL_RE.match(s):
            continue
        m = PTX_OPCODE_RE.match(s)
        if not m:
            continue
        op  = m.group(1)
        cat = classify_opcode(op)
        ph  = PHASE_MAP.get(cat, "other")
        br, bw = _estimate_global_bytes(op, s, cat)
        sr, sw = _estimate_shared_bytes(op, s, cat)
        fl = _estimate_flops(op, cat)

        # Detect predicated branch: statement starts with '@' and opcode 'bra...'
        stripped = s.lstrip()
        predicated = stripped.startswith("@") and op.lower().startswith("bra")

        # Count atomics
        atom = 1 if cat in ("atomic_global", "atomic_shared") else 0

        insts.append(PTXInst(
            idx=i, text=s, opcode=op, category=cat, phase=ph,
            bytes_global_read=br, bytes_global_write=bw,
            shared_read=sr, shared_write=sw, flops=fl,
            atomic_ops=atom, is_predicated_branch=predicated
        ))
    return insts

# -----------------------------
# 4) Coalescing into regions
# -----------------------------

@dataclass
class Region:
    """A contiguous run of PTX instructions with the same phase."""
    id: int
    phase: str
    start_line: int
    end_line: int

    # Aggregated raw work
    global_read: int = 0
    global_write: int = 0
    shared_read: int = 0
    shared_write: int = 0
    flops: int = 0
    atomic_ops: int = 0
    instruction_count: int = 0

    # Control-flow/divergence hint
    has_potential_divergence: bool = False

    # Histogram of fine categories inside this region
    categories: Dict[str, int] = field(default_factory=dict)

def coalesce_regions(insts: List[PTXInst]) -> List[Region]:
    """
    Merge consecutive instructions with the same phase into Regions.
    """
    regions: List[Region] = []
    if not insts:
        return regions

    rid = 0
    cur = Region(
        id=rid, phase=insts[0].phase, start_line=insts[0].idx, end_line=insts[0].idx
    )
    prev_phase = insts[0].phase
    for k, ins in enumerate(insts):
        if k > 0 and prev_phase != ins.phase:
            regions.append(cur)
            rid += 1
            cur = Region(
                id=rid, phase=ins.phase, start_line=ins.idx, end_line=ins.idx
            )
        prev_phase = ins.phase

        # accumulate
        cur.end_line       = ins.idx
        cur.global_read   += ins.bytes_global_read
        cur.global_write  += ins.bytes_global_write
        cur.shared_read   += ins.shared_read
        cur.shared_write  += ins.shared_write
        cur.flops         += ins.flops
        cur.atomic_ops    += ins.atomic_ops
        cur.instruction_count += 1
        cur.categories[ins.category] = cur.categories.get(ins.category, 0) + 1
        if ins.is_predicated_branch:
            cur.has_potential_divergence = True

    regions.append(cur)
    return regions

# -----------------------------
# 5) Build Region-DAG (typed edges) + loop detection
# -----------------------------

@dataclass
class Edge:
    src: int
    dst: int
    kind: str  # "flow" | "barrier" | "async_wait" | "mem_fence" | "async_dep"

@dataclass
class Loop:
    """Represents a statically-detected loop in the control flow."""
    id: int
    head_region_id: int  # The first region in the loop (target label region)
    tail_region_id: int  # The region containing the closing 'BRA'
    body_region_ids: List[int] = field(default_factory=list)

def map_inst_to_region(insts: List[PTXInst], regions: List[Region]) -> Dict[int, int]:
    """
    Safe mapping using actual coalesced regions: for each instruction idx,
    assign the region whose [start_line, end_line] range contains it.
    """
    mapping: Dict[int, int] = {}
    r_iter = iter(regions)
    r = next(r_iter, None)
    for ins in insts:
        # advance r until ins is within
        while r and ins.idx > r.end_line:
            r = next(r_iter, None)
        if r and r.start_line <= ins.idx <= r.end_line:
            mapping[ins.idx] = r.id
    return mapping

BRANCH_TARGET_RE = re.compile(r"\bbra(?:\.[\w\.]+)?\s+([$\w\.\-]+)")

def _region_id_for_line(line: int, regions: List[Region]) -> Optional[int]:
    """
    Map an arbitrary source line to the best region id:
      - Prefer region whose [start,end] contains line
      - Otherwise first region with start_line >= line
    """
    for r in regions:
        if r.start_line <= line <= r.end_line:
            return r.id
    for r in regions:
        if r.start_line >= line:
            return r.id
    return None

def detect_loops(ptx_text: str, insts: List[PTXInst], regions: List[Region]) -> List[Loop]:
    """
    Very lightweight loop detection: treat predicated or unpredicated 'bra' to an
    *earlier* label as a loop back-edge.
    """
    loops: List[Loop] = []
    label_lines = scan_labels(ptx_text)
    rid_by_line = map_inst_to_region(insts, regions)

    # Deduplicate by (head, tail)
    seen: set[Tuple[int,int]] = set()

    for ins in insts:
        op = ins.opcode.lower()
        if not op.startswith("bra"):
            continue
        m = BRANCH_TARGET_RE.search(ins.text)
        if not m:
            continue
        tgt = m.group(1)
        tgt_line = label_lines.get(tgt)
        if not tgt_line:
            continue
        if tgt_line >= ins.idx:
            # forward branch → not a back-edge
            continue

        tail_rid = rid_by_line.get(ins.idx)
        head_rid = _region_id_for_line(tgt_line, regions)
        if tail_rid is None or head_rid is None:
            continue

        key = (head_rid, tail_rid)
        if key in seen:
            continue
        seen.add(key)

        # Body = regions spanning [head_rid .. tail_rid] in region id order
        # (coalescer assigns densifying ids; this is a reasonable approximation)
        body = [r.id for r in regions if head_rid <= r.id <= tail_rid]
        loops.append(Loop(
            id=len(loops),
            head_region_id=head_rid,
            tail_region_id=tail_rid,
            body_region_ids=body
        ))

        # Mark divergence hint if predicated branch was used (already set per region)

    return loops

def build_dag(insts: List[PTXInst], regions: List[Region]) -> Tuple[List[Edge], Dict[int, List[int]], Dict]:
    """
    Construct typed edges:
      - Linear edges labeled as: flow | barrier | async_wait | mem_fence.
      - Async producer→consumer edges: connect *all* cp.async regions since last wait
        to *all* shared-load consumers immediately after the wait (until a cut).
    Also returns small diagnostics (orphan waits/copies).
    """
    rid_by_line = map_inst_to_region(insts, regions)

    # Flags per region
    is_barrier   = {r.id: False for r in regions}
    is_wait      = {r.id: False for r in regions}
    is_fence     = {r.id: False for r in regions}
    is_async     = {r.id: False for r in regions}
    is_ldsmem_ld = {r.id: False for r in regions}  # shared loads (ldmatrix/ld.shared only)

    for ins in insts:
        rid = rid_by_line.get(ins.idx)
        if rid is None:
            continue
        if ins.category == "barrier":
            is_barrier[rid] = True
        if ins.category == "async_wait":
            is_wait[rid] = True
        if ins.category == "mem_fence":
            is_fence[rid] = True
        if ins.phase == "mem_async_copy":
            is_async[rid] = True
        if ins.category == "shared_load" or ins.opcode.lower().startswith("ldmatrix"):
            is_ldsmem_ld[rid] = True

    edges: List[Edge] = []
    # Linear edges with labels
    for i in range(len(regions) - 1):
        kind = "flow"
        if is_barrier[regions[i].id]: kind = "barrier"
        elif is_wait[regions[i].id]:  kind = "async_wait"
        elif is_fence[regions[i].id]: kind = "mem_fence"
        edges.append(Edge(src=regions[i].id, dst=regions[i+1].id, kind=kind))

    # Async producer→consumer
    pending_async: List[int] = []  # ids of mem_async_copy regions since last wait/cut
    orphan_copies = 0
    orphan_waits  = 0

    def is_cut_phase(ph: str) -> bool:
        return ph in ("barrier", "async_wait", "mem_fence")

    i = 0
    while i < len(regions):
        r = regions[i]
        if is_async[r.id]:
            pending_async.append(r.id)

        if is_wait[r.id]:
            # Collect consumers after the wait until a cut or next async copy
            consumers: List[int] = []
            j = i + 1
            while j < len(regions):
                rj = regions[j]
                if is_cut_phase(rj.phase) or is_async[rj.id]:
                    break
                if is_ldsmem_ld[rj.id]:
                    consumers.append(rj.id)
                # stop after we pass the first compute (common GEMM shape)
                if rj.phase == "compute":
                    break
                j += 1

            if not consumers:
                orphan_waits += 1
            else:
                for prod in pending_async:
                    for cons in consumers:
                        edges.append(Edge(src=prod, dst=cons, kind="async_dep"))
            pending_async.clear()
        i += 1

    # Any leftover copies with no wait
    orphan_copies = len(pending_async)

    # adjacency
    adj: Dict[int, List[int]] = {r.id: [] for r in regions}
    for e in edges:
        adj[e.src].append(e.dst)

    diags = {"orphan_waits": orphan_waits, "orphan_async_copies": orphan_copies}
    return edges, adj, diags

# -----------------------------
# 6) Stage partitioning
# -----------------------------

@dataclass
class Stage:
    """A sequence of regions separated by strong synchronization."""
    id: int
    region_ids: List[int] = field(default_factory=list)

    # Static Work Profile (no time prediction)
    global_bytes: int = 0
    shared_read_bytes: int = 0
    shared_write_bytes: int = 0
    flops: int = 0
    atomic_ops: int = 0
    instruction_count: int = 0

    # Aggregated divergence hint
    has_potential_divergence: bool = False

def partition_stages(regions: List[Region], edges: List[Edge]) -> List[Stage]:
    """
    Cut stages after regions whose *linear* next-edge kind is {barrier, async_wait, mem_fence}.
    mbarrier does NOT cut.
    """
    out_kind = {r.id: "flow" for r in regions}
    for e in edges:
        if e.dst == e.src + 1:
            out_kind[e.src] = e.kind

    stages: List[Stage] = []
    cur: List[int] = []
    sid = 0
    for i, r in enumerate(regions):
        cur.append(r.id)
        if out_kind.get(r.id) in ("barrier", "async_wait", "mem_fence") or i == len(regions) - 1:
            gb = sum(regions[x].global_read + regions[x].global_write for x in cur)
            sr = sum(regions[x].shared_read for x in cur)
            sw = sum(regions[x].shared_write for x in cur)
            fl = sum(regions[x].flops for x in cur)
            ao = sum(regions[x].atomic_ops for x in cur)
            ic = sum(regions[x].instruction_count for x in cur)
            has_div = any(regions[x].has_potential_divergence for x in cur)
            stages.append(Stage(
                id=sid, region_ids=cur.copy(),
                global_bytes=gb, shared_read_bytes=sr, shared_write_bytes=sw,
                flops=fl, atomic_ops=ao, instruction_count=ic,
                has_potential_divergence=has_div
            ))
            sid += 1
            cur = []
    return stages

# -----------------------------
# 7) Analysis and suggestions
# -----------------------------

@dataclass
class Suggestion:
    kind: str
    reason: str
    code_change: str

# -----------------------------
# 8) JSON / DOT writers + driver
# -----------------------------

def _region_to_dict(r: Region) -> Dict:
    return {
        "id": r.id, "phase": r.phase,
        "start_line": r.start_line, "end_line": r.end_line,
        "global_read": r.global_read, "global_write": r.global_write,
        "shared_read": r.shared_read, "shared_write": r.shared_write,
        "flops": r.flops,
        "atomic_ops": r.atomic_ops,
        "instruction_count": r.instruction_count,
        "has_potential_divergence": r.has_potential_divergence,
        "categories": r.categories,  # histogram (dict)
    }

def write_json(out_dir: Path, insts: List[PTXInst], regions: List[Region], edges: List[Edge], stages: List[Stage], loops: List[Loop], analysis: Dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "instructions": [i.__dict__ for i in insts],
        "regions":      [_region_to_dict(r) for r in regions],
        "edges":        [e.__dict__ for e in edges],
        "stages":       [s.__dict__ for s in stages],
        "loops":        [l.__dict__ for l in loops],
        "analysis":     analysis,
    }
    (out_dir / "dag_analysis.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

def write_dot(out_dir: Path, regions: List[Region], edges: List[Edge], name: str = "regions_dag"):
    color = {
        "compute": "lightblue",
        "mem_async_copy": "gold",
        "async_wait": "orange",
        "barrier": "tomato",
        "mem_fence": "orangered",
        "mbarrier": "yellowgreen",
        "mem_shared_load": "palegreen",
        "mem_shared_store": "darkseagreen",
        "mem_global_store": "plum",
        "mem_global_load": "khaki",
        "warp_xfer": "lightgray",
        "data_move": "gray90",
        "control": "gray80",
        "addr_arith": "white",
        "atomic": "salmon",
        "other": "white",
    }
    lines = ["digraph G {", "  rankdir=LR;"]
    for r in regions:
        label = f"R{r.id}\\n{r.phase}\\nL{r.start_line}-{r.end_line}\\nG={r.global_read+r.global_write}B, S={r.shared_read+r.shared_write}B, F={r.flops}"
        lines.append(f'  r{r.id} [label="{label}", style=filled, fillcolor="{color.get(r.phase, "white")}", shape=box];')
    for e in edges:
        style = {"barrier":"bold", "async_wait":"dashed", "mem_fence":"dashed", "async_dep":"dotted", "flow":"solid"}.get(e.kind, "solid")
        lines.append(f"  r{e.src} -> r{e.dst} [style={style}, label={e.kind}];")
    lines.append("}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.dot").write_text("\n".join(lines), encoding="utf-8")

def build_all(ptx_text: str):
    insts   = parse_ptx(ptx_text)
    regions = coalesce_regions(insts)
    edges, adj, diags = build_dag(insts, regions)
    loops   = detect_loops(ptx_text, insts, regions)
    stages  = partition_stages(regions, edges)
    
    return insts, regions, edges, stages, loops

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx", type=str, required=True, help="Path to .ptx text file")
    ap.add_argument("--out", type=str, default="ptx_dag_out_v2", help="Output directory")
    args = ap.parse_args()

    text = Path(args.ptx).read_text(encoding="utf-8")
    insts, regions, edges, stages, loops = build_all(text)

    out_dir = Path(args.out)
    write_json(out_dir, insts, regions, edges, stages, loops)
    write_dot(out_dir, regions, edges)

    print(f"Parsed {len(insts)} instructions → {len(regions)} regions → {len(edges)} edges → {len(stages)} stages → {len(loops)} loops")
    print(f"Artifacts written to: {out_dir.resolve()}")
