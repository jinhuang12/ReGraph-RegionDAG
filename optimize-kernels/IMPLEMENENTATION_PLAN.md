# Implementation roadmap

I’ll break this into phases. Phases 0–1 are “make the current system functional and solid”. Phases 2–3 move toward “superhuman”.

## Progress tracker (2025-11-20)

- ✅ 0.1 Nsight Compute wired into CUDA worker (`find_ncu_cli`, wrap runner, emit `.ncu-rep`).
- ✅ 0.2 Root NodeStats now store PTX/NCU paths.
- ✅ 0.3 RegionDag/Ncu/PTX contexts set before LLM proposals (selection + rollout).
- ✅ Triton path now dumps real PTX (and best-effort SASS/CUBIN) and can be profiled via Nsight wrapper.
- ✅ Logging baseline added in orchestrator.
- ✅ Added noise-robust stats (mean/median/min/trimmed) in CUDA & Triton runners.
- ✅ Optional correctness checks against user-supplied Python reference in CUDA & Triton runners.
- ✅ Added a minimal CUDA smoke-test spec (`kernels/vec_add_cuda.json`) + reference (`reference/vec_add_ref.py`) for quick end-to-end validation.
- ✅ Smoke test executed on H200 host (`p5e-cmh`): `vec_add_cuda` baseline completed; Nsight report produced.
- ✅ Metric-aware reward penalties wired (configurable; currently DRAM bytes and warp activity).
- 🔜 Tighten relabel/catalog and metric selections; expand correctness coverage.
- 🔜 Phase 2/3 vLLM integration, representative shapes, cost models, playbook retrieval.

Short completion notes:
- Nsight (CUDA/Triton): added `find_ncu_cli`, wrap runners with `ncu`, capture report path into `result.json`.
- Baseline artifact propagation: root node now records PTX/NCU paths from baseline run.
- Tool contexts: orchestrator builds RegionDag/Ncu/PTX correlators per node and sets `kernel_opt_tooling.CURRENT_*` before LLM calls.
- Triton artifacts: runner writes PTX/SASS/CUBIN when available; mean/min/median/std returned to reduce noise.

### Phase 0 – Make Nsight & tools fully operational (3 → 5/10)

**Goal:** every kernel evaluation on your P5E48XL produces:

* Valid `kernel.ptx` and `kernel.cubin`.
* A **real Nsight Compute report** (`*.ncu-rep` or `*.nsight-cuprof-report`).
* `RegionDagContext`, `NcuMetricsContext`, `PtxSourceCorrelator` correctly initialised and accessible to tools.

#### 0.0. Environment sanity

On the P5E48XL (H200):

1. Make sure Nsight Compute CLI is installed and works:

   ```bash
   ncu --version
   # or, depending on the AMI:
   nv-nsight-cu-cli --version
   ```

   CLI usage is typically:
   `ncu -f -o report_name ./app args...` (or `nv-nsight-cu-cli -f -o`). This produces `report_name.ncu-rep` or `report_name.nsight-cuprof-report`. ([about.gitlab.com][4])

2. Ensure the **Python ncu_report module** is usable:

   ```bash
   python -c "import ncu_report; print('ok')"
   ```

   Nsight installs `ncu_report` under its `extras/python` directory; your code already tries common locations and can be extended with an env var override. ([NVIDIA Developer Forums][1])

3. Set:

   ```bash
   export NSIGHT_COMPUTE_CLI=ncu   # or nv-nsight-cu-cli
   ```

   If needed: `export NSIGHT_COMPUTE_PYTHON_PATH=/path/to/nsight-compute/extras/python`.

#### 0.1. Wire Nsight Compute into CUDA worker

**Changes (details in section 3):**

* Add a helper in `optimize_kernels.py`:

  ```python
  def find_ncu_cli() -> Optional[str]:
      # 1. NSIGHT_COMPUTE_CLI env override
      # 2. fall back to 'ncu' or 'nv-nsight-cu-cli'
  ```

* In `worker_main`, CUDA path:

  * After writing `runner_config.json`, build `runner_cmd = [sys.executable, "-m", "workers.cuda_runner", runner_cfg_path]`.
  * Check `control["ncu"]["enabled"]` and `["collect"]`.
  * If enabled and CLI found:

    * Run **Nsight Compute** wrapping the runner:

      ```python
      cmd = [ncu_cli, "-f", "-o", "ncu_report", "-k", kernel_name] + runner_cmd
      ```

    * This generates `ncu_report.ncu-rep` or `ncu_report.nsight-cuprof-report`.

    * Capture stdout/stderr to `runner_stdout.log` / `runner_stderr.log`.
  * If disabled or CLI missing:

    * Fall back to the previous direct runner call.

* After running:

  * Detect `ncu_report_path` by looking for

    * `ncu_report.ncu-rep` or
    * `ncu_report.nsight-cuprof-report`.
  * Include that path in `result.json`:

    ```python
    "ptx_path": str(workdir / "kernel.ptx"),
    "ncu_report_path": ncu_report_path,
    ```

#### 0.2. Propagate PTX & NCU paths into NodeStats

In `baseline_or_resume`:

* When creating the **root NodeStats**, include PTX and NCU paths from `run`:

  ```python
  root_node = NodeStats(
      visits=0,
      source_code=baseline_source,
      ncu=run.get("ncu_metrics", {"kernel_time_ms": run.get("mean_ms", 0.0)}),
      edges={}, actions={}, applied_actions=[], applied_methods=[],
      last_method="START",
      impl_hash=worker_root_hash,
      ptx_path=run.get("ptx_path"),
      ncu_report_path=run.get("ncu_report_path"),
  )
  ```

For non-root child nodes, you already propagate `ptx_path` and `ncu_report_path` from worker results; keep that.

#### 0.3. Hook Region/NCU/PTX correlator into LLM tools

In `optimize_kernels.py`:

* Add:

  ```python
  import kernel_opt_tooling  # new
  from kernel_opt_tooling import RegionDagContext, NcuMetricsContext, PtxSourceCorrelator
  ```

* Right before each call to `self.llm.propose(...)` (both in **selection** and **rollout**):

  1. Build `RegionDagContext`, `NcuMetricsContext`, `PtxSourceCorrelator` if possible from the current node’s `ptx_path` and `ncu_report_path`.

  2. Compute `region_summary = rctx.overview()` and `ncu_summary = nctx.summary()` to feed into the system prompt.

  3. Crucially, set:

     ```python
     kernel_opt_tooling.CURRENT_REGION_DAG_CTX = rctx
     kernel_opt_tooling.CURRENT_NCU_CTX = nctx
     kernel_opt_tooling.CURRENT_PTX_CORRELATOR = correlator
     ```

  4. On any exception, set all three to `None` and fall back to `{}` summaries.

This makes the tool calls like `region_dag_inspect`, `ncu_metrics_inspect`, `ptx_source_correlate` actually operate on the **current kernel** instead of empty globals.

#### 0.4. Minimal logging hygiene

You already have:

* `events.jsonl` with structured logs.
* `trace.md` with human-readable trace.
* Per-worker logs (`nvcc_*_stdout.log`, `runner_stdout.log`, etc).

I’d **keep that**, but add:

* A simple `logging` setup at the top of `optimize_kernels.py`:

  ```python
  import logging

  logging.basicConfig(
      level=os.environ.get("KERNEL_OPT_LOGLEVEL", "INFO"),
      format="[%(asctime)s] [%(levelname)s] %(message)s",
  )
  logger = logging.getLogger(__name__)
  ```

* A few key logs:

  * When baseline finishes.
  * At each iteration, log best-so-far and which method was chosen.

This gives you a clear **console view** while the richer detail lives in `events.jsonl` / `trace.md`.

#### 0.5. Smoke tests

1. Start with a **simple CUDA kernel** (e.g., naive vector add) in `specs/`.

2. Run:

   ```bash
   CUDA_VISIBLE_DEVICES=0 \
   NSIGHT_COMPUTE_CLI=ncu \
   python optimize_kernels.py specs/vec_add.yaml \
     --llm-model gpt-5.1 \
     --iterations 3
   ```

3. Check:

   * Baseline dir has `kernel.ptx`, `kernel.cubin`, `ncu_report.*`.
   * `result.json` has non-null `ptx_path` and `ncu_report_path`.
   * `events.jsonl` contains region/ncu summaries.
   * The LLM tool calls don’t crash.

At this point, your system is **functionally sound** and gives the LLM real PTX + Nsight data.

---

### Phase 1 – Make search robust and informative (5 → 7/10)

Now that you can profile real kernels, you need to **harden search and correctness**.

#### 1.1. Robust measurement and noise modelling

* In `workers.cuda_runner` / `workers.triton_runner`:

  * Run each candidate **multiple times**:

    * Warmup: e.g., 10 iterations.
    * Measurement: e.g., 50–100 iterations per repeat, 3–5 repeats.
  * Within each repeat:

    * Aggregate time using min or median.
  * Across repeats:

    * Use **median** or **trimmed mean**; compute stddev.
  * Return:

    * `mean_ms`, `std_ms`, `min_ms`.

* In the orchestrator:

  * Use these stats to:

    * **Break ties** (prefer lower std, or lower min_ms for close means).
    * Detect outliers (if std is huge, re-run or mark as unstable).

This is similar to what AutoTVM and other auto-tuners do to cope with GPU variance. ([Apache TVM][3])

#### 1.2. Correctness harness

For generic kernels (non-vLLM yet):

* For each `KernelCode` spec:

  * Include a **Python reference implementation**.
  * In the runner:

    * Generate random inputs in a range that exercises edge-cases.
    * Run both kernel and reference.
    * Check:

      * Max absolute / relative error (`<= tol`).
      * No NaNs/Infs unless expected.
    * If incorrect:

      * Mark `ok=False`, add error type (`"incorrect_output"`) to result.

* In the orchestrator:

  * Treat `ok=False` as a **hard constraint**: discard the candidate (large negative reward, no expansion from that node).

This is crucial to avoid the LLM “optimizing” by dropping work.

#### 1.3. Canonical method names and ReGraph clean-up

Right now you:

* Let the LLM propose informal method names.
* Run `LLMRelabeler` to canonicalise them.
* Log transitions into a ReGraph dataset.

To make this robust:

1. Maintain a **central catalogue** of canonical methods:

   ```yaml
   methods:
     - name: block_tiling
       aliases: ["tile blocks", "block tiling", "tiling"]
       description: ...
     - name: shared_mem_cache
       aliases: ["shared cache", "cache in shared memory", ...]
     ...
   ```

2. Use this catalogue in **both**:

   * The system prompt (so LLM prefers known names).
   * The `LLMRelabeler` (so it maps to a finite set).

3. Add **hard constraints** in relabelling:

   * If the model maps to a new name, either:

     * Force it into the closest known name, or
     * Add it to a “sandbox” set and only accept it if it appears several times in successful trajectories.

4. Use ReGraph more aggressively:

   * When expanding from method M, only allow transitions:

     * With enough support in the dataset, or
     * That preserve functional constraints (e.g., don’t undo necessary changes without reason).

This moves you toward a **playbook** of good method sequences, like a learned meta-policy.

#### 1.4. Better node evaluation

* Incorporate **profiled metrics** into reward:

  * Example: penalize high `dram__bytes_read.sum`, high `inst_executed`, low `sm__warps_active.avg.pct_of_peak_sustained_active`. ([NVIDIA Developer Forums][1])
  * Use `NcuMetricsContext.summary()` to generate a short metric vector.

* Reward = combination of:

  * Speed-up vs baseline.
  * Soft penalties from metrics (e.g., high DRAM traffic).
  * Penalty for instability / correctness risk.

This makes the LLM’s tools more informative: it can ask for `ncu_metrics_inspect` and see exactly where the kernel is struggling.

At the end of Phase 1, you have a **robust, introspection-rich search engine** that can reasonably optimize hand-written kernels in isolation.

---

### Phase 2 – Deep vLLM integration (7 → 9/10)

Now we make it vLLM-specific.

#### 2.1. Enumerate and spec vLLM kernels

From the vLLM codebase: ([Apache TVM][2])

* Identify key GPU kernels worth optimizing:

  * Fused attention (e.g., FlashAttention variants).
  * RMSNorm / LayerNorm.
  * Activation + bias + quantization fusions.
  * KV cache operations.

For each kernel:

1. Create a **spec file** (YAML or JSON) for `KernelCode`:

   * `source_path` / `source_code`.
   * `kernel_name`.
   * `io_contract` (tensor shapes, strides, dtype).
   * Typical shapes (batch size, seq length, hidden size).

2. Add a **reference path**:

   * Either a slower but clear version within vLLM.
   * Or a PyTorch implementation that is mathematically equivalent.

#### 2.2. vLLM harness

Extend `workers.cuda_runner` with a mode like:

* `mode: "standalone"` (current behavior) vs `mode: "vllm"`:

  * In `vllm` mode:

    * Import vLLM.
    * Use vLLM’s internal tensor layouts, random seeds, etc.
    * Launch the target kernel exactly as vLLM would during inference.
    * Measure latency via CUDA events at the **operator level**.

This ensures:

* **Correctness**: compare your optimized kernel against vLLM’s own reference.
* **Relevance**: performance is measured in the exact context vLLM cares about.

#### 2.3. Representative workload sampling

* For each kernel spec:

  * Predefine a small set of **representative shapes**:

    * Small, medium, large sequence lengths.
    * Different batch sizes (1, 4, 16, …).
  * For each candidate kernel:

    * Benchmark across these shapes.
    * Summarize with a **weighted mean** or worst-case performance.

This ensures you don’t overfit to a single microbenchmark shape.

#### 2.4. Cross-check with end-to-end throughput

Eventually:

* Integrate a mode where:

  * After each “best so far” candidate is found,
  * You plug it into vLLM and measure **tokens/sec** on a small synthetic workload.

That gives you a real-world validation: “does this actually make vLLM faster on H200?”

---

### Phase 3 – Toward superhuman (9 → 10/10)

Now we go from “strong optimizer” to "likely better than most humans most of the time".

#### 3.1. Template + parameter search

Borrowing from AutoTVM / Ansor: ([Apache TVM][3])

1. For each kernel family, define **parametric templates**:

   ```cpp
   // Example: a block-tiled GEMM-like kernel
   template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS>
   __global__ void kernel(...) { ... }
   ```

2. Let the LLM operate at two levels:

   * **Method-level edits**: change algorithm structure, memory layout, fusion decisions.
   * **Param-level search**: propose ranges / priors for BLOCK_M, BLOCK_N, etc.

3. Behind the scenes:

   * Have a **fast local search** or **Bayesian optimization** over parameters.
   * The LLM selects **regions** of the parameter space and structural edits; the param search fine-tunes them.

This allows fine-grained optimization beyond what a human would typically try by hand.

#### 3.2. Learned cost model + ReGraph

* Train a **cost model** on all accumulated trajectories:

  * Input: PTX structural features, Nsight metrics from a few warm-up runs, method sequence features.
  * Output: predicted speed-up vs baseline.

* Use this in the search as:

  * A **prior**: bias expansion toward high-predicted-gain nodes.
  * An **early-stopping** heuristic: if a branch is predicted to be hopeless, prune it.

This is exactly the kind of approach that has made AutoTVM and related frameworks powerful in practice. ([Apache TVM][3])

ReGraph comes in as:

* A graph of **methods** and their typical “contexts” and outcomes.
* A knowledge store for LLM to retrieve “playbooks”:

  * “When warp occupancy is low and memory latency is high, humans often: [block_tiling → shared_mem_cache → vectorize]”.

#### 3.3. Multi-kernel, multi-hardware generalization

* Gather data across:

  * Different kernels (fused attention, layernorm, etc).
  * Different GPUs (H100, H200, A100).

* Use that to:

  * Learn hardware-agnostic features (e.g., “this pattern causes bank conflicts”).
  * Let the LLM reason with meta-prompts like:

    * “This kernel underperforms on H200 vs H100; what changes in SM layout and memory hierarchy explain that, and how should we adapt?”

At this stage, the system is:

* Profiling real workloads.
* Learning from its own history.
* Using both explicit logic (PTX & Nsight) and implicit patterns (ReGraph + cost model) to search.

That’s about as close to “superhuman” as you can realistically get in a programmable sense.

---

## 3. Concrete code: make the current system functional with Nsight Compute

Below is a **coherent set of changes** that you can hand to GPT‑5.1 Codex to implement. They’re written as “drop-in” replacements / additions for your existing repo.

### 3.1. `optimize_kernels.py`: add Nsight helper

Near your imports (top of file, after `import shutil` etc.):

```python
import logging
from typing import Optional

# Simple logging setup; adjustable via env var.
logging.basicConfig(
    level=os.environ.get("KERNEL_OPT_LOGLEVEL", "INFO"),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Existing imports...
# from kernel_opt_tooling import LLMCandidateGenerator, RegionDagContext, NcuMetricsContext, PtxSourceCorrelator
import kernel_opt_tooling
from kernel_opt_tooling import (
    LLMCandidateGenerator,
    RegionDagContext,
    NcuMetricsContext,
    PtxSourceCorrelator,
)
```

Add this helper somewhere near your other small utilities:

```python
def find_ncu_cli() -> Optional[str]:
    """
    Locate Nsight Compute CLI.

    Order:
      1. NSIGHT_COMPUTE_CLI env var (if it resolves via shutil.which).
      2. 'ncu'
      3. 'nv-nsight-cu-cli'
    """
    override = os.environ.get("NSIGHT_COMPUTE_CLI")
    if override:
        path = shutil.which(override)
        if path:
            return path

    for name in ("ncu", "nv-nsight-cu-cli"):
        path = shutil.which(name)
        if path:
            return path

    return None
```

> NOTE: this function is primarily used in `worker_main` (below), but you can also expose it elsewhere as needed.

### 3.2. `optimize_kernels.py`: baseline NodeStats carries PTX & NCU paths

In `baseline_or_resume`, after you load the baseline `run` result and compute `worker_root_hash`, adjust root node creation:

```python
# Existing:
# worker_root_hash = run.get("state_hash", "")
# root_hash = worker_root_hash if USE_HASH_FOR_NODE_ID else str(uuid.uuid4())

worker_root_hash = run.get("state_hash", "")
root_hash = worker_root_hash if USE_HASH_FOR_NODE_ID else str(uuid.uuid4())

root_node = NodeStats(
    visits=0,
    source_code=baseline_source,
    ncu=run.get("ncu_metrics", {"kernel_time_ms": run.get("mean_ms", 0.0)}),
    edges={},
    actions={},
    applied_actions=[],
    applied_methods=[],
    last_method="START",
    impl_hash=worker_root_hash,
    # NEW: propagate paths for PTX and Nsight report
    ptx_path=run.get("ptx_path"),
    ncu_report_path=run.get("ncu_report_path"),
)

search_state = {
    "root": root_hash,
    "nodes": {root_hash: dataclasses.asdict(root_node)},
    "edges": {},
    "rollouts": 0,
}
```

This makes the baseline node’s PTX/NCU info available to selection/rollout.

### 3.3. `optimize_kernels.py`: wire RegionDag/Ncu/PTX contexts before LLM calls

#### 3.3.1. Selection (leaf node) context

In `Orchestrator.run_mcgs_for_kernel`, where you build `region_summary` and `ncu_summary` for the **leaf node**, replace your current block:

```python
region_summary = {}
ncu_summary = {}
try:
    if leaf_ns.ptx_path and Path(leaf_ns.ptx_path).exists():
        rctx = RegionDagContext(Path(leaf_ns.ptx_path).read_text(encoding="utf-8"), kernel_name="kernel")
        region_summary = rctx.overview()
    if leaf_ns.ncu_report_path and Path(leaf_ns.ncu_report_path).exists():
        nctx = NcuMetricsContext(leaf_ns.ncu_report_path, range_idx=0, action_idx=0)
        ncu_summary = nctx.summary()
except Exception:
    pass
```

with:

```python
region_summary: Dict[str, Any] = {}
ncu_summary: Dict[str, Any] = {}

try:
    rctx: Optional[RegionDagContext] = None
    nctx: Optional[NcuMetricsContext] = None
    correlator: Optional[PtxSourceCorrelator] = None

    # Region DAG from PTX
    if leaf_ns.ptx_path and Path(leaf_ns.ptx_path).exists():
        ptx_text = Path(leaf_ns.ptx_path).read_text(encoding="utf-8")
        kernel_name = get_metadata_value(k.metadata, "kernel_name", "kernel")
        rctx = RegionDagContext(ptx_text, kernel_name=kernel_name)
        region_summary = rctx.overview()

    # Nsight Compute metrics and PTX/source correlation
    if leaf_ns.ncu_report_path and Path(leaf_ns.ncu_report_path).exists():
        nctx = NcuMetricsContext(leaf_ns.ncu_report_path, range_idx=0, action_idx=0)
        ncu_summary = nctx.summary()
        correlator = PtxSourceCorrelator(Path(leaf_ns.ncu_report_path))

    # Make contexts globally available to tools
    kernel_opt_tooling.CURRENT_REGION_DAG_CTX = rctx
    kernel_opt_tooling.CURRENT_NCU_CTX = nctx
    kernel_opt_tooling.CURRENT_PTX_CORRELATOR = correlator

except Exception:
    # Fail-soft: clear contexts and fall back to empty summaries
    kernel_opt_tooling.CURRENT_REGION_DAG_CTX = None
    kernel_opt_tooling.CURRENT_NCU_CTX = None
    kernel_opt_tooling.CURRENT_PTX_CORRELATOR = None
    region_summary = {}
    ncu_summary = {}
```

Then the subsequent `self.llm.propose(k, region_summary=region_summary, ncu_summary=ncu_summary)` call becomes **informative** and tools can introspect.

#### 3.3.2. Rollout context

In the rollout part of `run_mcgs_for_kernel`, find this block (approximately):

```python
if not cur_ns.edges:
    # Ask LLM to propose new actions
    try:
        props_obj = self.llm.propose(k, region_summary={}, ncu_summary={})
    except TypeError:
        props_obj = self.llm.propose(k, region_summary={})
```

Replace it with something like:

```python
if not cur_ns.edges:
    # Build PTX/NCU contexts for this rollout node
    region_summary: Dict[str, Any] = {}
    ncu_summary: Dict[str, Any] = {}

    try:
        rctx: Optional[RegionDagContext] = None
        nctx: Optional[NcuMetricsContext] = None
        correlator: Optional[PtxSourceCorrelator] = None

        if cur_ns.ptx_path and Path(cur_ns.ptx_path).exists():
            ptx_text = Path(cur_ns.ptx_path).read_text(encoding="utf-8")
            kernel_name = get_metadata_value(k.metadata, "kernel_name", "kernel")
            rctx = RegionDagContext(ptx_text, kernel_name=kernel_name)
            region_summary = rctx.overview()

        if cur_ns.ncu_report_path and Path(cur_ns.ncu_report_path).exists():
            nctx = NcuMetricsContext(cur_ns.ncu_report_path, range_idx=0, action_idx=0)
            ncu_summary = nctx.summary()
            correlator = PtxSourceCorrelator(Path(cur_ns.ncu_report_path))

        kernel_opt_tooling.CURRENT_REGION_DAG_CTX = rctx
        kernel_opt_tooling.CURRENT_NCU_CTX = nctx
        kernel_opt_tooling.CURRENT_PTX_CORRELATOR = correlator

    except Exception:
        kernel_opt_tooling.CURRENT_REGION_DAG_CTX = None
        kernel_opt_tooling.CURRENT_NCU_CTX = None
        kernel_opt_tooling.CURRENT_PTX_CORRELATOR = None
        region_summary = {}
        ncu_summary = {}

    # Ask LLM to propose new actions with summaries
    try:
        props_obj = self.llm.propose(k, region_summary=region_summary, ncu_summary=ncu_summary)
    except TypeError:
        # Backward-compat in case the signature changed
        props_obj = self.llm.propose(k, region_summary=region_summary)
```

This ensures rollouts also benefit from PTX/NCU info.

---

### 3.4. `optimize_kernels.py`: Nsight Compute integration in `worker_main`

Here is a **full updated version** of `worker_main` that wires Nsight Compute into the CUDA path, while preserving your existing behaviour and state tracking.

Replace your existing `worker_main` definition with this one (adapting only the surrounding helpers/imports to match your file):

```python
def worker_main(control_path: str) -> int:
    """
    Worker entry point.

    Reads control.json, materializes kernel source, compiles (Triton or CUDA),
    runs the appropriate runner, and writes result.json.

    For CUDA kernels, if control["ncu"]["enabled"] and Nsight Compute CLI is
    available, wraps the runner in Nsight Compute and emits an .ncu-rep (or
    .nsight-cuprof-report) file. The path is returned as ncu_report_path.
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
            json_dump(
                {"ok": False, "error": f"unknown kernel_type {k.kernel_type}"},
                workdir / "result.json",
            )
            return 1

        target_path = workdir / target_rel
        workdir.mkdir(parents=True, exist_ok=True)

        # Materialize source: either apply a patch or replace full source.
        if "full_source_code" in variant and variant["full_source_code"]:
            target_path.write_text(variant["full_source_code"], encoding="utf-8")
        elif "patch" in variant and variant["patch"]:
            # You already have some patch application logic; keep or extend it.
            original = base_source_code
            patch_text = variant["patch"]
            # TODO: integrate your actual patching implementation here.
            target_path.write_text(original, encoding="utf-8")
        else:
            target_path.write_text(base_source_code, encoding="utf-8")

        materialized_source = target_path.read_text(encoding="utf-8")
        state_hash = sha256_str(
            materialized_source + json.dumps(launch_update, sort_keys=True)
        )
        save_partial_state(
            workdir,
            "materialized",
            {
                "source_hash": state_hash,
                "source_path": str(target_path),
                "kernel_type": k.kernel_type,
            },
        )

        if k.kernel_type == "triton":
            # --- TRITON PATH (unchanged except for minor cleanup) ---
            # Build runner config, call workers.triton_runner, etc.
            # You can leave your existing Triton implementation here.
            # Make sure result.json has ptx_path (if/when you implement it)
            # and ncu_report_path=None for now.
            ...
            return 0

        # --- CUDA PATH ---
        nvcc = shutil.which("nvcc")
        nvdisasm = shutil.which("nvdisasm") or shutil.which("cuobjdump")
        if not nvcc:
            json_dump(
                {"ok": False, "error": "nvcc not found on PATH"},
                workdir / "result.json",
            )
            return 1

        arch = k.device_profile.arch if k.device_profile else "sm_90"
        src_text = target_path.read_text(encoding="utf-8")
        src_hash = sha256_str(arch + "::" + src_text)

        COMPILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_ptx = COMPILE_CACHE_DIR / f"{src_hash}.ptx"
        cached_cubin = COMPILE_CACHE_DIR / f"{src_hash}.cubin"

        if cached_ptx.exists() and cached_cubin.exists():
            shutil.copy(cached_ptx, workdir / "kernel.ptx")
            shutil.copy(cached_cubin, workdir / "kernel.cubin")
        else:
            # Compile PTX
            rc, out, err = run_subprocess(
                [
                    nvcc,
                    "-O3",
                    f"-arch={arch}",
                    "-Xptxas",
                    "-v",
                    "-ptx",
                    target_rel,
                    "-o",
                    "kernel.ptx",
                ],
                cwd=workdir,
            )
            write_text(workdir / "nvcc_ptx_stdout.log", out)
            write_text(workdir / "nvcc_ptx_stderr.log", err)
            if rc != 0:
                json_dump(
                    {"ok": False, "error": "nvcc -ptx failed", "rc": rc},
                    workdir / "result.json",
                )
                return 1

            # Compile CUBIN
            rc, out, err = run_subprocess(
                [
                    nvcc,
                    "-O3",
                    f"-arch={arch}",
                    "-Xptxas",
                    "-v",
                    "-cubin",
                    target_rel,
                    "-o",
                    "kernel.cubin",
                ],
                cwd=workdir,
            )
            write_text(workdir / "nvcc_cubin_stdout.log", out)
            write_text(workdir / "nvcc_cubin_stderr.log", err)
            if rc != 0:
                json_dump(
                    {"ok": False, "error": "nvcc -cubin failed", "rc": rc},
                    workdir / "result.json",
                )
                return 1

            # Populate cache best-effort
            try:
                shutil.copy(workdir / "kernel.ptx", cached_ptx)
                shutil.copy(workdir / "kernel.cubin", cached_cubin)
            except Exception:
                pass

        save_partial_state(
            workdir,
            "compiled",
            {
                "ptx_path": str(workdir / "kernel.ptx"),
                "cubin_path": str(workdir / "kernel.cubin"),
                "arch": arch,
            },
        )

        # Optionally disassemble to SASS for later inspection
        if nvdisasm:
            if "nvdisasm" in nvdisasm:
                rc, out, err = run_subprocess(
                    [nvdisasm, "kernel.cubin"], cwd=workdir
                )
            else:
                rc, out, err = run_subprocess(
                    [nvdisasm, "--dump-sass", "kernel.cubin"], cwd=workdir
                )
            write_text(
                workdir / "kernel.sass",
                out if rc == 0 else f"disasm failed: {err}",
            )

        # Effective IO contract (may incorporate launch updates)
        effective_io = k.io.to_dict() if k.io else None
        # ... if you have launch_update logic, keep it here ...

        runner_config = {
            "root_path": str(Path(__file__).parent.resolve()),
            "kernel_cubin_path": str((workdir / "kernel.cubin").resolve()),
            "kernel_name": get_metadata_value(k.metadata, "kernel_name", "kernel"),
            "io_contract": effective_io or {},
            "timing": timing,
            "result_path": str((workdir / "runner_result.json").resolve()),
        }
        runner_cfg_path = workdir / "runner_config.json"
        runner_cfg_path.write_text(
            json.dumps(runner_config, indent=2), encoding="utf-8"
        )

        env = os.environ.copy()
        # Ensure workers.* modules can be imported
        env["PYTHONPATH"] = str(Path(__file__).parent.resolve())

        ncu_cfg = control.get("ncu", {}) or {}
        use_ncu = bool(ncu_cfg.get("enabled", False) and ncu_cfg.get("collect", True))
        ncu_cli = find_ncu_cli()
        runner_cmd = [
            sys.executable,
            "-m",
            "workers.cuda_runner",
            str(runner_cfg_path.resolve()),
        ]

        ncu_report_path: Optional[str] = None
        report_basename = "ncu_report"

        if use_ncu and ncu_cli:
            # Wrap the runner in Nsight Compute
            kernel_name_for_ncu = get_metadata_value(
                k.metadata, "kernel_name", "kernel"
            )
            cmd = [
                ncu_cli,
                "-f",  # force overwrite
                "-o",
                report_basename,
                "-k",
                kernel_name_for_ncu,
            ] + runner_cmd

            rc, out, err = run_subprocess(
                cmd, cwd=workdir, env=env, timeout=900
            )
            write_text(workdir / "runner_stdout.log", out)
            write_text(workdir / "runner_stderr.log", err)

            # Discover the report file
            for ext in (".ncu-rep", ".nsight-cuprof-report"):
                cand = workdir / f"{report_basename}{ext}"
                if cand.exists():
                    ncu_report_path = str(cand)
                    break

        else:
            # Run without Nsight Compute
            rc, out, err = run_subprocess(
                runner_cmd, cwd=workdir, env=env, timeout=900
            )
            write_text(workdir / "runner_stdout.log", out)
            write_text(workdir / "runner_stderr.log", err)

        if rc != 0 or not (workdir / "runner_result.json").exists():
            json_dump(
                {
                    "ok": False,
                    "error": "runner failed",
                    "rc": rc,
                },
                workdir / "result.json",
            )
            return 1

        r = json_load(workdir / "runner_result.json")

        save_partial_state(
            workdir,
            "timed",
            {
                "mean_ms": r.get("mean_ms", 1e9),
                "std_ms": r.get("std_ms", 0.0),
                "ok": r.get("ok", False),
            },
        )

        final_state_hash = sha256_str(
            materialized_source + json.dumps(effective_io or {}, sort_keys=True)
        )

        outj = {
            "ok": bool(r.get("ok", False)),
            "mean_ms": float(r.get("mean_ms", 1e9)),
            "std_ms": float(r.get("std_ms", 0.0)),
            "state_hash": final_state_hash,
            "ncu_metrics": {
                # Minimal summary; detailed analysis is done later via NcuMetricsContext
                "kernel_time_ms": float(r.get("mean_ms", 0.0))
            },
            "materialized_source": materialized_source,
            "ptx_path": str(workdir / "kernel.ptx"),
            "ncu_report_path": ncu_report_path,
        }
        json_dump(outj, workdir / "result.json")
        return 0

    except Exception as e:
        tb = traceback.format_exc()
        out = {"ok": False, "error": str(e), "traceback": tb}
        try:
            workdir = (
                Path(control["workdir"])
                if "control" in locals()
                else Path(control_path).parent
            )
            partial_state = get_latest_phase_data(workdir)
            if partial_state:
                out["partial_state"] = partial_state
        except Exception:
            pass
        json_dump(out, Path(control_path).parent / "result.json")
        return 1
```

**Notes:**

* I left the **Triton path** as `...` so you can keep your existing implementation; you only really need Nsight fully wired for CUDA first.
* The **only new dependency** here is `find_ncu_cli()` we added earlier.
* The worker now always tries to produce:

  * `kernel.ptx`
  * `kernel.cubin`
  * `kernel.sass` (optional)
  * `ncu_report.*` when Nsight is enabled and available.

---

## 4. What you get after applying these patches

Once you apply these changes and run on your P5E48XL with H200:

* **Every CUDA evaluation** produces:

  * Valid PTX + cubin (and optionally SASS).
  * A real Nsight Compute report file.
* The **orchestrator**:

  * Stores PTX/NCU paths in NodeStats (for baseline and variants).
  * Initializes `RegionDagContext`, `NcuMetricsContext`, `PtxSourceCorrelator` per node.
  * Sets `kernel_opt_tooling.CURRENT_*` globals so tools operate in a real context.
* The **LLM**:

  * Sees PTX- and Nsight-derived summaries in its system prompt.
  * Can call tools to, e.g., examine specific regions, loops, and metrics.

That gets you to a **functional, introspection-rich kernel optimizer** that you can then iterate into the full 10/10 design using the Phase 1–3 roadmap above.

If you’d like, next step we can do is:

* Design the **vLLM harness spec format** and implement the first vLLM kernel integration (e.g., RMSNorm) end-to-end, including correctness checks and profiling on your H200.

[1]: https://forums.developer.nvidia.com/t/extract-data-from-roofline-plot/322219?utm_source=chatgpt.com "Extract data from roofline plot - Nsight Compute"
[2]: https://tvm.apache.org/docs/v0.8.0/reference/api/python/autotvm.html?utm_source=chatgpt.com "tvm.autotvm — tvm 0.8.dev0 documentation - Apache TVM"
[3]: https://tvm.apache.org/2018/10/03/auto-opt-all?utm_source=chatgpt.com "Automatic Kernel Optimization for Deep Learning on All ..."
[4]: https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tree/master/ExamplePlots?utm_source=chatgpt.com "NERSC / Roofline-on-NVIDIA-GPUs"
