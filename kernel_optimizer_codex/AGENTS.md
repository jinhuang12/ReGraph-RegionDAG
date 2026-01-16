# AGENTS.md — Lightweight GPU Kernel Optimization Guide for Codex

This repository is set up so that **codex** can act as a GPU performance
engineer for Triton / CUDA kernels backed by an IronFist-style contract
dataset.

Codex has **full shell access** in this repo.

This file is a **reference**, not a rigid checklist. Use it to understand:
- where kernels live,
- how to run them and benchmark them,
- what tools are available,
- what **not** to touch.

Your primary objective is still:
> Make target kernels faster while keeping correctness, using whatever
> combination of custom code and existing tools is most effective.

---

## 1. Repository Layout (from repo root)

- `contracts/`
  - IronFist JSON contracts such as:
    - `vec_matmul_kernel_small_contract.json`
    - `gemm_split_k_kernel_large_contract.json`
  - Contracts are organized in subfolders (e.g. `contracts/ironfist`, `contracts/triton_bench`); generated kernels/suites mirror these subfolders under `kernels/` and `suites/`.
  - Each contract typically contains:
    - `kernel.kernel_type` (often `"multi_kernel"`)
    - `kernel.metadata.kernel_name` (e.g. `"gemm_split_k_kernel"`)
    - `kernel.metadata.entry_point` (usually `"run"`)
    - `kernel.io.args` (meta params + sometimes tensor specs)
    - `kernel.source_code` (Python+Triton or CUDA wrapper code)

- `kernels/`
  - One Python module per kernel family, e.g.:
    - `gemm_split_k_kernel.py`
    - `vec_matmul_kernel.py`
  - **You should treat files under `kernels/` as the main optimization surface.**
    This is where you adjust meta-parameters and kernel implementation.

- `suites/`
  - One JSON suite per kernel family, e.g.:
    - `gemm_split_k_kernel_suite.json`
  - Each suite groups multiple contracts (small/medium/large/rectangular/etc)
    and defines a suite objective (typically geomean runtime/speedup).

- `tools/`
  - Harness + helper scripts (see Section 2).

- `runs/`
  - Created by tools / your scripts.
  - Contains per-kernel, per-tag, per-contract profiling artifacts and summaries.
  - May also contain per-kernel `trace.jsonl` logs of your optimization attempts.

- Root utilities:
  - `kernel_opt_tooling.py` / `ptx_dag_tool_v2.py`
    - Internal helpers for Nsight metrics & Region-DAG analysis.
    - You can **read** these to understand outputs, but don’t modify them unless
      the user explicitly asks.

---

## 2. Tools Overview

All commands are run from the **repo root** unless noted otherwise.

Before using a tool script for the first time in a session, you may:
- run `python tools/<script>.py --help`, or
- open the top of the file (`sed -n '1,120p tools/<script>.py'`)
to confirm arguments and behavior. Do not guess CLI arguments.

### 2.1 Primary benchmarking tools

These are the main scripts you are *likely* to use for apples-to-apples
measurements. You may also write your own small tuning harnesses when that’s
easier, as long as you keep comparisons fair.

#### `tools/init_kernels_and_suites.py`

**Purpose**
- Scan `contracts/` and:
  - create/update `kernels/<kernel_name>.py` from contract `source_code`,
  - create/update `suites/<kernel_name>_suite.json` grouping contracts.

**Typical usage**

```bash
python tools/init_kernels_and_suites.py \
  --contracts-subdir ironfist \
  --kernels-dir kernels \
  --suites-dir suites
```

Run this:

* once after checkout,
* and again if new contracts are added or contract `source_code` changes.

> Do **not** edit `contracts/*.json` directly; treat them as your dataset.
> `--contracts-subdir` is required: it scopes to a subfolder under `contracts/` (e.g. `ironfist`, `triton_bench`) and mirrors that subfolder under `kernels/` and `suites/`. Suites record `contracts_dir` accordingly; profiling must point to the mirrored subpaths (e.g. `suites/ironfist/...`, `kernels/ironfist/...`).

---

#### `tools/profile_suite.py`  — suite-level baseline & candidates

**Purpose**

* Run all contracts in a suite against a kernel module.
* Always checks correctness by running a baseline kernel and comparing outputs; uses stored baseline snapshot if available.
* Works with subdir-scoped suites (e.g. `suites/ironfist/...`, `suites/triton_bench/...`); the repo root is inferred by locating the nearest `suites/` ancestor.
* Write:

  * per-contract results:
    `runs/<kernel_name>/<tag>/<contract_name>/result.json`
  * aggregate summary:
    `runs/<kernel_name>/<tag>/summary.json`

**Typical usage**

> Tip: for hash-suffixed kernel modules, use the suite’s `kernel_name` field.
> Example: `suites/ironfist/gemm_split_k_kernel_suite.json` → `kernel_name: gemm_split_k_kernel_a4a9473a`
> so the module is `kernels/ironfist/gemm_split_k_kernel_a4a9473a.py` and runs are under `runs/gemm_split_k_kernel_a4a9473a/...`.

Baseline:

```bash
python tools/profile_suite.py \
  --suite suites/ironfist/gemm_split_k_kernel_suite.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --device cuda:0 \
  --tag baseline \
  --warmup 10 --iters 100 --repeat 5
```

After a change (meta or structural):

```bash
python tools/profile_suite.py \
  --suite suites/ironfist/gemm_split_k_kernel_suite.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --baseline-module runs/gemm_split_k_kernel_a4a9473a/baseline/baseline_module.py \
  --device cuda:0 \
  --tag struct_001 \
  --warmup 10 --iters 100 --repeat 5 \
  --timing-mode cuda_graphs  # optional: events or cuda_graphs (default)
```

You should:

* use the **same warmup/iters/repeat** when comparing tags,
* interpret `summary.json` as the canonical suite-level comparison.
* For the first baseline run, `--baseline-module` defaults to `--module` and writes a snapshot to `runs/<kernel>/baseline/baseline_module.py`. Subsequent tags will automatically use that snapshot if `--baseline-module` is omitted; provide an explicit path to compare against a different baseline.
* `--timing-mode` controls timing: `events` uses CUDA events; `cuda_graphs` (default) captures/replays to strip launch overhead and will fall back to events if capture fails.
  * For fairness, non-baseline tags will only use `cuda_graphs` if the stored baseline summary was also graph-timed; otherwise they force `events`.
  * You should try to use cuda_graphs by default unless the kernel is un-capturable to isolate just the latency for execution 

---

#### `tools/profile_contract.py`  — single-contract microbenchmark

**Purpose**

* Run a single contract (one shape/config) against a kernel module with:

  * warmup iterations,
  * timed iterations.

* Writes `result.json` under `--out` with:

  * `mean_ms`, `std_ms`,
  * a simple output fingerprint,
  * optional `ncu_report` path if `--with-ncu` is used.

**Typical usage**

Plain timing of one contract:

```bash
python tools/profile_contract.py \
  --contract contracts/ironfist/gemm_split_k_kernel_large_contract.json \
  --module   kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --entry-point run \
  --device cuda:0 \
  --warmup 10 --iters 100 --repeat 5 \
  --out runs/gemm_split_k_kernel_a4a9473a/large_baseline
```

With Nsight Compute:

```bash
python tools/profile_contract.py \
  --contract contracts/ironfist/gemm_split_k_kernel_large_contract.json \
  --module   kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --entry-point run \
  --device cuda:0 \
  --warmup 10 --iters 100 --repeat 5 \
  --baseline-module runs/gemm_split_k_kernel_a4a9473a/baseline/baseline_module.py \
  --out runs/gemm_split_k_kernel_a4a9473a/ncu_large \
  --timing-mode cuda_graphs  # optional: events (default) or cuda_graphs \
  --backend triton \
  --with-ncu \
  --ncu-bin ncu \
  --ncu-call auto \
  --ncu-args "--kernel-name gemm_split_k_kernel"
```

Use this when:

* zooming into one representative shape,
* validating kernel-only performance,
* or generating an Nsight report for one contract.
* Correctness is always checked vs baseline before timing; if no `--baseline-module` is supplied, it defaults to the candidate module or an existing baseline snapshot.
* Triton-only contracts without a Python wrapper are supported: entry_point defaults to `metadata.kernel_name`, meta args are pulled from `is_meta` fields, and launch uses the contract `io.launch.grid/num_warps/num_stages` to call `kernel[grid](...)`.
* Nsight: the wrapper uses `ncu -f --set full ... --export <out>/ncu_report` and writes the exact command to `<out>/ncu_cmd.txt`.
  * For custom Nsight flags (filters/metrics), pass `--ncu-args "..."` or run `ncu` directly (below).
  * For `multi_kernel` contracts, the default (`--ncu-call auto`) profiles `benchmark_kernel` when it is used for timing.

**Backend / language switching (Option A)**

* Treat `multi_kernel` contracts as “shape-only” and put the real implementation behind `benchmark_kernel`.
* Use `--backend <name>` to select an implementation (e.g. `triton`, `cuda`, `cutlass`). The tools pass this via:
  * `contract_args["_backend"]` (for `benchmark_kernel`), and
  * `KO_BACKEND` environment variable (for any code path).
* If you implement the candidate backend only in the entrypoint (`run()`), use `--timing-target entrypoint` and `--ncu-call entrypoint` so the harness measures/profiles that path.

**Direct Nsight Compute (recommended when you want “just ncu”)**

```bash
ncu -f --set full --target-processes all --import-source yes \
  --export runs/gemm_split_k_kernel_a4a9473a/ncu_large/ncu_report \
  python tools/run_contract_once.py \
    --contract contracts/ironfist/gemm_split_k_kernel_large_contract.json \
    --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
    --call benchmark_kernel \
    --warmup 0 --iters 1 --repeat 1 \
    --backend triton \
    --device cuda:0
```

---

#### `tools/append_trace.py`  — optional structured logging

**Purpose**

* Append a JSON record to `runs/<kernel_name>/trace.jsonl`.
* Each line = one attempt (meta or structural), including:

  * tag,
  * high-level description,
  * baseline & candidate summaries,
  * optional Nsight summary path.

**Typical usage**

```bash
python tools/append_trace.py \
  --kernel-name gemm_split_k_kernel_a4a9473a \
  --summary runs/gemm_split_k_kernel_a4a9473a/struct_010/summary.json \
  --baseline runs/gemm_split_k_kernel_a4a9473a/baseline/summary.json \
  --tag struct_010 \
  --edit-kind structural \
  --description "Shape-aware split_k clamp + tiling tweaks; ~1.05x geomean" \
  --contract contracts/ironfist/gemm_split_k_kernel_large_contract.json \
  --ncu-summary runs/gemm_split_k_kernel_a4a9473a/ncu_struct_010_large/ncu_summary.json
```

Use this when you want a durable history of good attempts. You don’t
have to log every throwaway experiment.

---

### 2.2 Analysis tools (advanced / optional)

Use these when you need deeper insight, especially during structural work.
They are **not required** for every attempt.

#### `tools/summarize_ncu.py` — Nsight Compute summary

**Purpose**

* Consume an `.ncu-rep` file and emit a small JSON summary via
  `NcuMetricsContext` in `kernel_opt_tooling.py`.

**Typical usage**

```bash
python tools/summarize_ncu.py \
  --report runs/gemm_split_k_kernel_a4a9473a/ncu_large/ncu_report.ncu-rep \
  --mode summary \
  > runs/gemm_split_k_kernel_a4a9473a/ncu_large/ncu_summary.json
```

Use this to answer:

* Is the kernel compute-bound vs memory-bound?
* How’s SM throughput, occupancy, and major stall reasons?

There are extra modes (`get_values`, `search_names`) for specific metrics
if you need them. Check `--help`.

---

#### `tools/region_dag_summary.py` — PTX Region-DAG

**Purpose**

* Given a `.ptx` file, build a Region-DAG summary:

  * stages,
  * hot regions,
  * loops, barriers, divergence.

**Typical usage**

```bash
python tools/region_dag_summary.py \
  --ptx runs/my_cuda_kernel/some_candidate/kernel.ptx \
  --mode overview \
  > runs/my_cuda_kernel/some_candidate/region_dag_overview.json
```

Use this when:

- You want to map Nsight stall/memory bottlenecks onto structural regions of the code.
- Kernels have non-trivial loops/pipelines (async copy, ldmatrix, blocked reductions) and you need to see staging depth and barrier placement.
- Nsight shows long scoreboard, barrier, or divergence stalls and you need to localize them to specific regions/loops.
- Sparse/indirect or heavily masked workloads where you want to spot where address arithmetic and divergence cluster.
- After structural changes, to confirm barriers/regions were removed or moved as intended before re-profiling.

For pure Triton flows, PTX exists in the triton cache; just make sure you're pointing at the correct directory / kernel PTX.

---

#### `tools/ptx_source_summary.py`

**What it does**

* Wraps `PtxSourceCorrelator` (from `ptx_source_correlator.py`) to pull PTX/SASS correlated to source lines from an Nsight Compute `.ncu-rep`.
* Can emit a focused snippet for a source span or a full per-line mapping filtered to a source file, optionally attaching per-PC metric values (e.g., `inst_executed`, stall counters).

**When to use**

* After collecting an `.ncu-rep` with `--import-source yes`, to map hot PCs back to Triton/CUDA lines.

Good use cases:

- When Nsight shows stalls (e.g., long scoreboard/barrier) and you need exact source lines carrying those PCs/metrics.
- After changing async copy/ldmatrix pipelines to confirm hot PCs shifted to the intended overlapped region and off old barriers.
- Fused kernels (matmul + epilogue/normalization) to see which phase owns hot PCs before editing.
- Divergent/masked workloads to pinpoint masking-heavy lines that dominate `inst_executed` or divergence samples.
- Reductions/scans to see which reduction phase (tree vs. write-out) holds the hot PCs and should get barrier/shuffle tweaks.

**How to use**

Snippet for a source span:

```bash
python tools/ptx_source_summary.py \
  --report runs/gemm_split_k_kernel_a4a9473a/ncu_import_source_test3/ncu_report.ncu-rep \
  --kernel gemm_split_k_kernel \
  --source-file kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --start-line 90 --end-line 130 \
  --include-sass \
  --include-metric \
  --extra-metric smsp__pcsamp_warps_issue_stalled_long_scoreboard \
  > runs/gemm_split_k_kernel_a4a9473a/ncu_import_source_test3/source_span.json
```

Full mapping filtered to a file:

```bash
python tools/ptx_source_summary.py \
  --report runs/gemm_split_k_kernel_a4a9473a/ncu_import_source_test3/ncu_report.ncu-rep \
  --kernel gemm_split_k_kernel \
  --source-file kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --mode mapping \
  > runs/gemm_split_k_kernel_a4a9473a/ncu_import_source_test3/source_mapping.json
```

> Tip: pass `--nvtx <range>` and `--kernel <name>` when profiling multiple actions in one report so the correlator picks the right kernel without fallback.

---

### 2.3 Internal helpers (read-only)

These are part of the harness. You typically do **not** call or modify them.

* `tools/contracts_common.py`

  * Shared logic for loading contract JSON and building inputs.

* `tools/run_contract_once.py`

  * Helper used under `profile_contract.py` (especially in `--with-ncu` mode).

* `kernel_opt_tooling.py`, `ptx_dag_tool_v2.py`, `ptx_source_correlator.py`

  * Internal libraries backing `summarize_ncu.py`, `region_dag_summary.py` & `ptx_source_summary.py`.
  * Safe to read for understanding; avoid editing.

---

## 3. Guardrails

To keep the setup sane and reproducible:

1. **Don’t edit `contracts/`** unless the user explicitly says so.

   * Treat contracts as the “task dataset”.

2. **Prefer editing kernel implementations under `kernels/`.**

   * Meta-parameter logic,
   * tiling / blocking,
   * async pipelines,
   * split-K behavior, etc.

3. **Keep comparisons apples-to-apples.**

   * When comparing two tags (`baseline` vs `struct_010`), keep
     `--warmup`, `--iters`, `--repeat` the same.
   * If you change these, consider that a new baseline.

4. **Avoid dangerous shell commands.**

   * Don’t run `rm -rf /`, `chmod` on system paths, or anything that
     obviously risks the environment.

5. **Respect the user’s intent.**

   * If the user says “just optimize this one kernel using your own harness,”
     it is fine to write a small custom `benchmark_kernel` or tuning script,
     and only use `profile_suite.py` / `profile_contract.py` for validation.

---

## 4. Example Optimization Workflow (Non-binding)

This section describes a **recommended loop**. It’s not a rigid checklist:
you can deviate if you have a better plan that still preserves correctness
and fair measurement.

Use this as a mental model:

> **Loop:**
>
> 1. Profile baseline →
> 2. Analyze results (decide if tools help) →
> 3. Plan the next change →
> 4. Implement & validate →
> 5. Profile vs baseline & reassess → repeat until improvements dry up.

Below is how that maps onto this repo.

### 4.1 One-time setup for a kernel family

For a kernel family, e.g. `gemm_split_k_kernel`:

1. **Initialize kernels and suites (if not done yet)**

   ```bash
   python tools/init_kernels_and_suites.py \
     --contracts-subdir ironfist \
     --kernels-dir kernels \
     --suites-dir suites
   ```

2. **Read the suite and kernel**

   ```bash
   cat suites/ironfist/gemm_split_k_kernel_suite.json
   sed -n '1,200p' kernels/ironfist/gemm_split_k_kernel_a4a9473a.py
   ```

   Understand:

   * which contracts are included (small/medium/large/high_split/etc),
   * which ones are closest to realistic workloads (prefill vs decode).

### 4.2 The iterative optimization loop

For each target kernel, follow this loop. You may implement additional
helper functions (e.g. `benchmark_kernel`) inside `kernels/<kernel>.py`
if that helps, but keep comparisons fair.

---

#### Step 1 — Profile baseline

Get a suite-level baseline:

```bash
python tools/profile_suite.py \
  --suite suites/ironfist/gemm_split_k_kernel_suite.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --device cuda:0 \
  --tag baseline \
  --warmup 10 --iters 100 --repeat 5
```

Optional but recommended: pick one representative contract (often `*_large`
or a realistic config) and generate an Nsight baseline:

```bash
python tools/profile_contract.py \
  --contract contracts/ironfist/gemm_split_k_kernel_large_contract.json \
  --module   kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --entry-point run \
  --device cuda:0 \
  --warmup 10 --iters 100 --repeat 5 \
  --out runs/gemm_split_k_kernel_a4a9473a/ncu_large \
  --backend triton \
  --with-ncu --ncu-bin ncu \
  --ncu-call auto \
  --ncu-args "--kernel-name gemm_split_k_kernel"

python tools/summarize_ncu.py \
  --report runs/gemm_split_k_kernel_a4a9473a/ncu_large/ncu_report.ncu-rep \
  --mode summary \
  > runs/gemm_split_k_kernel_a4a9473a/ncu_large/ncu_summary.json
```

Treat:

* `runs/<kernel>/baseline/summary.json` as your suite baseline.
* `runs/<kernel>/ncu_<contract>/ncu_summary.json` as your device-level
  baseline for that shape.

---

#### Step 2 — Analyze results and use tools

After each baseline or candidate run:

1. **Read the suite summary**, e.g.:

   ```bash
   cat runs/gemm_split_k_kernel_a4a9473a/baseline/summary.json
   ```

   Note which contracts dominate time and whether any are outliers.

2. If you have an Nsight report, **summarize and interpret it**:

   * `summarize_ncu.py` (already shown above),
   * optionally `region_dag_summary.py` or `ptx_source_summary.py` if you
     need structural / source-correlated insight.

3. Explicitly decide and state (in your own “thinking”):

   * Is the kernel currently:

     * compute-bound?
     * memory-bound?
     * stall-limited (long scoreboard, barriers)?
     * occupancy-limited?
   * Which contracts are bottlenecking suite geomean?
   * Would an analysis tool help right now?

     * Nsight summary: to separate compute vs memory vs stalls.
     * Region DAG: to map stalls onto PTX regions (mostly CUDA).
     * PTX source correlator: to map hot PCs back to kernel source lines.


---

#### Step 3 — Plan the next change

Based on Step 2, update your plan before touching code:

* If the problem looks like **meta tuning**:

  * adjust tile sizes, `num_warps`, `num_stages`, split-K, etc.
  * consider writing a small `benchmark_kernel` or tuning helper inside
    `kernels/<kernel>.py` to sweep configs for a few key shapes.

* If it looks like a **structural issue**:

  * rework tiling/blocking,
  * introduce or deepen async pipelines,
  * restructure epilogues, reduce atomics, improve memory access patterns.

You can mix both, but it helps to label each candidate as primarily
“meta” vs “structural” in your own reasoning and in `struct_XXX` tag names.

Pick the **single most promising next change** and commit to testing it
before you move on to another idea.

---

#### Step 4 — Implement and validate correctness

Make a small, coherent change under `kernels/<kernel_name>.py` only.

Then:

1. **Check correctness** against a reference behavior:

   * If the kernel module has a convenient `test_*` or simple Python
     fallback, compare outputs.
   * Otherwise, you can:

     * temporarily keep a “baseline” copy of the kernel file, or
     * use contracts to run before/after and compare outputs (within
       tolerances for FP16/BF16).

2. Make sure your change **does not modify**:

   * the meaning of `contracts/*.json` (unless explicitly allowed),
   * the evaluation protocol (`warmup`, `iters`, `repeat`),
   * shapes/dtypes in the suite.

If a candidate breaks correctness, fix it or revert before treating it
as a real optimization.

---

#### Step 5 — Profile candidate vs baseline and reassess

With your new change in place, run a suite benchmark with a new tag:

```bash
python tools/profile_suite.py \
  --suite suites/ironfist/gemm_split_k_kernel_suite.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --device cuda:0 \
  --tag struct_001 \
  --warmup 10 --iters 100 --repeat 5
```

Compare:

* `runs/<kernel>/baseline/summary.json`
* `runs/<kernel>/struct_001/summary.json`

Key indicators:

* `aggregate.geomean_ms`
* `aggregate.geomean_speedup_vs_baseline` (if present)
* Per-contract changes (did you regress small shapes to fix large ones, etc.).

If helpful, run Nsight again on the same representative contract:

```bash
python tools/profile_contract.py \
  --contract contracts/ironfist/gemm_split_k_kernel_large_contract.json \
  --module   kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --entry-point run \
  --device cuda:0 \
  --warmup 10 --iters 100 --repeat 5 \
  --out runs/gemm_split_k_kernel_a4a9473a/ncu_struct_001_large \
  --backend triton \
  --with-ncu --ncu-bin ncu \
  --ncu-call auto \
  --ncu-args "--kernel-name gemm_split_k_kernel"

python tools/summarize_ncu.py \
  --report runs/gemm_split_k_kernel_a4a9473a/ncu_struct_001_large/ncu_report.ncu-rep \
  --mode summary \
  > runs/gemm_split_k_kernel_a4a9473a/ncu_struct_001_large/ncu_summary.json
```

Then go back to **Step 2** with the new results:

* analyze the new suite and Nsight summaries,
* decide whether your current plan still makes sense,
* either double down on the same direction or change course.

---

#### Loop and stopping

Repeat Steps 2–5:

* `analyze → plan → implement → validate → profile → analyze → …`

until:

* improvements in suite geomean are clearly < ~3–5% across several tries, **and**
* Nsight indicates you’re reasonably saturating the primary bottleneck
  (e.g. compute-bound with healthy SM throughput).

You do **not** need a formal proof of a performance ceiling—just a clear
signal of diminishing returns + sane hardware metrics.

Use `append_trace.py` when you have a candidate worth remembering
(best tags, interesting Nsight profiles, etc.).

---

## 5. Reporting Back to the User

When you explain what you did to the user, try to mention:

* Which kernel and suite you worked on:

  * `suite` path,
  * kernel module path.

* Baseline vs best candidate:

  * baseline geomean,
  * candidate geomean,
  * speedup factor.

* Any key Nsight signals (if used):

  * SM throughput,
  * occupancy,
  * stall reasons,
  * memory throughput.

* A brief description of the code changes:

  * e.g. “shape-aware split_k clamp + tiling tweaks for large K”.

By following this pattern and using tools when they genuinely help
your reasoning (instead of mechanically), you’ll behave like a careful
GPU performance engineer with a reproducible, traceable workflow.
