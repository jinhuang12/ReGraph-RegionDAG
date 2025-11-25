# AGENTS.md — GPU Kernel Optimization Playbook for Codex

This repo is set up so that **codex** can act as a GPU performance
engineer for Triton / CUDA kernels backed by an IronFist-style contract dataset.

The agent (Codex) has **full shell access**. It must follow the workflow and
guardrails below.

You should treat this file as the source of truth for:

- how to run kernels,
- how to benchmark them fairly,
- how to log attempts,
- when to use profiling and structural analysis tools.

---

## 1. Repository Layout

From the repo root:

- `contracts/`
  - IronFist contracts such as:
    - `vec_matmul_kernel_small_contract.json`
    - `vec_matmul_kernel_medium_contract.json`
    - `layer_norm_kernel_large_contract.json`
  - Each contains:
    - `kernel.kernel_type` (often `"multi_kernel"`)
    - `kernel.metadata.kernel_name` (e.g. `"vec_matmul_kernel"`)
    - `kernel.metadata.entry_point` (e.g. `"run"`)
    - `kernel.io.args` (meta params and sometimes tensor specs)
    - `kernel.source_code` (Python+Triton or CUDA wrapper code)

- `kernels/`
  - One Python module per kernel family, e.g.:
    - `vec_matmul_kernel.py`
    - `layer_norm_kernel.py`
  - **Codex should only edit kernel code under this directory.**

- `suites/`
  - One JSON suite per kernel family, e.g.:
    - `vec_matmul_kernel_suite.json`
  - Each suite groups multiple contracts (shapes / sizes) and defines a suite
    objective (e.g. geometric mean speedup).

- `tools/`
  - Core harness:
    - `init_kernels_and_suites.py` — build `kernels/` and `suites/` from contracts.
    - `contracts_common.py`     — shared helpers to load contracts and build inputs.
    - `profile_contract.py`     — time a single contract & module.
    - `run_contract_once.py`    — run one contract once (used under Nsight Compute).
    - `profile_suite.py`        — time a whole suite and compute aggregate metrics.
  - Optional analysis tools:
    - `summarize_ncu.py`        — summarize Nsight Compute `.ncu-rep` reports.
    - `region_dag_summary.py`   — summarize PTX into Region-DAG JSON (stages/regions).
    - `ptx_source_summary.py`   — pull PTX/SASS correlated to source lines from `.ncu-rep`.
    - `append_trace.py`         — append structured records to `runs/<kernel>/trace.jsonl`.

- `runs/`
  - Created by the tools; contains per-kernel, per-tag, per-contract profiling
    artifacts and summary JSONs.
  - Also contains per-kernel `trace.jsonl` files when `append_trace.py` is used.

---

### 2. CLI Tools and How to Use Them

All commands are run from the **repo root** unless explicitly stated.

There are three categories:

* **Profiling & Correctness** – core benchmarking + logging.
* **Analysis** – Nsight / PTX analysis when doing structural work.
* **Internal helpers (read‑only)** – used by the primary tools; you generally don’t call them directly.

Also, **before using any tool for the first time in a session**, Codex should:

* Either run `python tools/<script>.py --help`, **or**
* Open the top of the file (`sed -n '1,120p tools/<script>.py'`) to read the docstring and CLI.

Never guess argument names.

---

### 2.1 Profiling & Correctness tools

These form the canonical workflow for baseline, meta tuning, structural variants, and logging.

#### `tools/init_kernels_and_suites.py`

**What it does**

* Scans `contracts/` (IronFist JSON contracts).
* For each `kernel.metadata.kernel_name`, it:

  * Creates/updates a kernel module under `kernels/<kernel_name>.py` (using `kernel.source_code` from contracts).
  * Creates/updates `suites/<kernel_name>_suite.json` that groups all contracts for that kernel (small/medium/large/high_split/etc).

**When to use**

* Once per repo setup or if the kernels/ or suites/ directory is empty.
* Again whenever **new contracts** are added or existing contract source changes significantly.

**How to use**

```bash
python tools/init_kernels_and_suites.py \
  --contracts-dir contracts \
  --kernels-dir kernels \
  --suites-dir suites
```

**Rules**

* **Do NOT edit files under `contracts/`**. Those are your “dataset”.
* Codex may edit `kernels/<kernel_name>.py` (that’s the optimization surface).
* Codex may inspect `suites/*.json` but should not rewrite them unless explicitly asked.

---

#### `tools/profile_suite.py`  ✅ main benchmark entrypoint

**What it does**

* Runs **all contracts in a suite** with a given kernel module and writes:

  * Per‑contract results under
    `runs/<kernel_name>/<tag>/<contract_name>/result.json`
  * Aggregate summary under
    `runs/<kernel_name>/<tag>/summary.json`

**Key fields in `summary.json`**

* Per‑contract metrics (per contract in the suite).
* `aggregate.geomean_ms`
* `aggregate.geomean_speedup_vs_baseline` (if a `baseline` tag exists).

**When to use**

* **Baseline** for a kernel family.
* Every **meta‑parameter** and **structural** variant.
* This is what we compare to decide whether a change is a win.

**How to use**

Baseline:

```bash
python tools/profile_suite.py \
  --suite suites/gemm_split_k_kernel_suite.json \
  --module kernels/gemm_split_k_kernel.py \
  --device cuda:0 \
  --tag baseline \
  --warmup 10 --iters 100 --repeat 5
```

After a change (meta or structural):

```bash
python tools/profile_suite.py \
  --suite suites/gemm_split_k_kernel_suite.json \
  --module kernels/gemm_split_k_kernel.py \
  --device cuda:0 \
  --tag meta_001 \
  --warmup 10 --iters 100 --repeat 5
```

**Optional Nsight per contract**

If you add `--with-ncu`, the harness will **also** generate NCU reports per contract by invoking `profile_contract.py` under `ncu`:

```bash
python tools/profile_suite.py \
  --suite suites/gemm_split_k_kernel_suite.json \
  --module kernels/gemm_split_k_kernel.py \
  --device cuda:0 \
  --tag struct_001 \
  --warmup 10 --iters 100 --repeat 5 \
  --with-ncu --ncu-bin ncu
```

---

#### `tools/profile_contract.py`  ✅ single‑contract micro‑benchmark

**What it does**

* Runs **one contract** (single shape/config) against a kernel module, with:

  * `warmup` iterations
  * `iters × repeat` timed iterations
* Writes `result.json` with:

  * `mean_ms`, `std_ms`
  * a simple output fingerprint
  * optional `ncu_report` path if `--with-ncu` is used

**When to use**

* Debugging a **single shape** in isolation.
* Zooming into a “representative” contract from the suite (e.g., `*_large`, `*_small`).
* Running Nsight Compute on exactly one contract.

**How to use**

Plain timing:

```bash
python tools/profile_contract.py \
  --contract contracts/gemm_split_k_kernel_large_contract.json \
  --module   kernels/gemm_split_k_kernel.py \
  --entry-point run \
  --device cuda:0 \
  --warmup 10 --iters 100 --repeat 5 \
  --out runs/gemm_split_k_kernel/large_baseline
```

With Nsight Compute:

```bash
python tools/profile_contract.py \
  --contract contracts/gemm_split_k_kernel_large_contract.json \
  --module   kernels/gemm_split_k_kernel.py \
  --entry-point run \
  --device cuda:0 \
  --warmup 10 --iters 100 --repeat 5 \
  --out runs/gemm_split_k_kernel/ncu_large \
  --with-ncu \
  --ncu-bin ncu
```

> **Important:** For `--with-ncu`, the script wraps a **single `run_contract_once` call** under `ncu` (one timed iteration after warmup), so the NCU report isn’t bloated by hundreds of iterations. The `warmup/iters/repeat` parameters still control normal non‑NCU timing.

---

#### `tools/append_trace.py`  ✅ structured logging

**What it does**

* Appends a JSON object to `runs/<kernel_name>/trace.jsonl` (one JSON per line).
* This is your **single source of truth** for:

  * which attempts were run,
  * what type of edit they were (meta vs structural),
  * and which summary / NCU files correspond.

**When to use**

* After every **meaningful** candidate (meta or structural) you keep.
* Optional for obviously-bad experiments, but recommended even for failures.

**How to use**

Example for `gemm_split_k_kernel` structural variant:

```bash
python tools/append_trace.py \
  --kernel-name gemm_split_k_kernel \
  --summary runs/gemm_split_k_kernel/struct_001/summary.json \
  --baseline runs/gemm_split_k_kernel/baseline/summary.json \
  --tag struct_001 \
  --edit-kind structural \
  --description "Clamp shared memory usage + cap split_k to reduce atomic contention" \
  --ncu-summary runs/gemm_split_k_kernel/ncu_large_abs/ncu_summary.json
```

The script will fill in `timestamp` and copy core metrics from the summaries.

---

### 2.2 Analysis tools

These aren’t required for **every** iteration, but they are expected when you’re doing serious structural optimization (tiling changes, async pipelines, etc.).

#### `tools/summarize_ncu.py`

**What it does**

* Wraps `NcuMetricsContext` (from `kernel_opt_tooling.py`) to consume `.ncu-rep` and emit **JSON summaries**.

**Modes**

1. `summary` – compact high‑level view (this is what Codex should use first):

   ```bash
   python tools/summarize_ncu.py \
     --report runs/gemm_split_k_kernel/ncu_large_abs/ncu_report.ncu-rep \
     --mode summary \
     > runs/gemm_split_k_kernel/ncu_large_abs/ncu_summary.json
   ```

   Typical fields:

   * `kernel_name`
   * `sm__throughput_pct`
   * `dram__throughput_pct`
   * occupancy / warp limits
   * major stall reasons

2. `get_values` – fetch specific metrics by canonical or raw NCU name:

   ```bash
   python tools/summarize_ncu.py \
     --report runs/gemm_split_k_kernel/ncu_large_abs/ncu_report.ncu-rep \
     --mode get_values \
     --names sm__throughput_pct dram__throughput_pct \
     --name-kind canonical
   ```

3. `search_names` – **search metric names**, not kernel names:

   ```bash
   python tools/summarize_ncu.py \
     --report runs/gemm_split_k_kernel/ncu_large_abs/ncu_report.ncu-rep \
     --mode search_names \
     --query "sm__"
   ```

> **Clarification:** `search_names` is for metric names/descriptions only.
> It will **not** find “gemm” as a kernel; use `--mode summary` to see `kernel_name` and then query metrics.

**When to use**

* After any `--with-ncu` run that you care about.
* Especially when:

  * deciding if a kernel is mem‑bound vs compute‑bound,
  * understanding whether occupancy or stalls are limiting,
  * checking whether a change actually improved hardware utilization.

---

#### `tools/region_dag_summary.py`

**What it does**

* Wraps `RegionDagContext` (in `kernel_opt_tooling.py` / `ptx_dag_tool_v2.py`) to analyze PTX and build a **Region DAG**:

  * stages (phases of the kernel),
  * hot regions,
  * loops, barriers, divergent regions.

**When to use**

* Mainly for **CUDA kernels** that you can compile to PTX (`.ptx`).
* When Nsight says “memory bound” or “stall limited” and you want to see exactly *where* in the PTX.

**How to use**

Overview:

```bash
python tools/region_dag_summary.py \
  --ptx runs/gemm_split_k_kernel/some_candidate/kernel.ptx \
  --mode overview \
  > runs/gemm_split_k_kernel/some_candidate/region_dag_overview.json
```

Specific stage:

```bash
python tools/region_dag_summary.py \
  --ptx runs/gemm_split_k_kernel/some_candidate/kernel.ptx \
  --mode stage_detail \
  --stage-id 2 \
  > runs/gemm_split_k_kernel/some_candidate/stage_2.json
```

Specific region:

```bash
python tools/region_dag_summary.py \
  --ptx runs/gemm_split_k_kernel/some_candidate/kernel.ptx \
  --mode region_detail \
  --region-id 7 \
  > runs/gemm_split_k_kernel/some_candidate/region_7.json
```

---

#### `tools/ptx_source_summary.py`

**What it does**

* Wraps `PtxSourceCorrelator` (from `ptx_source_correlator.py`) to pull PTX/SASS correlated to source lines from an Nsight Compute `.ncu-rep`.
* Can emit a focused snippet for a source span or a full per-line mapping filtered to a source file, optionally attaching per-PC metric values (e.g., `inst_executed`, stall counters).

**When to use**

* After collecting an `.ncu-rep` with `--import-source yes`, to map hot PCs back to Triton/CUDA lines.
* When diagnosing stalls: combine with `--extra-metric smsp__pcsamp_sample_count` (or other source-correlated metrics) to see which lines are busiest.

**How to use**

Snippet for a source span:

```bash
python tools/ptx_source_summary.py \
  --report runs/gemm_split_k_kernel/ncu_import_source_test3/ncu_report.ncu-rep \
  --kernel gemm_split_k_kernel \
  --source-file gemm_split_k_kernel.py \
  --start-line 90 --end-line 130 \
  --include-sass \
  --include-metric \
  --extra-metric smsp__pcsamp_sample_count \
  > runs/gemm_split_k_kernel/ncu_import_source_test3/source_span.json
```

Full mapping filtered to a file:

```bash
python tools/ptx_source_summary.py \
  --report runs/gemm_split_k_kernel/ncu_import_source_test3/ncu_report.ncu-rep \
  --kernel gemm_split_k_kernel \
  --source-file gemm_split_k_kernel.py \
  --mode mapping \
  > runs/gemm_split_k_kernel/ncu_import_source_test3/source_mapping.json
```

> Tip: pass `--nvtx <range>` and `--kernel <name>` when profiling multiple actions in one report so the correlator picks the right kernel without fallback.

---

### 2.3 Internal helpers (read‑only)

These are used by the main tools; Codex should generally not call them directly.

#### `tools/contracts_common.py`

* Shared helpers for:

  * loading contract JSON,
  * building PyTorch tensors according to `kernel.io.args`,
  * generating launch configs, etc.
* Treat as a library; read it to understand how inputs are constructed, but don’t change it unless explicitly asked.

#### `tools/run_contract_once.py`

* Internal helper used by `profile_contract.py` when `--with-ncu` is enabled.
* Creates inputs, runs `entry_point` **once** (after warmup), and prints a JSON result.
* Nsight wraps this script; you normally shouldn’t invoke it yourself.

---
