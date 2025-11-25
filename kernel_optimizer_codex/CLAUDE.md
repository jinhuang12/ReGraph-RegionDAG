# GPU Kernel Optimization Workflow for Claude

This repository enables Claude to act as a GPU performance engineer for Triton/CUDA kernels, backed by an IronFist-style contract dataset.

---

## 1. Core Principles

**Authority**: Claude has full shell access and should execute commands confidently when following established workflows.

**Discovery First**: Before using any tool for the first time in a session, Claude MUST either:
- Run `python tools/<script>.py --help`, OR
- Read the file header: `sed -n '1,120p tools/<script>.py`

**Never guess** argument names or command syntax.

---

## 2. Repository Layout

```
├── contracts/                    # IronFist JSON contracts (READ-ONLY dataset)
│   ├── ironfist/                 # Main IronFist dataset (~49 contracts)
│   │   ├── gemm_split_k_kernel_large_contract.json
│   │   ├── layer_norm_kernel_transformer_contract.json
│   │   └── ...
│   └── triton_bench/             # External Triton benchmarks
│       ├── token_attn_llama2_contract.json
│       └── ...
│
├── kernels/                      # Generated kernel modules (OPTIMIZATION SURFACE)
│   ├── ironfist/                 # Mirrors contracts subdirs
│   │   ├── gemm_split_k_kernel_a4a9473a.py   # Note: hash-suffixed filenames
│   │   ├── layer_norm_kernel_78a8ed7f.py
│   │   └── ...
│   └── triton_bench/
│       └── _fwd_kernel_token_att1_43338dca.py
│
├── suites/                       # Generated suite definitions (mostly READ-ONLY)
│   ├── ironfist/
│   │   ├── gemm_split_k_kernel_suite.json    # Groups all contracts for a kernel
│   │   └── ...
│   └── triton_bench/
│       └── _fwd_kernel_token_att1_suite.json
│
├── tools/                        # Workflow automation scripts
│   ├── init_kernels_and_suites.py    # Setup: contracts → kernels + suites
│   ├── profile_suite.py               # Main: benchmark entire suite
│   ├── profile_contract.py            # Micro: benchmark single contract
│   ├── append_trace.py                # Logging: record attempts to trace.jsonl
│   ├── summarize_ncu.py               # Analysis: Nsight Compute reports
│   ├── region_dag_summary.py          # Analysis: PTX Region-DAG
│   ├── ptx_source_summary.py          # Analysis: PTX/SASS source correlation
│   └── [internal helpers...]
│
└── runs/                         # Generated artifacts (per-kernel, per-tag)
    └── <kernel_name>/
        ├── trace.jsonl                # Structured attempt log (JSONL format)
        ├── baseline/
        │   ├── baseline_module.py     # Snapshot of baseline kernel
        │   ├── summary.json           # Suite-level metrics
        │   └── <contract_name>/
        │       └── result.json        # Per-contract timing
        └── <tag>/
            ├── summary.json
            └── <contract_name>/
                ├── result.json
                └── ncu_report.ncu-rep # Optional NCU report
```

### Contract Structure

Each contract JSON contains:
- `kernel.kernel_type`: Usually `"multi_kernel"`
- `kernel.metadata.kernel_name`: e.g., `"gemm_split_k_kernel"`
- `kernel.metadata.entry_point`: e.g., `"run"`
- `kernel.io.args`: Meta-parameters and tensor specifications
- `kernel.source_code`: Python+Triton or CUDA wrapper code

### Key Organizational Notes

- **Subdirectory structure**: Contracts, kernels, and suites are organized by source (e.g., `ironfist/`, `triton_bench/`)
- **Hash-suffixed kernels**: Kernel filenames include an 8-character hash (e.g., `_a4a9473a`) for versioning
- **Suite grouping**: Each suite groups all contracts for a single kernel family

---

## 3. Tools Overview

All commands run from the **repo root**. Before first use, always check `--help`.

### 3.1 Primary Benchmarking Tools

#### `tools/init_kernels_and_suites.py`

**Purpose**: Bootstrap workspace from contracts.

**Usage**:
```bash
python tools/init_kernels_and_suites.py \
  --contracts-subdir ironfist \
  --kernels-dir kernels \
  --suites-dir suites
```

**Key arguments**:
- `--contracts-subdir` (REQUIRED): Subdirectory under `contracts/` (e.g., `ironfist`, `triton_bench`)
- `--kernels-dir`: Output directory for kernel modules (default: `kernels`)
- `--suites-dir`: Output directory for suite JSONs (default: `suites`)
- `--overwrite-modules`: Force overwrite existing kernel files

**When to run**:
- Once per repo setup (if `kernels/` or `suites/` is empty)
- After adding new contracts or updating contract source code

**Critical rules**:
- NEVER edit `contracts/*.json` (immutable dataset)
- ALWAYS edit `kernels/<subdir>/<kernel>.py` (optimization surface)
- READ `suites/*.json` but don't modify unless explicitly instructed

---

#### `tools/profile_suite.py` — Primary Benchmark Tool

**Purpose**: Benchmark all contracts in a suite and compute aggregate metrics.

**Outputs**:
- Per-contract: `runs/<kernel>/<tag>/<contract>/result.json`
- Aggregate: `runs/<kernel>/<tag>/summary.json`
  - Contains `aggregate.geomean_ms` (weighted geometric mean)
  - Contains `aggregate.geomean_speedup_vs_baseline` (when baseline exists)

**Baseline run** (required first):
```bash
python tools/profile_suite.py \
  --suite suites/ironfist/gemm_split_k_kernel_suite.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --device cuda:0 \
  --tag baseline \
  --warmup 10 --iters 100 --repeat 5
```

**Candidate run** (after modifications):
```bash
python tools/profile_suite.py \
  --suite suites/ironfist/gemm_split_k_kernel_suite.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --device cuda:0 \
  --tag meta_001 \
  --warmup 10 --iters 100 --repeat 5 \
  --timing-mode cuda_graphs
```

**Key arguments**:
- `--suite` (REQUIRED): Path to suite JSON
- `--module`: Path to kernel module (defaults based on suite)
- `--baseline-module`: Path to baseline kernel for correctness comparison
- `--tag` (REQUIRED): Tag name for this run (e.g., `baseline`, `meta_001`, `struct_001`)
- `--device`: CUDA device (default: `cuda:0`)
- `--warmup`: Warmup iterations (default: 10)
- `--iters`: Timed iterations (default: 100)
- `--repeat`: Repeat trials (default: 5)
- `--timing-mode`: `events` (default) or `cuda_graphs`
- `--with-ncu`: Enable Nsight Compute profiling
- `--ncu-bin`: Path to ncu binary (default: `ncu`)

**Baseline snapshot mechanism**:
- First baseline run auto-saves module to `runs/<kernel>/baseline/baseline_module.py`
- Subsequent tags auto-use this snapshot if `--baseline-module` not specified
- Ensures fair comparison even if kernel.py modified after baseline

**Timing mode fairness**:
- For fair comparison, candidate timing mode is forced to match baseline
- If baseline used `events` but candidate requests `cuda_graphs`, tool forces `events`

---

#### `tools/profile_contract.py` — Single-Contract Microbenchmark

**Purpose**: Deep-dive into a single shape/configuration.

**Plain timing**:
```bash
python tools/profile_contract.py \
  --contract contracts/ironfist/gemm_split_k_kernel_large_contract.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --entry-point run \
  --device cuda:0 \
  --warmup 10 --iters 100 --repeat 5 \
  --out runs/gemm_split_k_kernel/large_baseline
```

**With Nsight Compute**:
```bash
python tools/profile_contract.py \
  --contract contracts/ironfist/gemm_split_k_kernel_large_contract.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --entry-point run \
  --device cuda:0 \
  --warmup 10 --iters 100 --repeat 5 \
  --out runs/gemm_split_k_kernel/ncu_large \
  --with-ncu --ncu-bin ncu
```

**Key arguments**:
- `--contract` (REQUIRED): Path to contract JSON
- `--module` (REQUIRED): Path to kernel module
- `--baseline-module`: Path to baseline kernel for correctness comparison
- `--entry-point`: Entry point function name (default: from contract or `run`)
- `--out` (REQUIRED): Output directory for results
- `--timing-mode`: `events` or `cuda_graphs` (default: `cuda_graphs`)
- `--rtol`: Relative tolerance for correctness (default: 1e-4)
- `--atol`: Absolute tolerance for correctness (default: 1e-3)
- `--with-ncu`: Enable Nsight Compute profiling
- `--ncu-bin`: Path to ncu binary (default: `ncu`)

**When to use**:
- Debugging single problematic shapes
- Focused Nsight analysis on representative contracts
- Validating correctness before full suite run

---

#### `tools/append_trace.py` — Structured Logging

**Purpose**: Maintain audit trail of optimization attempts.

**Output**: Appends to `runs/<kernel_name>/trace.jsonl` (one JSON object per line).

**Usage**:
```bash
python tools/append_trace.py \
  --kernel-name gemm_split_k_kernel \
  --summary runs/gemm_split_k_kernel/meta_001/summary.json \
  --baseline runs/gemm_split_k_kernel/baseline/summary.json \
  --tag meta_001 \
  --edit-kind meta_params \
  --description "Increase BLOCK_SIZE from 64 to 128"
```

**Key arguments**:
- `--kernel-name` (REQUIRED): Logical kernel name
- `--summary` (REQUIRED): Path to candidate summary.json
- `--baseline`: Path to baseline summary.json (optional)
- `--tag` (REQUIRED): Candidate tag
- `--edit-kind` (REQUIRED): Type of edit — choices: `meta_params`, `structural`, `mixed`
- `--description` (REQUIRED): Short human-readable description
- `--contract`: Optional contract name or path
- `--ncu-summary`: Optional path to ncu_summary.json
- `--files-modified`: Optional list of modified files

**When to use**:
- After EVERY meaningful optimization attempt
- Recommended even for failed experiments (documents what didn't work)
- Required for any candidate you decide to keep

---

### 3.2 Analysis Tools (Advanced)

Use these when basic timing isn't sufficient—typically for structural optimizations.

#### `tools/summarize_ncu.py`

**Purpose**: Extract insights from Nsight Compute `.ncu-rep` files.

**Summary mode** (start here):
```bash
python tools/summarize_ncu.py \
  --report runs/gemm_split_k_kernel/ncu_large/ncu_report.ncu-rep \
  --mode summary \
  > runs/gemm_split_k_kernel/ncu_large/ncu_summary.json
```

**Get specific metrics**:
```bash
python tools/summarize_ncu.py \
  --report runs/gemm_split_k_kernel/ncu_large/ncu_report.ncu-rep \
  --mode get_values \
  --names sm__throughput_pct dram__throughput_pct \
  --name-kind canonical
```

**Search available metrics**:
```bash
python tools/summarize_ncu.py \
  --report runs/gemm_split_k_kernel/ncu_large/ncu_report.ncu-rep \
  --mode search_names \
  --query "sm__"
```

**Key fields in summary output**:
- `sm__throughput_pct` (compute utilization)
- `dram__throughput_pct` (memory bandwidth utilization)
- Occupancy metrics, warp limits
- Dominant stall reasons

---

#### `tools/region_dag_summary.py`

**Purpose**: Structural PTX analysis via Region-DAG representation.

**Overview mode**:
```bash
python tools/region_dag_summary.py \
  --ptx runs/gemm_split_k_kernel/some_candidate/kernel.ptx \
  --mode overview \
  > runs/gemm_split_k_kernel/some_candidate/region_dag_overview.json
```

**When to use**:
- When Nsight shows stalls but you need to pinpoint WHERE
- Before major structural refactoring (understand current control flow)
- To map stalls onto PTX regions (mostly CUDA kernels)
- Kernels with non-trivial loops/pipelines (async copy, ldmatrix, blocked reductions)
- Long scoreboard, barrier, or divergence stalls that need localization
- After structural changes, to confirm barriers/regions moved as intended

**Note**: For Triton kernels, PTX exists in the triton cache; point to the correct kernel PTX.

---

#### `tools/ptx_source_summary.py`

**Purpose**: Correlate PTX/SASS back to source lines using NCU import-source data.

**Snippet for source span**:
```bash
python tools/ptx_source_summary.py \
  --report runs/gemm_split_k_kernel/ncu_large/ncu_report.ncu-rep \
  --kernel gemm_split_k_kernel \
  --source-file gemm_split_k_kernel.py \
  --start-line 90 --end-line 130 \
  --include-sass \
  --include-metric \
  --extra-metric smsp__pcsamp_sample_count \
  > source_span.json
```

**Full mapping**:
```bash
python tools/ptx_source_summary.py \
  --report runs/gemm_split_k_kernel/ncu_large/ncu_report.ncu-rep \
  --kernel gemm_split_k_kernel \
  --source-file gemm_split_k_kernel.py \
  --mode mapping \
  > source_mapping.json
```

**When to use**:
- After collecting `.ncu-rep` with `--import-source yes`
- To map hot program counters back to Triton/CUDA source lines
- When Nsight shows stalls and you need exact source lines with those PCs/metrics
- After changing async copy/ldmatrix pipelines to confirm hot PCs shifted
- Fused kernels (matmul + epilogue) to see which phase owns hot PCs
- Reductions/scans to identify which phase holds hot PCs

**Tip**: Pass `--nvtx <range>` and `--kernel <name>` for multi-kernel reports.

---

### 3.3 Internal Helpers (Read-Only)

These are used by the main tools. Understand their purpose but don't invoke directly.

- `tools/contracts_common.py`: Contract loading, tensor construction, entry point resolution
- `tools/run_contract_once.py`: NCU target script (used by `--with-ncu`)
- `kernel_opt_tooling.py`, `ptx_dag_tool_v2.py`, `ptx_source_correlator.py`: Analysis libraries

---

## 4. Guardrails

1. **Don't edit `contracts/`** — Treat as immutable dataset
2. **Edit kernels under `kernels/<subdir>/`** — This is the optimization surface
3. **Keep comparisons fair** — Same `--warmup`, `--iters`, `--repeat` when comparing tags
4. **Avoid dangerous shell commands** — No `rm -rf /`, `chmod` on system paths
5. **Respect user intent** — If user wants custom harness, that's fine

---

## 5. Standard Optimization Workflow

### Phase 1: Initial Baseline

```bash
# 1. Initialize workspace (if kernels/ or suites/ empty)
python tools/init_kernels_and_suites.py \
  --contracts-subdir ironfist \
  --kernels-dir kernels \
  --suites-dir suites

# 2. Run baseline suite
python tools/profile_suite.py \
  --suite suites/ironfist/gemm_split_k_kernel_suite.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --device cuda:0 \
  --tag baseline \
  --warmup 10 --iters 100 --repeat 5

# 3. Log baseline
python tools/append_trace.py \
  --kernel-name gemm_split_k_kernel \
  --tag baseline \
  --edit-kind meta_params \
  --description "Initial unmodified kernel" \
  --summary runs/gemm_split_k_kernel/baseline/summary.json
```

### Phase 2: Meta-Parameter Tuning

```bash
# 1. Edit kernels/ironfist/<kernel>.py (change BLOCK_SIZE, num_warps, etc.)

# 2. Benchmark candidate
python tools/profile_suite.py \
  --suite suites/ironfist/gemm_split_k_kernel_suite.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --device cuda:0 \
  --tag meta_001 \
  --warmup 10 --iters 100 --repeat 5

# 3. Log attempt
python tools/append_trace.py \
  --kernel-name gemm_split_k_kernel \
  --tag meta_001 \
  --edit-kind meta_params \
  --description "BLOCK_SIZE: 64 → 128" \
  --baseline runs/gemm_split_k_kernel/baseline/summary.json \
  --summary runs/gemm_split_k_kernel/meta_001/summary.json
```

### Phase 3: Structural Optimization

```bash
# 1. Deep analysis on representative contract
python tools/profile_contract.py \
  --contract contracts/ironfist/gemm_split_k_kernel_large_contract.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --entry-point run \
  --device cuda:0 \
  --warmup 10 --iters 100 --repeat 5 \
  --out runs/gemm_split_k_kernel/ncu_large \
  --with-ncu --ncu-bin ncu

# 2. Summarize NCU findings
python tools/summarize_ncu.py \
  --report runs/gemm_split_k_kernel/ncu_large/ncu_report.ncu-rep \
  --mode summary \
  > runs/gemm_split_k_kernel/ncu_large/ncu_summary.json

# 3. Make structural changes to kernels/ironfist/<kernel>.py

# 4. Benchmark full suite
python tools/profile_suite.py \
  --suite suites/ironfist/gemm_split_k_kernel_suite.json \
  --module kernels/ironfist/gemm_split_k_kernel_a4a9473a.py \
  --device cuda:0 \
  --tag struct_001 \
  --warmup 10 --iters 100 --repeat 5

# 5. Log structural attempt
python tools/append_trace.py \
  --kernel-name gemm_split_k_kernel \
  --tag struct_001 \
  --edit-kind structural \
  --description "Tile outer loop + prefetch shared memory" \
  --baseline runs/gemm_split_k_kernel/baseline/summary.json \
  --summary runs/gemm_split_k_kernel/struct_001/summary.json \
  --ncu-summary runs/gemm_split_k_kernel/ncu_large/ncu_summary.json
```

### Analysis Decision Framework

After each run, explicitly determine:

1. **Kernel bottleneck type**:
   - Compute-bound?
   - Memory-bound?
   - Stall-limited (long scoreboard, barriers)?
   - Occupancy-limited?

2. **Which contracts bottleneck suite geomean?**

3. **Would an analysis tool help?**
   - Nsight summary: separate compute vs memory vs stalls
   - Region DAG: map stalls onto PTX regions (CUDA)
   - PTX source correlator: map hot PCs to source lines

**Don't call every tool every time.** Use them when they clarify your next move.

### Planning the Next Change

- **Meta tuning** problem: adjust tile sizes, `num_warps`, `num_stages`, split-K
- **Structural issue**: rework tiling/blocking, async pipelines, epilogues, atomics, memory patterns

Pick the **single most promising change** and test it before moving to another idea.

### Loop and Stopping Criteria

Repeat: `analyze → plan → implement → validate → profile → analyze → …`

Stop when:
- Improvements in suite geomean are < ~3-5% across several tries
- Nsight indicates reasonable saturation of primary bottleneck

---

## 6. Key Decision Points

### When to Use Each Tool

| Tool | Use Case |
|------|----------|
| `init_kernels_and_suites.py` | Initial setup, new contracts added |
| `profile_suite.py` | Primary performance measurement, every candidate |
| `profile_contract.py` | Single-shape debugging, focused NCU analysis |
| `append_trace.py` | Document every meaningful attempt |
| `summarize_ncu.py` | Understand hardware utilization, bottleneck diagnosis |
| `region_dag_summary.py` | PTX-level structural understanding (CUDA) |
| `ptx_source_summary.py` | Map NCU hotspots to source lines |

### Optimization Type Classification

- **Meta-parameter** (`--edit-kind meta_params`): Changes to compile-time constants (BLOCK_SIZE, NUM_WARPS, num_stages, split_k)
  - Fast iteration, use `--tag meta_NNN`

- **Structural** (`--edit-kind structural`): Changes to algorithm, tiling strategy, memory access patterns
  - Requires deep analysis, use `--tag struct_NNN`
  - Should be accompanied by NCU profiling

- **Mixed** (`--edit-kind mixed`): Combination of both

### Performance Comparison

Always compare via `aggregate.geomean_speedup_vs_baseline` in `summary.json`. Individual contract results can be misleading; focus on geometric mean across the suite.

---

## 7. Critical Rules

1. **Never modify `contracts/*.json`** — immutable dataset
2. **Always run baseline first** — speedup metrics require baseline reference
3. **Log all meaningful attempts** — use `append_trace.py` religiously
4. **Check tool help before first use** — never guess command syntax
5. **Focus on suite metrics** — individual contracts can mislead; trust geomean
6. **Use NCU for structural work** — don't make complex changes blind
7. **Work from repo root** — all paths assume repo root as working directory
8. **Use correct subdirectory paths** — `suites/ironfist/...`, `kernels/ironfist/...`

---

## 8. Reporting Results

When explaining optimization work to the user, include:

- **Kernel and suite worked on**: suite path, kernel module path
- **Baseline vs best candidate**: baseline geomean, candidate geomean, speedup factor
- **Key Nsight signals** (if used): SM throughput, occupancy, stall reasons, memory throughput
- **Brief description of code changes**: e.g., "shape-aware split_k clamp + tiling tweaks for large K"

---

## 9. Success Criteria

An optimization is successful when:
1. `aggregate.geomean_speedup_vs_baseline > 1.0` in suite summary
2. No correctness failures (output fingerprints match)
3. Improvement is consistent across most contracts (check per-contract results)
4. Changes are logged in trace.jsonl with clear description

If speedup < 1.0: revert or iterate. If speedup is marginal (<5%): document and consider whether complexity is justified.
