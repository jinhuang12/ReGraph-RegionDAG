# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research tool for automated GPU kernel optimization using AI-guided Monte Carlo Graph Search (MCGS). It optimizes Triton and CUDA kernels through iterative refinement guided by LLM-based candidate generation and performance measurements.

## Core Commands

### Run optimization on a kernel
```bash
python optimize_kernels.py \
  --input kernels/lora_expand.json \
  --outdir out \
  --gpus 0,1,2 \
  --budget-builds 50 \
  --llm-model gpt-5-mini \
  --ncu-every 5
```

### Resume interrupted optimization
```bash
python optimize_kernels.py --input kernels/lora_expand.json --outdir out --resume
```

### Run without LLM (rule-based only)
```bash
python optimize_kernels.py --input kernels/ --no-llm --budget-builds 20
```

### Worker subprocess (internal use)
```bash
python optimize_kernels.py --worker /path/to/variant/control.json
```

### Key Arguments
- `--input`: JSON spec file or directory with kernel specs
- `--outdir`: Output directory for results (default: `out`)
- `--gpus`: Comma-separated GPU indices for parallel execution
- `--budget-builds`: Maximum optimization attempts per kernel
- `--min-pred-gain`: Prune threshold for weak moves (default: 0.02)
- `--llm-model`: OpenAI model (default: `gpt-5-mini`)
- `--no-llm`: Disable LLM, use rule-based moves only
- `--ncu-every`: Collect NCU metrics every N visits (0=disable)
- `--resume`: Resume from existing search_state.json

## Architecture

### High-Level Design: Monte Carlo Graph Search (MCGS)

The system implements a graph-based (not tree) search where kernel optimization states are nodes and transformation moves are edges. States can be reached via multiple paths (transpositions are detected).

**Core workflow:**
1. **Selection**: Walk from root using PUCT (Upper Confidence Bound) with progressive widening
2. **Expansion**: Generate optimization candidate at leaf node (LLM or rule-based)
3. **Evaluation**: Launch worker subprocess on available GPU to measure performance
4. **Backpropagation**: Update Q-values and visit counts along path
5. **Incumbent tracking**: Maintain best variant found so far

### Component Interaction

**Orchestrator (`optimize_kernels.py`):**
- Main entry point implementing MCGS algorithm
- Manages search graph state across iterations
- Coordinates parallel GPU workers via subprocess spawning
- Applies Dirichlet noise to root node for exploration
- Persists state to `search_state.json` and `events.jsonl`

**Region-DAG Prior Model:**
- Infers kernel bottlenecks from NCU (NVIDIA Compute Profiler) metrics
- Computes prior probabilities for optimization moves
- Categories: `compute`, `gmem`, `sfu`, `atomics`, `smem`, `overlap`
- Used to guide MCGS exploration toward promising optimizations

**LLM-Based Candidate Generation:**
- Uses OpenAI API (with Responses API support) to propose optimizations
- Constrains generation to canonical move vocabulary (see below)
- Produces unified diff patches for robust source transformation
- Fallback to rule-based candidates if LLM unavailable

**Worker Subprocess System:**
- **triton_runner.py**: Executes Triton kernels via JIT compilation
- **cuda_runner.py**: Executes CUDA kernels via CuPy, manages compile cache at `/tmp/kernel_compile_cache`
- **state_manager.py**: Incremental state persistence for crash recovery
- Workers compute state hash from materialized source + launch config
- Results written to `runner_result.json` with timing and validation

**IOContract System (`shared/io_contract/`):**
- Unified tensor specification format for kernel inputs/outputs/scalars
- Supports initialization modes: `randn`, `uniform`, `zeros`, `ones`, `full`, `arange`
- Handles tensor encoding/decoding with optional zlib compression
- Enables reproducible kernel invocation across Triton and CUDA backends

### Canonical Optimization Moves

The system defines 9 canonical optimization tactics that the LLM is constrained to:

1. **enable_async_pipeline**: Overlap gmem with compute via cp.async/TMA
2. **vectorize_global_loads**: Use ld/st.v2/v4/v8 when aligned
3. **switch_to_mma_or_wgmma**: Promote dot/FMA to MMA/WGMMA instructions
4. **pad_tail_and_mask**: Pad tails for contiguous vectorization
5. **change_block_sizes**: Retune BLOCK sizes or launch params for occupancy
6. **cache_policy_cg**: Use .cg cache policy to reduce L1 pollution
7. **avoid_atomics_reduce**: Restructure to avoid atomic operations
8. **grouped_gemm_colmajor**: Grouped scheduling for better cache reuse
9. **softmax_math_fusion**: Fuse/approximate math to reduce SFU usage

### Patch-Based Modifications

Optimizations are applied via the `patch` command using unified diffs. This provides:
- Robust context-aware transformations (not brittle line-based edits)
- Human-readable diffs stored in `candidate.diff`
- Ability to compose multiple transformations (though currently single-move)
- Fallback to full source override if patch fails

### State Persistence

**Search state** (`search_state.json`):
- Complete MCGS graph: nodes, edges, Q-values, visit counts
- Incumbent best variant reference
- Resumable across process restarts

**Event log** (`events.jsonl`):
- Chronological append-only log of all search events
- Enables post-hoc analysis and debugging

**Trace** (`trace.md`):
- Human-readable optimization narrative
- Shows selection path, candidates generated, results

## Directory Structure

```
/home/ec2-user/optimize-kernels/
├── optimize_kernels.py           # Main orchestrator (939 lines)
├── kernels/                      # Input kernel specifications
│   └── *.json                   # Kernel spec with source/IOContract
├── workers/                      # Worker subprocess modules
│   ├── state_manager.py         # Crash recovery state management
│   ├── triton_runner.py         # Triton kernel execution
│   └── cuda_runner.py           # CUDA kernel execution with compile cache
├── shared/                       # Shared models and utilities
│   ├── model.py                 # Pydantic data models (KernelSpec, SearchNode, etc.)
│   └── io_contract/             # Tensor I/O contract system
│       ├── manager.py           # IOContractManager
│       ├── spec_builders.py     # Helper builders
│       └── tensor_utils.py      # Tensor encode/decode
└── out/                         # Optimization outputs
    ├── global_tactic_graph.json # Cross-kernel tactic graph
    └── kernels/_<kernel_name>/  # Per-kernel results
        ├── baseline/            # Baseline measurements
        ├── variants/<uuid>/     # Generated optimizations
        ├── best/               # Symlink to best variant
        ├── trace.md            # Optimization trace
        ├── events.jsonl        # Event log
        └── search_state.json   # MCGS state (resumable)
```

## Output Artifacts

For each kernel optimization run, the system generates:

**Baseline directory:**
- `run.json`, `result.json` - Configuration and timing results
- `kernel_module.py` or `kernel.cu` - Materialized source
- `kernel.ptx`, `kernel.sass` - Compiled artifacts (CUDA only)
- `stdout.log`, `stderr.log` - Compilation/execution logs

**Variant directories (`variants/<uuid>/`):**
- `control.json` - Worker control specification
- `result.json` - Performance measurements and validation
- `kernel_module.py` - Modified source code
- `candidate.diff` - Unified diff patch applied
- Compilation and execution logs

**Top-level per-kernel files:**
- `search_state.json` - Complete MCGS graph state (resumable)
- `events.jsonl` - Chronological event log
- `trace.md` - Human-readable optimization narrative
- `best/` - Symlink to best performing variant

## Dependencies

**Required Python packages:**
- `torch` - Tensor operations, CUDA events
- `triton` - Triton kernel compilation/execution
- `cupy` - CUDA kernel loading via RawModule
- `pydantic` - Data modeling and validation
- `openai` - LLM API integration
- `numpy` - Array operations

**System tools:**
- `nvcc` - CUDA compilation
- `patch` - Apply unified diffs
- `nvdisasm` or `cuobjdump` - Disassembly (optional)

**Environment variables:**
- `OPENAI_BASE_URL` - Custom OpenAI API endpoint (optional)
- `COMPILE_CACHE_DIR` - Override compile cache location (default: `/tmp/kernel_compile_cache`)

## Worker Execution Flow

### Triton Kernels (triton_runner.py)
1. Load kernel module with `sys.modules` registration
2. Find `@triton.jit` decorated kernels
3. Generate inputs from IOContract via IOContractManager
4. Apply `launch_update` overrides (num_warps, grid dimensions)
5. Time with CUDA events (warmup + iterations)
6. Write `runner_result.json` with timing and hash

### CUDA Kernels (cuda_runner.py)
1. Check compile cache by content hash
2. Compile to PTX/cubin via `nvcc` if cache miss
3. Load cubin via CuPy `RawModule`
4. Generate inputs from IOContract (CuPy arrays on GPU)
5. Apply `launch_update` to grid/block dimensions
6. Time with CUDA events
7. Validate outputs (NaN/Inf check)
8. Write `runner_result.json`

**State hash:** Computed from materialized source + launch config to detect duplicate states in graph.

## Key Implementation Notes

- **No traditional build system**: Pure Python project with runtime compilation
- **No test suite**: Research/experimental codebase
- **Compile cache**: CUDA kernels cached at `/tmp/kernel_compile_cache` by content hash
- **Graph vs Tree**: MCGS graph detects transpositions (same state via different paths)
- **Progressive widening**: Limits child expansion based on visit count
- **PUCT tuning**: Hardcoded constants (PUCT_C, ROOT_NOISE_EPS) in optimize_kernels.py
- **Parallel execution**: One worker subprocess per GPU, managed by orchestrator
- **Stateful resumption**: `--resume` continues from search_state.json without re-running baseline

## Modifying the System

**Adding new optimization moves:**
1. Add move to canonical vocabulary in optimize_kernels.py
2. Update LLM prompt to describe the new move
3. Optionally add rule-based candidate generator

**Supporting new kernel types:**
1. Implement new runner in `workers/` following triton_runner.py pattern
2. Add kernel type detection in worker_main()
3. Ensure IOContract compatibility

**Tuning MCGS parameters:**
- `PUCT_C`: Exploration constant (higher = more exploration)
- `ROOT_NOISE_EPS`: Dirichlet noise weight at root
- `PROG_WIDEN_THRESH`: Progressive widening threshold
- Located in optimize_kernels.py global constants
