# Kernel Optimizer (Triton/CUDA + IronFist Contracts)

  GPU kernel optimization playground with IronFist-style contracts, Triton/CUDA kernel implementations, and profiling tools.
  Kernels live in `kernels/`, contracts/suites in `contracts/` and `suites/`, and profiling outputs in `runs/`.

  ## Contents
  - `contracts/`: IronFist JSON contracts (`kernel.metadata`, `kernel.io.args`, `kernel.source_code`).
  - `kernels/`: Python modules (Triton/CUDA wrappers). Performance work happens here.
  - `suites/`: Per-kernel suites grouping contracts; drive profiling.
  - `tools/`: Profiling + analysis harness (`profile_contract.py`, `profile_suite.py`, NCU summaries, Region-DAG, etc.).
  - `runs/`: Profiling outputs (`result.json`, `summary.json`, optional `.ncu-rep`/summaries, traces).

  ## Prereqs
  - Python 3.10+
  - CUDA-capable GPU with recent driver
  - PyTorch, Triton, CuPy (install via editable extras)
  ```bash
  pip install -e .[full]

  ## Key Concepts

  - Contracts describe args/meta + reference source. The harness builds inputs from contracts and calls a module entrypoint.
  - Suites collect multiple contracts for a kernel; aggregate geomean metrics.
  - benchmark_kernel fast-path (optional): If a kernel module defines benchmark_kernel(contract_args, warmup, iters, repeat),
    profile_contract.py will use it to time kernel-only loops (CUDA events) instead of wrapping the run entrypoint. contract_args
    include parsed scalars and materialized tensors on the requested device (_device).

  ## Typical Workflow

  1. Init kernels/suites from contracts (once or when contracts change)

  python tools/init_kernels_and_suites.py \
    --contracts-dir contracts --kernels-dir kernels --suites-dir suites

  2. Baseline a kernel family (suite-level)

  python tools/profile_suite.py \
    --suite suites/gemm_split_k_kernel_suite.json \
    --module kernels/gemm_split_k_kernel.py \
    --device cuda:0 \
    --tag baseline \
    --warmup 10 --iters 100 --repeat 5

  3. Profile a single contract (with kernel-only fast-path if present)

  python tools/profile_contract.py \
    --contract contracts/gemm_split_k_kernel_large_contract.json \
    --module kernels/gemm_split_k_kernel.py \
    --entry-point run \
    --device cuda:0 \
    --warmup 10 --iters 100 --repeat 5 \
    --out runs/gemm_split_k_kernel/large_test

  4. Optional Nsight Compute on a contract

  python tools/profile_contract.py ... --with-ncu --ncu-bin ncu
  python tools/summarize_ncu.py --report runs/.../ncu_report.ncu-rep --mode summary \
    > runs/.../ncu_summary.json

  5. Log attempts (optional)

  python tools/append_trace.py --kernel-name <name> --contract-name <cname> \
    --tag <tag> --edit-kind meta_params|structural --description "...descr..." \
    --baseline-summary runs/<name>/baseline/summary.json \
    --candidate-summary runs/<name>/<tag>/summary.json

  ## benchmark_kernel Convention

  Implement in kernels/<name>.py to get kernel-only timings:

  def benchmark_kernel(contract_args: dict, warmup: int, iters: int, repeat: int) -> dict:
      # allocate once (contract_args include tensors when type == "tensor")
      # warmup launches
      # timed CUDA events around kernel launches (iters × repeat)
      return {
          "ok": True,
          "mean_ms": ...,
          "std_ms": ...,
          "output_summary": {...},  # small fingerprint
          "output": tensor_optional,
      }

  profile_contract.py will:

  - Prefer benchmark_kernel when present; else time the run entrypoint.
  - Record benchmark_kernel_used/available and timing_mode in result.json.

  ## Outputs

  - Per-contract: runs/<kernel>/<tag>/<contract>/result.json
  - Per-suite:    runs/<kernel>/<tag>/summary.json (geomean, optional speedup vs baseline)
  - Optional Nsight: .ncu-rep, ncu_stdout/stderr.log, ncu_summary.json (if summarized)
  - Optional trace: runs/<kernel>/trace.jsonl (one JSON per attempt)

  ## Notes & Best Practices

  - Keep comparisons apples-to-apples: same warmup/iters/repeat when comparing tags.
  - Use suites (not single contracts) to judge overall wins.
  - Preserve contract compatibility; keep entrypoint signatures aligned with kernel.io.args.
  - Nsight: use representative contracts (often “large”) when diagnosing bottlenecks.
  - Region-DAG (tools/region_dag_summary.py) is available for PTX-based CUDA analysis.

  ## Quick Examples

  Baseline suite:

  python tools/profile_suite.py \
    --suite suites/vec_matmul_kernel_suite.json \
    --module kernels/vec_matmul_kernel.py \
    --device cuda:0 \
    --tag baseline \
    --warmup 10 --iters 100 --repeat 5

  Single-contract kernel-only timing (fast-path):

  python tools/profile_contract.py \
    --contract contracts/layer_norm_kernel_small_contract.json \
    --module kernels/layer_norm_kernel.py \
    --entry-point run \
    --device cuda:0 \
    --iters 5 --repeat 3 --warmup 2 \
    --out /tmp/layernorm

  Happy tuning!