# LoRA Fused Kernel Suite Optimization Report (Triton → CUDA)

Repo: `/home/jinhun/kernel_optimizer_codex`  
Suite: `suites/ironfist/lora_fused_kernel_suite.json`  
Kernel module: `kernels/ironfist/lora_fused_kernel_0318783b.py`  
CUDA extension: `kernels/ironfist/cuda/lora_fused_ext.cu` + `kernels/ironfist/cuda/lora_fused_ext.cpp`  
GPU: **NVIDIA L40S (SM89)**  
Date: **2026-01-12**

---

## 0) Problem signature (what this kernel does)

We implement a fused LoRA update per active LoRA id:

- Inputs:
  - `X`: `[M, K]` (tokens × hidden)
  - `A`: `[S, L, R, K]` (LoRA shrink weights; `R=16`)
  - `B`: `[S, L, N, R]` (LoRA expand weights; `N == hidden_size` for suite contracts)
  - `token_indices_sorted`, `num_tokens_per_lora`, `lora_token_start_loc`, `lora_ids` to group tokens by LoRA id.

- Computation (per slice `s` and active LoRA `l`):
  - `U = X @ A^T`, so `U: [M_lora, R]`
  - `Y = U @ B^T`, so `Y: [M_lora, N]`
  - Scatter `Y` back to `output[token, slice_offset + n]` using token indices.

Key runtime fact: tokens are *sorted by LoRA id*, but per-LoRA token counts can be very uneven (especially in the “high_throughput” contract). That creates load imbalance if we amortize shrink too aggressively.

---

## 1) How we measured (tools + fairness)

### Suite-level (canonical)

- Command (baseline):

  ```bash
  python tools/profile_suite.py \
    --suite suites/ironfist/lora_fused_kernel_suite.json \
    --module kernels/ironfist/lora_fused_kernel_0318783b.py \
    --device cuda:0 \
    --tag baseline \
    --warmup 10 --iters 100 --repeat 5 \
    --backend triton
  ```

- Command (candidate vs baseline):

  ```bash
  python tools/profile_suite.py \
    --suite suites/ironfist/lora_fused_kernel_suite.json \
    --module kernels/ironfist/lora_fused_kernel_0318783b.py \
    --baseline-module runs/lora_fused_kernel_0318783b/baseline/baseline_module.py \
    --device cuda:0 \
    --tag <tag> \
    --warmup 10 --iters 100 --repeat 5 \
    --backend cuda
  ```

- Outputs:
  - `runs/lora_fused_kernel_0318783b/<tag>/summary.json` (geomean + per-contract)
  - Per-contract results under the tag directory.

### Single-contract microbench (high-throughput focus)

- Command:

  ```bash
  python tools/profile_contract.py \
    --contract contracts/ironfist/lora_fused_kernel_high_throughput_contract.json \
    --module kernels/ironfist/lora_fused_kernel_0318783b.py \
    --device cuda:0 \
    --warmup 10 --iters 100 --repeat 5 \
    --baseline-module runs/lora_fused_kernel_0318783b/baseline/baseline_module.py \
    --backend cuda \
    --out runs/lora_fused_kernel_0318783b/<out>
  ```

### Nsight Compute (NCU)

- Command pattern:

  ```bash
  python tools/profile_contract.py \
    --contract contracts/ironfist/lora_fused_kernel_high_throughput_contract.json \
    --module kernels/ironfist/lora_fused_kernel_0318783b.py \
    --device cuda:0 \
    --warmup 10 --iters 100 --repeat 5 \
    --baseline-module runs/lora_fused_kernel_0318783b/baseline/baseline_module.py \
    --backend cuda \
    --with-ncu --ncu-bin ncu --ncu-call auto \
    --ncu-args "--kernel-name lora_fused_wmma_kernel" \
    --out runs/lora_fused_kernel_0318783b/<out>
  ```

- Then summarize:

  ```bash
  python tools/summarize_ncu.py \
    --report runs/lora_fused_kernel_0318783b/<out>/ncu_report.ncu-rep \
    --mode summary \
    > runs/lora_fused_kernel_0318783b/<out>/ncu_summary.json
  ```

Correctness:
- Harness checks against baseline module outputs (`rtol=1e-4`, `atol=1e-3`) before timing.

---

## 2) “Start → Final” performance delta (suite + key contract)

### Final baseline (Triton, after correctness fix)

Artifact: `runs/lora_fused_kernel_0318783b/baseline/summary.json`

- Suite geomean: **0.10859065 ms**
- High-throughput contract mean: **1.02278343 ms**

### Final CUDA result (fused WMMA kernel)

Artifact: `runs/lora_fused_kernel_0318783b/cuda_smopt_003/summary.json`

- Suite geomean: **0.07896075 ms**  (**1.375x faster**)
- High-throughput contract mean: **1.01799469 ms** (**~1.005x faster**)

Per-contract speedups (computed from the two summaries above):

| Contract | Triton baseline (ms) | CUDA final (ms) | Speedup |
|---|---:|---:|---:|
| `lora_fused_kernel_contract` | 0.066049 | 0.034911 | 1.892x |
| `lora_fused_kernel_high_throughput_contract` | 1.022783 | 1.017995 | 1.005x |
| `lora_fused_kernel_large_contract` | 0.076362 | 0.088008 | 0.868x (regression) |
| `lora_fused_kernel_multi_lora_contract` | 0.067002 | 0.046816 | 1.431x |
| `lora_fused_kernel_multi_slice_contract` | 0.071112 | 0.057425 | 1.238x |
| `lora_fused_kernel_small_contract` | 0.066711 | 0.028823 | 2.314x |

---

## 3) Important baseline change (why some older suite numbers don’t match)

We fixed an off-by-one in “how many LoRA buckets we launch for”:

- Fix location: `kernels/ironfist/lora_fused_kernel_0318783b.py:1151`
- Change:
  - **Before:** `actual_num_loras = min(len(num_tokens_per_lora_raw), max_loras)`
  - **After:**  `actual_num_loras = min(len(num_tokens_per_lora_raw), max_loras + 1)`  
    because the padded arrays include the optional “-1 (no-LoRA)” bucket.

Effect:
- For contracts with `num_loras=1`, tokens can map to `-1` and `0`. The old code dropped one bucket, under-launching and producing a much smaller (but semantically incorrect) workload.
- After the fix, Triton baseline suite geomean increased (it now includes the “no-LoRA bucket” launch behavior).
- We re-ran baseline: `runs/lora_fused_kernel_0318783b/baseline/summary.json`.

Important: Many older `runs/.../summary.json` were produced *before* this baseline reset, so their embedded `geomean_speedup_vs_baseline` may not correspond to the current baseline file. The “Start → Final” comparison above is computed using the two **final** summaries.

---

## 4) Optimization timeline (what we tried, what changed, what happened)

### Phase A — Initial CUDA backend (very slow)

Goal: get a working fused CUDA path, then optimize.

- Artifact: `runs/lora_fused_kernel_0318783b/cuda_001/summary.json`
- Suite geomean (in that earlier regime): `0.1071658769 ms`
- High-throughput: `2.5543978839 ms`

At this stage, CUDA was ~2.5x slower than Triton for high-throughput.

What it did (high level):
- Shrink and expand fused using WMMA fragments.
- Basic shared-memory staging.
- Lots of sync/barriers and redundant memory traffic.

NCU early signals (example):
- `runs/lora_fused_kernel_0318783b/ncu_cuda_curr_001/ncu_summary.json`
  - `memory_throughput_pct ~49.6%`
  - `arith_intensity_dram_bytes ~943 MB` (very high)
  - Strong hint: **memory-bound** + **too many DRAM bytes** (redundant reads / duplicated work).

### Phase B — Micro-optimizations (vectorization + small sync changes)

Goal: remove obvious instruction + memory inefficiencies.

Representative microbench tags (high-throughput contract):
- `runs/lora_fused_kernel_0318783b/cuda_half2_u_store_001/result.json` → `2.177 ms`
- `runs/lora_fused_kernel_0318783b/cuda_no_syncwarp_001/result.json` → `2.177 ms`
- `runs/lora_fused_kernel_0318783b/cuda_b_shared_001/result.json` → `2.181 ms`
- `runs/lora_fused_kernel_0318783b/cuda_half2loadstore_001/result.json` → `2.242 ms`

What we changed (themes):
- Prefer `half2`-vectorized stores where output is contiguous.
- Tweak sync placement (`__syncwarp()` vs `__syncthreads()`) in the expand loop.
- Small shared-memory layout and copy changes.

Outcome:
- Helped a bit (2.55 ms → ~2.18 ms), but still far from Triton.

### Phase C — “Amortize shrink” via `GROUP_N` experiments

Goal: avoid recomputing shrink when multiple CTAs cover N.

Context:
- Each CTA computes shrink (`U`) once, then expands over some N tiles.
- If we launch multiple CTAs along N for the same `(slice, lora, mblock)`, shrink (X/A reads) gets duplicated.

We tried several `GROUP_N` values:
- `runs/lora_fused_kernel_0318783b/cuda_group16_001/result.json` → `2.538 ms` (worse)
- `runs/lora_fused_kernel_0318783b/cuda_group32_fast_001/result.json` → `2.207 ms` (better than 2.54, still slow)

Conclusion:
- `GROUP_N` helps reduce duplicated shrink, but it wasn’t sufficient until shrink itself was redesigned (next phases).

### Phase D — Add `cp.async` shrink staging for X and A (asyncXA)

Goal: overlap global→shared copies with compute and reduce load stalls.

Representative runs (high-throughput):
- `runs/lora_fused_kernel_0318783b/cuda_asyncXA_default_001/result.json` → `2.046 ms`
- `runs/lora_fused_kernel_0318783b/cuda_asyncXA_gn32_nw4_001/result.json` → `2.046 ms`

Key idea implemented:
- Use `cp.async.cg.shared.global` (Ampere+) to stage X and A tiles into shared memory.

Outcome:
- Reduced time again (~2.18 → ~2.05 ms) but still far from the ~1.02 ms Triton path.

### Phase E — Split-kernels (store U to global) experiment (rejected)

Goal: remove cross-warp communication by writing U out, then expanding in a second kernel.

Representative run:
- `runs/lora_fused_kernel_0318783b/cuda_splitU_001/result.json` → `2.302 ms` (worse)

Why it lost:
- Extra global write+read for `U` plus another kernel launch.
- Even though `U` is “small” (`[M,16]`), at this throughput regime it’s still too expensive vs keeping `U` on-chip.

Status in final code:
- Still available for experiments but **opt-in only** via `KO_LORA_SPLIT_KERNELS=1`.

### Phase F — Big redesign: K-blocking at 64 + double-buffer X/A with `cp.async` (k64)

Goal: drastically cut overhead in shrink’s K loop and improve the pipeline.

Core change:
- Process shrink K dimension in blocks of 64:
  - `kernels/ironfist/cuda/lora_fused_ext.cu:24` → `constexpr int kKBlock = 64;`
- Double-buffer X and A for the next K-block while computing the current one.
- A is staged once per CTA stage (shared), while X is staged per *active* warp tile.

Representative runs (high-throughput):
- `runs/lora_fused_kernel_0318783b/cuda_k64_001/result.json` → `1.077 ms`
- `runs/lora_fused_kernel_0318783b/cuda_k64_awarp_nw2_001/result.json` → `1.067 ms` (best in that sub-phase)

NCU improvement:
- Before: `ncu_cuda_curr_001` → ~49.6% memory throughput, ~943 MB DRAM bytes
- After:  `ncu_cuda_k64_001` → ~68.8% memory throughput, ~699 MB DRAM bytes  
  (`runs/lora_fused_kernel_0318783b/ncu_cuda_k64_001/ncu_summary.json`)

This was the largest single-step win: **~2.05 ms → ~1.07 ms** on high-throughput.

### Phase G — Shared memory footprint reduction (smopt): don’t allocate X buffers for idle warps

Goal: increase occupancy / reduce SMEM pressure; re-enable higher warp variants safely.

Problem:
- With kKBlock=64, shared memory grew enough that `NUM_WARPS=8` couldn’t compile earlier (static SMEM limit issues).
- Even with 2/4 warps, SMEM limited blocks/SM (hurting latency hiding).

Fix:
- Allocate X shared memory proportional to the number of M-tiles, not `NUM_WARPS`.
  - `kernels/ironfist/cuda/lora_fused_ext.cu:114`:
    - `kMTiles = (BLOCK_M + 15) / 16`
    - `x_shared[kMTiles][2][16 * 64]` instead of `x_shared[NUM_WARPS][2][...]`

Effect:
- Reduced per-block SMEM for the high-throughput config.
- NCU shows SMEM drop:
  - `ncu_cuda_smopt_001` launch SMEM: `22656 B/block`
- Memory throughput rose closer to Triton:
  - `runs/lora_fused_kernel_0318783b/ncu_cuda_smopt_001/ncu_summary.json`
    - `memory_throughput_pct ~79.9%`
    - `dram_bytes ~686 MB`

We also re-enabled dispatch for `num_warps=8`:
- `kernels/ironfist/cuda/lora_fused_ext.cu:883` and dispatch tables.

Warp-count sweep (high-throughput, default meta):
- `runs/lora_fused_kernel_0318783b/cuda_smopt_nw2_001/result.json` → `1.068 ms`
- `runs/lora_fused_kernel_0318783b/cuda_smopt_nw4_001/result.json` → `1.065 ms` (best)
- `runs/lora_fused_kernel_0318783b/cuda_smopt_nw8_001/result.json` → `1.074 ms`

Conclusion:
- 8-warps became possible again, but 4-warps remained best for this kernel layout.

### Phase H — High-throughput meta tuning: `BLOCK_M=16` + `GROUP_N≈24`

Goal: beat Triton on the “high throughput” contract (the hardest case) without regressing the suite badly.

Observations:
- In the high-throughput contract, there are many active LoRAs and average tokens per LoRA can be small; `BLOCK_M=32` leaves unused rows and wastes shrink work.
- `GROUP_N=32` fully amortizes shrink but can create **load imbalance** (long-tail LoRAs dominate).

We swept:
- `BLOCK_M=16`:
  - `runs/lora_fused_kernel_0318783b/cuda_sweep_bm16_nw4_001/result.json` → `1.050 ms`
- Then `GROUP_N`:
  - `gn=16`: `runs/.../cuda_sweep_bm16_nw4_gn16_001/result.json` → `1.030 ms`
  - `gn=20`: `runs/.../cuda_sweep_bm16_nw4_gn20_001/result.json` → `1.026 ms`
  - `gn=24`: `runs/.../cuda_sweep_bm16_nw4_gn24_001/result.json` → `1.019 ms` (best)
  - `gn=28`: `runs/.../cuda_sweep_bm16_nw4_gn28_001/result.json` → `1.023 ms`
  - `gn=32`: `runs/.../cuda_sweep_bm16_nw4_gn8_001/result.json` demonstrates that small groups can be worse; full sweep suggests 24 is best for this distribution.

Finalized defaults for CUDA in the “high throughput regime”:
- `kernels/ironfist/lora_fused_kernel_0318783b.py:1099` sets:
  - `block_m = 16`
  - `group_n = 24`
  - `num_warps = 4`
  (all still overrideable via `KO_LORA_CUDA_*`)

Final NCU for tuned high-throughput:
- `runs/lora_fused_kernel_0318783b/ncu_cuda_ht_smopt_002/ncu_summary.json`
  - `memory_throughput_pct ~76.2%`
  - `sm_throughput_pct ~25.7%`
  - `dram_bytes ~695 MB`

Suite-level confirmation (after the correctness fix + re-baseline):
- `runs/lora_fused_kernel_0318783b/cuda_smopt_003/summary.json` → **geomean 0.07896 ms**, **1.375x vs Triton baseline**

---

## 5) Key NCU-guided “why we did it” (cause → action mapping)

### Symptom: CUDA was memory-bound and far below peak DRAM utilization

- Early CUDA NCU: `runs/lora_fused_kernel_0318783b/ncu_cuda_curr_001/ncu_summary.json`
  - `memory_throughput_pct ~49.6%`
  - `dram_bytes ~943 MB` (too high)

Interpretation:
- We were both *stalling* and doing *too much DRAM traffic*.

Action:
- Reduce redundant traffic:
  - Increase expand coverage per shrink (tune `GROUP_N`)
  - Avoid split-kernel U→DRAM unless absolutely necessary
- Increase overlap:
  - `cp.async` pipelines

### Symptom: Shrink inner loop overhead + poor overlap

Action:
- K-blocking:
  - `kKBlock=64` + double-buffer X/A (`kernels/ironfist/cuda/lora_fused_ext.cu:24` and shrink loop region around `kernels/ironfist/cuda/lora_fused_ext.cu:146`)

Result:
- High-throughput: ~2.05 ms → ~1.07 ms
- NCU: dram bytes dropped ~943 MB → ~699 MB; memory throughput rose ~50% → ~69%

### Symptom: Occupancy / SMEM pressure preventing latency hiding

Action:
- Don’t allocate X smem for idle warps:
  - `kernels/ironfist/cuda/lora_fused_ext.cu:114`

Result:
- NCU memory throughput improved further (to ~80% in `ncu_cuda_smopt_001`)
- Enabled re-instantiation of 8-warp kernels (even if they weren’t the best choice)

### Symptom: Load imbalance in high-throughput distribution with `GROUP_N=32`

Action:
- Trade some shrink duplication for more CTAs:
  - `GROUP_N=24` and `BLOCK_M=16` for high-throughput regime only (`kernels/ironfist/lora_fused_kernel_0318783b.py:1099`)

Result:
- High-throughput nudged over the Triton baseline (~1.0228 → ~1.0180 ms)
- Suite geomean improved substantially vs Triton baseline.

---

## 6) Final implementation notes (current constraints + tuning knobs)

### CUDA backend constraints

Enforced in `kernels/ironfist/cuda/lora_fused_ext.cu` / Python wrapper:
- FP16 only
- `lora_rank == 16`
- `offset_start == 0` (CUDA backend)
- `K % 64 == 0` (because `kKBlock=64`)

### Supported meta values (CUDA extension dispatch)

- `BLOCK_M ∈ {16, 32, 64}`
- `BLOCK_N ∈ {64, 128, 256}`
- `NUM_WARPS ∈ {2, 4, 8}`

### Runtime tuning overrides (no code edits)

- `KO_LORA_CUDA_BLOCK_M`
- `KO_LORA_CUDA_BLOCK_N`
- `KO_LORA_CUDA_GROUP_N`
- `KO_LORA_CUDA_NUM_WARPS`
- `KO_LORA_SPLIT_KERNELS=1` (enables split U->GMEM path; currently slower)

---

## 6.1) Final CUDA: structural vs meta-parameter changes

This section is specifically about what makes the **final fastest CUDA path** fast, and splits it into:

- **Architectural / structural changes**: changes to the kernel’s algorithm, memory hierarchy, pipelining, and work partitioning.
- **Meta-parameter tuning**: choosing launch tiles / warp counts / grouping parameters and heuristics.

### Architectural / structural changes (kernel + launch structure)

These are “real kernel engineering” changes that alter how work is done.

- **K-blocked shrink loop (`kKBlock=64`)**: shrink processes K in 64-wide blocks instead of a smaller inner step; this is a structural design choice baked into the CUDA code (`kernels/ironfist/cuda/lora_fused_ext.cu:24`).
- **`cp.async` pipeline for X/A staging (shrink)**: uses `cp.async.cg.shared.global` + commit/wait to overlap global→shared loads with compute (`kernels/ironfist/cuda/lora_fused_ext.cu:33`).
- **Double-buffered X/A staging (shrink)**: prefetch next K-block into the alternate shared-memory stage while computing the current one (`kernels/ironfist/cuda/lora_fused_ext.cu:173`).
- **Stage A once per CTA stage (avoid per-warp duplication)**: a single “A loader” warp stages A for the CTA, while compute warps stage their own scattered X (`kernels/ironfist/cuda/lora_fused_ext.cu:144`).
- **Keep U on-chip (shared memory), not in global**: shrink writes U into CTA shared memory and expand reads it directly (`kernels/ironfist/cuda/lora_fused_ext.cu:109`).
- **Vectorized U write to shared (half2)**: shrink stores the `[16x16]` accumulator to shared as `half2` pairs to reduce store instruction count (`kernels/ironfist/cuda/lora_fused_ext.cu:229`).
- **Warp-private, double-buffered B staging (expand)**: each warp streams B tiles via `cp.async` into a private shared buffer and overlaps prefetch with compute (`kernels/ironfist/cuda/lora_fused_ext.cu:254`).
- **Vectorized scatter stores (half2) when output is contiguous**: for `output_stride1 == 1`, expand writes `half2` to global memory (and optionally fuses the add) (`kernels/ironfist/cuda/lora_fused_ext.cu:305`).
- **Shared-memory footprint reduction tied to actual M-tiles, not warps**: `x_shared` is sized by `kMTiles = ceil(BLOCK_M/16)` rather than `NUM_WARPS`, which increases occupancy and also enables larger warp-count instantiations (`kernels/ironfist/cuda/lora_fused_ext.cu:114`).
- **Re-enabled 8-warp instantiations after SMEM reduction**: dispatch now allows `num_warps ∈ {2,4,8}` (`kernels/ironfist/cuda/lora_fused_ext.cu:883`).
- **Host-side weight packing (CUDA path)**: stacks weights into contiguous `a_packed` / `b_packed` to avoid pointer indirection inside the CUDA extension (`kernels/ironfist/lora_fused_kernel_0318783b.py:657`).
- **Launch-grid capping to reduce overlaunch**: compute `grid_m` from max tokens-per-LoRA and `grid_loras` from number of unique IDs present (`kernels/ironfist/lora_fused_kernel_0318783b.py:1162`).

### Meta-parameter tuning (tile sizes, warp counts, grouping)

These do not fundamentally change the kernel structure; they pick *which instantiation / launch config* to run.

- **High-throughput CUDA defaults (`BLOCK_M=16`, `GROUP_N=24`, `NUM_WARPS=4`)**: the final speedup on the hardest contract came from selecting a different tiling and grouping than the Triton defaults for the CUDA backend (`kernels/ironfist/lora_fused_kernel_0318783b.py:1099`).
  - `BLOCK_M` changes how many token rows a CTA handles (load-balance vs waste on partially-filled tiles).
  - `GROUP_N` changes how many `BLOCK_N` tiles each CTA covers (shrink reuse vs more CTAs to handle load imbalance).
  - `NUM_WARPS` changes CTA parallelism/resources (and interacts with expand-side scheduling and shared-memory usage).
- **Env-var overrides for quick tuning**: `KO_LORA_CUDA_BLOCK_M`, `KO_LORA_CUDA_BLOCK_N`, `KO_LORA_CUDA_GROUP_N`, `KO_LORA_CUDA_NUM_WARPS` (`kernels/ironfist/lora_fused_kernel_0318783b.py:1078`).
- **Opt-in split-kernel path**: keeping `split_kernels` disabled by default is a policy choice (a meta “which algorithm variant” selector), not a structural part of the fast path (`kernels/ironfist/lora_fused_kernel_0318783b.py:1125`).

Rule of thumb for this project:
- The **big** win came from structural changes (K-blocking + `cp.async` pipelining + SMEM redesign).
- The **last few percent** to edge past Triton on the high-throughput shape came from meta tuning (`BLOCK_M`/`GROUP_N`/`NUM_WARPS`) on top of that structure.

---

## 7) What we didn’t do (but would be the next frontier)

If you want to push further beyond Triton on the high-throughput case (beyond the ~0.5% win):

- Use Ada MMA intrinsics directly (e.g., `mma.sync` variants + `ldmatrix`) instead of WMMA API to improve scheduling and reduce overhead.
- Optimize the “no-LoRA bucket” handling to avoid launching blocks that immediately return (small/large/multi-slice contracts are sensitive to launch overhead).
- Make expand-side B staging more bandwidth-friendly (e.g., larger vector loads, better L2 reuse policy) and reduce scatter-store overhead further.

---

## Appendix A — Suite-level tags recorded (geomean + high-throughput)

All from `runs/lora_fused_kernel_0318783b/<tag>/summary.json`:

- `baseline`: geomean `0.10859065`, HT `1.02278343`
- `cuda_001`: geomean `0.10716588`, HT `2.55439788`
- `cuda_asyncXA_001`: geomean `0.06468909`, HT `2.06611418`
- `cuda_k64_001`: geomean `0.05395250`, HT `1.07525478`
- `cuda_smopt_001`: geomean `0.05261170`, HT `1.06608000`
- `cuda_smopt_003` (final): geomean `0.07896075`, HT `1.01799469`

Note: Several older suite tags were produced before the `actual_num_loras` fix; their `geomean_speedup_vs_baseline` fields may be stale relative to the current baseline artifacts.

---

## Appendix B — High-throughput contract microbench tags recorded

All from `runs/lora_fused_kernel_0318783b/<tag>/result.json` where `contract_path == ...high_throughput...`:

- Early (2.55 ms class): `cuda_half2store_001`, `cuda_group16_001`, `cuda_warps8_001`, …
- Mid (2.05 ms class): `cuda_asyncXA_default_001`, `cuda_asyncXA_gn32_nw4_001`, …
- k64 era (1.07 ms class): `cuda_k64_001`, `cuda_k64_awarp_nw2_001`, …
- Final tuning (1.02 ms class): `cuda_sweep_bm16_nw4_gn24_001`
