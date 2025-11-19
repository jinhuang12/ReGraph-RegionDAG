# Self‑Improving LLMs for GPU Kernel Optimization

**Monte‑Carlo Graph Search + ReGraphT‑style Method Canonicalization + Tool‑Augmented Analysis**

> *Goal:* Turn a frontier LLM into a GPU performance engineer that **iterates** on CUDA/Triton kernels, **profiles** the results, **learns** which optimization methods work across kernels, and **reuses that knowledge** to be more sample‑efficient over time.

---

## TL;DR

* We implement a **True Monte‑Carlo Graph Search (MCGS)** loop that:

  * calls an LLM to propose **concrete code edits** (or diffs) to a kernel,
  * compiles + runs + profiles the variant,
  * scores it via **ground‑truth runtime** (Nsight/runner) and backpropagates reward along the selected path,
  * and **logs transitions** to build an **offline ReGraph** of “method → method” edges with empirical success.

* Our LLM proposals are **tool‑augmented**:

  * **Region‑DAG** from PTX (static structure + work profile + control hints),
  * **Nsight Compute metrics** (ground truth behavior),
  * **PTX↔source correlation** to inspect bottlenecks at source lines.

* We mirror the **ReGraphT paper**’s method taxonomy dynamic:

  * First, the LLM emits **free‑form “method” labels** with each candidate (e.g., “Shared Memory Tiling”).
  * Then we run **`relabel(LLM, τ, O)`**: canonicalize each step’s `method` into the evolving set `O`. If no match, keep original → **new method** that grows `O`.
  * We **persist transitions** `(from_method → to_method, reward)` to a dataset; an offline pass aggregates a **ReGraph** used to bias/limit online expansion.

* **What we explicitly do *not* do**:
  No static time model. Region‑DAG is **structure + work**, Nsight **is ground truth**.

---

## Why this exists (context)

Competing groups try to make LLMs “write faster kernels.” Two recurring problems:

1. **Noisy reward**: compile breakage, correctness issues, run‑to‑run variance.
2. **No reuse**: even if an LLM discovers a good pattern (e.g., async staging, tiling), it doesn’t leverage that **across kernels**.

Our approach aims to fix both:

* Use **MCGS** with a **rollout** and **regularization** to spend builds on actions that historically improve runtime.
* Build an **offline ReGraph** of **method transitions** that we reuse online to **constrain/guide** LLM expansion.

---

## Architecture

```
┌─────────────────────────┐
│ optimize_kernels.py     │  ← Orchestrator: MCGS + rollout + dataset logging
│  • Selection (P-UCB)    │
│  • Expansion (LLM)      │
│  • Relabel (LLM, O)     │ ← canonicalize methods into evolving set O
│  • Materialize worker   │
│  • Backprop + Rollout   │
│  • Dataset logging      │ → regraph_dataset.jsonl
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│ kernel_opt_tooling.py   │  ← LLM tools & wrappers
│  • RegionDagContext     │  ← PTX → Region-DAG (static work profile)
│  • NcuMetricsContext    │  ← Nsight Compute metrics (canonical API)
│  • PtxSourceCorrelator  │  ← Source↔PTX/SASS mapping via .ncu-rep
│  • LLMCandidateGenerator│  ← Responses API tool loop
└─────────────┬───────────┘
              │
              ▼
┌─────────────────────────┐
│ ptx_dag_tool_v2.py      │  ← Deterministic PTX parser → Regions/Edges/Stages
│  • No time model        │  ← counts bytes/FLOPs, flags barriers/async/divergence
└─────────────────────────┘
```

Workers (not shown) handle compilation/execution:

* **CUDA**: `nvcc` → `.ptx`/`.cubin`, optional disassembly, run via `workers.cuda_runner`.
* **Triton**: run via `workers.triton_runner` (PTX extraction TODOs noted below).

---

## What is implemented (and how it maps to the paper)

| Paper concept                   | Our implementation                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Trajectory generation (G.3)** | `LLMCandidateGenerator.propose(...)` returns `candidates[]` with `method`, `detail`, `code`.                                    |
| **Relabel step (G.4)**          | `LLMRelabeler.relabel(existing_methods=O, steps=candidates)` → per step canonical `method_canonical`, grow `O` if unmatched.    |
| **Graph building**              | Online: search graph over **code states** keyed by **state_hash** (transpositions). Offline: `ReGraphDataset` → `regraph.json`. |
| **Selection policy**            | Graph‑aware **P‑UCB** with progressive widening; **anti‑cycle** penalty within path.                                            |
| **Expansion**                   | First visit: register **all** relabeled candidates as actions at node; choose one to evaluate.                                  |
| **Rollout & regularization**    | ε‑greedy argmax on `Qbar - λ·Nsa` for up to `ROLLOUT_MAX_STEPS`; early stop on failure.                                         |
| **Reward**                      | From runtime: success → speedup mapping; failure → `-1.0`. Backprop **max rollout reward** along path.                          |
| **Reuse across problems**       | **ReGraph** edges give an empirical prior on `from_method → to_method`; online expansion can filter to allowed next methods.    |

> **Key difference from paper:**
> We directly optimize **code states** (materialized source) in the online MCGS. The **offline** ReGraph is maintained over **canonical methods** and can be used to **constrain** LLM expansion when desired.

---

## Repo layout (expected)

```
.
├── optimize_kernels.py            # Orchestrator (MCGS + rollout + dataset + relabel)
├── kernel_opt_tooling.py          # LLM tool adapters: Region-DAG, NCU, PTX-source; Responses API loop
├── ptx_dag_tool_v2.py             # PTX → Region DAG (static work)
├── workers/
│   ├── cuda_runner.py             # Runs a CUDA cubin with timing & checks  (you provide)
│   ├── triton_runner.py           # Runs a Triton kernel with timing       (you provide)
│   └── state_manager.py           # save_partial_state()/get_latest_phase_data()
├── shared/
│   └── model.py                   # KernelCode, DeviceProfile, etc.
└── specs/
    └── example.json               # Example kernel spec (see below)
```

---

## Prerequisites

* **Python 3.10+**
* **CUDA toolkit**: `nvcc`, `nvdisasm` or `cuobjdump`
* **Nsight Compute** (optional but recommended for metrics/correlation)
* **OpenAI Python SDK** if using LLM tools (`pip install openai`)
* Environment variables:

  * `OPENAI_API_KEY` – for LLM calls
  * `COMPILE_CACHE_DIR` (optional) – where to cache `.ptx`/`.cubin`

---

## Quickstart

### 1) Create a kernel spec

`specs/example.json`:

```json
{
  "name": "vector_add_cuda",
  "kernel": {
    "kernel_type": "cuda",
    "source_code": "// minimal CUDA kernel here\nextern \"C\" __global__ void kernel(float* a, float* b, float* c, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }\n",
    "metadata": {
      "kernel_name": "kernel"
    },
    "invocation_example": "",
    "io": {
      "launch": {
        "grid":  {"x": 256, "y": 1, "z": 1},
        "block": {"x": 256, "y": 1, "z": 1}
      }
    }
  },
  "device_profile": {
    "gpu_name": "H100",
    "arch": "sm_90"
  }
}
```

> For Triton, set `"kernel_type": "triton"` and provide a Python module source containing the kernel. The Triton PTX export is currently a TODO (we write a placeholder PTX file).

### 2) Run baseline + search (MCGS)

```bash
python optimize_kernels.py \
  --input specs/example.json \
  --outdir runs/example \
  --gpus 0 \
  --budget-builds 30 \
  --llm-model gpt-5-mini
```

This will:

* Materialize the baseline kernel, compile, and time it.
* Start MCGS:

  * At an unvisited node, call LLM for candidates (with tool assistance if artifacts exist).
  * **Relabel** methods into the evolving catalog **O**.
  * Register all candidates as actions; pick one; compile, time, and reward it.
  * Backpropagate reward; optionally **rollout** further steps.
  * Log dataset rows into `runs/example/regraph_dataset.jsonl`.

### 3) Build and reuse offline ReGraph

Build from accumulated dataset:

```bash
python optimize_kernels.py --outdir runs/example --build-regraph
```

This writes `runs/example/regraph.json` with node/edge stats (`count`, `success`, `mean_reward`).
You can also auto‑rebuild before search:

```bash
python optimize_kernels.py ... --regraph-autobuild
```

Online, we can (optionally) **filter** expansions: only allow methods that are neighbors of `last_method` in `regraph.json`. This is already wired in the sample rollout path; enable/disable per your needs (see code comments).

---

## Outputs & Directory Structure

Inside `--outdir`:

```
out/
└── kernels/<safe_kernel_name>/
    ├── baseline/                      # Baseline build & run artifacts
    │   ├── control.json
    │   ├── result.json                # mean_ms, state_hash, ncu_metrics, materialized_source
    │   ├── kernel.ptx / .cubin / .sass (CUDA)
    │   └── runner_result.json         # runner-reported timings
    ├── variants/<uuid>/*              # One folder per evaluated candidate
    │   ├── control.json
    │   ├── result.json
    │   └── logs...
    ├── search_state.json              # Entire MCGS graph state (nodes, edges, catalog O)
    ├── events.jsonl                   # Human-readable event log
    ├── trace.md                       # Quick summary (best improvements, timestamps)
    ├── best -> variants/<uuid>        # Symlink/copy to current best variant
    └── best_summary.json
```

Top level (per `--outdir`):

```
out/
├── regraph_dataset.jsonl              # Cross-kernel dataset of transitions
└── regraph.json                       # Offline aggregated ReGraph (nodes + edges stats)
```

---

## Orchestrator (optimize_kernels.py) – Detailed Notes

### Data structures

```python
@dataclass
class EdgeStats:
    Qsum: float = 0.0   # sum of observed rewards on this edge
    Nsa:  float = 0.0   # visit count for (s,a)
    Qbar: float = 0.0   # mean reward (Qsum / max(1, Nsa))
    child: Optional[str] = None  # state id of child once expanded

@dataclass
class NodeStats:
    visits: int = 0
    source_code: str = ""                  # materialized full source for this state
    ncu: Dict[str, Any] = field(default_factory=dict)
    edges: Dict[str, EdgeStats] = field(default_factory=dict)   # action_id -> EdgeStats
    actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # action_id -> candidate payload
    applied_actions: List[str] = field(default_factory=list)    # action ids along path to this node
    applied_methods: List[str] = field(default_factory=list)    # canonical methods along the path
    last_method: Optional[str] = "START"
    impl_hash: str = ""                                         # worker-reported state hash
    ptx_path: Optional[str] = None
    ncu_report_path: Optional[str] = None
```

* **Node identity**: by default we use the **worker’s state hash** (transpositions enabled).
* We store the **full `source_code`** at *every* node to let you diff or re‑materialize later.

### Selection (graph‑aware P‑UCB + progressive widening)

* At node `s` with `visits = N` and actions `a` with `Nsa`, `Qbar`:

  ```
  score(s,a) = Qbar + sqrt( 2 * ln(N+1) / (Nsa+1) )   // exploration
               - λ_cycle * used_in_this_path(s,a)     // anti-cycle penalty
  ```
* **Anti-cycle**: if `child` already appears in the current path, penalize harshly to avoid loops.

### Expansion

* If a node has **no actions yet**, we:

  1. Call **LLM** for `candidates[]` (with region/NCU context if available).
  2. **Relabel**: `LLMRelabeler.relabel(O, steps)` → `method_canonical` for each.
  3. Update `O` (method catalog) by adding unmatched items.
  4. Register **all** candidates as edges (`actions[aid] = candidate`), each with empty `EdgeStats`.
  5. Choose one to evaluate (greedy by `Qbar`, tie-break deterministic).

> **Candidate payload** (LLM output):
>
> ```json
> {
>   "think": "...",
>   "method": "Shared Memory Tiling",
>   "detail": "Tile 128x...; cp.async ...",
>   "code": "<unified diff or full source>",
>   "launch_update": { "...": "..." }  // optional
> }
> ```
>
> After relabel, we add `"method_canonical"`.

### Worker & reward

* **Materialize** the chosen candidate into a variant dir (`full_source_code` or `patch`).
* Run **CUDA** or **Triton** runner with timing (and optional NCU—hook points exist).
* **Reward**:

  * Failure → `-1.0`.
  * Success → speedup‐based mapping (if `speedup≥1.0` reward=speedup, else reward can be `speedup-1.0`).
* Attach `child` pointer to the new state id (worker’s hash).
* Update **incumbent best** if `mean_ms` improves.

### Rollout (regularized)

* From the first successful child, perform up to `ROLLOUT_MAX_STEPS` steps:

  * If node has no actions: request LLM, **relabel**, **optionally filter** by **ReGraph** adjacency (neighbors of `last_method`).
  * Policy: ε‑greedy on `Qbar - λ·Nsa`.
  * Early stop on failure.
* Backprop **maximum** reward observed during rollout along the initial path.

### Logging

* Every evaluated transition is appended to **`regraph_dataset.jsonl`** with:

  ```
  t, kernel, arch, from_method, to_method, ok, mean_ms, baseline_ms, speedup, reward, candidate_id, variant_dir
  ```
* The **offline aggregator** builds `regraph.json` (node/edge counts, successes, mean rewards).

---

## LLM tools & prompts (kernel_opt_tooling.py)

* **RegionDagContext**:
  Construct from `.ptx` text. Provides:

  * `overview(max_hot_stages, max_hot_regions)`: stages with work breakdown (global/shared bytes, FLOPs, insts), **no time model**, loops, potential divergence flags.
  * `stage_detail(stage_id)`, `region_detail(region_id)`: drilldown; PTX snippet for a region.
* **NcuMetricsContext**:

  * Canonical summary (`sm_throughput_pct`, `gpu_dram_throughput_pct`, occupancy, hit rates, etc.).
  * `get_values(names, name_kind)` and `search_names(query)` for ad‑hoc metrics.
* **PtxSourceCorrelator**:

  * `get_ptx_snippet_for_source_span(source_file, start_line, end_line, ...)`: correlate PTX/SASS to source with optional per‑PC metrics.
* **LLMCandidateGenerator**:

  * Manages the **Responses API** tool loop using the above tools.
  * Enforces **STRICT JSON** output schema for candidates.

> **Important**: Our **system prompt** instructs the model to first analyze NCU (ground truth), then map to Region‑DAG (structure), then propose code edits. No static time model is ever used.

---

## CLI & Options

```
python optimize_kernels.py \
  --input specs/example.json \
  --outdir runs/example \
  --gpus 0 \
  --budget-builds 30 \
  --llm-model gpt-5-mini \
  [--resume] \
  [--build-regraph] \
  [--regraph-autobuild]
```

* `--resume` – continue from existing `search_state.json`.
* `--build-regraph` – rebuild `regraph.json` from `regraph_dataset.jsonl` and exit.
* `--regraph-autobuild` – rebuild `regraph.json` before running search.

---

## Known gaps & TODOs (near‑term)

* **Triton PTX export**: the worker writes placeholders; wire real PTX extraction and optionally a `.ncu-rep` capture path for the correlator.
* **Correctness gates**: add property-based checks and numeric tolerance to abort bad branches early.
* **De‑duplication**: normalize code and skip duplicates (hash after formatting).
* **Hot‑region verification**: compute PTX stage/region diffs for candidates to ensure the edit touches the bottleneck.
* **Noise reduction**: run multiple timing repeats and use trimmed mean or bootstrap CIs in reward.
* **Method hygiene**: periodic offline synonym merge in `regraph.json` (string/embedding similarity) to keep `O` tight.
* **Parametric templates**: encourage LLM to emit edits with knobs (tile sizes, unroll, vector width) and sweep locally.

---

## Troubleshooting

* **Baseline fails**: check `baseline/stderr.log`, `runner_stderr.log`. Missing `nvcc` or bad `kernel_name`.
* **LLM returns non‑JSON**: ensure `OPENAI_API_KEY` is set; network okay; retry. We already nudge once on malformed output.
* **No speedups**: inspect `events.jsonl` and `regraph_dataset.jsonl` to see method distribution. Ensure correctness gates aren’t silently killing variants.
* **NCU not found**: `NcuMetricsContext` requires Nsight Compute Python interface; we degrade gracefully when it’s missing.

---

## Extending the system

* **Add new tools** (e.g., SASS diff, ptxas diagnostics, occupancy models):
  Expose as a function tool in `kernel_opt_tooling.py` and reference in the system prompt.
* **Alternate selection policy**: swap P‑UCB for your favorite bandit or add diversity bonuses when `Qbar`s tie.
* **Graph constraints**: switch from hard neighbor filtering to a soft prior (e.g., bias scores by `regraph.mean_reward(to_method)`).

---

## FAQ

**Q: Why a **graph** instead of a tree?**
A: Different action sequences can converge to the same materialized code. Reusing states via `state_hash` (transpositions) saves budget.

**Q: Why not a static time model?**
A: We’ve seen static models frequently mislead search in realistic kernels. Here, **NCU is ground truth**; Region‑DAG is **structure only**.

**Q: What is the relabel step doing again?**
A: It **canonicalizes** free‑form `method` strings into the evolving set **O**. If no match, the original string is kept and becomes a **new** method in `O`. This mirrors Appendix G.4 of the paper.

---

## Code reading guide (documented)

### `optimize_kernels.py`

* **Top constants**: knobs for selection (`PW_*`), rollout (`EPSILON_ROLLOUT`, `LAMBDA_REG`, `ROLLOUT_MAX_STEPS`), anti‑cycle regularization.
* **`ReGraphDataset`**: append‑only JSONL; aggregator builds `regraph.json` with node/edge `count/success/mean_reward`.
* **`LLMRelabeler`**: small wrapper around the Responses API; returns `[ {index, existed, method}, ... ]`.
* **`Orchestrator.baseline_or_resume`**: builds baseline, seeds the root node with full `source_code` and baseline metrics.
* **`Orchestrator.run_mcgs_for_kernel`**:

  * *Selection*: P‑UCB with progressive widening & anti‑cycle.
  * *Expansion*: ask LLM → relabel → register all actions; pick one to build now.
  * *Worker*: materialize → compile/run → reward.
  * *Dataset logging*: write `(from_method, to_method, reward)` row.
  * *Rollout*: limited horizon ε‑greedy argmax `Qbar - λNsa`; optional ReGraph neighbor filter.
  * *Backprop*: add `rollout_best` to each edge along the initial path.
* **`worker_main`**: materialize source (`full_source_code` or unified diff), compile, optionally capture PTX/SASS, run runner, save `result.json`. CUDA path is complete; Triton PTX extraction is TODO (placeholder PTX written).

### `kernel_opt_tooling.py`

* **RegionDagContext**:

  * `overview()` returns counts + stage/region work; **no times**.
  * `stage_detail()`, `region_detail()` provide drilldown & PTX snippets.
* **NcuMetricsContext**: canonical metric names; `summary()`, `get_values()`, `search_names()`.
* **PtxSourceCorrelator**: `.ncu-rep` → per‑source‑line PTX/SASS/metric entries.
* **LLMCandidateGenerator**: system prompt, tool schemas, tool loop; enforces **STRICT JSON** candidate schema.

### `ptx_dag_tool_v2.py`

* Deterministic PTX parsing:

  * classifies ops → phases, accounts global/shared bytes, counts FLOPs (tensor ops / FMA), flags barriers, async waits, fences, and detects **loops**/potential divergence via predicated branches.
  * builds regions & typed edges; partitions into **stages** at strong sync boundaries.
  * **No** timing model.

---

## License / Attribution

* This repo implements ideas inspired by **ReGraphT** (reasoning graphs for CUDA optimization). Our implementation differs in details; any mistakes are ours.
* CUDA® and Nsight® Compute are NVIDIA marks; Triton is an OpenAI project.

---

## Final word

Treat this as an **experiment** with tight feedback loops: measure compile/correctness rates, early speedups, and the utility of the offline ReGraph. If the priors start to help (higher win‑rate along top ReGraph edges), you’re on a promising path. If not, invest in guardrails or pivot to LLM‑generated **parametric templates** + classical tuning.

PRs that improve correctness gates, Triton PTX export, and ReGraph quality will have outsized impact.
