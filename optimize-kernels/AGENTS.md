# AGENTS

This repo already implements a small set of cooperating “agents.” Use the roles below when extending or plugging into the system. All paths are workspace‑relative.

- **Search Orchestrator** (`optimize_kernels.py`)
  - Entry: `python optimize_kernels.py --input <spec.json> --outdir <runs/...> --gpus 0,1 --budget-builds 30 --llm-model gpt-5-mini [--resume|--build-regraph|--regraph-autobuild]`.
  - Responsibilities: baseline build/run, maintain the Monte‑Carlo Graph Search (P‑UCB + progressive widening, anti‑cycle), expansion via LLM, rollout regularization, reward mapping (speedup→reward, failure→-1), backprop of the max rollout reward, incumbent tracking (`best/` symlink, `trace.md`), and persistence (`search_state.json`).
  - Data it owns: per‑node `NodeStats` (source, NCU summary, actions, method history, edges with Q/N counts, ptx/ncu paths), evolving `method_catalog` (O), `events.jsonl`, `regraph_dataset.jsonl`, `regraph.json` (offline prior), and `regraph_autobuild` support.
  - Expansion policy: first visit → call LLM, **relabel** methods → register all candidates → optionally filter by offline ReGraph adjacency (neighbors of `last_method`) → greedily materialize one.
  - Rollout: ε‑greedy on `Qbar - λ·Nsa` for up to `ROLLOUT_MAX_STEPS`, early stop on failure, same relabel+ReGraph filtering at first visit.

- **LLM Candidate Generator** (`kernel_opt_tooling.py::LLMCandidateGenerator`)
  - Inputs: `KernelCode`, optional `region_summary` (Region‑DAG work profile) and `ncu_summary` (Nsight metrics).
  - Prompt guarantees: system prompt enforces “analyze NCU (ground truth) then Region‑DAG (structure) then propose code,” STRICT JSON output with array `candidates[{think, method, detail, code, launch_update?}]`.
  - Tooling exposed to the model (Responses API): `region_dag_inspect`, `ncu_metrics_inspect`, `get_ptx_by_source`. Tool registry dispatches to local handlers using global contexts.
  - Session management: tracks `previous_response_id` per kernel to continue tool loops across turns; retries once on malformed JSON.

- **Method Relabeler** (`optimize_kernels.py::LLMRelabeler` + `relabel_methods`)
  - Purpose: canonicalize free‑form `candidate["method"]` into the evolving catalog O (ReGraphT “relabel” step). Adds new methods when no match. Fallback heuristic keeps search alive if the LLM relabel API fails.
  - Used both in expansion and rollout; canonical methods drive dataset logging and offline ReGraph adjacency filtering.

- **Analysis Context Providers** (`kernel_opt_tooling.py`)
  - `RegionDagContext`: PTX → regions/stages/loops via `ptx_dag_tool_v2.build_all`, returns static work profile only (bytes/FLOPs, divergence hints). Feeds the `region_dag_inspect` tool.
  - `NcuMetricsContext`: wraps `ncu_report` to canonicalize Nsight metrics; powers `ncu_metrics_inspect` (`summary`, `get_values`, `search_names`).
  - `PtxSourceCorrelator`: optional `.ncu-rep` → source/PTX/SASS mapping with per‑PC metrics; backs `get_ptx_by_source`.

- **Worker Agent** (`optimize_kernels.py --worker <control.json>`)
  - Input: `control.json` with `kernel`, `variant` (`full_source_code` or `patch` + optional `launch_update`), `workdir`, `ncu` flags, `timing`, and optional `base_source_code`.
  - Materialization: writes source, applies unified diff if provided, hashes materialized source + launch_update for `state_hash`, and saves partial phases via `workers/state_manager.py`.
  - Triton branch: executes `workers.triton_runner` (IOContract aware), returns timing; currently writes placeholder PTX/SASS until Triton PTX export is wired.
  - CUDA branch: compiles via `nvcc` to PTX+CUBIN (compile cache at `$COMPILE_CACHE_DIR`), optional disasm, then runs `workers.cuda_runner` (CuPy) with hash including launch cfg. Outputs `result.json` with `{ok, mean_ms, std_ms, state_hash, ncu_metrics≈kernel_time_ms, materialized_source, ptx_path, ncu_report_path}`.

- **Offline ReGraph Builder** (`ReGraphDataset.build_regraph()` inside `optimize_kernels.py`)
  - Aggregates `regraph_dataset.jsonl` (rows: `from_method → to_method`, reward, ok, timing, variant_dir) into `regraph.json` with node/edge counts, successes, and mean rewards. Orchestrator can auto-build at start and use adjacency to filter expansions.

Usage tips:
- Set `OPENAI_API_KEY` for LLM calls; `COMPILE_CACHE_DIR` optional for CUDA compile cache.
- Run `python optimize_kernels.py --build-regraph --outdir <run>` to refresh offline priors without starting search.
- For debugging a single candidate, you can manually invoke worker mode with a crafted `control.json` in a variant directory.
