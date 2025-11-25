#!/usr/bin/env python3
"""
init_kernels_and_suites.py

Scan an IronFist-style contracts/ directory and:

  1. Materialize ONE Python module per kernel family in kernels/<kernel_name>.py
     (using kernel.metadata.kernel_name and kernel.source_code).

  2. Build one suite JSON per kernel family in suites/<kernel_name>_suite.json
     that groups all contracts for that kernel and defines a suite objective.

Usage (from repo root):

  python tools/init_kernels_and_suites.py \
      --contracts-dir contracts \
      --kernels-dir kernels \
      --suites-dir suites \
      [--overwrite-modules]

You can safely re-run this script; it will not overwrite existing modules unless
--overwrite-modules is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_source_hash(src: str) -> str:
    """Return a stable hash for the source code string."""
    return hashlib.sha1(src.encode("utf-8")).hexdigest()


def longest_common_prefix(names: List[str]) -> str:
    """Simple longest common prefix (character-wise)."""
    if not names:
        return ""
    prefix = names[0]
    for name in names[1:]:
        # Shrink until it matches or is empty
        while prefix and not name.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix


def derive_base_group_name(kernel_names: List[str]) -> str:
    """
    Choose a human-friendly base name for a group:
      * longest common prefix of kernel_names (stripping trailing underscores),
      * else fall back to the first kernel_name.
    """
    if not kernel_names:
        return "kernel"
    prefix = longest_common_prefix(kernel_names).rstrip("_")
    if prefix and len(prefix) >= 3:
        return prefix
    return kernel_names[0]


def tokenize(text: str) -> List[str]:
    """Simple tokenization on identifiers; suitable for loose similarity."""
    import re

    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)


def cosine_similarity(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Cosine similarity between two sparse counters."""
    import math

    if not a or not b:
        return 0.0
    # Iterate over smaller dict for the dot product.
    if len(a) > len(b):
        a, b = b, a
    dot = sum(val * b.get(k, 0) for k, val in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--contracts-subdir",
        required=True,
        help="Required subdirectory under contracts/ to process (e.g. ironfist, triton_bench); outputs mirror this under kernels/ and suites/",
    )
    ap.add_argument("--kernels-dir", default="kernels", help="Output base directory for materialized kernel modules")
    ap.add_argument("--suites-dir", default="suites", help="Output base directory for suite JSONs")
    ap.add_argument("--overwrite-modules", action="store_true", help="Overwrite existing kernel modules")
    args = ap.parse_args()

    contracts_dir = Path("contracts").resolve()
    contracts_subdir = Path(args.contracts_subdir).as_posix()
    contracts_root = contracts_dir / contracts_subdir
    kernels_dir = Path(args.kernels_dir).resolve()
    suites_dir = Path(args.suites_dir).resolve()
    repo_root = Path.cwd()

    if not contracts_dir.is_dir():
        raise SystemExit(f"contracts-dir {contracts_dir} does not exist or is not a directory")
    if not contracts_root.is_dir():
        raise SystemExit(f"contracts subdir {contracts_root} does not exist or is not a directory")

    kernels_root = kernels_dir / contracts_subdir if contracts_subdir else kernels_dir
    suites_root = suites_dir / contracts_subdir if contracts_subdir else suites_dir

    kernels_root.mkdir(parents=True, exist_ok=True)
    suites_root.mkdir(parents=True, exist_ok=True)

    # 1) Parse contracts and compute source hashes
    records: List[Dict[str, Any]] = []
    contract_paths = sorted(contracts_root.rglob("*.json"))
    for path in contract_paths:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"[WARN] Failed to parse {path}: {e}")
            continue

        kernel = data.get("kernel", {})
        meta = kernel.get("metadata", {})
        kernel_name = meta.get("kernel_name") or data.get("name") or path.stem
        if kernel.get("kernel_type") == "triton":
            entry_point = meta.get("entry_point") or kernel_name or "run"
        else:
            entry_point = meta.get("entry_point", "run")
        kernel_type = kernel.get("kernel_type", "multi_kernel")
        source_code = kernel.get("source_code", "") or ""
        source_hash = compute_source_hash(source_code) if source_code else f"no_source::{path.name}"
        rel_path = path.relative_to(contracts_root).as_posix()

        records.append(
            {
                "path": path,
                "rel_path": rel_path,
                "name": data.get("name", path.stem),
                "kernel_name": kernel_name,
                "entry_point": entry_point,
                "kernel_type": kernel_type,
                "source_code": source_code,
                "source_hash": source_hash,
            }
        )

    if not records:
        print(f"[INFO] No contracts found under {contracts_dir}")
        return

    # 2) Group contracts by source hash first, then bucket by prefix for similarity-based suite merging
    by_hash: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_hash.setdefault(rec["source_hash"], []).append(rec)

    hash_groups: List[Dict[str, Any]] = []
    for src_hash, items in sorted(by_hash.items(), key=lambda kv: kv[0]):
        kernel_names = [it["kernel_name"] for it in items if it.get("kernel_name")]
        base_prefix = derive_base_group_name(kernel_names)
        hash_suffix = src_hash[:8]
        kernel_name = f"{base_prefix}_{hash_suffix}"

        entry_point = items[0]["entry_point"]
        kernel_type = items[0]["kernel_type"]

        if any(it["entry_point"] != entry_point for it in items):
            print(f"[WARN] Mixed entry_point values in hash group {src_hash[:8]}: using '{entry_point}'")
        if any(it["kernel_type"] != kernel_type for it in items):
            print(f"[WARN] Mixed kernel_type values in hash group {src_hash[:8]}: using '{kernel_type}'")

        # Precompute token bag for similarity comparisons (used later for merges)
        token_bag: Dict[str, int] = {}
        if items and items[0].get("source_code"):
            for tok in tokenize(items[0]["source_code"]):
                token_bag[tok] = token_bag.get(tok, 0) + 1

        hash_groups.append(
            {
                "prefix": base_prefix,
                "kernel_name": kernel_name,
                "source_hash": src_hash,
                "entry_point": entry_point,
                "kernel_type": kernel_type,
                "token_bag": token_bag,
                "items": items,
            }
        )

    contracts_dir_rel = os.path.relpath(contracts_root, start=repo_root)

    # 3) Within each prefix bucket, merge hash groups whose cosine similarity >= 0.98
    PREFIX_SIM_THRESHOLD = 0.98
    buckets: Dict[str, List[int]] = {}
    for idx, grp in enumerate(hash_groups):
        buckets.setdefault(grp["prefix"], []).append(idx)

    def build_clusters(indices: List[int]) -> List[List[int]]:
        # Simple union-find based on similarity threshold
        parent = {i: i for i in indices}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                gi = hash_groups[indices[i]]
                gj = hash_groups[indices[j]]
                sim = cosine_similarity(gi["token_bag"], gj["token_bag"])
                if sim >= PREFIX_SIM_THRESHOLD:
                    union(indices[i], indices[j])
                else:
                    print(
                        f"[INFO] Prefix '{gi['prefix']}' has divergent hashes "
                        f"{gi['source_hash'][:8]} vs {gj['source_hash'][:8]} (cosine={sim:.4f}); keeping separate suites"
                    )

        clusters: Dict[int, List[int]] = {}
        for idx in indices:
            root = find(idx)
            clusters.setdefault(root, []).append(idx)
        return list(clusters.values())

    suite_groups: List[Dict[str, Any]] = []
    used_suite_stems: Set[str] = set()

    for prefix, idxs in buckets.items():
        clusters = build_clusters(idxs)
        for cluster in clusters:
            # Deterministic order
            cluster = sorted(cluster, key=lambda i: hash_groups[i]["source_hash"])
            rep = hash_groups[cluster[0]]
            suite_stem = prefix
            if suite_stem in used_suite_stems:
                suite_stem = f"{prefix}_{rep['source_hash'][:8]}"
            used_suite_stems.add(suite_stem)
            if len(cluster) > 1:
                merged_hashes = ", ".join(hash_groups[i]["source_hash"][:8] for i in cluster)
                print(
                    f"[INFO] Merging {len(cluster)} hash groups under prefix '{suite_stem}' "
                    f"(hashes: {merged_hashes}) via cosine >= {PREFIX_SIM_THRESHOLD}"
                )
            suite_groups.append(
                {
                    "suite_stem": suite_stem,
                    "kernel_name": rep["kernel_name"],
                    "entry_point": rep["entry_point"],
                    "kernel_type": rep["kernel_type"],
                    "member_indices": cluster,
                }
            )

    print(f"[INFO] Found {len(hash_groups)} hash groups across {len(buckets)} prefix bucket(s) in {contracts_dir_rel}")
    print(f"[INFO] Emitting {len(suite_groups)} suite(s) after similarity merging")

    # 4) Materialize modules per hash group (code identity)
    for group in hash_groups:
        kernel_name = group["kernel_name"]
        items = group["items"]
        print(
            f"\n[INFO] Materializing module for hash {group['source_hash'][:8]} "
            f"as '{kernel_name}' with {len(items)} contract(s)"
        )

        rep_with_code = next((it for it in items if it.get("source_code")), None)
        source_code = rep_with_code["source_code"] if rep_with_code else ""

        module_path = kernels_root / f"{kernel_name}.py"
        if module_path.exists() and not args.overwrite_modules:
            print(f"  [SKIP] Module already exists: {module_path}")
        else:
            if not source_code:
                print(f"  [WARN] No source_code found for module '{kernel_name}', skipping module materialization")
            else:
                module_path.write_text(source_code, encoding="utf-8")
                print(f"  [OK] Wrote kernel module: {module_path}")

    # 5) Build suites after similarity-based merging
    for suite_group in suite_groups:
        suite_stem = suite_group["suite_stem"]
        suite_kernel_name = suite_group["kernel_name"]
        entry_point = suite_group["entry_point"]
        kernel_type = suite_group["kernel_type"]
        member_indices = suite_group["member_indices"]

        cases = []
        for idx in member_indices:
            for item in hash_groups[idx]["items"]:
                fname = item.get("rel_path") or item["path"].name
                cases.append(
                    {
                        "name": item["name"],
                        "filename": fname,
                        "weight": 1.0,
                    }
                )

        suite = {
            "suite_name": f"{suite_stem}_suite",
            "kernel_name": suite_kernel_name,
            "entry_point": entry_point,
            "kernel_type": kernel_type,
            "contracts_dir": contracts_dir_rel,
            "contracts": cases,
            "objective": "geomean_speedup",
            "notes": (
                "Auto-generated from contracts by init_kernels_and_suites.py; "
                "suite may include multiple source hashes merged by prefix + high similarity"
            ),
        }

        suite_path = suites_root / f"{suite_stem}_suite.json"
        suite_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
        print(
            f"[OK] Wrote suite: {suite_path} "
            f"(members: {len(cases)} contracts across {len(member_indices)} hash group(s))"
        )

    print("\n[INFO] Done. You can now profile suites with tools/profile_suite.py")


if __name__ == "__main__":
    main()
