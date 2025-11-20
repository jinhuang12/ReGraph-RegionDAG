import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Safely stream a JSONL file, yielding dict rows."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def extract_ptx_features(ptx_path: Path) -> Dict[str, float]:
    """Small bag-of-ops feature extractor for PTX text."""
    feats: Dict[str, float] = {}
    try:
        text = Path(ptx_path).read_text(encoding="utf-8")
    except Exception:
        return feats

    lines = [ln for ln in text.splitlines() if ln.strip()]
    feats["ptx_lines"] = float(len(lines))
    feats["ptx_bytes"] = float(len(text.encode("utf-8")))

    patterns = {
        "ld_global": r"ld\.global",
        "st_global": r"st\.global",
        "ld_shared": r"ld\.shared",
        "st_shared": r"st\.shared",
        "barrier": r"bar\.sync",
        "atom": r"atom\.",
    }
    for name, pat in patterns.items():
        feats[f"ptx_{name}_count"] = float(len(re.findall(pat, text)))
    return feats


def flatten_ncu_metrics(ncu: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric Nsight metrics into flat feature names."""
    feats: Dict[str, float] = {}
    for k, v in (ncu or {}).items():
        if isinstance(v, (int, float)):
            feats[f"ncu::{k}"] = float(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    feats[f"ncu::{k}.{kk}"] = float(vv)
    return feats


@dataclass
class OfflineModel:
    feature_weights: Dict[str, float]
    bias: float
    mean_speedup: float

    @classmethod
    def from_json(cls, path: Path) -> "OfflineModel":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_weights=obj.get("feature_weights", {}),
            bias=float(obj.get("bias", 0.0)),
            mean_speedup=float(obj.get("mean_speedup", 0.0)),
        )

    def to_json(self, path: Path) -> None:
        payload = {
            "feature_weights": self.feature_weights,
            "bias": float(self.bias),
            "mean_speedup": float(self.mean_speedup),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def predict(self, features: Dict[str, float]) -> float:
        score = float(self.bias)
        for k, v in features.items():
            if not isinstance(v, (int, float)):
                continue
            score += float(self.feature_weights.get(k, 0.0)) * float(v)
        # Guardrails: never dip below 0
        return max(0.0, score)


@dataclass
class OfflineExample:
    features: Dict[str, float]
    target_speedup: float


class OfflineTrainer:
    """Offline trainer that ingests trajectory logs to predict speedup."""

    def __init__(self, dataset_path: Path, model_path: Path):
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)

    def _method_prior(self, regraph: Dict[str, Any], prev_method: str, next_method: str) -> Tuple[float, float]:
        edges = (regraph or {}).get("edges", {})
        stats = edges.get(prev_method, {}).get(next_method, {}) if isinstance(edges, dict) else {}
        cnt = float(stats.get("count", 0.0) or 0.0)
        succ = float(stats.get("success", 0.0) or 0.0)
        reward_mean = float(stats.get("mean_reward", 0.0) or 0.0)
        success_rate = succ / max(1.0, cnt)
        return success_rate, reward_mean

    def _vector_keys(self, examples: List[OfflineExample]) -> List[str]:
        keys: List[str] = []
        for ex in examples:
            for k in ex.features.keys():
                if k not in keys:
                    keys.append(k)
        return sorted(keys)

    def _fit_model(self, examples: List[OfflineExample]) -> OfflineModel:
        keys = self._vector_keys(examples)
        if not keys:
            return OfflineModel(feature_weights={}, bias=0.0, mean_speedup=0.0)

        X: List[List[float]] = []
        y: List[float] = []
        for ex in examples:
            X.append([float(ex.features.get(k, 0.0) or 0.0) for k in keys])
            y.append(float(ex.target_speedup))

        bias = sum(y) / max(1, len(y))
        weights: Dict[str, float] = {}
        try:
            import numpy as np  # type: ignore

            Xmat = np.array(X, dtype=float)
            yvec = np.array(y, dtype=float)
            ones = np.ones((Xmat.shape[0], 1), dtype=float)
            X_aug = np.concatenate([ones, Xmat], axis=1)
            coeffs, *_ = np.linalg.lstsq(X_aug, yvec, rcond=None)
            bias = float(coeffs[0])
            weights = {k: float(w) for k, w in zip(keys, coeffs[1:])}
        except Exception:
            # Fall back to feature-wise averages (esp. useful when numpy is unavailable)
            for k in keys:
                vals: List[float] = []
                for ex in examples:
                    val = ex.features.get(k, 0.0)
                    if isinstance(val, (int, float)):
                        vals.append(float(val) * ex.target_speedup)
                if vals:
                    weights[k] = sum(vals) / max(1.0, len(vals))

        return OfflineModel(feature_weights=weights, bias=bias, mean_speedup=bias)

    def _gather_features(
        self, row: Dict[str, Any], regraph: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Optional[float]]:
        speedup = row.get("speedup")
        try:
            target_speedup = float(speedup if speedup is not None else 0.0)
        except Exception:
            target_speedup = 0.0

        prev_method = (row.get("from_method") or "START")
        next_method = (row.get("to_method") or "")
        variant_dir = Path(row.get("variant_dir", ""))

        feats: Dict[str, float] = {}
        if next_method:
            feats[f"method::{next_method}"] = 1.0
        feats[f"context::{prev_method}"] = 1.0
        feats["reward"] = float(row.get("reward", 0.0) or 0.0)

        # Add ReGraph prior on success
        success_rate, reward_mean = self._method_prior(regraph or {}, prev_method, next_method)
        feats["method_success_prior"] = success_rate
        feats["method_reward_prior"] = reward_mean

        # PTX features (if variant dir contains result/ptx)
        ptx_path: Optional[Path] = None
        if variant_dir.exists():
            res_path = variant_dir / "result.json"
            if res_path.exists():
                try:
                    res_obj = json.loads(res_path.read_text(encoding="utf-8"))
                    ptx_val = res_obj.get("ptx_path")
                    if ptx_val:
                        ptx_path = Path(ptx_val)
                    ncu_metrics = res_obj.get("ncu_metrics", {}) or {}
                    feats.update(flatten_ncu_metrics(ncu_metrics))
                except Exception:
                    pass
            if not ptx_path:
                for cand in variant_dir.glob("*.ptx"):
                    ptx_path = cand
                    break
        if ptx_path and ptx_path.exists():
            feats.update(extract_ptx_features(ptx_path))

        return feats, target_speedup

    def load_examples(self, regraph: Optional[Dict[str, Any]]) -> List[OfflineExample]:
        examples: List[OfflineExample] = []
        for row in _load_jsonl(self.dataset_path):
            feats, target = self._gather_features(row, regraph)
            if feats:
                examples.append(OfflineExample(features=feats, target_speedup=target))
        return examples

    def train_and_save(self, regraph: Optional[Dict[str, Any]] = None) -> Optional[OfflineModel]:
        examples = self.load_examples(regraph)
        if not examples:
            return None
        model = self._fit_model(examples)
        model.to_json(self.model_path)
        return model


class OfflinePredictor:
    """Lightweight inference wrapper around OfflineModel."""

    def __init__(self, model: OfflineModel):
        self.model = model

    @classmethod
    def load(cls, path: Path) -> "OfflinePredictor":
        return cls(OfflineModel.from_json(path))

    def predict_for_node(
        self,
        node: Dict[str, Any],
        next_method: str,
        baseline_ms: float,
        regraph: Optional[Dict[str, Any]] = None,
    ) -> float:
        feats: Dict[str, float] = {}
        feats[f"method::{next_method}"] = 1.0
        prev_method = (node.get("last_method") or "START") if isinstance(node, dict) else "START"
        feats[f"context::{prev_method}"] = 1.0
        feats["baseline_ms"] = float(baseline_ms)

        # Node metrics/PTX features
        ptx_path = node.get("ptx_path") if isinstance(node, dict) else None
        if ptx_path:
            feats.update(extract_ptx_features(Path(ptx_path)))
        ncu = node.get("ncu") if isinstance(node, dict) else None
        if isinstance(ncu, dict):
            feats.update(flatten_ncu_metrics(ncu))

        # ReGraph priors
        if regraph and isinstance(regraph, dict):
            edges = regraph.get("edges", {}) if isinstance(regraph.get("edges", {}), dict) else {}
            stats = edges.get(prev_method, {}).get(next_method, {}) if isinstance(edges, dict) else {}
            cnt = float(stats.get("count", 0.0) or 0.0)
            succ = float(stats.get("success", 0.0) or 0.0)
            feats["method_success_prior"] = succ / max(1.0, cnt)
            feats["method_reward_prior"] = float(stats.get("mean_reward", 0.0) or 0.0)

        return self.model.predict(feats)
