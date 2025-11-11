#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
state_manager.py
----------------
Incremental state persistence for crash recovery in subprocess workers.

Supports phases:
- "materialized": Source code has been written to disk
- "compiled": CUDA kernel has been compiled to PTX/cubin
- "timed": Kernel timing has completed successfully

State is saved to {workdir}/partial_state.json and updated incrementally.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def save_partial_state(workdir: Path, phase: str, data: Dict[str, Any]) -> None:
    """
    Save or update partial state for crash recovery.

    Args:
        workdir: Working directory for the variant
        phase: Current phase ("materialized", "compiled", "timed")
        data: Phase-specific data to save

    State file structure:
    {
        "phases": {
            "materialized": {"timestamp": "...", "data": {...}},
            "compiled": {"timestamp": "...", "data": {...}},
            "timed": {"timestamp": "...", "data": {...}}
        },
        "last_phase": "compiled",
        "last_update": "2025-11-11T12:34:56"
    }
    """
    state_path = workdir / "partial_state.json"

    # Load existing state if present
    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            # Corrupted state file, start fresh
            state = {"phases": {}}
    else:
        state = {"phases": {}}

    # Update phase data
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    state["phases"][phase] = {
        "timestamp": timestamp,
        "data": data
    }
    state["last_phase"] = phase
    state["last_update"] = timestamp

    # Write atomically (write to temp, then rename)
    temp_path = workdir / "partial_state.json.tmp"
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)

    temp_path.replace(state_path)


def load_partial_state(workdir: Path) -> Optional[Dict[str, Any]]:
    """
    Load partial state from workdir.

    Args:
        workdir: Working directory for the variant

    Returns:
        State dict if exists, None otherwise
    """
    state_path = workdir / "partial_state.json"

    if not state_path.exists():
        return None

    try:
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_latest_phase_data(workdir: Path) -> Optional[Dict[str, Any]]:
    """
    Get data from the most recent successful phase.

    Args:
        workdir: Working directory for the variant

    Returns:
        Dict with "phase" and "data" keys, or None if no state
    """
    state = load_partial_state(workdir)
    if not state:
        return None

    last_phase = state.get("last_phase")
    if not last_phase or last_phase not in state.get("phases", {}):
        return None

    return {
        "phase": last_phase,
        "data": state["phases"][last_phase]["data"],
        "timestamp": state["phases"][last_phase]["timestamp"]
    }
