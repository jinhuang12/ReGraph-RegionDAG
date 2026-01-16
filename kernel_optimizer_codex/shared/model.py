from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeviceProfile:
    """
    Minimal device roofline profile used by kernel_opt_tooling.py.

    All fields are optional; if unset, roofline classification falls back to
    "unknown".
    """

    peak_flops_tflops: Optional[float] = None
    mem_bandwidth_gbps: Optional[float] = None
    hbm_bw_gbps: Optional[float] = None

