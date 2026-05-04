#  LBFC: Load–Battery–Failure–Cloud aware heuristic (stress + water-filling offload).
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

from __future__ import annotations

import math
import os
import zlib
from typing import Sequence

from log import Log

MODULE = "LBFC"

# Stress onset: mean normalized queue above this triggers u > 0 (after EMA smoothing in Node).
TAU: float = 0.1

# EMA on instantaneous mean queue for LBFC: new = alpha*q_inst + (1-alpha)*old.
LBFC_EMA_ALPHA: float = float(os.getenv("LBFC_EMA_ALPHA", "0.1"))

# Optional debug: first N scheduler decisions log u and p_k (set LBFC_DEBUG_STEPS=0 to disable).
_DEBUG_STEPS_INITIAL: int = max(0, int(os.getenv("LBFC_DEBUG_STEPS", "48")))


def mean_normalized_load_active_workers(
    loads_cluster: Sequence[Sequence[int]],
    workers: Sequence[object],
    max_jobs_in_queue: int,
) -> float:
    """Mean Qi over workers with ``is_died() == False``. Same Qi definition as SCORE_SIMPLE."""
    max_q = max(1, int(max_jobs_in_queue))
    qi_values: list[float] = []
    for idx, worker in enumerate(workers):
        if idx >= len(loads_cluster):
            break
        if getattr(worker, "is_died")():
            continue
        qi = float(sum(loads_cluster[idx])) / float(max_q)
        qi_values.append(qi)
    if not qi_values:
        # No alive workers: treat as maximum load stress.
        return 1.0
    return sum(qi_values) / float(len(qi_values))


def stress_u(mean_q: float, tau: float = TAU) -> float:
    """u = clip((mean_q - tau) / (1 - tau), 0, 1); u = 0 if mean_q < tau."""
    if mean_q < tau:
        return 0.0
    denom = max(1e-12, 1.0 - tau)
    u = (mean_q - tau) / denom
    return float(min(1.0, max(0.0, u)))


def offload_fractions_lbfc(u: float, expected_service_weights: Sequence[float]) -> list[float]:
    """
    LBFC offload probabilities (stress ``u`` in ``[0, 1]``).

    * Heaviest class ``K-1`` always prefers cloud: ``p_{K-1} = 1`` (fixed; it
      does **not** consume width on the ``u`` axis).
    * Partition ``[0, 1]`` using **only** weights ``W_0 .. W_{K-2}`` with
      ``W_tot = sum_{t=0}^{K-2} W_t``. Segments run heaviest-to-lightest among
      those types (``K-2``, then ``K-3``, …, ``0``): within each segment the
      corresponding ``p_t`` ramps from 0 to 1 while earlier types in that order
      stay saturated at 1.

    ``expected_service_weights`` must follow job types ``0 .. K-1``. ``u >= 1``
    implies all classes prefer cloud.
    """
    k = len(expected_service_weights)
    if k <= 0:
        return []
    if k == 1:
        return [1.0]
    if u >= 1.0:
        return [1.0] * k

    weights_partial = [max(1e-12, float(w)) for w in expected_service_weights[:-1]]
    w_tot = float(sum(weights_partial))
    if w_tot <= 0.0:
        return [1.0] * k

    boundaries: list[float] = [0.0]
    for j in range(k - 1):
        boundaries.append(boundaries[-1] + weights_partial[k - 2 - j] / w_tot)
    boundaries[-1] = 1.0

    u_clamped = min(1.0, max(0.0, float(u)))
    probs = [0.0] * k
    probs[k - 1] = 1.0

    for typ in range(0, k - 1):
        j = (k - 2) - typ
        left = boundaries[j]
        right = boundaries[j + 1]
        span = max(right - left, 1e-15)
        if u_clamped <= left:
            probs[typ] = 0.0
        elif u_clamped >= right:
            probs[typ] = 1.0
        else:
            probs[typ] = (u_clamped - left) / span

    return probs


def stable_unit_interval(job_uid: str) -> float:
    """Reproducible [0, 1) from job uid (independent of PYTHONHASHSEED)."""
    c = zlib.crc32(job_uid.encode("utf-8")) & 0xFFFFFFFF
    return c / 4294967296.0


def prefer_cloud(job_uid: str, job_type: int, probs: Sequence[float]) -> bool:
    """Route to cloud with probability p[job_type] using a stable per-job coin flip."""
    if not probs:
        return False
    jt = min(max(0, int(job_type)), len(probs) - 1)
    r = stable_unit_interval(job_uid)
    return r < probs[jt]


class LBFC:
    """Namespace + debug counter for LBFC scheduling helpers."""

    _debug_remaining: int = _DEBUG_STEPS_INITIAL

    @staticmethod
    def maybe_log_debug(
        sim_time: float,
        q_inst: float,
        ema_q: float,
        u: float,
        probs: Sequence[float],
    ) -> None:
        if LBFC._debug_remaining <= 0:
            return
        LBFC._debug_remaining -= 1
        p_str = "[" + ", ".join(f"{p:.4f}" for p in probs) + "]"
        Log.minfo(
            MODULE,
            f"[LBFC debug] t={sim_time:.3f} q_inst={q_inst:.4f} ema_Q={ema_q:.4f} u={u:.4f} p_k={p_str}",
        )

    @staticmethod
    def reset_debug_counter() -> None:
        LBFC._debug_remaining = max(0, int(os.getenv("LBFC_DEBUG_STEPS", str(_DEBUG_STEPS_INITIAL))))
