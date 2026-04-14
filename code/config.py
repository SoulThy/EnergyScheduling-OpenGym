from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Tuple

from default_config import (
    DEFAULT_MAX_PARALLEL_SIMULATIONS,
    DEFAULT_POWER_MAX_TRANSMISSION_W,
    DEFAULT_WORKER_BATTERY_CAPACITIES,
    DEFAULT_NET_SPEED_CLIENT_SCHEDULER_MBIT,
    DEFAULT_NET_SPEED_SCHEDULER_WORKER_MBIT,
    DEFAULT_NET_SPEED_SCHEDULER_CLOUD_MBIT,
)

# Single knob to control how many simulations run in parallel across all runners.
# Can be overridden at runtime with the MAX_PARALLEL_SIMULATIONS environment variable.
MAX_PARALLEL_SIMULATIONS: Final[int] = int(
    os.getenv("MAX_PARALLEL_SIMULATIONS", str(DEFAULT_MAX_PARALLEL_SIMULATIONS))
)

# High-level model selector (LEGACY, SMALL_JOBS_V2, SMALL_JOBS_V3, ...).
MODEL_VERSION: Final[str] = os.getenv("MODEL_VERSION", "LEGACY")

# Repository layout: ``code/config.py`` -> parent is repo root.
_CODE_DIR: Final[Path] = Path(__file__).resolve().parent
_REPO_ROOT: Final[Path] = _CODE_DIR.parent


def get_results_data_dir() -> Path:
    """Directory that contains ``_log/`` (same root ``ServiceDataStorage`` uses).

    Default: ``<repo>/results/data`` so paths do not depend on the process cwd.

    Override: set env ``RESULTS_DATA_DIR`` to the absolute path of that ``data``
    directory (the folder that should contain ``_log``).
    """
    override = os.getenv("RESULTS_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (_REPO_ROOT / "results" / "data").resolve()


_RESULTS_DATA_PATH: Final[Path] = get_results_data_dir()
RESULTS_DATA_DIR: Final[str] = str(_RESULTS_DATA_PATH)
RESULTS_TABLE_DIR: Final[str] = str(_RESULTS_DATA_PATH.parent / "table")
RESULTS_PLOT_DIR: Final[str] = str(_RESULTS_DATA_PATH.parent / "plot")


def _parse_worker_battery_capacities(env_value: str | None) -> Tuple[int, int, int] | None:
    if not env_value:
        return None

    parts = [p.strip() for p in env_value.split(",") if p.strip()]
    if len(parts) != 3:
        return None

    try:
        values = tuple(int(p) for p in parts)
    except ValueError:
        return None

    return values  # type: ignore[return-value]


# Worker battery capacities (Wh) for nodes 1, 2 and 3.
#
# The order is (node_1_batt, node_2_batt, node_3_batt).
# - If the WORKER_BATTERY_CAPACITIES env var is set and valid, it is used.
# - Otherwise, DEFAULT_WORKER_BATTERY_CAPACITIES is used.
WORKER_BATTERY_CAPACITIES: Final[Tuple[int, int, int]] = (
    _parse_worker_battery_capacities(os.getenv("WORKER_BATTERY_CAPACITIES"))
    or DEFAULT_WORKER_BATTERY_CAPACITIES
)

# ---------------------------------------------------------------------------
# Network and radio parameters
# ---------------------------------------------------------------------------
#
# These constants control the effective link speeds (in Mbit/s) used in the
# simulator and the maximum radio transmission power. Adjusting them lets us
# emulate different fog/edge environments without touching the core logic.
#
# The defaults below are taken from `default_config.py` and reproduce the
# original code (200/1000/1000 Mbit/s). Environment variables can override
# them to explore alternative regimes (e.g., slower fog links).

NET_SPEED_CLIENT_SCHEDULER_MBIT: Final[float] = float(
    os.getenv("NET_SPEED_CLIENT_SCHEDULER_MBIT", str(DEFAULT_NET_SPEED_CLIENT_SCHEDULER_MBIT))
)
NET_SPEED_SCHEDULER_WORKER_MBIT: Final[float] = float(
    os.getenv("NET_SPEED_SCHEDULER_WORKER_MBIT", str(DEFAULT_NET_SPEED_SCHEDULER_WORKER_MBIT))
)
NET_SPEED_SCHEDULER_CLOUD_MBIT: Final[float] = float(
    os.getenv("NET_SPEED_SCHEDULER_CLOUD_MBIT", str(DEFAULT_NET_SPEED_SCHEDULER_CLOUD_MBIT))
)

POWER_MAX_TRANSMISSION_W: Final[float] = float(
    os.getenv("POWER_MAX_TRANSMISSION_W", str(DEFAULT_POWER_MAX_TRANSMISSION_W))
)

# ---------------------------------------------------------------------------
# Probing energy cost
# ---------------------------------------------------------------------------
#
# We assume:
# - Probes have a fixed payload size (in bytes), e.g. a 200 B control packet.
# - They use the same radio and link as scheduler→worker transmissions.
# - There is an additional "crossfactor" energy cost that is independent of
#   payload size (protocol stack, wake-up, etc.).
#
# The per-probe energy (in joules) is:
#   E_probe_J = P_tx * (probe_bits / (speed_bits_per_s)) + E_crossfactor_J
# and we expose the equivalent in Wh for use in the battery model.

PROBE_SIZE_BYTES: Final[int] = int(os.getenv("PROBE_SIZE_BYTES", "200"))
PROBE_CROSSFACTOR_J: Final[float] = 0.0002  # 0.2 mJ total (e.g., 0.1 mJ RX + 0.1 mJ TX)

def _compute_default_probing_energy_cost_wh() -> float:
    """Compute default per-probe energy cost (Wh) from radio parameters."""
    probe_bits = PROBE_SIZE_BYTES * 8
    speed_bits_per_s = NET_SPEED_SCHEDULER_WORKER_MBIT * 1e6
    if speed_bits_per_s <= 0.0:
        return 0.0
    time_s = probe_bits / speed_bits_per_s
    energy_j = POWER_MAX_TRANSMISSION_W * time_s + PROBE_CROSSFACTOR_J
    return energy_j / 3600.0


DEFAULT_PROBING_ENERGY_COST_WH: Final[float] = _compute_default_probing_energy_cost_wh()

PROBING_ENERGY_COST_WH: Final[float] = float(
    os.getenv("PROBING_ENERGY_COST_WH", str(DEFAULT_PROBING_ENERGY_COST_WH))
)

# Intermittent probing control:
# - 1 => probe on every scheduler state request (legacy behavior)
# - K>1 => probe once every K scheduler state requests, reusing cached worker
#   state features for the intermediate requests.
PROBING_STATE_REFRESH_EVERY_K_JOBS: Final[int] = max(
    1, int(os.getenv("PROBING_STATE_REFRESH_EVERY_K_JOBS", "1"))
)

# ---------------------------------------------------------------------------
# SCORE_SIMPLE heuristic policy weights
# ---------------------------------------------------------------------------
# Scheduler score for a worker i:
#   score_i = wq * Qi + wb * Bi
# Dead workers are excluded (hard filter), same spirit as least-loaded baselines.
# where:
#   Qi = normalized queue load
#   Bi = battery penalty (1 - residual_percentage)
SCORE_SIMPLE_WEIGHT_Q: Final[float] = float(os.getenv("SCORE_SIMPLE_WEIGHT_Q", "0.5"))
SCORE_SIMPLE_WEIGHT_B: Final[float] = float(os.getenv("SCORE_SIMPLE_WEIGHT_B", "0.3"))
