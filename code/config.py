from __future__ import annotations

import os
from typing import Final, Tuple

DEFAULT_MAX_PARALLEL_SIMULATIONS: Final[int] = 2

# Single knob to control how many simulations run in parallel across all runners.
# Can be overridden at runtime with the MAX_PARALLEL_SIMULATIONS environment variable.
MAX_PARALLEL_SIMULATIONS: Final[int] = int(
    os.getenv("MAX_PARALLEL_SIMULATIONS", str(DEFAULT_MAX_PARALLEL_SIMULATIONS))
)

DEFAULT_WORKER_BATTERY_CAPACITIES: Final[Tuple[int, int, int]] = (7, 8, 9)


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
# The defaults below reduce bandwidth compared to the original code (which
# used 200/1000 Mbit/s) to make job-communication costs more impactful while
# remaining in a plausible range for fog devices.

DEFAULT_NET_SPEED_CLIENT_SCHEDULER_MBIT: Final[float] = 10.0
DEFAULT_NET_SPEED_SCHEDULER_WORKER_MBIT: Final[float] = 20.0

# The thesis focuses on only workers, so it does not really care about this value.
DEFAULT_NET_SPEED_SCHEDULER_CLOUD_MBIT: Final[float] = 50.0

NET_SPEED_CLIENT_SCHEDULER_MBIT: Final[float] = float(
    os.getenv("NET_SPEED_CLIENT_SCHEDULER_MBIT", str(DEFAULT_NET_SPEED_CLIENT_SCHEDULER_MBIT))
)
NET_SPEED_SCHEDULER_WORKER_MBIT: Final[float] = float(
    os.getenv("NET_SPEED_SCHEDULER_WORKER_MBIT", str(DEFAULT_NET_SPEED_SCHEDULER_WORKER_MBIT))
)
NET_SPEED_SCHEDULER_CLOUD_MBIT: Final[float] = float(
    os.getenv("NET_SPEED_SCHEDULER_CLOUD_MBIT", str(DEFAULT_NET_SPEED_SCHEDULER_CLOUD_MBIT))
)

# Maximum radio transmission power (W). This is used to convert job/probes
# transmission times into energy. Increasing this value or reducing link
# speeds above will make job and probes communication more energy-expensive.
DEFAULT_POWER_MAX_TRANSMISSION_W: Final[float] = 0.5

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

PROBE_SIZE_BYTES: Final[int] = 200
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
