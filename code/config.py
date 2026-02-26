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
