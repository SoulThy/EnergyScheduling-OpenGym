from __future__ import annotations

"""
Legacy default configuration values for the simulator.

These constants mirror the original thesis codebase parameters. The main
`config.py` module should import from here and then allow overriding via
environment variables, so that:

- Importing the simulator with no env overrides reproduces the original
  behaviour.
- Experiments can tweak parameters (e.g., link speeds, transmission power)
  without losing the legacy baseline.
"""

from typing import Final, Tuple

# ---------------------------------------------------------------------------
# Parallel simulations and batteries
# ---------------------------------------------------------------------------

DEFAULT_MAX_PARALLEL_SIMULATIONS: Final[int] = 2

# Worker battery capacities (Wh) for nodes 1, 2 and 3.
DEFAULT_WORKER_BATTERY_CAPACITIES: Final[Tuple[int, int, int]] = (7, 8, 9)

# ---------------------------------------------------------------------------
# Network speeds (Mbit/s) — legacy values
# ---------------------------------------------------------------------------

DEFAULT_NET_SPEED_CLIENT_SCHEDULER_MBIT: Final[float] = 200.0
DEFAULT_NET_SPEED_SCHEDULER_WORKER_MBIT: Final[float] = 1000.0
DEFAULT_NET_SPEED_SCHEDULER_CLOUD_MBIT: Final[float] = 1000.0

# ---------------------------------------------------------------------------
# Radio power — legacy value
# ---------------------------------------------------------------------------

DEFAULT_POWER_MAX_TRANSMISSION_W: Final[float] = 0.2

