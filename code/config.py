from __future__ import annotations

import os
from typing import Final

DEFAULT_MAX_PARALLEL_SIMULATIONS: Final[int] = 2

# Single knob to control how many simulations run in parallel across all runners.
# Can be overridden at runtime with the MAX_PARALLEL_SIMULATIONS environment variable.
MAX_PARALLEL_SIMULATIONS: Final[int] = int(
    os.getenv("MAX_PARALLEL_SIMULATIONS", str(DEFAULT_MAX_PARALLEL_SIMULATIONS))
)

