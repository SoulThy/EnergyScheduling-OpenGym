from __future__ import annotations

import argparse

from run_simulation_d_sarsa_cloud_failure import run_simulation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single D-SARSA WORKERS_OR_CLOUD simulation with the built-in failure schedule."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Reward trade-off alpha (FPS vs battery). Higher => more FPS priority.",
    )
    args = parser.parse_args()

    run_simulation(args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
