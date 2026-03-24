#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing
import os
import random
from datetime import datetime
from typing import List

"""
Sweep intermittent probing refresh interval K for SMALL_JOBS_V3.

Definitions:
- K=1: probe state on every scheduler state request (legacy behavior).
- K=2: probe one job, reuse previous probed state for the next.
- K=3: probe once every 3 jobs, etc.

Each run writes a regular `log.db` under:
results/data/_log/learning/D_SARSA/ONLY_WORKERS/SIM_SMALL_JOBS_V3/<session_id>/log.db
with session_id containing both probe packet size and K.
"""

MODULE = "SmallJobsV3KSweep"

SIMULATION_TIME = 10000
SIMULATION_TOTAL_TIME = SIMULATION_TIME
SOLAR_PANEL_ENABLED = False

SESSION_ID = datetime.now().strftime("%Y%m%d")
ALPHA = 0.50
PROBE_SIZE_BYTES = 200
MODEL_VERSION = "SMALL_JOBS_V3"
RANDOM_SEED = 42


def get_die_after(node_id: int) -> int:
    if node_id == 1:
        return 4000
    return 0


def _k_values(max_k: int) -> List[int]:
    if max_k < 1:
        raise ValueError("max_k must be >= 1")
    return list(range(1, max_k + 1))


def run_simulation_for_k(step_index: int, k_value: int) -> None:
    os.environ["MODEL_VERSION"] = MODEL_VERSION
    os.environ["PROBE_SIZE_BYTES"] = str(PROBE_SIZE_BYTES)
    os.environ["PROBING_STATE_REFRESH_EVERY_K_JOBS"] = str(k_value)

    random.seed(RANDOM_SEED)
    import numpy as np

    np.random.seed(RANDOM_SEED)

    from log import Log
    from models import SolarPanelSpec
    from node import Node
    from sim_builder import build_simulator

    session_id = f"{SESSION_ID}_{ALPHA:.2f}_{PROBE_SIZE_BYTES}B_K{k_value}"

    tilt_list = [i for i in range(0, 72, 8)]
    azimuth_list = [i for i in range(0, 360, 40)]
    nodes_id_list = [1, 2, 3]
    tilt_list = [tilt_list[i] for i in nodes_id_list]
    azimuth_list = [azimuth_list[i] for i in nodes_id_list]

    def gen_spec_solar_panels(
        n_nodes: int,
        simulation_time: int,
        start_date_str: str,
        latitude: float = 41.80,
        longitude: float = 12.36,
        altitude: float = 5.0,
        tilt_list: List[float] | None = None,
        azimuth_list: List[float] | None = None,
        efficiency: float = 0.20,
        panel_surface_m2: float = 1.0,
        station_file: str = "",
    ) -> List[SolarPanelSpec]:
        panels_specs: List[SolarPanelSpec] = []
        for i in range(n_nodes):
            panels_specs.append(
                SolarPanelSpec(
                    node_id=i,
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude,
                    timezone=dt.timezone.utc,
                    start_date_str=start_date_str,
                    simulation_time_seconds=simulation_time,
                    tilt=tilt_list[i],
                    azimuth=azimuth_list[i],
                    efficiency=efficiency,
                    panel_surface_m2=panel_surface_m2,
                    station_file=station_file,
                )
            )
        return panels_specs

    panels_mapping = gen_spec_solar_panels(
        len(tilt_list),
        SIMULATION_TIME,
        "12-01-2020",
        tilt_list=tilt_list,
        azimuth_list=azimuth_list,
        efficiency=0.2,
        panel_surface_m2=0.4 * 0.4,
        station_file="723170TYA.CSV",
    )
    solar_panel_spec_by_node_id = {}
    if SOLAR_PANEL_ENABLED:
        for panel_spec in panels_mapping:
            solar_panel_spec_by_node_id[panel_spec.node_id] = panel_spec

    Log.minfo(
        MODULE,
        f"Starting K-sweep step={step_index}, K={k_value}, probe_size={PROBE_SIZE_BYTES}B, session_id={session_id}",
    )

    env, nodes, cloud, discovery, data_storage = build_simulator(
        sim_time=SIMULATION_TIME,
        session_uid=SESSION_ID,
        data_storage_session_id=session_id,
        learning_type=Node.LearningType.D_SARSA,
        no_learning_policy=Node.NoLearningPolicy.RANDOM,
        actions_space=Node.ActionsSpace.ONLY_WORKERS,
        state_type=Node.StateType.JOB_TYPE,
        reward_alpha=ALPHA,
        episode_length=60,
        get_die_after_seconds=get_die_after,
        solar_panel_enabled=SOLAR_PANEL_ENABLED,
        solar_panel_spec_by_node_id=solar_panel_spec_by_node_id if SOLAR_PANEL_ENABLED else None,
    )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env.run(until=SIMULATION_TOTAL_TIME)

    data_storage.done_simulation()

    Log.minfo(
        MODULE,
        f"Simulation ended: session_id={session_id}, K={k_value}, ALPHA={ALPHA}, probe_size={PROBE_SIZE_BYTES}B",
    )


def main() -> None:
    # Import here (not at module import time) to keep behavior aligned with other
    # sweep runners that delay config-dependent imports.
    from config import MAX_PARALLEL_SIMULATIONS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-k",
        type=int,
        default=9,
        help=(
            "Sweep K from 1 to max-k inclusive "
            "(default: 9 => K = 1..9, multiple of 3 recommended)."
        ),
    )
    parser.add_argument(
        "--only-k",
        type=int,
        default=None,
        help="Run only one K value (debug).",
    )
    args = parser.parse_args()

    if args.only_k is not None and args.only_k < 1:
        raise ValueError("--only-k must be >= 1")

    k_values = [args.only_k] if args.only_k is not None else _k_values(args.max_k)

    num_cores = MAX_PARALLEL_SIMULATIONS
    # Use spawned children so each process imports config fresh after we set
    # per-run env vars (MODEL_VERSION, PROBE_SIZE_BYTES, K).
    ctx = multiprocessing.get_context("spawn")
    processes: list[multiprocessing.Process] = []

    for step_index, k_value in enumerate(k_values):
        p = ctx.Process(
            target=run_simulation_for_k,
            args=(step_index, k_value),
        )
        processes.append(p)
        p.start()

        if len(processes) >= num_cores:
            for proc in processes:
                proc.join()
            processes = []

    for proc in processes:
        proc.join()

    print("All K-sweep simulations completed.")


if __name__ == "__main__":
    main()

