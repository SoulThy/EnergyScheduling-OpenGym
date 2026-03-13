#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing
import os
from datetime import datetime
import datetime as dt
from typing import List

"""
Sweep over different probing packet sizes to study how probing energy share and
job success ratio change as probes become more/less expensive.

This runner is intentionally self-contained and avoids importing `config` at
module import time so that each child process can set `PROBE_SIZE_BYTES`
through the environment before the simulator and energy model are imported.
"""

MODULE = "ProbeSizeSweep"

SIMULATION_TIME = 10000
SIMULATION_TOTAL_TIME = SIMULATION_TIME
SOLAR_PANEL_ENABLED = False

SESSION_ID = datetime.now().strftime("%Y%m%d")

# We fix the learning configuration to the standard D-SARSA setup used in the
# other runners; imports of Node, SolarPanelSpec, build_simulator are delayed
# until inside the child processes.
ALPHA = 0.50

# ---------------------------------------------------------------------------
# Probe-size sweep configuration
# ---------------------------------------------------------------------------
#
# We choose a small set of representative probe sizes instead of a dense grid:
# - Start from the default 200 B control packet.
# - Increase approximately geometrically up to around the order of magnitude
#   of typical job payloads (tens of kB).
#
# This yields clear curves without oversampling:
PROBE_SIZES_BYTES: List[int] = [
    200,      # baseline
    400,
    800,
    1600,
    3200,
    6400,
    12_800,
    25_600,
    51_200,   # ~50 kB
]


def get_die_after(node_id: int) -> int:
    if node_id == 1:
        return 4000
    return 0


def run_simulation_for_probe_size(step_index: int, probe_size_bytes: int) -> None:
    """
    Run a single simulation for a given probing packet size.

    We set PROBE_SIZE_BYTES via the environment before importing any modules
    that depend on `config`, so that the probing energy model and sim_config
    metadata are consistent for this run.
    """
    # Configure probe size for this child process before importing simulator modules.
    os.environ["PROBE_SIZE_BYTES"] = str(probe_size_bytes)

    # Delayed imports so that they see the per-process PROBE_SIZE_BYTES.
    from log import Log
    from models import SolarPanelSpec
    from node import Node
    from sim_builder import build_simulator

    # Derive a descriptive session id that encodes alpha and probe size.
    session_id = f"{SESSION_ID}_{ALPHA:.2f}_{probe_size_bytes}B_PS"

    # Solar panel specs (copied from other D-SARSA runners for consistency).
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
        latitude_list: List[float] | None = None,
        longitude_list: List[float] | None = None,
        altitude_list: List[float] | None = None,
    ) -> List[SolarPanelSpec]:
        panels_specs: List[SolarPanelSpec] = []
        for i in range(n_nodes):
            if latitude_list is not None and longitude_list is not None and altitude_list is not None:
                panels_specs.append(
                    SolarPanelSpec(
                        node_id=i,
                        latitude=latitude_list[i],
                        longitude=longitude_list[i],
                        altitude=altitude_list[i],
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
            else:
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
        latitude_list=None,
        longitude_list=None,
        altitude_list=None,
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
        f"Starting probe-size sweep step={step_index}, "
        f"probe_size_bytes={probe_size_bytes}, session_id={session_id}",
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

    env.run(until=SIMULATION_TOTAL_TIME)
    Log.minfo(
        MODULE,
        f"Simulation ended: SESSION_ID={session_id}, LEARNING_TYPE={Node.LearningType.D_SARSA.name}, "
        f"NO_LEARNING_POLICY={Node.NoLearningPolicy.RANDOM.name}, "
        f"ACTIONS_SPACE={Node.ActionsSpace.ONLY_WORKERS.name}, "
        f"ALPHA={ALPHA}, probe_size_bytes={probe_size_bytes}",
    )

    data_storage.done_simulation()


if __name__ == "__main__":
    # Use the same convention as `config.MAX_PARALLEL_SIMULATIONS` but avoid
    # importing config here to keep PROBE_SIZE_BYTES controllable per child.
    num_cores = int(os.getenv("MAX_PARALLEL_SIMULATIONS", "4"))

    processes: list[multiprocessing.Process] = []
    for step_index, probe_size in enumerate(PROBE_SIZES_BYTES):
        p = multiprocessing.Process(
            target=run_simulation_for_probe_size,
            args=(step_index, probe_size),
        )
        processes.append(p)
        p.start()

        if len(processes) >= num_cores:
            for proc in processes:
                proc.join()
            processes = []

    for proc in processes:
        proc.join()

    print("All probe-size sweep simulations completed.")

