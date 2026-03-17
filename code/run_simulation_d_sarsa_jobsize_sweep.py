#!/usr/bin/env python3
from __future__ import annotations

import multiprocessing
from datetime import datetime
import datetime as dt

from config import MAX_PARALLEL_SIMULATIONS
from log import Log
from models import SolarPanelSpec
from node import Node
from sim_builder import build_simulator

MODULE = "JobSizeSweep"

SIMULATION_TIME = 10000
SIMULATION_TOTAL_TIME = SIMULATION_TIME
SOLAR_PANEL_ENABLED = False

SESSION_ID = datetime.now().strftime("%Y%m%d")
LEARNING_TYPE = Node.LearningType.D_SARSA
NO_LEARNING_POLICY = Node.NoLearningPolicy.RANDOM
ACTIONS_SPACE = Node.ActionsSpace.ONLY_WORKERS

ALPHA = 0.50

# Base job sizes (MB) for periodic and exponential jobs, matching sim_builder defaults.
BASE_PERIODIC_SIZE_MB = 0.050  # 50 kB
BASE_EXPONENTIAL_SIZE_MB = 0.100  # 100 kB

# Byte-to-MB conversion (1 MB = 1e6 bytes).
BYTE_TO_MB = 1_000_000.0

# Decrement per sweep step: 996 bytes.
DELTA_SIZE_MB = 996.0 / BYTE_TO_MB

# Minimum job size: 200 bytes.
MIN_JOB_SIZE_MB = 200.0 / BYTE_TO_MB

N_SWEEP_STEPS = 50


def gen_spec_solar_panels(
    n_nodes,
    simulation_time,
    start_date_str,
    latitude=41.80,
    longitude=12.36,
    altitude=5.0,
    tilt_list=None,
    azimuth_list=None,
    efficiency=0.20,
    panel_surface_m2=1.0,
    station_file: str = "",
    latitude_list=None,
    longitude_list=None,
    altitude_list=None,
):
    panels_specs = []
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


def get_die_after(node_id: int) -> int:
    if node_id == 1:
        return 4000
    return 0


def _compute_avg_job_size_mb(periodic_size_mb: float, exp_size_mb: float) -> float:
    """
    Compute the expected average job size given the current payload sizes and
    fixed arrival rates:
      periodic rates: 60, 30, 15 fps -> 105 total
      exponential rate: 10 fps       -> 10 total
    """
    p_periodic = 105.0 / 115.0
    p_exponential = 1.0 - p_periodic
    return p_periodic * periodic_size_mb + p_exponential * exp_size_mb


def run_simulation_for_step(step_index: int) -> None:
    """
    Run a single simulation for the given sweep step:
    periodic and exponential job sizes are reduced by step_index * DELTA_SIZE_MB.
    """
    periodic_size_mb = max(BASE_PERIODIC_SIZE_MB - step_index * DELTA_SIZE_MB, MIN_JOB_SIZE_MB)
    exp_size_mb = max(BASE_EXPONENTIAL_SIZE_MB - step_index * DELTA_SIZE_MB, MIN_JOB_SIZE_MB)

    avg_job_size_mb = _compute_avg_job_size_mb(periodic_size_mb, exp_size_mb)
    avg_job_size_kb = avg_job_size_mb * 1000.0

    session_id = f"{SESSION_ID}_{ALPHA:.2f}_{avg_job_size_kb:.0f}K_P"

    tilt_list = [i for i in range(0, 72, 8)]
    azimuth_list = [i for i in range(0, 360, 40)]
    nodes_id_list = [1, 2, 3]
    tilt_list = [tilt_list[i] for i in nodes_id_list]
    azimuth_list = [azimuth_list[i] for i in nodes_id_list]

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
        f"Starting sweep step={step_index}, periodic_size_mb={periodic_size_mb:.6f}, "
        f"exp_size_mb={exp_size_mb:.6f}, avg_job_size_mb={avg_job_size_mb:.6f}, "
        f"session_id={session_id}",
    )

    env, nodes, cloud, discovery, data_storage = build_simulator(
        sim_time=SIMULATION_TIME,
        session_uid=SESSION_ID,
        data_storage_session_id=session_id,
        learning_type=LEARNING_TYPE,
        no_learning_policy=NO_LEARNING_POLICY,
        actions_space=ACTIONS_SPACE,
        state_type=Node.StateType.JOB_TYPE,
        reward_alpha=ALPHA,
        episode_length=60,
        get_die_after_seconds=get_die_after,
        solar_panel_enabled=SOLAR_PANEL_ENABLED,
        solar_panel_spec_by_node_id=solar_panel_spec_by_node_id if SOLAR_PANEL_ENABLED else None,
        job_periodic_payload_sizes_mbytes=(periodic_size_mb, periodic_size_mb, periodic_size_mb),
        job_exponential_payload_sizes_mbytes=[exp_size_mb],
    )

    env.run(until=SIMULATION_TOTAL_TIME)
    Log.minfo(
        MODULE,
        f"Simulation ended: SESSION_ID={session_id}, LEARNING_TYPE={LEARNING_TYPE.name}, "
        f"NO_LEARNING_POLICY={NO_LEARNING_POLICY.name}, ACTIONS_SPACE={ACTIONS_SPACE.name}, "
        f"ALPHA={ALPHA}, avg_job_size_mb={avg_job_size_mb:.6f}",
    )

    data_storage.done_simulation()


if __name__ == "__main__":
    num_cores = MAX_PARALLEL_SIMULATIONS

    processes: list[multiprocessing.Process] = []
    for step in range(N_SWEEP_STEPS):
        p = multiprocessing.Process(target=run_simulation_for_step, args=(step,))
        processes.append(p)
        p.start()

        if len(processes) >= num_cores:
            for proc in processes:
                proc.join()
            processes = []

    for proc in processes:
        proc.join()

    print("All sweep simulations completed.")

