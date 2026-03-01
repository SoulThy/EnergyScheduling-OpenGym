#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

from __future__ import annotations

import os
import shutil
import signal
import sys
import multiprocessing
from datetime import datetime
import datetime as dt

from config import MAX_PARALLEL_SIMULATIONS
from log import Log
from models import SolarPanelSpec
from node import Node
from sim_builder import build_simulator

"""
Run the simulation of deadline scheduling
"""

MODULE = "Main"

SIMULATION_TIME = 10000 #int(2 * 24 * 3600)
SIMULATION_TOTAL_TIME = SIMULATION_TIME
SOLAR_PANEL_ENABLED = False

SESSION_ID = datetime.now().strftime("%Y%m%d")
LEARNING_TYPE = Node.LearningType.D_SARSA
NO_LEARNING_POLICY = Node.NoLearningPolicy.RANDOM
ACTIONS_SPACE = Node.ActionsSpace.ONLY_WORKERS


ALPHA_INCREMENT = 0.05

# Define a function to run the simulation for a given alpha
def run_simulation(alpha):
    session_id = f'{SESSION_ID}_{alpha:.2f}'

    tilt_list = [i for i in range(0, 72, 8)]
    azimuth_list = [i for i in range(0, 360, 40)]
    nodes_id_list = [1, 2, 3]
    tilt_list = [tilt_list[i] for i in nodes_id_list]
    azimuth_list = [azimuth_list[i] for i in nodes_id_list]

    panels_mapping = gen_spec_solar_panels(
        len(tilt_list), SIMULATION_TIME, "12-01-2020",
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

    env, nodes, cloud, discovery, data_storage = build_simulator(
        sim_time=SIMULATION_TIME,
        session_uid=SESSION_ID,
        data_storage_session_id=session_id,
        learning_type=LEARNING_TYPE,
        no_learning_policy=NO_LEARNING_POLICY,
        actions_space=ACTIONS_SPACE,
        state_type=Node.StateType.JOB_TYPE,
        reward_alpha=alpha,
        episode_length=60,
        get_die_after_seconds=get_die_after,
        solar_panel_enabled=SOLAR_PANEL_ENABLED,
        solar_panel_spec_by_node_id=solar_panel_spec_by_node_id if SOLAR_PANEL_ENABLED else None,
    )

    Log.minfo(MODULE, f"Started simulation for alpha={alpha}")
    env.run(until=SIMULATION_TOTAL_TIME)
    Log.minfo(MODULE, f"Simulation ended: SESSION_ID={session_id}, LEARNING_TYPE={LEARNING_TYPE.name}, "
                      f"NO_LEARNING_POLICY={NO_LEARNING_POLICY.name}, ACTIONS_SPACE={ACTIONS_SPACE.name}, "
                      f"ALPHA={alpha}")

    data_storage.done_simulation()


def get_die_after(node_id):
    if node_id == 1:
        return 4000
    return 0


def gen_spec_solar_panels(n_nodes, simulation_time, start_date_str, latitude=41.80, longitude=12.36, altitude=5.0,
                        tilt_list = None, azimuth_list = None,
                        efficiency=0.20, panel_surface_m2=1.0,
                        station_file: str = "",
                        latitude_list = None,
                        longitude_list = None,
                        altitude_list = None):
    panels_specs = []
    for i in range(n_nodes):
        if latitude_list is not None and longitude_list is not None and altitude_list is not None:
            panels_specs.append(SolarPanelSpec(node_id=i, latitude=latitude_list[i], longitude=longitude_list[i],
                                                altitude=altitude_list[i], timezone=dt.timezone.utc,
                                                start_date_str=start_date_str,
                                                simulation_time_seconds=simulation_time, tilt=tilt_list[i],
                                                azimuth=azimuth_list[i], efficiency=efficiency,
                                                panel_surface_m2=panel_surface_m2,
                                                station_file=station_file))
        else:
            panels_specs.append(SolarPanelSpec(node_id=i, latitude=latitude, longitude=longitude, altitude=altitude,
                                                timezone=dt.timezone.utc, start_date_str=start_date_str,
                                                simulation_time_seconds=simulation_time, tilt=tilt_list[i],
                                                azimuth=azimuth_list[i], efficiency=efficiency,
                                                panel_surface_m2=panel_surface_m2,
                                                station_file=station_file))

    return panels_specs


if __name__ == "__main__":
    # Calculate the number of processes to launch based on CPU cores
    num_cores = MAX_PARALLEL_SIMULATIONS

    # Calculate the range of alphas to cover
    alpha_values = [i * ALPHA_INCREMENT for i in range(0, 21)]
    
    # Launch processes
    processes = []
    for alpha in alpha_values:
        process = multiprocessing.Process(target=run_simulation, args=(alpha,))
        processes.append(process)
        process.start()

        # Control the number of running processes
        if len(processes) >= num_cores:
            for p in processes:
                p.join()
            processes = []

    # Wait for remaining processes to finish
    for p in processes:
        p.join()

    print("All simulations completed.")
