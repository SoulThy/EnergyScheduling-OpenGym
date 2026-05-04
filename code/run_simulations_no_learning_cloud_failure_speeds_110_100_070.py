from __future__ import annotations

import multiprocessing
from datetime import datetime

import simpy

from cloud import Cloud
from config import MAX_PARALLEL_SIMULATIONS
import lbfc
from log import Log
from node import Node
from service_data_storage import ServiceDataStorage
from service_discovery import ServiceDiscovery

"""
Run NO-LEARNING simulations (WORKERS_OR_CLOUD) with a temporary worker crash,
using worker speeds 1.1, 1.0, 0.7 to match historical D-SARSA failure runs.

Failure model:
- Crash the most powerful worker at t=4000 s
- Keep it crashed until t=8000 s (duration 4000 s)
"""

MODULE = "RunNoLearningFailureWorkersOrCloud_110_100_070"

SIMULATION_TIME = 10_000
SIMULATION_TOTAL_TIME = SIMULATION_TIME

CRASH_START_S = 4000
CRASH_END_S = 8000
CRASH_DURATION_S = CRASH_END_S - CRASH_START_S

SESSION_ID = datetime.now().strftime("%Y%m%d")
LEARNING_TYPE = Node.LearningType.NO_LEARNING
ACTIONS_SPACE = Node.ActionsSpace.WORKERS_OR_CLOUD


def run_simulation(policy: Node.NoLearningPolicy) -> None:
    session_id = f"{SESSION_ID}_{policy.name}_WORKERS_OR_CLOUD_FAILURE_110_100_070"

    env = simpy.Environment()
    cloud = Cloud(env, latency_roundtrip_ms=20)

    # Match the D-SARSA failure runner: speeds 1.1, 1.0, 0.7 and batteries 9, 8, 7.
    worker_specs = [
        (1, 1.1, 9),
        (2, 1.0, 8),
        (3, 0.7, 7),
    ]
    crash_worker_id = max(worker_specs, key=lambda x: x[1])[0]

    nodes: list[Node] = []
    nodes.append(create_node(env, 0, 0, Node.NodeType.SCHEDULER, 1.0, 10, policy, crash_worker_id))
    for uid, speed, batt in worker_specs:
        nodes.append(create_node(env, uid, 0, Node.NodeType.WORKER, speed, batt, policy, crash_worker_id))

    discovery = ServiceDiscovery(1, nodes, cloud)
    data_storage = ServiceDataStorage(nodes, session_id, LEARNING_TYPE, policy, ACTIONS_SPACE)

    for node in nodes:
        node.set_service_discovery(discovery)
        node.set_service_data_storage(data_storage)
    lbfc.LBFC.reset_debug_counter()
    for node in nodes:
        node.init()
    cloud.set_service_discovery(discovery)

    Log.minfo(
        MODULE,
        f"Started simulation for {policy.name} (crash worker={crash_worker_id} at t={CRASH_START_S}s..{CRASH_END_S}s)",
    )
    if policy == Node.NoLearningPolicy.LBFC:
        Log.minfo(
            MODULE,
            f"LBFC active: EMA on mean cluster load (LBFC_EMA_ALPHA={lbfc.LBFC_EMA_ALPHA}), "
            f"stress_u uses lbfc.TAU={lbfc.TAU}, weighted water-filling (see code/lbfc.py).",
        )
    env.run(until=SIMULATION_TOTAL_TIME)
    Log.minfo(
        MODULE,
        f"Simulation ended: SESSION_ID={session_id}, LEARNING_TYPE={LEARNING_TYPE.name}, "
        f"NO_LEARNING_POLICY={policy.name}, ACTIONS_SPACE={ACTIONS_SPACE.name}",
    )

    data_storage.done_simulation()


def create_node(
    env: simpy.Environment,
    node_id: int,
    belong_to_cluster_id: int,
    node_type: Node.NodeType,
    machine_speed: float,
    batt: float,
    policy: Node.NoLearningPolicy,
    crash_worker_id: int,
) -> Node:
    die_simulation = node_type == Node.NodeType.WORKER and node_id == crash_worker_id
    die_after_seconds = CRASH_START_S if die_simulation else 0

    return Node(
        env,
        node_id,
        SESSION_ID,
        simulation_time=SIMULATION_TIME,
        skip_plots=True,
        node_belong_to_cluster=belong_to_cluster_id,
        node_type=node_type,
        die_simulation=die_simulation,
        die_after_seconds=die_after_seconds,
        die_duration=CRASH_DURATION_S,
        # rates
        machine_speed=machine_speed,
        rate_l=30.0,
        # solar panel
        solar_panel_enabled=False,
        solar_panel_spec=None,
        # traffic model
        rate_l_model_path_shift=0,
        rate_l_model_path_cycles=3,
        rate_l_model_path_parse_x_max=None,
        rate_l_model_path_steady=False,
        rate_l_model_path_steady_for=2000,
        rate_l_model_path_steady_every=2000,
        # net
        net_speed_client_scheduler_mbits=200,
        net_speed_scheduler_scheduler_mbits=300,
        net_speed_scheduler_worker_mbits=1000,
        net_speed_scheduler_cloud_mbits=1000,
        # job info
        job_periodic_types=3,
        job_periodic_payload_sizes_mbytes=(0.050, 0.050, 0.050),
        job_periodic_duration_std_devs=(0.0003, 0.0003, 0.0003),
        job_periodic_percentages=(0.33, 0.33, 0.34),
        job_periodic_deadlines=(0.016, 0.033, 0.070),
        job_periodic_durations=(0.010, 0.020, 0.055),
        job_periodic_arrival_time_std_devs=(0.001, 0.002, 0.01),
        job_periodic_rates_fps=(60, 30, 15),
        job_periodic_desired_rates_fps=(60, 30, 15),
        job_periodic_desired_rates_fps_max=(60, 30, 15),
        job_periodic_desired_rates_fps_min=(50, 20, 10),
        job_exponential_types=1,
        job_exponential_payload_sizes_mbytes=[0.1],
        job_exponential_duration_std_devs=[0.01],
        job_exponential_arrival_time_std_devs=[0.01],
        job_exponential_percentages=[1],
        job_exponential_deadlines=[0.300],
        job_exponential_durations=[0.100],
        job_exponential_rates_fps=[10],
        job_exponential_desired_rates_fps=[1],
        job_exponential_desired_rates_fps_min=[0],
        job_exponential_desired_rates_fps_max=[10],
        # node info
        max_jobs_in_queue=5,
        distribution_arrivals=Node.DistributionArrivals.POISSON,
        delay_probing=0.003,
        # learning
        sarsa_alpha=0.01,
        sarsa_beta=0.01,
        state_type=Node.StateType.JOB_TYPE,
        learning_type=LEARNING_TYPE,
        no_learning_policy=policy,
        actions_space=ACTIONS_SPACE,
        pwr2_binary_policy="001111",
        tiling_num_tilings=26,
        # distributions
        distribution_network_probing_sigma=0.0001,
        distribution_network_forwarding_sigma=0.00002,
        episode_length=60,
        eps=0.90,
        eps_decay=0.9995,
        eps_dynamic=True,
        eps_min=0.05,
        logging_info=True,
        battery_total_capacity_wh=batt,
        battery_initial_capacity_wh=batt,
    )


if __name__ == "__main__":
    num_cores = MAX_PARALLEL_SIMULATIONS
    policies = [
        Node.NoLearningPolicy.LEAST_LOADED_AWARE_CLOUD,
        Node.NoLearningPolicy.MAXIMUM_LIFESPANE,
        Node.NoLearningPolicy.RANDOM,
        # LBF baseline (SCORE_SIMPLE): keep in repo but disabled for this LBFC-focused batch.
        # Node.NoLearningPolicy.SCORE_SIMPLE,
        Node.NoLearningPolicy.LBFC,
    ]

    processes: list[multiprocessing.Process] = []
    for policy in policies:
        p = multiprocessing.Process(target=run_simulation, args=(policy,))
        processes.append(p)
        p.start()

        if len(processes) >= num_cores:
            for jp in processes:
                jp.join()
            processes = []

    for jp in processes:
        jp.join()

    print("All failure simulations (WORKERS_OR_CLOUD, speeds 1.1/1.0/0.7) completed.")

