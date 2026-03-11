#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

"""
Shared simulator builder for the energy-aware scheduling simulator.

Provides a single function that creates the SimPy environment, Cloud, scheduler and
worker Nodes, ServiceDiscovery, and ServiceDataStorage, so that legacy runners
(run_simulation_d_sarsa) and the Gym wrapper (gym_env) use identical setup.
"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple, List

import simpy

from cloud import Cloud
from config import WORKER_BATTERY_CAPACITIES
from config import (
    NET_SPEED_CLIENT_SCHEDULER_MBIT,
    NET_SPEED_SCHEDULER_CLOUD_MBIT,
    NET_SPEED_SCHEDULER_WORKER_MBIT,
)
from node import Node
from service_data_storage import ServiceDataStorage
from service_discovery import ServiceDiscovery


def build_simulator(
    sim_time: int,
    session_uid: str,
    *,
    data_storage_session_id: str | None = None,
    learning_type: Node.LearningType = Node.LearningType.D_SARSA,
    no_learning_policy: Node.NoLearningPolicy = Node.NoLearningPolicy.RANDOM,
    actions_space: Node.ActionsSpace = Node.ActionsSpace.ONLY_WORKERS,
    state_type: Node.StateType = Node.StateType.JOB_TYPE,
    reward_alpha: float = 0.5,
    episode_length: int = 60,
    get_die_after_seconds: Callable[[int], int] | None = None,
    solar_panel_enabled: bool = False,
    solar_panel_spec_by_node_id: dict[int, object] | None = None,
    cloud_latency_roundtrip_ms: int = 20,
    gym_mode: bool = False,
    job_periodic_payload_sizes_mbytes: Tuple[float, float, float] | None = None,
    job_exponential_payload_sizes_mbytes: List[float] | None = None,
) -> tuple[simpy.Environment, list[Node], Cloud, ServiceDiscovery, ServiceDataStorage]:
    """
    Build the full simulator: SimPy env, Cloud, scheduler + 3 workers, discovery, data storage.

    Args:
        sim_time: Simulation time limit in seconds (passed to Node).
        session_uid: Session identifier used for Node session_uid.
        data_storage_session_id: Session id for ServiceDataStorage; if None, uses session_uid.
        learning_type: Node learning type (e.g. D_SARSA for legacy, NO_LEARNING for Gym).
        no_learning_policy: Policy when learning_type is NO_LEARNING.
        actions_space: Node actions space (ONLY_WORKERS, WORKERS_OR_CLOUD, etc.).
        state_type: Node state type (JOB_TYPE, ONLY_NUMBER).
        reward_alpha: Reward trade-off alpha (FPS vs battery).
        episode_length: Jobs per episode.
        get_die_after_seconds: Callable(node_id) -> seconds after which node "dies"; default no death.
        solar_panel_enabled: Whether solar panel is enabled on nodes.
        solar_panel_spec_by_node_id: Optional dict node_id -> solar panel spec.
        cloud_latency_roundtrip_ms: Cloud roundtrip latency in ms.
        gym_mode: If True, scheduler waits for external action via store (for Gym env).

    Returns:
        (env, nodes, cloud, discovery, data_storage). nodes[0] is the scheduler.
    """
    if data_storage_session_id is None:
        data_storage_session_id = session_uid
    if get_die_after_seconds is None:
        get_die_after_seconds = lambda _: 0
    if solar_panel_spec_by_node_id is None:
        solar_panel_spec_by_node_id = {}

    # Fallback to legacy defaults if job sizes are not explicitly provided.
    if job_periodic_payload_sizes_mbytes is None:
        job_periodic_payload_sizes_mbytes = (0.050, 0.050, 0.050)
    if job_exponential_payload_sizes_mbytes is None:
        job_exponential_payload_sizes_mbytes = [0.1]

    env = simpy.Environment()
    cloud = Cloud(env, latency_roundtrip_ms=cloud_latency_roundtrip_ms)
    worker_batt_1, worker_batt_2, worker_batt_3 = WORKER_BATTERY_CAPACITIES

    nodes: list[Node] = []
    nodes.append(
        _create_scheduler_node(
            env,
            sim_time,
            session_uid,
            state_type=state_type,
            learning_type=learning_type,
            no_learning_policy=no_learning_policy,
            actions_space=actions_space,
            reward_alpha=reward_alpha,
            episode_length=episode_length,
            die_after_seconds=get_die_after_seconds(0),
            solar_panel_enabled=solar_panel_enabled,
            solar_panel_spec=solar_panel_spec_by_node_id.get(0),
            gym_mode=gym_mode,
            job_periodic_payload_sizes_mbytes=job_periodic_payload_sizes_mbytes,
            job_exponential_payload_sizes_mbytes=job_exponential_payload_sizes_mbytes,
        )
    )
    nodes.append(
        _create_worker_node(
            env,
            1,
            worker_batt_1,
            sim_time,
            session_uid,
            state_type=state_type,
            learning_type=learning_type,
            no_learning_policy=no_learning_policy,
            actions_space=actions_space,
            reward_alpha=reward_alpha,
            episode_length=episode_length,
            die_after_seconds=get_die_after_seconds(1),
            solar_panel_enabled=solar_panel_enabled,
            solar_panel_spec=solar_panel_spec_by_node_id.get(1),
            machine_speed=1.8,
            job_periodic_payload_sizes_mbytes=job_periodic_payload_sizes_mbytes,
            job_exponential_payload_sizes_mbytes=job_exponential_payload_sizes_mbytes,
        )
    )
    nodes.append(
        _create_worker_node(
            env,
            2,
            worker_batt_2,
            sim_time,
            session_uid,
            state_type=state_type,
            learning_type=learning_type,
            no_learning_policy=no_learning_policy,
            actions_space=actions_space,
            reward_alpha=reward_alpha,
            episode_length=episode_length,
            die_after_seconds=get_die_after_seconds(2),
            solar_panel_enabled=solar_panel_enabled,
            solar_panel_spec=solar_panel_spec_by_node_id.get(2),
            machine_speed=1.7,
            job_periodic_payload_sizes_mbytes=job_periodic_payload_sizes_mbytes,
            job_exponential_payload_sizes_mbytes=job_exponential_payload_sizes_mbytes,
        )
    )
    nodes.append(
        _create_worker_node(
            env,
            3,
            worker_batt_3,
            sim_time,
            session_uid,
            state_type=state_type,
            learning_type=learning_type,
            no_learning_policy=no_learning_policy,
            actions_space=actions_space,
            reward_alpha=reward_alpha,
            episode_length=episode_length,
            die_after_seconds=get_die_after_seconds(3),
            solar_panel_enabled=solar_panel_enabled,
            solar_panel_spec=solar_panel_spec_by_node_id.get(3),
            machine_speed=1.4,
            job_periodic_payload_sizes_mbytes=job_periodic_payload_sizes_mbytes,
            job_exponential_payload_sizes_mbytes=job_exponential_payload_sizes_mbytes,
        )
    )

    discovery = ServiceDiscovery(1, nodes, cloud)
    data_storage = ServiceDataStorage(
        nodes,
        data_storage_session_id,
        learning_type,
        no_learning_policy,
        actions_space,
    )
    for node in nodes:
        node.set_service_discovery(discovery)
        node.set_service_data_storage(data_storage)
    for node in nodes:
        node.init()
    cloud.set_service_discovery(discovery)

    return (env, nodes, cloud, discovery, data_storage)


def _create_scheduler_node(
    env: simpy.Environment,
    sim_time: int,
    session_uid: str,
    *,
    state_type: Node.StateType = Node.StateType.JOB_TYPE,
    learning_type: Node.LearningType = Node.LearningType.D_SARSA,
    no_learning_policy: Node.NoLearningPolicy = Node.NoLearningPolicy.RANDOM,
    actions_space: Node.ActionsSpace = Node.ActionsSpace.ONLY_WORKERS,
    reward_alpha: float = 0.5,
    episode_length: int = 60,
    die_after_seconds: int = 0,
    solar_panel_enabled: bool = False,
    solar_panel_spec: object = None,
    gym_mode: bool = False,
    job_periodic_payload_sizes_mbytes: Tuple[float, float, float] | None = None,
    job_exponential_payload_sizes_mbytes: List[float] | None = None,
) -> Node:
    """Create scheduler node (node_id=0). Same kwargs as legacy create_node for scheduler."""
    return Node(
        env,
        0,
        session_uid,
        simulation_time=sim_time,
        skip_plots=True,
        node_belong_to_cluster=0,
        node_type=Node.NodeType.SCHEDULER,
        die_after_seconds=die_after_seconds,
        die_duration=4000,
        machine_speed=1.0,
        rate_l=30.0,
        solar_panel_enabled=solar_panel_enabled,
        solar_panel_spec=solar_panel_spec,
        rate_l_model_path_shift=0,
        rate_l_model_path_cycles=3,
        rate_l_model_path_parse_x_max=None,
        rate_l_model_path_steady=False,
        rate_l_model_path_steady_for=2000,
        rate_l_model_path_steady_every=2000,
        net_speed_client_scheduler_mbits=NET_SPEED_CLIENT_SCHEDULER_MBIT,
        net_speed_scheduler_scheduler_mbits=300,
        net_speed_scheduler_worker_mbits=NET_SPEED_SCHEDULER_WORKER_MBIT,
        net_speed_scheduler_cloud_mbits=NET_SPEED_SCHEDULER_CLOUD_MBIT,
        job_periodic_types=3,
        job_periodic_payload_sizes_mbytes=job_periodic_payload_sizes_mbytes
        if job_periodic_payload_sizes_mbytes is not None
        else (0.050, 0.050, 0.050),
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
        job_exponential_payload_sizes_mbytes=job_exponential_payload_sizes_mbytes
        if job_exponential_payload_sizes_mbytes is not None
        else [0.1],
        job_exponential_duration_std_devs=[0.01],
        job_exponential_arrival_time_std_devs=[0.01],
        job_exponential_percentages=[1],
        job_exponential_deadlines=[0.300],
        job_exponential_durations=[0.100],
        job_exponential_rates_fps=[10],
        job_exponential_desired_rates_fps=[1],
        job_exponential_desired_rates_fps_min=[0],
        job_exponential_desired_rates_fps_max=[10],
        max_jobs_in_queue=5,
        distribution_arrivals=Node.DistributionArrivals.POISSON,
        delay_probing=0.003,
        sarsa_alpha=0.01,
        sarsa_beta=0.01,
        state_type=state_type,
        learning_type=learning_type,
        no_learning_policy=no_learning_policy,
        actions_space=actions_space,
        pwr2_binary_policy="001111",
        tiling_num_tilings=26,
        reward_alpha=reward_alpha,
        distribution_network_probing_sigma=0.0001,
        distribution_network_forwarding_sigma=0.00002,
        episode_length=episode_length,
        eps=0.90,
        eps_decay=0.9995,
        eps_dynamic=True,
        eps_min=0.05,
        logging_info=True,
        battery_total_capacity_wh=10,
        battery_initial_capacity_wh=10,
        gym_mode=gym_mode,
    )


def _create_worker_node(
    env: simpy.Environment,
    node_id: int,
    batt: int,
    sim_time: int,
    session_uid: str,
    *,
    state_type: Node.StateType = Node.StateType.JOB_TYPE,
    learning_type: Node.LearningType = Node.LearningType.D_SARSA,
    no_learning_policy: Node.NoLearningPolicy = Node.NoLearningPolicy.RANDOM,
    actions_space: Node.ActionsSpace = Node.ActionsSpace.ONLY_WORKERS,
    reward_alpha: float = 0.5,
    episode_length: int = 60,
    die_after_seconds: int = 0,
    solar_panel_enabled: bool = False,
    solar_panel_spec: object = None,
    machine_speed: float = 1.0,
    job_periodic_payload_sizes_mbytes: Tuple[float, float, float] | None = None,
    job_exponential_payload_sizes_mbytes: List[float] | None = None,
) -> Node:
    """Create worker node. Same kwargs as legacy create_node for workers."""
    return Node(
        env,
        node_id,
        session_uid,
        simulation_time=sim_time,
        skip_plots=True,
        node_belong_to_cluster=0,
        node_type=Node.NodeType.WORKER,
        die_after_seconds=die_after_seconds,
        die_duration=4000,
        machine_speed=machine_speed,
        rate_l=30.0,
        solar_panel_enabled=solar_panel_enabled,
        solar_panel_spec=solar_panel_spec,
        rate_l_model_path_shift=0,
        rate_l_model_path_cycles=3,
        rate_l_model_path_parse_x_max=None,
        rate_l_model_path_steady=False,
        rate_l_model_path_steady_for=2000,
        rate_l_model_path_steady_every=2000,
        net_speed_client_scheduler_mbits=NET_SPEED_CLIENT_SCHEDULER_MBIT,
        net_speed_scheduler_scheduler_mbits=300,
        net_speed_scheduler_worker_mbits=NET_SPEED_SCHEDULER_WORKER_MBIT,
        net_speed_scheduler_cloud_mbits=NET_SPEED_SCHEDULER_CLOUD_MBIT,
        job_periodic_types=3,
        job_periodic_payload_sizes_mbytes=job_periodic_payload_sizes_mbytes
        if job_periodic_payload_sizes_mbytes is not None
        else (0.050, 0.050, 0.050),
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
        job_exponential_payload_sizes_mbytes=job_exponential_payload_sizes_mbytes
        if job_exponential_payload_sizes_mbytes is not None
        else [0.1],
        job_exponential_duration_std_devs=[0.01],
        job_exponential_arrival_time_std_devs=[0.01],
        job_exponential_percentages=[1],
        job_exponential_deadlines=[0.300],
        job_exponential_durations=[0.100],
        job_exponential_rates_fps=[10],
        job_exponential_desired_rates_fps=[1],
        job_exponential_desired_rates_fps_min=[0],
        job_exponential_desired_rates_fps_max=[10],
        max_jobs_in_queue=5,
        distribution_arrivals=Node.DistributionArrivals.POISSON,
        delay_probing=0.003,
        sarsa_alpha=0.01,
        sarsa_beta=0.01,
        state_type=state_type,
        learning_type=learning_type,
        no_learning_policy=no_learning_policy,
        actions_space=actions_space,
        pwr2_binary_policy="001111",
        tiling_num_tilings=26,
        reward_alpha=reward_alpha,
        distribution_network_probing_sigma=0.0001,
        distribution_network_forwarding_sigma=0.00002,
        episode_length=episode_length,
        eps=0.90,
        eps_decay=0.9995,
        eps_dynamic=True,
        eps_min=0.05,
        logging_info=True,
        battery_total_capacity_wh=batt,
        battery_initial_capacity_wh=batt,
    )
