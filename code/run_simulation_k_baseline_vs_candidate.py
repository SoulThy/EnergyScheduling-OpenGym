#!/usr/bin/env python3
from __future__ import annotations

"""
Run two simulations in parallel for the same workload preset:

- K=1 (baseline: probe on every scheduler state request).
- K=CANDIDATE_K (hardcoded below), e.g. 5.

MODEL_VERSION must be set in the environment before starting this script, e.g.:

    MODEL_VERSION=SMALL_JOBS_V2 python code/run_simulation_k_baseline_vs_candidate.py

Each run writes ``log.db`` under:

    results/data/_log/learning/D_SARSA/ONLY_WORKERS/STALENESS_EVALUATION/<session_id>/log.db

with ``session_id`` containing ``_K<value>`` so ``scripts/evaluate_k_policy_table.py``
can pair baseline (K=1) and candidate runs inside the same model regime.

After both runs finish, point ``evaluate_k_policy_table.py`` at the K=1 ``log.db`` as
baseline and use ``.../ONLY_WORKERS/STALENESS_EVALUATION/`` as ``runs_dir`` (or any
parent that globs both run folders).
"""

import datetime as dt
import multiprocessing
import os
import random
import sys
from datetime import datetime
from typing import FrozenSet, List

# Keep in sync with sim_builder._JOB_PARAMS_BY_MODEL keys (avoid top-level sim_builder import).
_VALID_MODEL_VERSIONS: FrozenSet[str] = frozenset(
    ("LEGACY", "SMALL_JOBS_V1", "SMALL_JOBS_V2", "SMALL_JOBS_V3")
)

MODULE = "KBaselineVsCandidate"

# All logs go under ONLY_WORKERS/STALENESS_EVALUATION/ (see ServiceDataStorage paths).
STALENESS_EVALUATION_OUTPUT_SUBDIR = "STALENESS_EVALUATION"

SIMULATION_TIME = 10000
SIMULATION_TOTAL_TIME = SIMULATION_TIME
SOLAR_PANEL_ENABLED = False

SESSION_ID = datetime.now().strftime("%Y%m%d")
ALPHA = 0.50
PROBE_SIZE_BYTES = 200
RANDOM_SEED = 42

# Change this in source to test K=2, K=5, etc. Must be >= 2 so it differs from baseline.
CANDIDATE_K = 5


def get_die_after(node_id: int) -> int:
    if node_id == 1:
        return 4000
    return 0


def run_simulation_for_k(step_index: int, k_value: int, model_version: str) -> None:
    # config.PROBING_STATE_REFRESH_EVERY_K_JOBS is fixed at first import of config.
    # Set env *before* any import that pulls in config/node. Using multiprocessing
    # "spawn" gives each child a fresh interpreter (no copied parent modules).
    # With "fork", a parent that had already imported config would leak the wrong K.
    os.environ["MODEL_VERSION"] = model_version
    os.environ["PROBE_SIZE_BYTES"] = str(PROBE_SIZE_BYTES)
    os.environ["PROBING_STATE_REFRESH_EVERY_K_JOBS"] = str(k_value)

    random.seed(RANDOM_SEED)
    import numpy as np

    np.random.seed(RANDOM_SEED)

    from log import Log
    from models import SolarPanelSpec
    from node import Node
    from sim_builder import build_simulator
    from config import PROBING_STATE_REFRESH_EVERY_K_JOBS

    if PROBING_STATE_REFRESH_EVERY_K_JOBS != k_value:
        raise RuntimeError(
            f"PROBING_STATE_REFRESH_EVERY_K_JOBS={PROBING_STATE_REFRESH_EVERY_K_JOBS} "
            f"!= expected k_value={k_value}: config was imported before env was set, "
            f"or start method is fork with a polluted parent."
        )

    session_id = (
        f"{SESSION_ID}_{model_version}_{ALPHA:.2f}_{PROBE_SIZE_BYTES}B_K{k_value}"
    )
    data_storage_session_id = f"{STALENESS_EVALUATION_OUTPUT_SUBDIR}/{session_id}"

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
        f"Starting step={step_index}, MODEL_VERSION={model_version}, K={k_value}, "
        f"probe_size={PROBE_SIZE_BYTES}B, data_storage_session_id={data_storage_session_id}",
    )

    env, nodes, cloud, discovery, data_storage = build_simulator(
        sim_time=SIMULATION_TIME,
        session_uid=SESSION_ID,
        data_storage_session_id=data_storage_session_id,
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
        f"Simulation ended: data_storage_session_id={data_storage_session_id}, K={k_value}, "
        f"ALPHA={ALPHA}, probe_size={PROBE_SIZE_BYTES}B",
    )


def main() -> None:
    model_version = os.environ.get("MODEL_VERSION", "").strip()
    if not model_version:
        print(
            "ERROR: set MODEL_VERSION in the environment, e.g.\n"
            "  MODEL_VERSION=SMALL_JOBS_V3 python code/run_simulation_k_baseline_vs_candidate.py",
            file=sys.stderr,
        )
        sys.exit(1)
    if model_version not in _VALID_MODEL_VERSIONS:
        known = ", ".join(sorted(_VALID_MODEL_VERSIONS))
        print(
            f"ERROR: unknown MODEL_VERSION={model_version!r}. Expected one of: {known}",
            file=sys.stderr,
        )
        sys.exit(1)
    if CANDIDATE_K < 2:
        print(
            f"ERROR: CANDIDATE_K must be >= 2 (got {CANDIDATE_K}); "
            "edit the constant in this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    ctx = multiprocessing.get_context("spawn")
    jobs = [
        (0, 1, model_version),
        (1, CANDIDATE_K, model_version),
    ]
    processes: list[multiprocessing.Process] = []
    for step_index, k_value, mv in jobs:
        p = ctx.Process(
            target=run_simulation_for_k,
            args=(step_index, k_value, mv),
        )
        processes.append(p)
        p.start()

    for proc in processes:
        proc.join()

    for proc in processes:
        if proc.exitcode != 0:
            print(
                f"ERROR: child pid={proc.pid} exited with code {proc.exitcode}",
                file=sys.stderr,
            )
            sys.exit(proc.exitcode or 1)

    print(
        f"Completed MODEL_VERSION={model_version}: K=1 vs K={CANDIDATE_K} "
        f"under ONLY_WORKERS/{STALENESS_EVALUATION_OUTPUT_SUBDIR}/ "
        f"(session prefix {SESSION_ID}_{model_version}_...)."
    )


if __name__ == "__main__":
    main()
