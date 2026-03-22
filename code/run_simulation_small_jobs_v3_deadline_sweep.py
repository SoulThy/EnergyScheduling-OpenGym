#!/usr/bin/env python3
from __future__ import annotations

"""
Sweep periodic deadline triplets for MODEL_VERSION=SMALL_JOBS_V3.

Fixes reward_alpha=0.50 and PROBE_SIZE_BYTES=200. Scales the default V3
periodic deadlines (same shape as V2: 0.016, 0.033, 0.070 s) by ~9 factors
from tighter to looser so you can pick a regime with a meaningful
over_deadline ratio.

Run from the ``code/`` directory (same as other run_simulation_*.py scripts).

Options: ``--dry-run`` prints all deadline triples without importing the simulator;
``--quiet`` skips the plan before a real sweep. Each run checks that
``build_simulator(..., job_periodic_deadlines_s=...)`` is applied on every node.
"""

import argparse
import csv
import multiprocessing
import os
import random
import sqlite3
from datetime import datetime
from typing import List, Tuple

MODULE = "SmallJobsV3DeadlineSweep"

# Must match ``sim_builder.SMALL_JOBS_V3_JOB_PARAMS["periodic_deadlines_s"]`` (inherited from V2).
# Used for --dry-run / validation without importing sim_builder (which pulls in simpy).
SMALL_JOBS_V3_BASE_PERIODIC_DEADLINES_S: Tuple[float, float, float] = (0.016, 0.033, 0.070)

SIMULATION_TIME = 10000
SIMULATION_TOTAL_TIME = SIMULATION_TIME
SOLAR_PANEL_ENABLED = False

SESSION_ID = datetime.now().strftime("%Y%m%d")

ALPHA = 0.50
PROBE_SIZE_BYTES = 200

# Reproducibility (same pattern as run_simulation_d_sarsa_probe_sweep.py).
RANDOM_SEED = 42

# Nine multipliers applied uniformly to (d0, d1, d2). <1 = tighter deadlines.
DEADLINE_SCALE_FACTORS: List[float] = [
    0.40,
    0.52,
    0.64,
    0.76,
    0.88,
    1.00,
    1.15,
    1.35,
    1.60,
]


def get_die_after(node_id: int) -> int:
    if node_id == 1:
        return 4000
    return 0


def _scale_deadlines(
    base_s: Tuple[float, float, float], factor: float
) -> Tuple[float, float, float]:
    return (base_s[0] * factor, base_s[1] * factor, base_s[2] * factor)


def _deadlines_close(
    a: Tuple[float, float, float], b: Tuple[float, float, float], eps: float = 1e-9
) -> bool:
    return all(abs(a[i] - b[i]) <= eps for i in range(3))


def validate_sweep_config(
    base_deadlines: Tuple[float, float, float],
    scales: List[float],
) -> None:
    """Ensure each sweep step uses a distinct scale and distinct deadline triple."""
    if len(scales) != len(set(scales)):
        raise ValueError(
            f"DEADLINE_SCALE_FACTORS must be unique (duplicates: {scales})."
        )
    triples = [_scale_deadlines(base_deadlines, s) for s in scales]
    if len(set(triples)) != len(triples):
        raise ValueError(
            "Scaled deadline triples are not all distinct; check DEADLINE_SCALE_FACTORS."
        )


def _over_deadline_stats(db_path: str) -> Tuple[float, int, int]:
    """Return (ratio among executed non-rejected jobs, over count, total count)."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COALESCE(SUM(over_deadline), 0), COUNT(*) "
            "FROM jobs WHERE executed = 1 AND rejected = 0"
        )
        row = cur.fetchone()
        if row is None:
            return (0.0, 0, 0)
        over = int(row[0])
        n = int(row[1])
        ratio = float(over) / float(n) if n > 0 else 0.0
        return (ratio, over, n)
    finally:
        conn.close()


def run_one(
    step_index: int,
    scale: float,
    results_rows: list | None,
    results_lock: object | None,
) -> None:
    os.environ["MODEL_VERSION"] = "SMALL_JOBS_V3"
    os.environ["PROBE_SIZE_BYTES"] = str(PROBE_SIZE_BYTES)

    random.seed(RANDOM_SEED)
    import numpy as np

    np.random.seed(RANDOM_SEED)

    from log import Log
    from node import Node
    from sim_builder import SMALL_JOBS_V3_JOB_PARAMS, build_simulator

    base_deadlines = SMALL_JOBS_V3_JOB_PARAMS["periodic_deadlines_s"]
    if not _deadlines_close(base_deadlines, SMALL_JOBS_V3_BASE_PERIODIC_DEADLINES_S):
        raise RuntimeError(
            "sim_builder SMALL_JOBS_V3 periodic_deadlines_s changed; update "
            "SMALL_JOBS_V3_BASE_PERIODIC_DEADLINES_S in run_simulation_small_jobs_v3_deadline_sweep.py"
        )
    deadlines_s = _scale_deadlines(base_deadlines, scale)
    for i in range(3):
        expected = base_deadlines[i] * scale
        if abs(deadlines_s[i] - expected) > 1e-12:
            raise RuntimeError(
                f"Internal deadline scale error: idx={i} expected={expected} got={deadlines_s[i]}"
            )

    scale_tag = f"{scale:.4f}".replace(".", "p")
    session_id = f"{SESSION_ID}_{ALPHA:.2f}_200B_V3dl{scale_tag}"

    Log.minfo(
        MODULE,
        f"Starting step={step_index} scale={scale} deadlines_s={deadlines_s} session_id={session_id}",
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
        solar_panel_spec_by_node_id=None,
        job_periodic_deadlines_s=deadlines_s,
    )

    for node in nodes:
        got_deadlines = tuple(node._job_periodic_deadlines)
        if not _deadlines_close(got_deadlines, deadlines_s):
            raise RuntimeError(
                "build_simulator did not apply job_periodic_deadlines_s consistently: "
                f"node_uid={node.get_uid()} expected {deadlines_s}, got {got_deadlines}"
            )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    env.run(until=SIMULATION_TOTAL_TIME)

    data_storage.done_simulation()

    db_path = os.path.join(data_storage.get_log_dir(), "log.db")
    ratio, over_c, n = _over_deadline_stats(db_path)

    row = {
        "step": step_index,
        "deadline_scale": scale,
        "deadline_s_type0": deadlines_s[0],
        "deadline_s_type1": deadlines_s[1],
        "deadline_s_type2": deadlines_s[2],
        "over_deadline_ratio": f"{ratio:.6f}",
        "over_deadline_count": over_c,
        "executed_jobs": n,
        "session_id": session_id,
        "log_db": db_path,
    }

    Log.minfo(
        MODULE,
        f"Done step={step_index} scale={scale} over_deadline_ratio={ratio:.4f} "
        f"({over_c}/{n}) log={db_path}",
    )

    if results_rows is not None:
        if results_lock is not None:
            with results_lock:
                results_rows.append(row)
        else:
            results_rows.append(row)


def print_sweep_plan(
    base_deadlines: Tuple[float, float, float],
    scales: List[float],
) -> None:
    print("SMALL_JOBS_V3 periodic deadline sweep (seconds; same shape 0/1/2 as sim_builder)")
    print(f"  Base triple (scale 1.0): {base_deadlines}")
    print("  step  scale    d0         d1         d2")
    for step_index, scale in enumerate(scales):
        t = _scale_deadlines(base_deadlines, scale)
        print(
            f"  {step_index:3d}  {scale:5.2f}   "
            f"{t[0]:.6f}   {t[1]:.6f}   {t[2]:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-out",
        type=str,
        default="../results/small_jobs_v3_deadline_sweep.csv",
        help="Path to write summary CSV (relative to cwd, usually code/).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sweeps one after another (easier debugging).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the nine deadline triples and exit (no simulation).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the sweep plan before running.",
    )
    args = parser.parse_args()

    base_deadlines = SMALL_JOBS_V3_BASE_PERIODIC_DEADLINES_S
    validate_sweep_config(base_deadlines, DEADLINE_SCALE_FACTORS)

    if args.dry_run:
        print_sweep_plan(base_deadlines, DEADLINE_SCALE_FACTORS)
        print("Dry run: no simulations started.")
        return

    if not args.quiet:
        print_sweep_plan(base_deadlines, DEADLINE_SCALE_FACTORS)
        print()

    tasks = list(enumerate(DEADLINE_SCALE_FACTORS))

    if args.sequential:
        results_rows: list = []
        for step_index, scale in tasks:
            run_one(step_index, scale, results_rows, None)
        rows_sorted = sorted(results_rows, key=lambda r: int(r["step"]))
    else:
        manager = multiprocessing.Manager()
        results_rows = manager.list()
        results_lock = manager.Lock()
        num_cores = int(os.getenv("MAX_PARALLEL_SIMULATIONS", "4"))
        processes: list[multiprocessing.Process] = []
        for step_index, scale in tasks:
            p = multiprocessing.Process(
                target=run_one,
                args=(step_index, scale, results_rows, results_lock),
            )
            processes.append(p)
            p.start()
            if len(processes) >= num_cores:
                for q in processes:
                    q.join()
                processes = []
        for q in processes:
            q.join()
        rows_sorted = sorted(list(results_rows), key=lambda r: int(r["step"]))

    out_path = os.path.abspath(args.csv_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if rows_sorted:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
            w.writeheader()
            w.writerows(rows_sorted)
        print(f"Wrote {len(rows_sorted)} rows to {out_path}")
    else:
        print("No result rows collected.")


if __name__ == "__main__":
    main()
