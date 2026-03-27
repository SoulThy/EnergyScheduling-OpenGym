#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

def compute_stats(db_path: Path) -> Dict[str, Any]:
    """Compute thesis-oriented aggregate metrics from a simulation log.db."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()

        # Simulation configuration (best effort).
        sim_config: Dict[str, str] = {}
        try:
            cur.execute("SELECT key, value FROM sim_config;")
            for key, value in cur.fetchall():
                sim_config[str(key)] = str(value)
        except sqlite3.OperationalError:
            sim_config = {}

        # Core job metrics.
        cur.execute("SELECT COUNT(*) FROM jobs;")
        (total_jobs,) = cur.fetchone()

        cur.execute("SELECT COALESCE(SUM(executed), 0), COALESCE(SUM(rejected), 0) FROM jobs;")
        executed_jobs, rejected_jobs = cur.fetchone()

        # Deadline misses among executed and not rejected jobs.
        cur.execute(
            "SELECT COALESCE(SUM(over_deadline), 0), COUNT(*) "
            "FROM jobs WHERE executed = 1 AND rejected = 0"
        )
        deadline_miss_jobs, executed_not_rejected_jobs = cur.fetchone()

        # Success = executed and within deadline.
        cur.execute(
            "SELECT COALESCE(SUM(CASE WHEN executed = 1 AND over_deadline = 0 "
            "THEN 1 ELSE 0 END), 0) FROM jobs;"
        )
        (successful_jobs,) = cur.fetchone()

        # Battery indicators from round logs.
        battery_mean_by_worker: Dict[int, float] = {}
        battery_last_by_worker: Dict[int, float] = {}
        try:
            cur.execute(
                "SELECT worker_id, AVG(battery_residual), MAX(time) "
                "FROM round GROUP BY worker_id;"
            )
            rows = cur.fetchall()
            for worker_id, avg_battery, _max_time in rows:
                battery_mean_by_worker[int(worker_id)] = float(avg_battery or 0.0)

            for worker_id in battery_mean_by_worker.keys():
                cur.execute(
                    "SELECT battery_residual FROM round "
                    "WHERE worker_id = ? ORDER BY time DESC LIMIT 1;",
                    (worker_id,),
                )
                row = cur.fetchone()
                battery_last_by_worker[worker_id] = float(row[0] if row else 0.0)
        except sqlite3.OperationalError:
            battery_mean_by_worker = {}
            battery_last_by_worker = {}

        battery_last_values = list(battery_last_by_worker.values())
        if battery_last_values:
            battery_last_avg = sum(battery_last_values) / float(len(battery_last_values))
            battery_last_min = min(battery_last_values)
        else:
            battery_last_avg = None
            battery_last_min = None

        return {
            "sim_config": sim_config,
            "total_jobs": int(total_jobs),
            "executed_jobs": int(executed_jobs),
            "rejected_jobs": int(rejected_jobs),
            "successful_jobs": int(successful_jobs),
            "success_rate": (
                float(successful_jobs) / float(total_jobs) if total_jobs > 0 else 0.0
            ),
            "deadline_miss_jobs": int(deadline_miss_jobs),
            "deadline_miss_rate": (
                float(deadline_miss_jobs) / float(executed_not_rejected_jobs)
                if executed_not_rejected_jobs > 0
                else 0.0
            ),
            "reject_rate": (
                float(rejected_jobs) / float(total_jobs) if total_jobs > 0 else 0.0
            ),
            "executed_not_rejected_jobs": int(executed_not_rejected_jobs),
            "battery_mean_by_worker_wh": battery_mean_by_worker,
            "battery_last_by_worker_wh": battery_last_by_worker,
            "battery_last_avg_wh": battery_last_avg,
            "battery_last_min_wh": battery_last_min,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute compact, thesis-oriented aggregate metrics from a log.db file."
        )
    )
    parser.add_argument(
        "db_file",
        type=Path,
        help="Path to log.db",
    )
    args = parser.parse_args()

    try:
        stats = compute_stats(args.db_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Database: {args.db_file}")

    # Simulation configuration overview.
    cfg = stats["sim_config"]
    print("Simulation configuration (useful fields):")
    if cfg:
        def _get(name: str, default: str = "n/a") -> str:
            return cfg.get(name, default)

        print(f"- MODEL_VERSION: {_get('MODEL_VERSION')}")
        print(f"- SCORE_SIMPLE_WEIGHT_Q: {_get('SCORE_SIMPLE_WEIGHT_Q')}")
        print(f"- SCORE_SIMPLE_WEIGHT_B: {_get('SCORE_SIMPLE_WEIGHT_B')}")
        print(f"- SCORE_SIMPLE_WEIGHT_F: {_get('SCORE_SIMPLE_WEIGHT_F')}")
        print(f"- WORKER_BATTERY_CAPACITIES: {_get('WORKER_BATTERY_CAPACITIES')}")
        print(f"- NODE_MACHINE_SPEEDS: {_get('NODE_MACHINE_SPEEDS')}")
    else:
        print("  (no sim_config table in this log.db)")

    print()
    print("Core policy metrics:")
    print(f"- total_jobs: {stats['total_jobs']}")
    print(f"- executed_jobs: {stats['executed_jobs']}")
    print(f"- rejected_jobs: {stats['rejected_jobs']}")
    print(f"- successful_jobs: {stats['successful_jobs']}")
    print(f"- success_rate: {stats['success_rate'] * 100:.2f}%")
    print(
        f"- deadline_miss_jobs: {stats['deadline_miss_jobs']} "
        f"({stats['deadline_miss_rate'] * 100:.2f}% of executed & non-rejected jobs)"
    )

    print(f"- reject_rate: {stats['reject_rate'] * 100:.2f}%")

    print()
    print("Battery indicators:")
    if stats["battery_last_by_worker_wh"]:
        for worker_id, value in sorted(stats["battery_last_by_worker_wh"].items()):
            print(f"- worker {worker_id} last_battery_wh: {value:.4f}")
        print(f"- battery_last_avg_wh: {stats['battery_last_avg_wh']:.4f}")
        print(f"- battery_last_min_wh: {stats['battery_last_min_wh']:.4f}")
        for worker_id, value in sorted(stats["battery_mean_by_worker_wh"].items()):
            print(f"- worker {worker_id} avg_battery_wh: {value:.4f}")
    else:
        print("- n/a (round table not available in this log.db)")


if __name__ == "__main__":
    main()
