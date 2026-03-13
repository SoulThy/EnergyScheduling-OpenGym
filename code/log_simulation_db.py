#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

from config import WORKER_BATTERY_CAPACITIES


def compute_stats(db_path: Path) -> Dict[str, Any]:
    """
    Compute aggregate statistics from a log.db file produced by ServiceDataStorage.

    This is intended for quick, script-friendly comparisons between runs
    (e.g., D-SARSA with vs without probing energy cost).
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()

        # ----------------------------------------------------------------------------
        # Simulation configuration (if available)
        # ----------------------------------------------------------------------------
        sim_config: Dict[str, str] = {}
        try:
            cur.execute("SELECT key, value FROM sim_config;")
            for key, value in cur.fetchall():
                sim_config[str(key)] = str(value)
        except sqlite3.OperationalError:
            # Older log.db files won't have this table.
            sim_config = {}

        # ----------------------------------------------------------------------------
        # Core job / timing statistics
        # ----------------------------------------------------------------------------
        # Total jobs (created)
        cur.execute("SELECT COUNT(*) FROM jobs;")
        (total_jobs,) = cur.fetchone()

        # Executed / rejected
        cur.execute("SELECT COALESCE(SUM(executed), 0), COALESCE(SUM(rejected), 0) FROM jobs;")
        executed_jobs, rejected_jobs = cur.fetchone()

        # Over-deadline jobs (regardless of executed/rejected)
        cur.execute("SELECT COALESCE(SUM(over_deadline), 0) FROM jobs;")
        (over_deadline_jobs,) = cur.fetchone()

        # Jobs that were executed and not over deadline: our notion of "successful" jobs.
        cur.execute(
            "SELECT COALESCE(SUM(CASE WHEN executed = 1 AND over_deadline = 0 "
            "THEN 1 ELSE 0 END), 0) FROM jobs;"
        )
        (successful_jobs,) = cur.fetchone()

        # Basic timing window (simulation span) using generation and end-to-end time_total.
        # time_total is the Job.get_total_time(), so end_time ~= generated_at + time_total.
        cur.execute(
            "SELECT MIN(generated_at), MAX(generated_at + time_total) FROM jobs "
            "WHERE generated_at IS NOT NULL AND time_total IS NOT NULL;"
        )
        row = cur.fetchone()
        min_generated, max_done = row if row is not None else (None, None)

        if min_generated is not None and max_done is not None:
            sim_span_s = max_done - min_generated
        else:
            sim_span_s = 0.0

        # Effective FPS as "executed jobs per simulated second".
        effective_fps = (executed_jobs / sim_span_s) if sim_span_s > 0 else 0.0

        # Average end-to-end latency per job (s) and for executed jobs only.
        cur.execute("SELECT AVG(time_total) FROM jobs;")
        (avg_total_time_all,) = cur.fetchone()

        cur.execute("SELECT AVG(time_total) FROM jobs WHERE executed = 1;")
        (avg_total_time_executed,) = cur.fetchone()

        # Probing energy per node (Wh) – available for new simulations where the
        # probing_energy table exists.
        probing_energy_by_node: Dict[int, float] = {}
        try:
            cur.execute("SELECT node_uid, energy_wh FROM probing_energy;")
            for node_uid, energy_wh in cur.fetchall():
                probing_energy_by_node[int(node_uid)] = float(energy_wh)
        except sqlite3.OperationalError:
            # Older log.db files won't have this table; keep stats minimal.
            probing_energy_by_node = {}

        total_probing_energy_wh = sum(probing_energy_by_node.values())
        total_battery_capacity_wh = float(sum(WORKER_BATTERY_CAPACITIES))
        probing_energy_share = (
            (total_probing_energy_wh / total_battery_capacity_wh) if total_battery_capacity_wh > 0 else 0.0
        )

        # ----------------------------------------------------------------------------
        # Expected average job size (MB) under default arrival rates
        # ----------------------------------------------------------------------------
        # By design, periodic and exponential jobs use independent arrival processes
        # with fixed rates. In the default D-SARSA setup, the scheduler uses:
        #   periodic rates: 60, 30, 15 fps  -> total 105
        #   exponential rate: 10 fps        -> total 10
        # so the probability that the next job is periodic is 105/115 and
        # exponential is 10/115.
        #
        # We combine those with the representative payload sizes from sim_config.
        periodic_size_mb = 0.050
        exp_size_mb = 0.100
        if "JOB_PERIODIC_PAYLOAD_SIZE_MB" in sim_config:
            try:
                periodic_size_mb = float(sim_config["JOB_PERIODIC_PAYLOAD_SIZE_MB"])
            except ValueError:
                pass
        if "JOB_EXPONENTIAL_PAYLOAD_SIZE_MB" in sim_config:
            try:
                exp_size_mb = float(sim_config["JOB_EXPONENTIAL_PAYLOAD_SIZE_MB"])
            except ValueError:
                pass

        p_periodic = 105.0 / 115.0
        p_exponential = 1.0 - p_periodic
        avg_job_size_mb = p_periodic * periodic_size_mb + p_exponential * exp_size_mb

        return {
            "sim_config": sim_config,
            "total_jobs": int(total_jobs),
            "executed_jobs": int(executed_jobs),
            "rejected_jobs": int(rejected_jobs),
            "over_deadline_jobs": int(over_deadline_jobs),
            "successful_jobs": int(successful_jobs),
            "job_success_ratio": (
                float(successful_jobs) / float(total_jobs) if total_jobs > 0 else 0.0
            ),
            "over_deadline_ratio_all": (
                float(over_deadline_jobs) / float(total_jobs) if total_jobs > 0 else 0.0
            ),
            "sim_span_s": float(sim_span_s),
            "effective_fps": float(effective_fps),
            "avg_total_time_all_s": float(avg_total_time_all) if avg_total_time_all is not None else None,
            "avg_total_time_executed_s": (
                float(avg_total_time_executed) if avg_total_time_executed is not None else None
            ),
            "avg_job_size_mb": avg_job_size_mb,
            "probing_energy_by_node_wh": probing_energy_by_node,
            "total_probing_energy_wh": total_probing_energy_wh,
            "total_battery_capacity_wh": total_battery_capacity_wh,
            "probing_energy_share": probing_energy_share,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute aggregate statistics from a log.db file "
            "(jobs, deadlines, effective FPS, basic latencies)."
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

    # -------------------------------------------------------------------------
    # Simulation configuration overview (printed first)
    # -------------------------------------------------------------------------
    cfg = stats["sim_config"]
    print("Simulation configuration:")
    if cfg:
        def _get(name: str, default: str = "n/a") -> str:
            return cfg.get(name, default)

        print(f"- NET_SPEED_SCHEDULER_WORKER_MBIT: {_get('NET_SPEED_SCHEDULER_WORKER_MBIT')}")
        print(f"- PROBE_SIZE_BYTES: {_get('PROBE_SIZE_BYTES')}")
        print(f"- PROBE_CROSSFACTOR_J: {_get('PROBE_CROSSFACTOR_J')}")
        print(f"- PROBING_ENERGY_COST_WH: {_get('PROBING_ENERGY_COST_WH')}")
        print(f"- POWER_MAX_TRANSMISSION_W: {_get('POWER_MAX_TRANSMISSION_W')}")
        print(f"- WORKER_BATTERY_CAPACITIES: {_get('WORKER_BATTERY_CAPACITIES')}")
    else:
        print("  (no sim_config table in this log.db)")

    print()
    print("Aggregate job and timing statistics:")
    print(f"- total_jobs: {stats['total_jobs']}")
    print(f"- executed_jobs: {stats['executed_jobs']}")
    print(f"- rejected_jobs: {stats['rejected_jobs']}")
    print(
        f"- over_deadline_jobs: {stats['over_deadline_jobs']} "
        f"({stats['over_deadline_ratio_all'] * 100:.2f}% of all jobs)"
    )
    print(
        f"- successful_jobs: {stats['successful_jobs']} "
        f"({stats['job_success_ratio'] * 100:.2f}% of all jobs)"
    )
    print(f"- sim_span_s: {stats['sim_span_s']:.2f}")
    print(f"- effective_fps (executed_jobs / sim_span_s): {stats['effective_fps']:.3f}")

    avg_all = stats['avg_total_time_all_s']
    avg_exec = stats['avg_total_time_executed_s']
    if avg_all is not None:
        print(f"- avg_total_time_all_s: {avg_all:.4f}")
    else:
        print("- avg_total_time_all_s: n/a")
    if avg_exec is not None:
        print(f"- avg_total_time_executed_s: {avg_exec:.4f}")
    else:
        print("- avg_total_time_executed_s: n/a")

    print(f"- avg_job_size_mb (expected): {stats['avg_job_size_mb']:.6f}")

    # Probing-specific statistics (only for runs where we logged probing energy).
    pe_by_node = stats["probing_energy_by_node_wh"]
    if pe_by_node:
        print("- probing_energy_by_node_wh:")
        for node_uid, energy_wh in sorted(pe_by_node.items()):
            print(f"  - node {node_uid}: {energy_wh:.6e} Wh")
        print(f"- total_probing_energy_wh: {stats['total_probing_energy_wh']:.6e} Wh")
        print(f"- total_battery_capacity_wh: {stats['total_battery_capacity_wh']:.3f} Wh")
        print(
            f"- probing_energy_share: {stats['probing_energy_share'] * 100:.4f}% "
            "(total_probing_energy / sum_worker_capacities)"
        )
    else:
        print("- probing_energy: n/a (no probing_energy table in this log.db)")


if __name__ == "__main__":
    main()
