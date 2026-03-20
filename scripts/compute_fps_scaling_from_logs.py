#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class TypeStats:
    job_type: int
    count: int
    avg_time_total_s: float


REQUIRED_JOBS_COLUMNS = {"type", "executed", "rejected", "time_total"}


def _validate_db_schema(conn: sqlite3.Connection, db_path: Path) -> None:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"Missing required table 'jobs' in {db_path}")

    cur.execute("PRAGMA table_info(jobs)")
    columns = {str(r[1]) for r in cur.fetchall()}
    missing = REQUIRED_JOBS_COLUMNS - columns
    if missing:
        raise RuntimeError(
            f"Missing required columns in jobs table for {db_path}: {', '.join(sorted(missing))}"
        )


def _load_stats(db_path: Path, job_types: Iterable[int] | None = None) -> Dict[int, TypeStats]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        _validate_db_schema(conn, db_path)
        cur = conn.cursor()

        base_query = (
            "SELECT type, COUNT(*) AS n, AVG(time_total) AS avg_time_total "
            "FROM jobs "
            "WHERE executed = 1 AND rejected = 0 AND time_total > 0"
        )
        params: List[int] = []
        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            base_query += f" AND type IN ({placeholders})"
            params.extend(job_types)

        base_query += " GROUP BY type ORDER BY type"
        cur.execute(base_query, params)
        rows = cur.fetchall()
    finally:
        conn.close()

    result: Dict[int, TypeStats] = {}
    for job_type, count, avg_time_total in rows:
        if avg_time_total is None:
            continue
        result[int(job_type)] = TypeStats(
            job_type=int(job_type),
            count=int(count),
            avg_time_total_s=float(avg_time_total),
        )
    return result


def _parse_job_types(raw: str | None) -> List[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(p.strip()) for p in raw.split(",") if p.strip() != ""]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute scaling factor k from two simulation logs.\n"
            "k is computed as: k = avg_time_legacy / avg_time_target.\n"
            "First input DB is the legacy reference, second is the target model."
        )
    )
    parser.add_argument("legacy_db", type=Path, help="Path to legacy log.db")
    parser.add_argument("target_db", type=Path, help="Path to target log.db")
    parser.add_argument(
        "--job-types",
        type=str,
        default="0,1,2",
        help=(
            "Comma-separated job types to include (default: 0,1,2). "
            "Use empty string to include all available types."
        ),
    )
    parser.add_argument(
        "--legacy-periodic-min",
        type=str,
        default="50,20,10",
        help="LEGACY periodic desired min FPS triplet (default: 50,20,10).",
    )
    parser.add_argument(
        "--legacy-periodic-max",
        type=str,
        default="60,30,15",
        help="LEGACY periodic desired max FPS triplet (default: 60,30,15).",
    )
    args = parser.parse_args()

    selected_types = _parse_job_types(args.job_types)
    legacy_stats = _load_stats(args.legacy_db, selected_types)
    target_stats = _load_stats(args.target_db, selected_types)

    common_types = sorted(set(legacy_stats.keys()) & set(target_stats.keys()))
    if not common_types:
        raise RuntimeError(
            "No common job types found between DBs for the selected filter. "
            "Check --job-types and the input files."
        )
    missing_in_legacy = sorted(set(target_stats.keys()) - set(legacy_stats.keys()))
    missing_in_target = sorted(set(legacy_stats.keys()) - set(target_stats.keys()))

    print(f"legacy_db: {args.legacy_db}")
    print(f"target_db: {args.target_db}")
    if missing_in_legacy:
        print(f"warning: types missing in legacy and skipped: {missing_in_legacy}")
    if missing_in_target:
        print(f"warning: types missing in target and skipped: {missing_in_target}")
    print()
    print("Per-type k (k = avg_time_legacy / avg_time_target):")

    weighted_num = 0.0
    weighted_den = 0.0
    k_by_type: Dict[int, float] = {}
    for job_type in common_types:
        ls = legacy_stats[job_type]
        ts = target_stats[job_type]
        k = ls.avg_time_total_s / ts.avg_time_total_s
        k_by_type[job_type] = k
        print(
            f"- type {job_type}: "
            f"legacy_avg_time={ls.avg_time_total_s:.8f}s, "
            f"target_avg_time={ts.avg_time_total_s:.8f}s, "
            f"k={k:.6f}, "
            f"n_legacy={ls.count}, n_target={ts.count}"
        )

        # Weighted global k: give each type a weight proportional to the
        # amount of target samples seen for that type.
        weighted_num += k * ts.count
        weighted_den += ts.count

    if weighted_den > 0:
        k_weighted = weighted_num / weighted_den
    else:
        k_weighted = 0.0

    print()
    print(f"k_weighted_by_target_counts={k_weighted:.6f}")
    print(
        "Interpretation: multiply LEGACY FPS min/max objectives by k to obtain "
        "a first coherent scaled range for the target model."
    )

    def _parse_triplet(raw: str, label: str) -> List[float]:
        values = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
        if len(values) != 3:
            raise ValueError(f"{label} must contain exactly 3 comma-separated values.")
        return values

    legacy_periodic_min = _parse_triplet(args.legacy_periodic_min, "legacy-periodic-min")
    legacy_periodic_max = _parse_triplet(args.legacy_periodic_max, "legacy-periodic-max")

    periodic_min_scaled = [v * k_weighted for v in legacy_periodic_min]
    periodic_max_scaled = [v * k_weighted for v in legacy_periodic_max]

    def _fmt_triplet(values: List[float]) -> str:
        return ",".join(f"{v:.2f}" for v in values)

    periodic_min_int = [int(round(v)) for v in periodic_min_scaled]
    periodic_max_int = [int(round(v)) for v in periodic_max_scaled]

    print()
    print("Suggested periodic desired FPS range for target (scaled with k_weighted):")
    print(f"- periodic_desired_rates_fps_min (float): {_fmt_triplet(periodic_min_scaled)}")
    print(f"- periodic_desired_rates_fps_max (float): {_fmt_triplet(periodic_max_scaled)}")
    print(
        f"- periodic_desired_rates_fps_min (int, rounded): "
        f"{','.join(str(v) for v in periodic_min_int)}"
    )
    print(
        f"- periodic_desired_rates_fps_max (int, rounded): "
        f"{','.join(str(v) for v in periodic_max_int)}"
    )

    # Type-wise proposal (more faithful when k differs across job types).
    # Map periodic triplets index -> job type (0,1,2).
    typewise_min: List[int] = []
    typewise_max: List[int] = []
    for idx, (legacy_min, legacy_max) in enumerate(zip(legacy_periodic_min, legacy_periodic_max)):
        k_t = k_by_type.get(idx, k_weighted)
        typewise_min.append(int(round(legacy_min * k_t)))
        typewise_max.append(int(round(legacy_max * k_t)))

    print()
    print("Suggested periodic desired FPS range for target (scaled per-type k):")
    print(f"- periodic_desired_rates_fps_min (int, rounded): {','.join(str(v) for v in typewise_min)}")
    print(f"- periodic_desired_rates_fps_max (int, rounded): {','.join(str(v) for v in typewise_max)}")


if __name__ == "__main__":
    main()
