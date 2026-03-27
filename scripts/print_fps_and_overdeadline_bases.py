#!/usr/bin/env python3
from __future__ import annotations

"""
Print FPS base (per job type) and over-deadline base from a `log.db`.

Definitions match this repo's plotting/metrics:
- FPS_base_i is computed using the same bucketing as
  `code/utils_plot.py::UtilsPlot.plot_data_average_client_fps_time()`, i.e.:
  - jobs with: node_uid = <node_uid>, type = <job_type>, executed = 1, rejected = 0
  - finish_time = generated_at + time_total, and we group by cast(finish_time as int)
  - compute lag_time = finish_time - LAG(finish_time) over generated_at ordering
  - keep only rows with lag_time > 0
  - average buckets every `--average-every-secs` seconds (default 250)
  - print FPS_base_i as mean of those plotted bucket values.

- over_deadline_base is:
  SUM(over_deadline) / COUNT(*) over jobs with executed = 1 AND rejected = 0.
"""

import argparse
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class FpsBaseResult:
    job_type: int
    fps_base: float
    samples: int
    min_sample: float | None
    max_sample: float | None


def compute_over_deadline_base(
    db_path: str,
) -> Tuple[float, int, int]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(SUM(over_deadline), 0), COUNT(*)
            FROM jobs
            WHERE executed = 1 AND rejected = 0
            """
        )
        over_sum, n = cur.fetchone()
        over_sum_i = int(over_sum)
        n_i = int(n)
        ratio = float(over_sum_i) / float(n_i) if n_i > 0 else 0.0
        return ratio, over_sum_i, n_i
    finally:
        conn.close()


def _fetch_finish_time_counts(
    db_path: str,
    *,
    node_uid: int,
    job_type: int,
) -> Iterable[Tuple[int, int]]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        res = cur.execute(
            f"""
            SELECT
                cast(finish_time as int) AS t_sec,
                count(*)
            FROM (
                SELECT
                    id,
                    generated_at + time_total AS finish_time,
                    (generated_at + time_total)
                      - LAG(generated_at + time_total, 1) OVER (ORDER BY generated_at) AS lag_time
                FROM jobs
                WHERE node_uid = {node_uid}
                  AND type = {job_type}
                  AND executed = 1
                  AND rejected = 0
            ) sub
            WHERE lag_time > 0
            GROUP BY cast(finish_time as int)
            """
        )
        for t_sec, count in res:
            yield int(t_sec), int(count)
    finally:
        conn.close()


def compute_fps_base_for_type(
    db_path: str,
    *,
    node_uid: int,
    job_type: int,
    average_every_secs: int,
) -> FpsBaseResult:
    x_samples: List[int] = []
    y_samples: List[float] = []

    sum_reward = 0.0
    added = 0

    for t_sec, count in _fetch_finish_time_counts(
        db_path, node_uid=node_uid, job_type=job_type
    ):
        sum_reward += count
        added += 1
        if t_sec % average_every_secs == 0 and t_sec > 0:
            y_samples.append(sum_reward / added if added > 0 else 0.0)
            x_samples.append(t_sec)
            added = 0
            sum_reward = 0.0

    if not y_samples:
        return FpsBaseResult(
            job_type=job_type,
            fps_base=0.0,
            samples=0,
            min_sample=None,
            max_sample=None,
        )

    fps_base = sum(y_samples) / len(y_samples)
    return FpsBaseResult(
        job_type=job_type,
        fps_base=fps_base,
        samples=len(y_samples),
        min_sample=min(y_samples),
        max_sample=max(y_samples),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print per-type FPS base and over-deadline base from a log.db."
    )
    parser.add_argument("db_path", type=str, help="Path to a log.db file")
    parser.add_argument("--node-uid", type=int, default=0, help="node_uid used for FPS")
    parser.add_argument(
        "--average-every-secs",
        type=int,
        default=250,
        help="Bucket averaging window (matches plot_stacked_performances.py default)",
    )
    parser.add_argument(
        "--job-types",
        type=str,
        default="0,1,2",
        help="Comma-separated job types (default: 0,1,2)",
    )
    args = parser.parse_args()

    job_types: List[int] = [int(x.strip()) for x in args.job_types.split(",") if x.strip() != ""]

    over_ratio, over_sum, n = compute_over_deadline_base(args.db_path)

    print(f"Database: {args.db_path}")
    print(
        "over_deadline_base (executed=1 & rejected=0): "
        f"{over_sum}/{n} = {over_ratio * 100:.4f}%"
    )
    print()

    for jt in job_types:
        res = compute_fps_base_for_type(
            args.db_path,
            node_uid=args.node_uid,
            job_type=jt,
            average_every_secs=args.average_every_secs,
        )
        print(
            f"fps_base_{jt} (type={jt}) = {res.fps_base:.3f} fps "
            f"(samples={res.samples}"
            + (f", min={res.min_sample:.3f}, max={res.max_sample:.3f}" if res.samples > 0 else "")
            + ")"
        )


if __name__ == "__main__":
    main()

