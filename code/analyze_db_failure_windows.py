from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WindowStats:
    name: str
    n_total: int
    n_reject: int
    n_cloud: int
    n_edge: int
    n_ontime: int
    gamma_eff: float
    gamma_cond: float | None
    avg_time_queue_s: float | None
    p95_time_queue_s: float | None


def _q1(con: sqlite3.Connection, q: str, params: tuple[object, ...]) -> float | None:
    cur = con.cursor()
    cur.execute(q, params)
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return float(row[0])


def _count(con: sqlite3.Connection, q: str, params: tuple[object, ...]) -> int:
    cur = con.cursor()
    cur.execute(q, params)
    row = cur.fetchone()
    return int(row[0] or 0)


def _p95_time_queue(con: sqlite3.Connection, scheduler_uid: int, t0: float, t1: float) -> float | None:
    # Use ORDER BY + LIMIT/OFFSET (ok on SQLite, dataset size is manageable for thesis runs).
    n = _count(
        con,
        """
        SELECT COUNT(*)
        FROM jobs
        WHERE node_uid = ? AND generated_at >= ? AND generated_at < ?
          AND rejected = 0
        """,
        (scheduler_uid, t0, t1),
    )
    if n <= 0:
        return None
    idx = int(round(0.95 * (n - 1)))
    cur = con.cursor()
    cur.execute(
        """
        SELECT time_queue
        FROM jobs
        WHERE node_uid = ? AND generated_at >= ? AND generated_at < ?
          AND rejected = 0
        ORDER BY time_queue ASC
        LIMIT 1 OFFSET ?
        """,
        (scheduler_uid, t0, t1, idx),
    )
    row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _stats_for_window(
    con: sqlite3.Connection,
    scheduler_uid: int,
    t0: float,
    t1: float,
    name: str,
) -> WindowStats:
    n_total = _count(
        con,
        "SELECT COUNT(*) FROM jobs WHERE node_uid = ? AND generated_at >= ? AND generated_at < ?",
        (scheduler_uid, t0, t1),
    )
    n_reject = _count(
        con,
        """
        SELECT COUNT(*)
        FROM jobs
        WHERE node_uid = ? AND generated_at >= ? AND generated_at < ?
          AND rejected = 1
        """,
        (scheduler_uid, t0, t1),
    )
    n_cloud = _count(
        con,
        """
        SELECT COUNT(*)
        FROM jobs
        WHERE node_uid = ? AND generated_at >= ? AND generated_at < ?
          AND rejected = 0 AND forwarded_to_cloud = 1
        """,
        (scheduler_uid, t0, t1),
    )
    n_ontime = _count(
        con,
        """
        SELECT COUNT(*)
        FROM jobs
        WHERE node_uid = ? AND generated_at >= ? AND generated_at < ?
          AND rejected = 0 AND done = 1 AND over_deadline = 0
        """,
        (scheduler_uid, t0, t1),
    )
    # Conditional gamma (only among done & non-rejected).
    denom_cond = _count(
        con,
        """
        SELECT COUNT(*)
        FROM jobs
        WHERE node_uid = ? AND generated_at >= ? AND generated_at < ?
          AND rejected = 0 AND done = 1
        """,
        (scheduler_uid, t0, t1),
    )

    gamma_eff = float(n_ontime) / float(n_total) if n_total > 0 else 0.0
    gamma_cond = float(n_ontime) / float(denom_cond) if denom_cond > 0 else None

    n_edge = max(0, n_total - n_reject - n_cloud)

    avg_time_queue_s = _q1(
        con,
        """
        SELECT AVG(time_queue)
        FROM jobs
        WHERE node_uid = ? AND generated_at >= ? AND generated_at < ?
          AND rejected = 0
        """,
        (scheduler_uid, t0, t1),
    )
    p95 = _p95_time_queue(con, scheduler_uid=scheduler_uid, t0=t0, t1=t1)

    return WindowStats(
        name=name,
        n_total=n_total,
        n_reject=n_reject,
        n_cloud=n_cloud,
        n_edge=n_edge,
        n_ontime=n_ontime,
        gamma_eff=gamma_eff,
        gamma_cond=gamma_cond,
        avg_time_queue_s=avg_time_queue_s,
        p95_time_queue_s=p95,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze pre-failure / failure / post-recovery windows from a single log.db."
    )
    parser.add_argument("--db-path", type=Path, required=True, help="Path to log.db")
    parser.add_argument("--scheduler-uid", type=int, default=0, help="Scheduler node UID in jobs table")
    parser.add_argument("--t-fail", type=float, default=4000.0, help="Failure start time (s)")
    parser.add_argument("--t-recover", type=float, default=8000.0, help="Recovery time (s)")
    parser.add_argument("--pre-window-s", type=float, default=2000.0, help="Window length before failure")
    parser.add_argument("--post-window-s", type=float, default=2000.0, help="Window length after recovery")
    args = parser.parse_args()

    con = sqlite3.connect(str(args.db_path))
    try:
        pre0 = max(0.0, args.t_fail - args.pre_window_s)
        pre1 = args.t_fail
        fail0 = args.t_fail
        fail1 = args.t_recover
        post0 = args.t_recover
        post1 = args.t_recover + args.post_window_s

        stats = [
            _stats_for_window(con, args.scheduler_uid, pre0, pre1, "pre"),
            _stats_for_window(con, args.scheduler_uid, fail0, fail1, "failure"),
            _stats_for_window(con, args.scheduler_uid, post0, post1, "post"),
        ]

        def fmt_f(v: float | None, nd: int = 3) -> str:
            if v is None:
                return "-"
            return f"{v:.{nd}f}"

        print(f"DB: {args.db_path}")
        print(f"windows: pre=[{pre0:.0f},{pre1:.0f}) fail=[{fail0:.0f},{fail1:.0f}) post=[{post0:.0f},{post1:.0f})")
        print()
        for s in stats:
            rej_p = 100.0 * s.n_reject / s.n_total if s.n_total else 0.0
            cloud_p = 100.0 * s.n_cloud / s.n_total if s.n_total else 0.0
            edge_p = 100.0 * s.n_edge / s.n_total if s.n_total else 0.0
            print(f"[{s.name}] total={s.n_total} ontime={s.n_ontime} "
                  f"gamma_eff={fmt_f(s.gamma_eff,3)} gamma_cond={fmt_f(s.gamma_cond,3)}")
            print(f"       reject%={rej_p:.1f} cloud%={cloud_p:.1f} edge%={edge_p:.1f}")
            print(f"       avg_queue_s={fmt_f(s.avg_time_queue_s,4)} p95_queue_s={fmt_f(s.p95_time_queue_s,4)}")
            print()

    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

