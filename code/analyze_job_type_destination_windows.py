from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Window:
    name: str
    t0: float
    t1: float


def _fetch_rows(con: sqlite3.Connection, scheduler_uid: int, t0: float, t1: float) -> list[tuple[int, int, int]]:
    """
    Returns rows:
      (type, rejected, forwarded_to_cloud)
    for jobs generated in [t0, t1) by the scheduler.

    Note: `ServiceDataStorage.done_job()` inserts a row when the job completes, so each row is
    a terminal outcome for that job (with final flags).
    """
    cur = con.cursor()
    cur.execute(
        """
        SELECT type, rejected, forwarded_to_cloud
        FROM jobs
        WHERE node_uid = ?
          AND generated_at >= ? AND generated_at < ?
        """,
        (scheduler_uid, t0, t1),
    )
    return [(int(r[0]), int(r[1]), int(r[2])) for r in cur.fetchall()]


def _classify_destination(rejected: int, forwarded_to_cloud: int) -> str:
    if rejected == 1:
        return "reject"
    if forwarded_to_cloud == 1:
        return "cloud"
    return "edge"


def _print_table(window: Window, rows: list[tuple[int, int, int]]) -> None:
    # counts[type][dest] = n
    counts: dict[int, dict[str, int]] = {}
    totals_by_type: dict[int, int] = {}
    totals_by_dest: dict[str, int] = {}

    for typ, rej, cloud in rows:
        dest = _classify_destination(rej, cloud)
        counts.setdefault(typ, {}).setdefault(dest, 0)
        counts[typ][dest] += 1
        totals_by_type[typ] = totals_by_type.get(typ, 0) + 1
        totals_by_dest[dest] = totals_by_dest.get(dest, 0) + 1

    print(f"[{window.name}] jobs={len(rows)} window=[{window.t0:.0f},{window.t1:.0f})")
    if not rows:
        print("  (empty)")
        print()
        return

    dests = sorted(totals_by_dest.keys(), key=lambda d: (-totals_by_dest[d], d))
    types = sorted(counts.keys())

    # Header
    header = "type".ljust(6) + " " + "total".rjust(8)
    for d in dests:
        header += " " + f"{d}%".rjust(10)
    header += " " + "cloud|acc%".rjust(12)
    print("  " + header)
    print("  " + ("-" * len(header)))

    for typ in types:
        tot_t = totals_by_type[typ]
        line = f"{typ}".ljust(6) + " " + f"{tot_t}".rjust(8)
        for d in dests:
            n = counts[typ].get(d, 0)
            pct = 100.0 * float(n) / float(tot_t) if tot_t > 0 else 0.0
            line += " " + f"{pct:6.1f}".rjust(10)
        acc = tot_t - counts[typ].get("reject", 0)
        cloud_n = counts[typ].get("cloud", 0)
        cloud_given_acc = 100.0 * float(cloud_n) / float(acc) if acc > 0 else 0.0
        line += " " + f"{cloud_given_acc:6.1f}".rjust(12)
        print("  " + line)

    # Totals row (percent over all jobs in window)
    tot_all = len(rows)
    line = "ALL".ljust(6) + " " + f"{tot_all}".rjust(8)
    for d in dests:
        n = totals_by_dest.get(d, 0)
        pct = 100.0 * float(n) / float(tot_all) if tot_all > 0 else 0.0
        line += " " + f"{pct:6.1f}".rjust(10)
    acc_all = tot_all - totals_by_dest.get("reject", 0)
    cloud_all = totals_by_dest.get("cloud", 0)
    cloud_given_acc_all = 100.0 * float(cloud_all) / float(acc_all) if acc_all > 0 else 0.0
    line += " " + f"{cloud_given_acc_all:6.1f}".rjust(12)
    print("  " + line)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze per-job-type destination mix (edge_exec vs cloud vs reject) "
            "over time windows, using scheduler-generated jobs rows."
        )
    )
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--scheduler-uid", type=int, default=0)
    parser.add_argument("--t-fail", type=float, default=4000.0)
    parser.add_argument("--t-recover", type=float, default=8000.0)
    parser.add_argument("--pre-window-s", type=float, default=2000.0)
    parser.add_argument("--post-window-s", type=float, default=2000.0)
    args = parser.parse_args()

    pre0 = max(0.0, args.t_fail - args.pre_window_s)
    pre1 = args.t_fail
    fail0 = args.t_fail
    fail1 = args.t_recover
    post0 = args.t_recover
    post1 = args.t_recover + args.post_window_s

    windows = [
        Window("pre", pre0, pre1),
        Window("failure", fail0, fail1),
        Window("post", post0, post1),
    ]

    con = sqlite3.connect(str(args.db_path))
    try:
        print(f"DB: {args.db_path}")
        print(f"windows: pre=[{pre0:.0f},{pre1:.0f}) fail=[{fail0:.0f},{fail1:.0f}) post=[{post0:.0f},{post1:.0f})")
        print()
        for w in windows:
            rows = _fetch_rows(con, args.scheduler_uid, w.t0, w.t1)
            _print_table(w, rows)
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
