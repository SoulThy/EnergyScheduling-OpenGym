from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt


def _simulation_time_s(db_path: Path, scheduler_uid: int) -> int:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT MAX(generated_at) FROM jobs WHERE node_uid = ?", (scheduler_uid,))
        row = cur.fetchone()
        tmax = float(row[0] or 0.0)
    finally:
        con.close()
    return int(math.ceil(tmax))


def _counts_per_second(
    db_path: Path, scheduler_uid: int, warmup_time_s: float
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """
    Return dicts keyed by second:
      - total jobs
      - reject jobs (rejected=1)
      - cloud jobs  (rejected=0 AND forwarded_to_cloud=1)

    Worker/edge jobs are computed as: total - reject - cloud.
    """
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        total: dict[int, int] = {}
        rej: dict[int, int] = {}
        cloud: dict[int, int] = {}

        cur.execute(
            """
            SELECT CAST(generated_at AS INTEGER) AS t, COUNT(*)
            FROM jobs
            WHERE node_uid = ? AND generated_at > ?
            GROUP BY CAST(generated_at AS INTEGER)
            """,
            (scheduler_uid, warmup_time_s),
        )
        for t, c in cur.fetchall():
            total[int(t)] = int(c)

        cur.execute(
            """
            SELECT CAST(generated_at AS INTEGER) AS t, COUNT(*)
            FROM jobs
            WHERE node_uid = ? AND generated_at > ? AND rejected = 1
            GROUP BY CAST(generated_at AS INTEGER)
            """,
            (scheduler_uid, warmup_time_s),
        )
        for t, c in cur.fetchall():
            rej[int(t)] = int(c)

        cur.execute(
            """
            SELECT CAST(generated_at AS INTEGER) AS t, COUNT(*)
            FROM jobs
            WHERE node_uid = ? AND generated_at > ?
              AND rejected = 0 AND forwarded_to_cloud = 1
            GROUP BY CAST(generated_at AS INTEGER)
            """,
            (scheduler_uid, warmup_time_s),
        )
        for t, c in cur.fetchall():
            cloud[int(t)] = int(c)
    finally:
        con.close()

    return total, rej, cloud


def _windowed_percentages(
    total: dict[int, int],
    rej: dict[int, int],
    cloud: dict[int, int],
    simulation_time_s: int,
    window_s: int,
) -> tuple[list[int], list[float], list[float], list[float]]:
    x: list[int] = []
    y_rej: list[float] = []
    y_cloud: list[float] = []
    y_worker: list[float] = []

    if window_s <= 0:
        window_s = 1

    sum_total = 0
    sum_rej = 0
    sum_cloud = 0

    for t in range(simulation_time_s + 1):
        sum_total += total.get(t, 0)
        sum_rej += rej.get(t, 0)
        sum_cloud += cloud.get(t, 0)

        if t > 0 and (t % window_s == 0):
            if sum_total <= 0:
                pr = 0.0
                pc = 0.0
                pw = 0.0
            else:
                pr = 100.0 * float(sum_rej) / float(sum_total)
                pc = 100.0 * float(sum_cloud) / float(sum_total)
                pw = 100.0 * float(max(0, sum_total - sum_rej - sum_cloud)) / float(sum_total)
            x.append(t)
            y_rej.append(pr)
            y_cloud.append(pc)
            y_worker.append(pw)
            sum_total = 0
            sum_rej = 0
            sum_cloud = 0

    return x, y_rej, y_cloud, y_worker


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Cloud/Edge/Reject share over time for a single log.db.")
    parser.add_argument("--db-path", type=Path, required=True, help="Path to log.db")
    parser.add_argument("--window-s", type=int, default=120, help="Window size in seconds")
    parser.add_argument("--warmup-time-s", type=float, default=0.0, help="Ignore generated_at <= warmup_time_s")
    parser.add_argument("--scheduler-uid", type=int, default=0, help="Scheduler node UID in jobs table")
    parser.add_argument("--t-fail", type=float, default=None, help="Failure time marker (s)")
    parser.add_argument("--t-recover", type=float, default=None, help="Recovery time marker (s)")
    parser.add_argument("--out", type=Path, default=Path("results/plot/actions_over_time_single.pdf"))
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PNG output (if --out endswith .png)")
    args = parser.parse_args()

    sim_t = _simulation_time_s(args.db_path, scheduler_uid=args.scheduler_uid)
    total_s, rej_s, cloud_s = _counts_per_second(
        args.db_path, scheduler_uid=args.scheduler_uid, warmup_time_s=args.warmup_time_s
    )
    x, y_rej, y_cloud, y_worker = _windowed_percentages(
        total=total_s,
        rej=rej_s,
        cloud=cloud_s,
        simulation_time_s=sim_t,
        window_s=args.window_s,
    )

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 1.6,
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    col_worker = "#1f77b4"
    col_cloud = "#9467bd"
    col_rej = "#d62728"

    ax.plot(x, y_worker, color=col_worker, marker="o", markersize=3.0, label="Worker (edge)")
    ax.plot(x, y_cloud, color=col_cloud, marker="s", markersize=3.0, label="Cloud")
    ax.plot(x, y_rej, color=col_rej, marker="^", markersize=3.0, label="Reject")

    ax.grid(color="#cacaca", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_ylabel("Action share (%)")
    ax.set_ylim(0.0, 100.0)

    if args.t_fail is not None:
        ax.axvline(float(args.t_fail), color=col_rej, linestyle="--", linewidth=1.2, alpha=0.8)
    if args.t_recover is not None:
        ax.axvline(float(args.t_recover), color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8)

    # Legend inside plot + figure-level x-label avoids overlap with bottom ticks/label.
    ax.legend(frameon=False, loc="upper right", fontsize="small", ncol=1)
    fig.supxlabel("Time (s)", y=0.02)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".png":
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    else:
        fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

