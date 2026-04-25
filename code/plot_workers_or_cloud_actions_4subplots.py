from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PolicySpec:
    policy_dir: str
    pretty: str


POLICIES_WORKERS_OR_CLOUD: list[PolicySpec] = [
    PolicySpec("LEAST_LOADED_AWARE_CLOUD", "LLAC"),
    PolicySpec("MAXIMUM_LIFESPANE", "ML"),
    PolicySpec("RANDOM", "RAND"),
    PolicySpec("SCORE_SIMPLE", "LBF"),
]


def _latest_workers_or_cloud_db(no_learning_root: Path, policy_dir: str) -> Path:
    policy_root = no_learning_root / policy_dir
    if not policy_root.exists():
        raise FileNotFoundError(f"Policy directory not found: {policy_root}")

    # Expected: <DATE>_<POLICY>_WORKERS_OR_CLOUD/log.db
    runs = sorted(
        [p for p in policy_root.iterdir() if p.is_dir() and p.name.endswith("_WORKERS_OR_CLOUD")]
    )
    if not runs:
        raise FileNotFoundError(f"No *_WORKERS_OR_CLOUD runs found under {policy_root}")

    db_path = runs[-1] / "log.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing log.db: {db_path}")
    return db_path


def _simulation_time_s(db_path: Path, scheduler_uid: int) -> int:
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT MAX(generated_at) FROM jobs WHERE node_uid = ?",
            (scheduler_uid,),
        )
        row = cur.fetchone()
        tmax = float(row[0] or 0.0)
    finally:
        con.close()
    return int(math.ceil(tmax))


def _counts_per_second(
    db_path: Path, scheduler_uid: int
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """
    Return dicts keyed by second:
      - total jobs
      - reject jobs (rejected=1)
      - cloud jobs  (rejected=0 AND forwarded_to_cloud=1)

    Worker/edge jobs are computed as: total - reject - cloud (where cloud excludes rejected jobs).
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
            WHERE node_uid = ?
            GROUP BY CAST(generated_at AS INTEGER)
            """,
            (scheduler_uid,),
        )
        for t, c in cur.fetchall():
            total[int(t)] = int(c)

        cur.execute(
            """
            SELECT CAST(generated_at AS INTEGER) AS t, COUNT(*)
            FROM jobs
            WHERE node_uid = ? AND rejected = 1
            GROUP BY CAST(generated_at AS INTEGER)
            """,
            (scheduler_uid,),
        )
        for t, c in cur.fetchall():
            rej[int(t)] = int(c)

        cur.execute(
            """
            SELECT CAST(generated_at AS INTEGER) AS t, COUNT(*)
            FROM jobs
            WHERE node_uid = ? AND rejected = 0 AND forwarded_to_cloud = 1
            GROUP BY CAST(generated_at AS INTEGER)
            """,
            (scheduler_uid,),
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
    parser = argparse.ArgumentParser(
        description=(
            "Plot WORKERS_OR_CLOUD action percentages over time as 4 subplots "
            "(one per policy) from results/data/_log/no-learning/**/log.db"
        )
    )
    parser.add_argument(
        "--no-learning-root",
        type=Path,
        default=Path("results/data/_log/no-learning"),
        help="Root directory containing no-learning logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plot"),
        help="Output directory for thesis plots.",
    )
    parser.add_argument(
        "--out-stem",
        type=str,
        default="workers_or_cloud_actions_4subplots",
        help="Output filename stem (without extension).",
    )
    parser.add_argument(
        "--window-s",
        type=int,
        default=300,
        help="Window size in seconds for averaging action percentages.",
    )
    parser.add_argument(
        "--scheduler-uid",
        type=int,
        default=0,
        help="Scheduler node UID in the jobs table.",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="10,6.5",
        help="Figure size in inches as 'W,H' (e.g. '10,6.5').",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PNG output.",
    )
    args = parser.parse_args()

    policy_to_db: list[tuple[PolicySpec, Path]] = []
    for spec in POLICIES_WORKERS_OR_CLOUD:
        policy_to_db.append((spec, _latest_workers_or_cloud_db(args.no_learning_root, spec.policy_dir)))

    # Gather windowed series per policy.
    series_by_policy: list[tuple[PolicySpec, list[int], list[float], list[float], list[float]]] = []
    global_t_max = 0

    for spec, db_path in policy_to_db:
        sim_t = _simulation_time_s(db_path, scheduler_uid=args.scheduler_uid)
        total, rej, cloud = _counts_per_second(db_path, scheduler_uid=args.scheduler_uid)
        x, y_rej, y_cloud, y_worker = _windowed_percentages(
            total=total,
            rej=rej,
            cloud=cloud,
            simulation_time_s=sim_t,
            window_s=args.window_s,
        )
        series_by_policy.append((spec, x, y_rej, y_cloud, y_worker))
        if x:
            global_t_max = max(global_t_max, max(x))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = args.out_dir / f"{args.out_stem}.pdf"
    out_png = args.out_dir / f"{args.out_stem}.png"

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 1.4,
        }
    )

    try:
        fig_w_s, fig_h_s = args.figsize.split(",", 1)
        fig_w = float(fig_w_s.strip())
        fig_h = float(fig_h_s.strip())
    except Exception as e:
        raise SystemExit(f"Invalid --figsize '{args.figsize}', expected 'W,H'") from e

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(fig_w, fig_h))
    axes_list = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    # Fixed, high-contrast colors for the 3 macro-actions.
    col_worker = "#1f77b4"  # blue
    col_cloud = "#9467bd"   # purple
    col_rej = "#d62728"     # red

    legend_handles = None
    for ax, (spec, x, y_rej, y_cloud, y_worker) in zip(axes_list, series_by_policy):
        (l_worker,) = ax.plot(x, y_worker, color=col_worker, marker="o", markersize=3.0, label="Worker (edge)")
        (l_cloud,) = ax.plot(x, y_cloud, color=col_cloud, marker="s", markersize=3.0, label="Cloud")
        (l_rej,) = ax.plot(x, y_rej, color=col_rej, marker="^", markersize=3.0, label="Reject")

        if legend_handles is None:
            legend_handles = (l_worker, l_cloud, l_rej)

        ax.set_title(spec.pretty, fontweight="semibold")
        ax.grid(color="#cacaca", linestyle="--", linewidth=0.6, alpha=0.8)
        ax.set_ylim(0.0, 100.0)

    fig.supxlabel("Time (s)", y=0.08)
    fig.supylabel("Action share (%)")

    if legend_handles is not None:
        fig.legend(
            handles=list(legend_handles),
            labels=["Worker (edge)", "Cloud", "Reject"],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
        )

    # Shared x-limits for fair comparison.
    if global_t_max > 0:
        for ax in axes_list:
            ax.set_xlim(0.0, float(global_t_max))

    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Wrote {out_pdf}")
    print(f"[OK] Wrote {out_png}")
    for spec, db_path in policy_to_db:
        print(f"[DB] {spec.pretty}: {db_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

