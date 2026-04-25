from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PolicySpec:
    policy_dir: str
    pretty: str


POLICIES_ONLY_WORKERS: list[PolicySpec] = [
    PolicySpec("LEAST_LOADED_NOT_AWARE", "LL"),
    PolicySpec("MAXIMUM_LIFESPANE", "ML"),
    PolicySpec("RANDOM", "RAND"),
    PolicySpec("SCORE_SIMPLE", "LBF"),
]


def _latest_only_workers_db(no_learning_root: Path, policy_dir: str) -> Path:
    policy_root = no_learning_root / policy_dir
    if not policy_root.exists():
        raise FileNotFoundError(f"Policy directory not found: {policy_root}")

    # Expected: <DATE>_<POLICY>_ONLY_WORKERS/log.db
    runs = sorted([p for p in policy_root.iterdir() if p.is_dir() and p.name.endswith("_ONLY_WORKERS")])
    if not runs:
        raise FileNotFoundError(f"No *_ONLY_WORKERS runs found under {policy_root}")

    db_path = runs[-1] / "log.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing log.db: {db_path}")
    return db_path


def _read_round_batteries(db_path: Path) -> dict[int, list[tuple[float, float]]]:
    """
    Return per-worker battery traces from the `round` table.

    Output:
      {worker_id: [(time_s, battery_residual_wh), ...], ...}
    """
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT time, worker_id, battery_residual FROM round ORDER BY time ASC")
        rows = cur.fetchall()
    finally:
        con.close()

    out: dict[int, list[tuple[float, float]]] = {}
    for t, wid, b in rows:
        wid_i = int(wid)
        out.setdefault(wid_i, []).append((float(t), float(b)))
    return out


def _downsample(series: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    if max_points <= 0 or len(series) <= max_points:
        return series
    step = max(1, len(series) // max_points)
    out = series[::step]
    if out[-1] != series[-1]:
        out.append(series[-1])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ONLY_WORKERS battery residuals as 4 subplots (one per policy) "
            "from results/data/_log/no-learning/**/log.db"
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
        default="only_workers_batteries_4subplots",
        help="Output filename stem (without extension).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=3500,
        help="Max points per line (downsampling) to keep plots light and readable.",
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
    for spec in POLICIES_ONLY_WORKERS:
        policy_to_db.append((spec, _latest_only_workers_db(args.no_learning_root, spec.policy_dir)))

    traces_by_policy: list[tuple[PolicySpec, dict[int, list[tuple[float, float]]]]] = []
    all_worker_ids: set[int] = set()
    global_t_max = 0.0
    global_b_max = 0.0

    for spec, db_path in policy_to_db:
        traces = _read_round_batteries(db_path)
        traces_by_policy.append((spec, traces))
        all_worker_ids.update(traces.keys())
        for wid, series in traces.items():
            for t, b in series:
                global_t_max = max(global_t_max, t)
                global_b_max = max(global_b_max, b)

    worker_ids = sorted(all_worker_ids)
    if not worker_ids:
        raise SystemExit("No workers found in `round` table (empty traces).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = args.out_dir / f"{args.out_stem}.pdf"
    out_png = args.out_dir / f"{args.out_stem}.png"

    # Matplotlib styling tuned for thesis readability.
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 1.2,
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

    colors = plt.get_cmap("tab10")
    worker_to_color = {wid: colors(i % 10) for i, wid in enumerate(worker_ids)}

    xlim = (0.0, global_t_max if global_t_max > 0 else 1.0)
    ylim = (0.0, global_b_max * 1.02 if global_b_max > 0 else 1.0)

    legend_handles: list[Optional[plt.Line2D]] = [None for _ in worker_ids]

    for ax, (spec, traces) in zip(axes_list, traces_by_policy):
        for idx_w, wid in enumerate(worker_ids):
            series = traces.get(wid)
            if not series:
                continue
            series = _downsample(series, args.max_points)
            xs = [p[0] for p in series]
            ys = [p[1] for p in series]
            (line,) = ax.plot(xs, ys, color=worker_to_color[wid])
            if legend_handles[idx_w] is None:
                legend_handles[idx_w] = line

        ax.set_title(spec.pretty, fontweight="semibold")
        ax.grid(color="#cacaca", linestyle="--", linewidth=0.6, alpha=0.8)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    # Keep global x-label above the figure-level legend.
    fig.supxlabel("Time (s)", y=0.08)
    fig.supylabel("Battery residual (Wh)")

    handles = [h for h in legend_handles if h is not None]
    labels = [f"Worker {wid}" for wid, h in zip(worker_ids, legend_handles) if h is not None]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )

    # Reserve bottom space for supxlabel + legend (avoid overlap).
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

