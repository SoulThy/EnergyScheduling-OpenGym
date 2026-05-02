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

POLICIES_WORKERS_OR_CLOUD: list[PolicySpec] = [
    PolicySpec("LEAST_LOADED_AWARE_CLOUD", "LLAC"),
    PolicySpec("MAXIMUM_LIFESPANE", "ML"),
    PolicySpec("RANDOM", "RAND"),
    PolicySpec("SCORE_SIMPLE", "LBF"),
]


def _latest_db_for(
    no_learning_root: Path,
    policy_dir: str,
    scenario_suffix: str,
    name_contains: str | None,
    run_name_contains: str | None,
    failure_mode: str,
) -> Path:
    """
    Discover the latest run directory under:
      <no_learning_root>/<policy_dir>/
    matching endswith(scenario_suffix) or endswith(scenario_suffix + '_FAILURE'),
    plus optional name filters (same logic as plot_no_learning_gamma_over_time_4subplots).
    """
    policy_root = no_learning_root / policy_dir
    if not policy_root.exists():
        raise FileNotFoundError(f"Policy directory not found: {policy_root}")

    runs = []
    for p in policy_root.iterdir():
        if not p.is_dir():
            continue
        is_failure = p.name.endswith(f"{scenario_suffix}_FAILURE")
        is_non_failure = p.name.endswith(scenario_suffix)

        if failure_mode == "only":
            if not is_failure:
                continue
        elif failure_mode == "non":
            if not is_non_failure:
                continue
        elif failure_mode == "any":
            if not (is_non_failure or is_failure):
                continue
        else:
            raise RuntimeError(f"Invalid failure_mode='{failure_mode}', expected: any|only|non")

        if name_contains and name_contains not in p.name:
            continue
        if run_name_contains and run_name_contains not in p.name:
            continue
        runs.append(p)

    runs = sorted(runs)
    if not runs:
        extra = f" containing '{name_contains}'" if name_contains else ""
        raise FileNotFoundError(f"No run found under {policy_root} ending with '{scenario_suffix}'{extra}")

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
            "Plot battery residuals from `round` as 4 subplots (one per policy). "
            "Supports ONLY_WORKERS and WORKERS_OR_CLOUD (including *_FAILURE runs)."
        )
    )
    parser.add_argument(
        "--no-learning-root",
        type=Path,
        default=Path("results/data/_log/no-learning"),
        help="Root directory containing no-learning logs.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["ONLY_WORKERS", "WORKERS_OR_CLOUD"],
        default="ONLY_WORKERS",
        help="Scenario suffix to pick run directories.",
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        default="",
        help=(
            "Substring that must appear in the run directory name. "
            "Use 'FAILURE' for failure runs; pass '' to disable."
        ),
    )
    parser.add_argument(
        "--run-name-contains",
        type=str,
        default="",
        help="Extra substring to pick a specific date/tag among multiple runs (e.g. '20260429').",
    )
    parser.add_argument(
        "--failure-mode",
        type=str,
        choices=["any", "only", "non"],
        default="any",
        help=(
            "Choose which run type to pick when both are present: "
            "'non' for *_<SCENARIO>, 'only' for *_<SCENARIO>_FAILURE, 'any' for either."
        ),
    )
    parser.add_argument(
        "--t-fail",
        type=float,
        default=None,
        help="Optional failure time (s) for a vertical marker.",
    )
    parser.add_argument(
        "--t-recover",
        type=float,
        default=None,
        help="Optional recovery time (s) for a vertical marker.",
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
        default="",
        help="Output filename stem (without extension). Default depends on --scenario.",
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

    scenario_suffix = f"_{args.scenario}"
    name_contains = args.name_contains if args.name_contains.strip() else None
    run_name_contains = args.run_name_contains if args.run_name_contains.strip() else None

    policies = POLICIES_ONLY_WORKERS if args.scenario == "ONLY_WORKERS" else POLICIES_WORKERS_OR_CLOUD

    out_stem = args.out_stem.strip()
    if not out_stem:
        out_stem = (
            "only_workers_batteries_4subplots"
            if args.scenario == "ONLY_WORKERS"
            else "workers_or_cloud_batteries_4subplots"
        )

    policy_to_db: list[tuple[PolicySpec, Path]] = []
    for spec in policies:
        policy_to_db.append(
            (
                spec,
                _latest_db_for(
                    no_learning_root=args.no_learning_root,
                    policy_dir=spec.policy_dir,
                    scenario_suffix=scenario_suffix,
                    name_contains=name_contains,
                    run_name_contains=run_name_contains,
                    failure_mode=args.failure_mode,
                ),
            )
        )

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
    out_pdf = args.out_dir / f"{out_stem}.pdf"
    out_png = args.out_dir / f"{out_stem}.png"

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

    fail_color = "#d62728"
    rec_color = "#2ca02c"

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

        if args.t_fail is not None:
            ax.axvline(float(args.t_fail), color=fail_color, linestyle="--", linewidth=1.2)
        if args.t_recover is not None:
            ax.axvline(float(args.t_recover), color=rec_color, linestyle="--", linewidth=1.2)

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
