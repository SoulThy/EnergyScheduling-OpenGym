#!/usr/bin/env python3
"""
Plot probing-energy share and job success ratio vs probing packet size
from a probe-size sweep (run_simulation_d_sarsa_probe_sweep.py).

Finds all log.db under results/data/_log/learning/D_SARSA/ONLY_WORKERS/*B_PS/,
calls log_simulation_db.compute_stats() on each, and produces:
- One double y-axis figure (probing energy share + job success ratio vs probe size)
  stored under _log (e.g. .../ONLY_WORKERS/probe_sweep_graph.png).
- One pie chart of energy share (processing / transmission / probing) per simulation,
  stored in each simulation folder next to its log.db (e.g. .../200B_PS/energy_pie.png).

Note on sigma (battery variance) and 60FPS: When probing packet size increases,
the 60FPS worker often dies earlier than 15/30FPS workers because it runs more
jobs (highest load) and thus consumes more execution+transmission energy; extra
probing cost is applied to all workers equally, so the already busiest node hits
zero first and sigma (variance of battery levels) increases.

Usage:
  From repo root or code/:  python code/plot_probe_sweep.py [--results-dir PATH] [--out FILE]
  Default results dir: code/../results/data  (so results/data when run from code/)
  Default output: <results-dir>/_log/learning/D_SARSA/ONLY_WORKERS/probe_sweep_graph.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

# Import after potential path setup so we can run from repo root or code/
import sys
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_simulation_db import compute_stats


def find_probe_sweep_dbs(results_data: Path) -> List[Path]:
    """Return list of log.db paths from probe-sweep runs (dirs containing B_PS, e.g. 200B_PS or 200B_PS_s0)."""
    base = results_data / "_log" / "learning" / "D_SARSA" / "ONLY_WORKERS"
    if not base.exists():
        return []
    dbs: List[Path] = []
    for d in base.iterdir():
        if not d.is_dir() or "B_PS" not in d.name:
            continue
        db = d / "log.db"
        if db.exists():
            dbs.append(db)
    return dbs


def load_sweep_stats(
    db_paths: List[Path], aggregate_seeds: bool = True
) -> List[Dict[str, Any]]:
    """Load compute_stats for each db. If aggregate_seeds, group by probe size and average."""
    raw: List[Dict[str, Any]] = []
    for db_path in db_paths:
        try:
            stats = compute_stats(db_path)
        except Exception as e:
            print(f"Warning: skip {db_path}: {e}", file=sys.stderr)
            continue
        cfg = stats.get("sim_config") or {}
        try:
            probe_bytes = int(cfg.get("PROBE_SIZE_BYTES", 0))
        except (TypeError, ValueError):
            continue
        if "probing_energy_share" not in stats or "job_success_ratio" not in stats:
            continue
        row = {
            "probe_size_bytes": probe_bytes,
            "probing_energy_share": stats["probing_energy_share"],
            "job_success_ratio": stats["job_success_ratio"],
            "db_path": db_path,
        }
        if "execution_energy_share" in stats:
            row["execution_energy_share"] = stats["execution_energy_share"]
            row["transmission_energy_share"] = stats["transmission_energy_share"]
            row["idle_energy_share"] = stats.get("idle_energy_share", 0.0)
        raw.append(row)

    if not aggregate_seeds or not raw:
        return sorted(raw, key=lambda r: (r["probe_size_bytes"], str(r["db_path"])))

    # Group by probe_size_bytes and average (smooths curve when N_SEEDS > 1).
    from collections import defaultdict

    by_size: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in raw:
        by_size[r["probe_size_bytes"]].append(r)

    aggregated: List[Dict[str, Any]] = []
    for probe_bytes in sorted(by_size.keys()):
        group = by_size[probe_bytes]
        n = len(group)
        agg = {
            "probe_size_bytes": probe_bytes,
            "probing_energy_share": sum(r["probing_energy_share"] for r in group) / n,
            "job_success_ratio": sum(r["job_success_ratio"] for r in group) / n,
            "db_path": group[0]["db_path"],
        }
        if "execution_energy_share" in group[0]:
            agg["execution_energy_share"] = sum(r["execution_energy_share"] for r in group) / n
            agg["transmission_energy_share"] = sum(r["transmission_energy_share"] for r in group) / n
            agg["idle_energy_share"] = sum(r.get("idle_energy_share", 0) for r in group) / n
        if n > 1:
            agg["job_success_ratio_std"] = (
                (sum(r["job_success_ratio"] ** 2 for r in group) / n - agg["job_success_ratio"] ** 2) ** 0.5
            )
            agg["probing_energy_share_std"] = (
                (sum(r["probing_energy_share"] ** 2 for r in group) / n - agg["probing_energy_share"] ** 2) ** 0.5
            )
        aggregated.append(agg)
    return aggregated


def plot_sweep(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Draw probing_energy_share and job_success_ratio vs probe size (means, optional error bars)."""
    if not rows:
        print("No data to plot.", file=sys.stderr)
        return

    x = [r["probe_size_bytes"] for r in rows]
    share = [r["probing_energy_share"] for r in rows]
    success = [r["job_success_ratio"] for r in rows]
    share_std = [r.get("probing_energy_share_std", 0) for r in rows]
    success_std = [r.get("job_success_ratio_std", 0) for r in rows]

    fig, ax_left = plt.subplots(figsize=(6, 4))

    # Left y-axis: probing energy share (red)
    ax_left.plot(x, share, color="red", marker="o", markersize=5, label="Probing energy share")
    if any(share_std):
        ax_left.fill_between(x, [s - std for s, std in zip(share, share_std)], [s + std for s, std in zip(share, share_std)], color="red", alpha=0.2)
    ax_left.set_xlabel("Probing packet size (log scale)")
    ax_left.set_ylabel("Probing energy share [%]", color="red")
    ax_left.tick_params(axis="y", labelcolor="red")
    ax_left.set_ylim(0, (max(share) * 1.05 if share else 1.0))
    ax_left.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_left.grid(True, which="both", axis="both", alpha=0.3)

    # Logarithmic scale on x-axis to match geometric probe-size sweep.
    ax_left.set_xscale("log")
    # Force ticks exactly at the simulated probe sizes; show >= 10000 as KB, else as B.
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(
        [f"{v // 1000} KB" if v >= 10000 else f"{v} B" for v in x],
        rotation=45,
        ha="right",
    )

    # Right y-axis: job success ratio (green), zoomed to data range, ticks every 2%
    ax_right = ax_left.twinx()
    ax_right.plot(x, success, color="green", marker="s", markersize=5, label="Job success ratio")
    if any(success_std):
        ax_right.fill_between(x, [s - std for s, std in zip(success, success_std)], [s + std for s, std in zip(success, success_std)], color="green", alpha=0.2)
    ax_right.set_ylabel("Job success ratio [%]", color="green")
    ax_right.tick_params(axis="y", labelcolor="green")
    if success:
        pad = max(0.02, (max(success) - min(success)) * 0.3)
        ax_right.set_ylim(max(0, min(success) - pad), min(1.05, max(success) + pad))
    else:
        ax_right.set_ylim(0, 1.05)
    ax_right.yaxis.set_major_locator(MultipleLocator(0.02))  # 2% steps for zoomed view
    ax_right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _format_probe_size(probe_bytes: int) -> str:
    """Format probe size as 'N B' or 'N KB' for display (>= 10000 bytes -> KB)."""
    return f"{probe_bytes // 1000} KB" if probe_bytes >= 10000 else f"{probe_bytes} B"


def plot_energy_pie(row: Dict[str, Any], out_path: Path) -> None:
    """Draw a minimal pie chart of energy share: processing, transmission, probing (and idle)."""
    labels = []
    sizes = []
    # Soft, minimal palette: processing, transmission, probing, idle
    colors = ["#5b9bd5", "#70ad47", "#ed7d31", "#a5a5a5"]
    if row.get("execution_energy_share", 0) > 0:
        labels.append("Processing")
        sizes.append(row["execution_energy_share"])
    if row.get("transmission_energy_share", 0) > 0:
        labels.append("Transmission")
        sizes.append(row["transmission_energy_share"])
    if row.get("probing_energy_share", 0) > 0:
        labels.append("Probing")
        sizes.append(row["probing_energy_share"])
    idle = row.get("idle_energy_share", 0)
    if idle > 0:
        labels.append("Idle")
        sizes.append(idle)
    if not sizes:
        print("No energy breakdown available for pie chart.", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    wedgeprops = {"linewidth": 1.0, "edgecolor": "white"}
    textprops = {"size": 10, "color": "#333333"}
    ax.pie(
        sizes,
        labels=labels,
        colors=colors[: len(sizes)],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=wedgeprops,
        textprops=textprops,
        pctdistance=0.75,
        labeldistance=1.05,
    )
    probe_str = _format_probe_size(row["probe_size_bytes"])
    ax.set_title(f"Energy share — {probe_str}", fontsize=11, color="#444444")
    ax.set_axis_off()
    plt.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot probe-sweep results: probing energy share and job success ratio vs packet size."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".." / "results" / "data",
        help="Path to results/data (contains _log/...).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the double y-axis sweep figure (default: under _log/.../ONLY_WORKERS/).",
    )
    args = parser.parse_args()

    results_data = args.results_dir.resolve()
    base_log = results_data / "_log" / "learning" / "D_SARSA" / "ONLY_WORKERS"
    out_path = (
        args.out.resolve() if args.out is not None else base_log / "probe_sweep_graph.png"
    )
    out_path = out_path.resolve()

    db_paths = find_probe_sweep_dbs(results_data)
    if not db_paths:
        print(
            f"No probe-sweep log.db found under {results_data}/_log/learning/D_SARSA/ONLY_WORKERS/*B_PS/",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(db_paths)} probe-sweep runs.")
    rows = load_sweep_stats(db_paths)
    if not rows:
        print("No valid stats (PROBE_SIZE_BYTES, probing_energy_share, job_success_ratio).", file=sys.stderr)
        sys.exit(1)

    plot_sweep(rows, out_path)

    # One pie chart per simulation run, stored in that run's folder (next to log.db).
    for db_path in db_paths:
        try:
            stats = compute_stats(db_path)
        except Exception:
            continue
        if "execution_energy_share" not in stats:
            continue
        probe_bytes = int((stats.get("sim_config") or {}).get("PROBE_SIZE_BYTES", 0))
        row = {
            "probe_size_bytes": probe_bytes,
            "probing_energy_share": stats["probing_energy_share"],
            "job_success_ratio": stats["job_success_ratio"],
            "execution_energy_share": stats["execution_energy_share"],
            "transmission_energy_share": stats["transmission_energy_share"],
            "idle_energy_share": stats.get("idle_energy_share", 0.0),
        }
        plot_energy_pie(row, db_path.parent / "energy_pie.png")


if __name__ == "__main__":
    main()
