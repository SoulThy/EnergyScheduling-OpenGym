#!/usr/bin/env python3
"""
Plot probing-energy share and job success ratio vs probing packet size
from a probe-size sweep (run_simulation_d_sarsa_probe_sweep.py).

Finds all log.db under results/data/_log/learning/D_SARSA/ONLY_WORKERS/*B_PS/,
calls log_simulation_db.compute_stats() on each, and produces a single figure
with one x-axis (probing packet size) and two y-axes:
- left: probing energy share (red)
- right: job success ratio (green)

Usage:
  From repo root or code/:  python code/plot_probe_sweep.py [--results-dir PATH] [--out FILE]
  Default results dir: code/../results/data  (so results/data when run from code/)
  Default output: results/data/probe_sweep_graph.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Import after potential path setup so we can run from repo root or code/
import sys
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_simulation_db import compute_stats


def find_probe_sweep_dbs(results_data: Path) -> List[Path]:
    """Return list of log.db paths from probe-sweep runs (*B_PS session ids)."""
    base = results_data / "_log" / "learning" / "D_SARSA" / "ONLY_WORKERS"
    if not base.exists():
        return []
    dbs: List[Path] = []
    for d in base.iterdir():
        if not d.is_dir() or not d.name.endswith("B_PS"):
            continue
        db = d / "log.db"
        if db.exists():
            dbs.append(db)
    return dbs


def load_sweep_stats(db_paths: List[Path]) -> List[Dict[str, Any]]:
    """Load compute_stats for each db; return list of stats dicts with probe_size_bytes."""
    rows: List[Dict[str, Any]] = []
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
        rows.append({
            "probe_size_bytes": probe_bytes,
            "probing_energy_share": stats["probing_energy_share"],
            "job_success_ratio": stats["job_success_ratio"],
            "db_path": db_path,
        })
    return sorted(rows, key=lambda r: r["probe_size_bytes"])


def plot_sweep(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Draw probing_energy_share and job_success_ratio vs probe size."""
    if not rows:
        print("No data to plot.", file=sys.stderr)
        return

    x = [r["probe_size_bytes"] for r in rows]
    share = [r["probing_energy_share"] for r in rows]
    success = [r["job_success_ratio"] for r in rows]

    fig, ax_left = plt.subplots(figsize=(6, 4))

    # Left y-axis: probing energy share (red)
    ax_left.plot(x, share, color="red", marker="o", markersize=5, label="Probing energy share")
    ax_left.set_xlabel("Probing packet size [bytes] (log scale)")
    ax_left.set_ylabel("Probing energy share [%]", color="red")
    ax_left.tick_params(axis="y", labelcolor="red")
    ax_left.set_ylim(0, (max(share) * 1.05 if share else 1.0))
    ax_left.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_left.grid(True, which="both", axis="both", alpha=0.3)

    # Logarithmic scale on x-axis to match geometric probe-size sweep.
    ax_left.set_xscale("log")
    # Force ticks exactly at the simulated probe sizes so their values are
    # clearly readable on the log axis.
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([str(v) for v in x])

    # Right y-axis: job success ratio (green)
    ax_right = ax_left.twinx()
    ax_right.plot(x, success, color="green", marker="s", markersize=5, label="Job success ratio")
    ax_right.set_ylabel("Job success ratio [%]", color="green")
    ax_right.tick_params(axis="y", labelcolor="green")
    ax_right.set_ylim(0, 1.05)
    ax_right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    plt.tight_layout()
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
        help="Output figure path (default: <results-dir>/probe_sweep_graph.png).",
    )
    args = parser.parse_args()

    results_data = args.results_dir.resolve()
    out_path = args.out if args.out is not None else results_data / "probe_sweep_graph.png"
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


if __name__ == "__main__":
    main()
