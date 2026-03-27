#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _fetch_finish_time_counts(
    db_path: Path,
    *,
    node_uid: int,
    job_type: int,
) -> Iterable[Tuple[int, int]]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        res = cur.execute(
            f"""
            SELECT cast(finish_time as int), count(*)
            FROM (
                SELECT
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


def fps_by_type(
    db_path: Path,
    *,
    node_uid: int,
    job_types: List[int],
    average_every_secs: int,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for job_type in job_types:
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
                sum_reward = 0.0
                added = 0
        out[job_type] = (sum(y_samples) / len(y_samples)) if y_samples else 0.0
    return out


def over_deadline_ratio(db_path: Path) -> float:
    conn = sqlite3.connect(str(db_path))
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
        over_i = int(over_sum)
        n_i = int(n)
        return (float(over_i) / float(n_i)) if n_i > 0 else 0.0
    finally:
        conn.close()


def probe_over_execution_percent(db_path: Path) -> float:
    repo_root = Path(__file__).resolve().parents[1]
    code_dir = repo_root / "code"
    sys.path.insert(0, str(code_dir))
    from log_simulation_db import compute_stats  # type: ignore

    stats = compute_stats(db_path)
    return float(stats.get("probe_over_execution_percent", 0.0) or 0.0)


@dataclass
class Row:
    k: int
    fps_ratio: float
    fps_error_percent: float
    over_deadline_ratio: float
    over_deadline_increase_percent: float
    probe_over_execution_percent: float
    policy_ok: bool


def extract_k_from_path(db_path: Path) -> int | None:
    m = re.search(r"_K(\d+)", str(db_path))
    if not m:
        return None
    return int(m.group(1))


def collect_k_dbs(runs_dir: Path) -> List[Path]:
    dbs = sorted(runs_dir.glob("**/log.db"))
    out = [p for p in dbs if extract_k_from_path(p) is not None]
    return out


def is_db_readable(db_path: Path) -> bool:
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM jobs LIMIT 1")
        cur.fetchone()
        conn.close()
        return True
    except sqlite3.DatabaseError:
        return False
    except sqlite3.OperationalError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build K-policy table from baseline db and a folder of K runs."
    )
    parser.add_argument(
        "baseline_db",
        type=Path,
        help="Reference K=1 log.db used to compute bases.",
    )
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Directory containing K run folders (session names must include `_K<value>`).",
    )
    parser.add_argument(
        "--eps-fps",
        type=float,
        required=True,
        help="Max FPS relative error (fraction), e.g. 0.02 for 2%%.",
    )
    parser.add_argument(
        "--eps-deadline",
        type=float,
        required=True,
        help="Max deadline-ratio increase (fraction), e.g. 0.01 for +1%% point.",
    )
    parser.add_argument(
        "--average-every-secs",
        type=int,
        default=250,
        help="FPS bucket averaging window (must match your plotting setup).",
    )
    parser.add_argument(
        "--node-uid",
        type=int,
        default=0,
        help="node_uid used for FPS computation (default 0 = scheduler/client view).",
    )
    parser.add_argument(
        "--job-types",
        type=str,
        default="0,1,2",
        help="Comma-separated periodic types used in FPS_ratio (default: 0,1,2).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("results/k_policy_table.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed/unreadable DBs instead of skipping them.",
    )
    args = parser.parse_args()

    job_types = [int(x.strip()) for x in args.job_types.split(",") if x.strip() != ""]

    fps_base = fps_by_type(
        args.baseline_db,
        node_uid=args.node_uid,
        job_types=job_types,
        average_every_secs=args.average_every_secs,
    )
    over_base = over_deadline_ratio(args.baseline_db)

    dbs = collect_k_dbs(args.runs_dir)
    if not dbs:
        raise RuntimeError(f"No K run log.db found in {args.runs_dir} (need `_K<number>` in path).")

    rows: List[Row] = []
    skipped: List[Path] = []
    for db_path in dbs:
        k = extract_k_from_path(db_path)
        if k is None:
            continue
        if not is_db_readable(db_path):
            if args.strict:
                raise RuntimeError(f"Malformed or unreadable DB: {db_path}")
            skipped.append(db_path)
            print(f"[warn] skipping malformed/unreadable DB: {db_path}")
            continue

        fps_k = fps_by_type(
            db_path,
            node_uid=args.node_uid,
            job_types=job_types,
            average_every_secs=args.average_every_secs,
        )
        ratios = []
        for jt in job_types:
            b = fps_base[jt]
            r = (fps_k[jt] / b) if b > 0 else 0.0
            ratios.append(r)
        fps_ratio = min(ratios) if ratios else 0.0
        fps_error_percent = (1.0 - fps_ratio) * 100.0

        over_k = over_deadline_ratio(db_path)
        over_inc_percent = (over_k - over_base) * 100.0
        probe_pct = probe_over_execution_percent(db_path)

        policy_ok = (
            fps_error_percent <= args.eps_fps * 100.0
            and over_inc_percent <= args.eps_deadline * 100.0
        )

        rows.append(
            Row(
                k=k,
                fps_ratio=fps_ratio,
                fps_error_percent=fps_error_percent,
                over_deadline_ratio=over_k,
                over_deadline_increase_percent=over_inc_percent,
                probe_over_execution_percent=probe_pct,
                policy_ok=policy_ok,
            )
        )

    rows.sort(key=lambda r: r.k)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "k",
                "fps_ratio",
                "fps_error_%",
                "over_deadline_ratio",
                "over_deadline_increase_%",
                "probe_over_execution_%",
                "policy_OK",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.k,
                    f"{r.fps_ratio:.6f}",
                    f"{r.fps_error_percent:.4f}",
                    f"{r.over_deadline_ratio:.6f}",
                    f"{r.over_deadline_increase_percent:.4f}",
                    f"{r.probe_over_execution_percent:.4f}",
                    "YES" if r.policy_ok else "NO",
                ]
            )

    print("Policy table generated.")
    print(f"- baseline_db: {args.baseline_db}")
    print(f"- runs_dir: {args.runs_dir}")
    print(f"- eps_fps: {args.eps_fps} ({args.eps_fps * 100:.2f}%)")
    print(f"- eps_deadline: {args.eps_deadline} ({args.eps_deadline * 100:.2f}%)")
    print(f"- job_types: {job_types}, node_uid={args.node_uid}, average_every_secs={args.average_every_secs}")
    print(f"- over_deadline_ratio_base: {over_base:.6f} ({over_base * 100:.4f}%)")
    print("- fps_base_i:")
    for jt in job_types:
        print(f"  - type {jt}: {fps_base[jt]:.6f}")
    print(f"- rows: {len(rows)}")
    if skipped:
        print(f"- skipped_malformed_dbs: {len(skipped)}")
        for p in skipped:
            print(f"  - {p}")
    print(f"- csv_out: {args.csv_out.resolve()}")


if __name__ == "__main__":
    main()

