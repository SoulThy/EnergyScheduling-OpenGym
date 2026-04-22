import argparse
import csv
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_WARMUP_TIME_S = 0.0


@dataclass(frozen=True)
class MetricsRow:
    policy: str
    scenario: str  # ONLY_WORKERS | WORKERS_OR_CLOUD
    db_path: Path
    sigma_var_wh: Optional[float]
    delta_gap_wh_first_death: Optional[float]
    m_first_death_s: Optional[float]
    M_last_death_s: Optional[float]
    gamma_qos: Optional[float]
    gamma_qos_type0: Optional[float]
    gamma_qos_type1: Optional[float]
    gamma_qos_type2: Optional[float]
    reject_rate: Optional[float]
    t_service_total_s: Optional[float]


def _discover_log_dbs(no_learning_root: Path) -> list[Path]:
    return sorted(no_learning_root.glob("*/*/log.db"))


def _infer_policy_and_scenario(db_path: Path) -> tuple[str, str]:
    # Expected:
    # results/data/_log/no-learning/<POLICY>/<SESSION>_<POLICY>_<SCENARIO>/log.db
    policy = db_path.parents[1].name
    run_dir = db_path.parent.name
    if run_dir.endswith("_ONLY_WORKERS"):
        scenario = "ONLY_WORKERS"
    elif run_dir.endswith("_WORKERS_OR_CLOUD"):
        scenario = "WORKERS_OR_CLOUD"
    else:
        # Fallback: try regex search
        m = re.search(r"_(ONLY_WORKERS|WORKERS_OR_CLOUD)$", run_dir)
        scenario = m.group(1) if m else "UNKNOWN"
    return policy, scenario


def _q1(cur: sqlite3.Cursor, query: str, params: tuple[object, ...] = ()) -> Optional[float]:
    cur.execute(query, params)
    row = cur.fetchone()
    if not row:
        return None
    val = row[0]
    if val is None:
        return None
    return float(val)


def _q2(cur: sqlite3.Cursor, query: str, params: tuple[object, ...] = ()) -> tuple[Optional[float], Optional[float]]:
    cur.execute(query, params)
    row = cur.fetchone()
    if not row:
        return None, None
    a, b = row
    return (float(a) if a is not None else None), (float(b) if b is not None else None)


def _safe_ratio(numer: int, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return float(numer) / float(denom)


def _count(cur: sqlite3.Cursor, query: str, params: tuple[object, ...] = ()) -> int:
    cur.execute(query, params)
    row = cur.fetchone()
    return int(row[0] or 0)


def _compute_metrics(db_path: Path, warmup_time_s: float) -> MetricsRow:
    policy, scenario = _infer_policy_and_scenario(db_path)

    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()

        # --- Energy fairness metrics ---
        sigma = _q1(
            cur,
            """
            SELECT AVG(variance)
            FROM round
            WHERE time > ?
            """,
            (warmup_time_s,),
        )

        # Gap max-min residual at first death (as logged by simulator).
        delta = _q1(
            cur,
            """
            SELECT max_battery
            FROM end_batteries
            WHERE time = (SELECT MIN(time) FROM end_batteries)
            """,
        )

        m_first, M_last = _q2(
            cur,
            """
            SELECT MIN(time), MAX(time)
            FROM end_batteries
            """,
        )

        # --- QoS metrics (deadline) + rejects ---
        # We evaluate over the whole simulation by default (warmup_time_s=0).
        #
        # IMPORTANT (thesis): gamma should not penalize a policy for rejecting jobs, otherwise reject_rate is
        # counted twice. So we compute gamma over COMPLETED jobs only:
        #   gamma = (done within deadline) / (done and not rejected)
        total_jobs = _count(
            cur,
            """
            SELECT COUNT(*)
            FROM jobs
            WHERE node_uid = 0 AND generated_at > ?
            """,
            (warmup_time_s,),
        )
        rejected_jobs = _count(
            cur,
            """
            SELECT COUNT(*)
            FROM jobs
            WHERE node_uid = 0 AND generated_at > ? AND rejected = 1
            """,
            (warmup_time_s,),
        )

        done_non_rejected_total = _count(
            cur,
            """
            SELECT COUNT(*)
            FROM jobs
            WHERE node_uid = 0 AND generated_at > ?
              AND rejected = 0 AND done = 1
            """,
            (warmup_time_s,),
        )

        ontime_total = _count(
            cur,
            """
            SELECT COUNT(*)
            FROM jobs
            WHERE node_uid = 0 AND generated_at > ?
              AND rejected = 0 AND done = 1 AND over_deadline = 0
            """,
            (warmup_time_s,),
        )

        def ontime_for_type(job_type: int) -> tuple[int, int]:
            denom = _count(
                cur,
                """
                SELECT COUNT(*)
                FROM jobs
                WHERE node_uid = 0 AND generated_at > ?
                  AND type = ?
                  AND rejected = 0 AND done = 1
                """,
                (warmup_time_s, job_type),
            )
            numer = _count(
                cur,
                """
                SELECT COUNT(*)
                FROM jobs
                WHERE node_uid = 0 AND generated_at > ?
                  AND type = ?
                  AND rejected = 0 AND done = 1 AND over_deadline = 0
                """,
                (warmup_time_s, job_type),
            )
            return numer, denom

        n0, d0 = ontime_for_type(0)
        n1, d1 = ontime_for_type(1)
        # Bugfix vs legacy plot_tables.py: type=2 must filter type=2 (not type=1).
        n2, d2 = ontime_for_type(2)

        gamma = _safe_ratio(ontime_total, done_non_rejected_total)
        gamma0 = _safe_ratio(n0, d0)
        gamma1 = _safe_ratio(n1, d1)
        gamma2 = _safe_ratio(n2, d2)
        rej = _safe_ratio(rejected_jobs, total_jobs)

        # --- Total worker service time ---
        # "Tempo di servizio totale": sum of worker execution times (exclude cloud-executed jobs).
        t_service = _q1(
            cur,
            """
            SELECT SUM(time_execution)
            FROM jobs
            WHERE node_uid = 0 AND generated_at > ?
              AND rejected = 0 AND executed = 1 AND forwarded_to_cloud = 0
            """,
            (warmup_time_s,),
        )

        return MetricsRow(
            policy=policy,
            scenario=scenario,
            db_path=db_path,
            sigma_var_wh=sigma,
            delta_gap_wh_first_death=delta,
            m_first_death_s=m_first,
            M_last_death_s=M_last,
            gamma_qos=gamma,
            gamma_qos_type0=gamma0,
            gamma_qos_type1=gamma1,
            gamma_qos_type2=gamma2,
            reject_rate=rej,
            t_service_total_s=t_service,
        )
    finally:
        con.close()


def _fmt_opt_float(val: Optional[float], ndigits: int = 2) -> str:
    if val is None:
        return "-"
    return f"{val:.{ndigits}f}"


def _fmt_opt_percent(val: Optional[float], ndigits: int = 1) -> str:
    if val is None:
        return "-"
    return f"{val * 100.0:.{ndigits}f}"


def _policy_pretty_name(policy: str) -> str:
    mapping = {
        "LEAST_LOADED_AWARE_CLOUD": "LLAC",
        "LEAST_LOADED_NOT_AWARE": "LL",
        "MAXIMUM_LIFESPANE": "ML",
        "RANDOM": "RAND",
        "SCORE_SIMPLE": "LBF",
    }
    return mapping.get(policy, policy)


def _write_csv(out_path: Path, rows: Iterable[MetricsRow]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "alg",
                "sigma",
                "delta",
                "m",
                "M",
                "gamma",
                "gamma_0",
                "gamma_1",
                "gamma_2",
                "r_rej",
                "ts",
            ]
        )

        def fmt_2(val: Optional[float]) -> str:
            if val is None:
                return ""
            return f"{val:.2f}"

        def fmt_int(val: Optional[float]) -> str:
            if val is None:
                return ""
            return str(int(round(val)))

        for r in rows:
            w.writerow(
                [
                    _policy_pretty_name(r.policy),
                    fmt_2(r.sigma_var_wh),
                    fmt_2(r.delta_gap_wh_first_death),
                    fmt_int(r.m_first_death_s),
                    fmt_int(r.M_last_death_s),
                    fmt_2(r.gamma_qos),
                    fmt_2(r.gamma_qos_type0),
                    fmt_2(r.gamma_qos_type1),
                    fmt_2(r.gamma_qos_type2),
                    fmt_2(r.reject_rate),
                    fmt_int(r.t_service_total_s),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thesis CSV tables for NO-LEARNING policies (ONLY_WORKERS vs WORKERS_OR_CLOUD) "
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
        "--warmup-time-s",
        type=float,
        default=DEFAULT_WARMUP_TIME_S,
        help="Ignore initial transient by filtering generated_at/time > warmup_time_s.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/data/_tables/no-learning-thesis"),
        help="Output directory for CSV tables.",
    )
    args = parser.parse_args()

    dbs = _discover_log_dbs(args.no_learning_root)
    if not dbs:
        raise SystemExit(f"No log.db found under {args.no_learning_root}")

    rows: list[MetricsRow] = []
    for db_path in dbs:
        rows.append(_compute_metrics(db_path, warmup_time_s=args.warmup_time_s))

    # Split by scenario.
    scenarios = sorted({r.scenario for r in rows})
    for scenario in scenarios:
        scenario_rows = [r for r in rows if r.scenario == scenario]
        out_csv = args.out_dir / f"no_learning_metrics_{scenario}.csv"

        _write_csv(out_csv, scenario_rows)

        print(f"[OK] Wrote {out_csv}")

    # Also print a small, copy-paste friendly preview.
    print("\nPreview (policy, scenario, sigma, delta, m, M, gamma%, rej%, t_s):")
    for r in sorted(rows, key=lambda x: (x.scenario, x.policy)):
        print(
            f"- {r.policy:24s} {r.scenario:16s} "
            f"sigma={_fmt_opt_float(r.sigma_var_wh,2)} "
            f"delta={_fmt_opt_float(r.delta_gap_wh_first_death,2)} "
            f"m={_fmt_opt_float(r.m_first_death_s,0)} "
            f"M={_fmt_opt_float(r.M_last_death_s,0)} "
            f"gamma%={_fmt_opt_percent(r.gamma_qos,1)} "
            f"rej%={_fmt_opt_percent(r.reject_rate,1)} "
            f"t_s={_fmt_opt_float(r.t_service_total_s,0)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

