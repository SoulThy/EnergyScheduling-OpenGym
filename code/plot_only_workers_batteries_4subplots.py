from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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


def _hex_color(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


TAB10 = [
    _hex_color((31, 119, 180)),
    _hex_color((255, 127, 14)),
    _hex_color((44, 160, 44)),
    _hex_color((214, 39, 40)),
    _hex_color((148, 103, 189)),
    _hex_color((140, 86, 75)),
    _hex_color((227, 119, 194)),
    _hex_color((127, 127, 127)),
    _hex_color((188, 189, 34)),
    _hex_color((23, 190, 207)),
]


def _svg_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _polyline(points: Iterable[tuple[float, float]]) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" points="{pts}"/>'


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
    out_svg = args.out_dir / f"{args.out_stem}.svg"

    # Consistent colors across subplots.
    worker_to_color = {wid: TAB10[i % len(TAB10)] for i, wid in enumerate(worker_ids)}

    # ---- Simple SVG layout (no external deps) ----
    # Canvas size tuned to look decent in thesis and be easily converted to PDF.
    width = 1100
    height = 800

    margin_outer = 60
    legend_h = 70

    grid_w = width - 2 * margin_outer
    grid_h = height - 2 * margin_outer - legend_h

    cols = 2
    rows = 2
    cell_w = grid_w / cols
    cell_h = grid_h / rows

    pad_l = 55
    pad_r = 15
    pad_t = 28
    pad_b = 40

    t_max = max(global_t_max, 1e-9)
    b_max = max(global_b_max * 1.02, 1e-9)

    def x_map(t: float, x0: float, w: float) -> float:
        return x0 + (t / t_max) * w

    def y_map(b: float, y0: float, h: float) -> float:
        # SVG y grows downward.
        return y0 + h - (b / b_max) * h

    def add(s: str) -> None:
        svg_parts.append(s)

    svg_parts: list[str] = []
    add('<?xml version="1.0" encoding="UTF-8"?>')
    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    add(
        "<style>"
        ".axis{stroke:#000;stroke-width:1}"
        ".grid{stroke:#cacaca;stroke-width:0.8;stroke-dasharray:4 4;opacity:0.8}"
        ".line{stroke-width:1.2}"
        ".title{font-family:serif;font-size:16px;font-weight:600}"
        ".label{font-family:serif;font-size:14px}"
        ".tick{font-family:serif;font-size:12px;fill:#111}"
        "</style>"
    )

    # Global labels
    add(
        f'<text x="{width/2:.1f}" y="{height - margin_outer/2:.1f}" text-anchor="middle" class="label">'
        f"{_svg_escape('Time (s)')}</text>"
    )
    add(
        f'<text x="{margin_outer/2:.1f}" y="{height/2:.1f}" text-anchor="middle" class="label" '
        f'transform="rotate(-90 {margin_outer/2:.1f},{height/2:.1f})">'
        f"{_svg_escape('Battery residual (Wh)')}</text>"
    )

    # Ticks (keep minimal and consistent)
    x_ticks = 5
    y_ticks = 5

    for idx, (spec, traces) in enumerate(traces_by_policy):
        r = idx // cols
        c = idx % cols

        cell_x0 = margin_outer + c * cell_w
        cell_y0 = margin_outer + r * cell_h

        plot_x0 = cell_x0 + pad_l
        plot_y0 = cell_y0 + pad_t
        plot_w = cell_w - pad_l - pad_r
        plot_h = cell_h - pad_t - pad_b

        # Title
        add(
            f'<text x="{cell_x0 + cell_w/2:.1f}" y="{cell_y0 + 18:.1f}" text-anchor="middle" class="title">'
            f"{_svg_escape(spec.pretty)}</text>"
        )

        # Grid + ticks
        for i in range(x_ticks + 1):
            tx = plot_x0 + (i / x_ticks) * plot_w
            add(f'<line x1="{tx:.2f}" y1="{plot_y0:.2f}" x2="{tx:.2f}" y2="{plot_y0 + plot_h:.2f}" class="grid"/>')
            t_val = (i / x_ticks) * t_max
            add(
                f'<text x="{tx:.2f}" y="{plot_y0 + plot_h + 18:.2f}" text-anchor="middle" class="tick">'
                f"{int(round(t_val))}</text>"
            )
        for i in range(y_ticks + 1):
            ty = plot_y0 + plot_h - (i / y_ticks) * plot_h
            add(f'<line x1="{plot_x0:.2f}" y1="{ty:.2f}" x2="{plot_x0 + plot_w:.2f}" y2="{ty:.2f}" class="grid"/>')
            b_val = (i / y_ticks) * b_max
            add(
                f'<text x="{plot_x0 - 8:.2f}" y="{ty + 4:.2f}" text-anchor="end" class="tick">'
                f"{b_val:.1f}</text>"
            )

        # Axes
        add(f'<line x1="{plot_x0:.2f}" y1="{plot_y0:.2f}" x2="{plot_x0:.2f}" y2="{plot_y0 + plot_h:.2f}" class="axis"/>')
        add(f'<line x1="{plot_x0:.2f}" y1="{plot_y0 + plot_h:.2f}" x2="{plot_x0 + plot_w:.2f}" y2="{plot_y0 + plot_h:.2f}" class="axis"/>')

        # Lines
        for wid in worker_ids:
            series = traces.get(wid)
            if not series:
                continue
            pts = [(x_map(t, plot_x0, plot_w), y_map(b, plot_y0, plot_h)) for t, b in series]
            add(
                _polyline(pts).replace(
                    "/>",
                    f' class="line" stroke="{worker_to_color[wid]}"/>',
                )
            )

    # Legend
    leg_y0 = height - margin_outer - legend_h + 25
    leg_x0 = margin_outer + 10
    step = min(240, (width - 2 * margin_outer) / max(1, len(worker_ids)))
    for i, wid in enumerate(worker_ids):
        x = leg_x0 + i * step
        add(
            f'<line x1="{x:.1f}" y1="{leg_y0:.1f}" x2="{x + 28:.1f}" y2="{leg_y0:.1f}" '
            f'stroke="{worker_to_color[wid]}" stroke-width="3"/>'
        )
        add(
            f'<text x="{x + 36:.1f}" y="{leg_y0 + 4:.1f}" text-anchor="start" class="tick">'
            f"{_svg_escape(f'Worker {wid}')}</text>"
        )

    add("</svg>")
    out_svg.write_text("\n".join(svg_parts), encoding="utf-8")

    print(f"[OK] Wrote {out_svg}")
    for spec, db_path in policy_to_db:
        print(f"[DB] {spec.pretty}: {db_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

