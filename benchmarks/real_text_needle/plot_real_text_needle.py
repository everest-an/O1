"""Plot real-text needle retrieval supplement.

Provenance: original plotting script (contributed 2026-07) for the real-text
needle experiment. Reads results/<suite>/raw/real_text_needle_raw.csv produced
by run_real_text_needle.py in the same directory, and writes the summary table
+ recall-vs-length curve.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

COLORS = {"Transformer": "#4C566A", "LNN": "#7E8DBA", "Mamba": "#2EA69A",
          "MT-LNN": "#B44E4A", "MT-LNN-routeB": "#E0952A"}


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def summarize(rows):
    buckets = defaultdict(list)
    failures = []
    for row in rows:
        if row.get("status") != "ok":
            failures.append(row)
            continue
        buckets[(row["model"], int(row["context_length"]), float(row["needle_depth"]))].append(row)
    out = []
    for (model, length, depth), group in sorted(buckets.items()):
        vals = np.array([fnum(r["recall_accuracy"]) for r in group], dtype=float)
        params = np.array([fnum(r["num_params"]) for r in group], dtype=float)
        out.append({
            "model": model,
            "context_length": length,
            "needle_depth": depth,
            "recall_mean": float(np.nanmean(vals)),
            "recall_std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "num_params_mean": float(np.nanmean(params)),
            "n_seeds": len(group),
        })
    return out, failures


def write_dicts(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_all(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def make_curve(rows, path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for model in sorted({r["model"] for r in rows}):
        pts = []
        for length in sorted({int(r["context_length"]) for r in rows}):
            vals = [r for r in rows if r["model"] == model and int(r["context_length"]) == length]
            if vals:
                pts.append((length, np.nanmean([r["recall_mean"] for r in vals]), np.nanmean([r["recall_std"] for r in vals])))
        if not pts:
            continue
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        e = np.array([p[2] for p in pts], dtype=float)
        color = COLORS.get(model, "#555555")
        ax.plot(x, y, marker="o", linewidth=1.8, markersize=3.8, color=color, label=model)
        ax.fill_between(x, np.maximum(0, y - e), np.minimum(1, y + e), color=color, alpha=0.16, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(r["context_length"]) for r in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Real-text context length")
    ax.set_ylabel("Needle recall accuracy")
    ax.set_title("Real-text needle retrieval", loc="left", fontsize=9, fontweight="bold")
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_all(fig, path)
    plt.close(fig)


def make_table(rows, path: Path):
    models = sorted({r["model"] for r in rows})
    lengths = sorted({int(r["context_length"]) for r in rows})
    cells = []
    for model in models:
        vals = []
        sub_model = [r for r in rows if r["model"] == model]
        params = np.nanmean([r["num_params_mean"] for r in sub_model])
        for length in lengths:
            sub = [r for r in sub_model if int(r["context_length"]) == length]
            if sub:
                vals.append(f"{np.nanmean([r['recall_mean'] for r in sub]):.3f} +/- {np.nanmean([r['recall_std'] for r in sub]):.3f}")
            else:
                vals.append("skipped")
        cells.append([model, f"{params/1000:.1f}K", *vals])
    headers = ["Model", "Params"] + [f"L={x}" for x in lengths]
    fig, ax = plt.subplots(figsize=(max(5.8, 1.05 * len(headers)), max(2.0, 0.36 * (len(cells) + 2))))
    ax.axis("off")
    ax.set_title("Real-text needle retrieval", loc="left", fontsize=9, fontweight="bold")
    table = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D8D8")
        cell.set_linewidth(0.35)
        if r == 0:
            cell.set_facecolor("#F1F1F1")
            cell.set_text_props(weight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
    save_all(fig, path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_name", choices=["smoke", "core_full", "extended_full", "full"], default="full")
    parser.add_argument("--out_root", default=None, help="results root (default: this script's dir/results)")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.out_root) if args.out_root else (EXP_DIR / "results")
    raw = root / args.suite_name / "raw" / "real_text_needle_raw.csv"
    summary = root / args.suite_name / "summary" / "real_text_needle_summary.csv"
    figure_dir = root / args.suite_name / "figures"
    rows, failures = summarize(read_csv(raw))
    write_dicts(summary, rows, ["model", "context_length", "needle_depth", "recall_mean", "recall_std", "num_params_mean", "n_seeds"])
    write_dicts(summary.with_name("real_text_needle_failures.csv"), failures, ["model", "context_length", "needle_depth", "status", "notes"])
    make_table(rows, figure_dir / "real_text_needle_table.png")
    make_curve(rows, figure_dir / "real_text_needle_curve.png")
    print(f"wrote {summary}")
    print(f"wrote {figure_dir}")


if __name__ == "__main__":
    main()
