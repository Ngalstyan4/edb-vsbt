"""Plot edb-vsbt benchmark results from all_results.csv.

Produces the same 6-panel faceted chart as plot.py but reads from CSV
and supports N systems (suite_types) rather than the hardcoded 2.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from plotnine import (
    aes,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_blank,
    geom_col,
    geom_point,
    geom_text,
    ggplot,
    guide_legend,
    guides,
    labs,
    position_dodge,
    position_nudge,
    scale_fill_manual,
    scale_shape_manual,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from plot import (
    parse_index_size_mb,
    dense_breaks,
    pgvector_ivf_build_ram,
    vchordrq_build_ram,
    build_manual_legend,
    _build_qps_overlay,
    FONT,
    MARKER,
    DODGE_W,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS_PALETTE = [
    "#4C9AE8",  # blue
    "#BFBB54",  # gold
    "#E8744C",  # orange
    "#54BFA0",  # teal
    "#9B59B6",  # purple
    "#E84C7F",  # pink
    "#2ECC71",  # green
    "#F39C12",  # amber
]

METRIC_COLS = [
    ("QPS (peak)\nhigher is better", "max_qps"),
    ("P50 (P99) Latency (ms)\nlower is better", "latency_avg_ms"),
    ("Recall %\nhigher is better", "recall"),
    ("Index Size (MB)\nlower is better", "index_size_mb"),
    ("Build RAM (MB)\nlower is better", "build_ram_mb"),
    ("Build Time (s)\nlower is better", "build_time_s"),
]
METRIC_ORDER = [m[0] for m in METRIC_COLS]

DIM_MAP = {"openai": 1536, "laion": 768}


def infer_dim(dataset: str) -> int:
    for key, dim in DIM_MAP.items():
        if key in dataset.lower():
            return dim
    return 1536


def infer_n_rows(dataset: str) -> int | None:
    m = re.search(r"(\d+)(k|m)", dataset.lower())
    if not m:
        return None
    val = int(m.group(1))
    if m.group(2) == "m":
        return val * 1_000_000
    return val * 1_000


def compute_build_ram(row) -> float | None:
    suite = row["suite_type"]
    dataset = row["dataset"]
    N = infer_n_rows(dataset)
    if N is None:
        return None
    dim = infer_dim(dataset)

    if suite == "ivfflat_bq_rerank":
        lists = row.get("lists")
        if pd.isna(lists) or str(lists).strip() == "N/A":
            lists = max(1, int(math.sqrt(N)))
        else:
            lists = int(float(lists))
        return pgvector_ivf_build_ram(N, lists, dim, quant="bit") / (1024 * 1024)
    elif suite in ("vectorchord", "pgpu"):
        lists_raw = row.get("lists")
        if pd.isna(lists_raw) or str(lists_raw).strip() == "N/A":
            base = N / 625
            leaf_lists = max(1, int(base))
        else:
            cleaned = str(lists_raw).strip().strip("[]")
            parts = cleaned.split(",")
            leaf_lists = int(float(parts[-1].strip()))
        sf = row.get("sampling_factor")
        sf = 256 if (pd.isna(sf) or str(sf).strip() == "N/A") else int(float(sf))
        return vchordrq_build_ram(N, leaf_lists, dim, F=sf) / (1024 * 1024)
    elif suite == "pgvector":
        m_val = row.get("m")
        if pd.isna(m_val) or str(m_val).strip() == "N/A":
            m_val = 16
        else:
            m_val = int(float(m_val))
        return N * (4 * dim + 8 + m_val * 12) / (1024 * 1024)
    return None


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------


def load_csv(csv_path: Path, filter_dataset=None, filter_suite=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if filter_dataset:
        df = df[df["dataset"].isin(filter_dataset)]
    if filter_suite:
        df = df[df["suite_type"].isin(filter_suite)]
    df = df[df["recall"] > 0].copy()
    df = df[df["qps"] > 0].copy()
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a DataFrame matching the shape of plot.py's load_data() output.

    Columns: system, index_type, index_family, dataset_n, dataset_size,
             recall, max_qps, conc_qps_list, conc_num_list,
             latency_avg_ms, latency_p95_ms, latency_p99_ms,
             index_size_mb, build_time_s, build_ram_mb, index_params
    """
    group_cols = ["run_id", "suite_type", "dataset", "benchmark_name"]
    records = []

    for key, grp in df.groupby(group_cols):
        run_id, suite_type, dataset, benchmark_name = key
        grp_sorted = grp.sort_values("query_clients")

        conc_nums = grp_sorted["query_clients"].tolist()
        conc_qps = grp_sorted["qps"].tolist()

        # Latency from lowest-concurrency row
        lat_row = grp_sorted.iloc[0]
        first = grp_sorted.iloc[0]

        idx_size = parse_index_size_mb(str(first["index_size"])) if pd.notna(first["index_size"]) else None
        build_ram = compute_build_ram(first)

        build_time = first.get("index_build_time_s")
        if pd.isna(build_time) or str(build_time).strip() == "N/A":
            build_time = 0
        else:
            build_time = float(build_time)

        N = infer_n_rows(dataset)

        p50 = float(lat_row["p50_latency_ms"]) if pd.notna(lat_row["p50_latency_ms"]) else 0
        p99 = float(lat_row["p99_latency_ms"]) if pd.notna(lat_row["p99_latency_ms"]) else 0

        records.append({
            "system": "_all",
            "index_type": suite_type,
            "index_family": suite_type,
            "dataset_n": N or 0,
            "dataset_size": dataset,
            "recall": float(first["recall"]),
            "max_qps": max(conc_qps),
            "conc_qps_list": conc_qps if len(conc_qps) > 1 else [],
            "conc_num_list": conc_nums if len(conc_nums) > 1 else [],
            "latency_avg_ms": p50,
            "latency_p95_ms": 0,
            "latency_p99_ms": p99,
            "index_size_mb": idx_size,
            "build_time_s": build_time,
            "build_ram_mb": build_ram,
            "index_params": {},
            "benchmark_name": benchmark_name,
        })

    agg = pd.DataFrame(records)
    if agg.empty:
        return agg

    # Deduplicate: keep highest recall per (index_family, dataset_size)
    agg = agg.sort_values("recall", ascending=False).drop_duplicates(
        subset=["index_family", "dataset_size"], keep="first"
    ).reset_index(drop=True)
    return agg


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to long format matching plot.py's filter_and_pivot() output."""
    long_rows = []
    for _, row in df.iterrows():
        base = {
            "system": row["system"],
            "dataset_n": row["dataset_n"],
            "dataset_size": row["dataset_size"],
            "index_family": row["index_family"],
            "index_type": row["index_type"],
            "recall": row["recall"],
        }
        for mname, col in METRIC_COLS:
            val = row[col] * 100 if col == "recall" else row[col]
            extra = {}
            if col == "latency_avg_ms":
                extra["lat_p95"] = row["latency_p95_ms"]
                extra["lat_p99"] = row["latency_p99_ms"]
            if col == "max_qps":
                extra["conc_qps_list"] = row["conc_qps_list"]
                extra["conc_num_list"] = row["conc_num_list"]
            long_rows.append({**base, "metric": mname, "value": val, **extra})

    out = pd.DataFrame(long_rows)
    out["metric"] = pd.Categorical(out["metric"], categories=METRIC_ORDER, ordered=True)
    return out


# ---------------------------------------------------------------------------
# Latency overlay (p99 only, since CSV has no p95)
# ---------------------------------------------------------------------------


def _build_latency_overlay(sdf: pd.DataFrame, ordered: list[str], families: list[str],
                           metric_order: list[str] | None = None):
    m_order = metric_order or METRIC_ORDER
    lat_mask = sdf["lat_p99"].notna() & (sdf["lat_p99"] > 0)
    lat_pts = sdf[lat_mask].copy()
    pts_rows = []
    for _, r in lat_pts.iterrows():
        base = {"dataset_size": r["dataset_size"], "index_family": r["index_family"], "metric": r["metric"]}
        pts_rows.append({**base, "lat_y": r["lat_p99"], "percentile": "p99"})
    pts_df = pd.DataFrame(pts_rows)
    if not pts_df.empty:
        pts_df["metric"] = pd.Categorical(pts_df["metric"], categories=m_order, ordered=True)
        pts_df["dataset_size"] = pd.Categorical(pts_df["dataset_size"], categories=ordered, ordered=True)
        dodge_order = [f"{fam}_p99" for fam in families]
        pts_df["dodge_group"] = pts_df["index_family"] + "_" + pts_df["percentile"]
        pts_df["dodge_group"] = pd.Categorical(pts_df["dodge_group"], categories=dodge_order, ordered=True)
    return pts_df


# ---------------------------------------------------------------------------
# Plotting — mirrors plot.py build_figure() exactly
# ---------------------------------------------------------------------------


def build_figure(long: pd.DataFrame, title: str = ""):
    sdf = long.copy()
    if sdf.empty:
        print("No data, skipping.")
        return None

    families = sorted(sdf["index_family"].unique())
    colors = {f: COLORS_PALETTE[i % len(COLORS_PALETTE)] for i, f in enumerate(families)}

    ordered = sdf.drop_duplicates("dataset_n").sort_values("dataset_n")["dataset_size"].tolist()
    sdf["dataset_size"] = pd.Categorical(sdf["dataset_size"], categories=ordered, ordered=True)

    # Pre-scan concurrency levels to decide QPS panel label
    qps_pts_df_tmp = _build_qps_overlay(sdf, ordered)
    single_conc = None
    if not qps_pts_df_tmp.empty:
        conc_labels = sorted(
            qps_pts_df_tmp["conc_label"].unique(),
            key=lambda s: int(s.split("=")[1]),
        )
        if len(conc_labels) == 1:
            single_conc = int(conc_labels[0].split("=")[1])

    # Determine metric order (rename QPS panel when single concurrency)
    qps_metric = METRIC_ORDER[0]
    qps_base = qps_metric
    if single_conc is not None:
        conc_tag = "single client" if single_conc == 1 else f"c={single_conc}"
        qps_label = f"QPS ({conc_tag})\nhigher is better"
    else:
        qps_label = qps_metric
    metric_order = [qps_label if m == qps_base else m for m in METRIC_ORDER]

    if qps_label != qps_base:
        sdf["metric"] = sdf["metric"].cat.rename_categories({qps_base: qps_label})

    # Invisible anchors to force recall panel 0–100%
    recall_metric = METRIC_ORDER[2]
    anchor_df = pd.DataFrame([
        {"dataset_size": ordered[0], "value": 0, "metric": recall_metric},
        {"dataset_size": ordered[0], "value": 100, "metric": recall_metric},
    ])
    anchor_df["metric"] = pd.Categorical(anchor_df["metric"], categories=metric_order, ordered=True)
    anchor_df["dataset_size"] = pd.Categorical(anchor_df["dataset_size"], categories=ordered, ordered=True)

    # Build overlay data
    pts_df = _build_latency_overlay(sdf, ordered, families, metric_order)
    qps_pts_df = _build_qps_overlay(sdf, ordered, metric_order)

    # Shape definitions
    _conc_shapes = {"^": "^", "s": "s", "v": "v", "P": "P", "X": "X", "*": "*", "p": "p", "h": "h"}
    conc_labels_sorted = []
    conc_shape_map = {}
    if not qps_pts_df.empty and single_conc is None:
        conc_labels_sorted = sorted(
            qps_pts_df["conc_label"].unique(),
            key=lambda s: int(s.split("=")[1]),
        )
        shapes_list = list(_conc_shapes.values())
        for i, cl in enumerate(conc_labels_sorted):
            conc_shape_map[cl] = shapes_list[i % len(shapes_list)]
    if single_conc is not None:
        qps_pts_df = qps_pts_df.iloc[0:0]

    lat_shape_map = {"p99": "D"}
    all_shape_map = {**lat_shape_map, **conc_shape_map}

    # --- Manual legends ---
    extra_layers = []

    # QPS concurrency legend
    if conc_shape_map:
        qps_max = sdf[sdf["metric"] == qps_label]["value"].max()
        entries = [{"label": cl, "shape": conc_shape_map[cl]} for cl in conc_labels_sorted]
        layers, _, _ = build_manual_legend(
            qps_label, "Concurrency", entries, ordered, qps_max,
            y_offset_frac=0.90, extend_y=False, metric_order=metric_order,
            width_scale=1.65,
        )
        extra_layers.extend(layers)

    lat_metric = METRIC_ORDER[1]

    # Recall bar labels
    recall_df = sdf[sdf["metric"] == recall_metric].copy()
    recall_df["label"] = recall_df["value"].apply(lambda v: f"{v:.1f}%")
    recall_df["label_y"] = recall_df["value"] / 2

    # --- Assemble plot ---
    p = (
        ggplot(sdf, aes(x="dataset_size", y="value", fill="index_family"))
        + geom_col(position=position_dodge(width=DODGE_W), width=0.6)
        + geom_blank(data=anchor_df, mapping=aes(x="dataset_size", y="value"), inherit_aes=False)
        # Recall value labels rotated inside bars
        + geom_text(
            data=recall_df,
            mapping=aes(x="dataset_size", y="label_y", label="label", group="index_family"),
            position=position_dodge(width=DODGE_W),
            inherit_aes=False, va="center", ha="center", angle=90,
            size=FONT["bar_label"], color="black",
            show_legend=False,
        )
        # Latency p99 markers
        + geom_point(
            data=pts_df,
            mapping=aes(x="dataset_size", y="lat_y", fill="index_family",
                        shape="percentile", group="dodge_group"),
            position=position_dodge(width=DODGE_W),
            inherit_aes=False, color="black",
            size=MARKER["latency"], stroke=MARKER["latency_stroke"],
            show_legend=False,
        )
        + facet_wrap("~metric", scales="free_y", ncol=6)
        + scale_fill_manual(values=colors)
        + scale_shape_manual(values=all_shape_map)
        + scale_y_continuous(breaks=dense_breaks)
        + guides(
            fill=guide_legend(title="", position="top"),
            shape=False,
        )
    )

    for layer in extra_layers:
        p = p + layer

    # QPS concurrency markers: per-nudge layers (manual dodge + overlap spreading)
    if not qps_pts_df.empty:
        for nudge_val, sub_df in qps_pts_df.groupby("total_nudge"):
            p = p + geom_point(
                data=sub_df,
                mapping=aes(x="dataset_size", y="qps_y", fill="index_family", shape="conc_label"),
                inherit_aes=False, color="black",
                size=MARKER["qps"], stroke=MARKER["qps_stroke"],
                position=position_nudge(x=nudge_val),
                show_legend=False,
            )

    if not title:
        title = "Vector Index Benchmark Comparison"
    subtitle = f"Systems: {', '.join(families)} | Datasets: {', '.join(ordered)}"

    p = (
        p
        + labs(
            title=title,
            subtitle=subtitle,
            caption="",
            x="-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   Dataset -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  ",
            y="",
        )
        + theme_minimal()
        + theme(
            figure_size=(16.5, 6.25),
            legend_position="top",
            plot_title=element_text(size=FONT["title"], weight="bold"),
            plot_subtitle=element_text(size=FONT["subtitle"], color="#555555"),
            plot_caption=element_text(size=FONT["caption"], ha="left"),
            strip_text=element_text(size=FONT["strip"]),
            axis_text=element_text(size=FONT["axis_tick"]),
            axis_text_y=element_text(size=FONT["axis_tick"]),
            axis_title_x=element_text(size=FONT["axis_label"]),
            panel_grid_major_x=element_line(color="none"),
            panel_grid_minor_y=element_line(color="#dddddd", size=0.3),
            plot_background=element_rect(color="#999999", fill="#f7f7f7", size=0.8),
            legend_text=element_text(size=FONT["top_legend"]),
            legend_background=element_rect(fill="#eeeeeecc", color="#bbbbbb", size=0.5),
        )
    )
    multi_conc = single_conc is None and not qps_pts_df.empty
    return p, multi_conc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Path to all_results.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--filter-dataset", type=str, default=None,
                        help="Comma-separated dataset names to include")
    parser.add_argument("--filter-suite", type=str, default=None,
                        help="Comma-separated suite_types to include")
    parser.add_argument("--title", type=str, default="", help="Plot title")
    parser.add_argument("--show", action="store_true", help="Open interactive window")
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    filter_dataset = args.filter_dataset.split(",") if args.filter_dataset else None
    filter_suite = args.filter_suite.split(",") if args.filter_suite else None

    print("")
    print("============================================================")
    print("  plot_csv: Vector Index Benchmark Comparison")
    print("============================================================")
    print("")

    print(f"  CSV:      {args.csv}")
    if filter_dataset:
        print(f"  Datasets: {', '.join(filter_dataset)}")
    if filter_suite:
        print(f"  Suites:   {', '.join(filter_suite)}")
    print("")

    df = load_csv(args.csv, filter_dataset=filter_dataset, filter_suite=filter_suite)
    if df.empty:
        print("No data found after filtering.")
        sys.exit(1)

    print(f"--- Loading ---")
    print(f"  Rows:        {len(df)}")
    print(f"  Suite types: {sorted(df['suite_type'].unique())}")
    print(f"  Datasets:    {sorted(df['dataset'].unique())}")
    print("")

    agg = aggregate(df)
    if agg.empty:
        print("No data after aggregation.")
        sys.exit(1)

    print(f"--- Aggregated ({len(agg)} records) ---")
    for _, row in agg.iterrows():
        print(f"  [{row['index_family']}] {row['dataset_size']} "
              f"| {row['benchmark_name']} "
              f"| recall={row['recall']:.4f} qps={row['max_qps']:.1f}")
    print("")

    print("--- Rendering ---")
    long = to_long(agg)
    result = build_figure(long, title=args.title)
    if result is None:
        sys.exit(1)

    p, multi_conc = result
    mpl_fig = p.draw()

    # Caption
    cap_size = FONT["caption"]
    notes_lines = ["Plot notes:"]
    if multi_conc:
        notes_lines.append("  QPS bars = peak; markers = other concurrency levels.")
    notes_lines.append("  Latency bars = p50; markers = p99.")
    notes_lines.append("  Build time = CREATE INDEX time.")
    caption_left = "\n".join(notes_lines)
    mpl_fig.text(0.01, -0.02, caption_left, fontsize=cap_size,
                 ha="left", va="top", color="#333333")

    print("--- Output ---")
    for ext in ("png", "pdf"):
        out = args.output_dir / f"comparison.{ext}"
        mpl_fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved {out}")

    print("")
    print("============================================================")
    print("  Done!")
    print("============================================================")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
