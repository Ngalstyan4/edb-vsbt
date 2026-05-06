"""Compare pgvector vs VectorChord (vchordrq) on each system independently."""

import argparse
import json
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
    geom_rect,
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CASE_ID_SIZE = {50: 50_000, 10: 500_000, 11: 5_000_000}
SIZE_LABELS = {
    50_000: "50K",
    500_000: "500K",
    1_000_000: "1M",
    3_000_000: "3M",
    5_000_000: "5M",
}
COLORS = {"pgvector": "#4C9AE8", "VectorChord": "#BFBB54"}
DODGE_W = 0.75

FIXED_INDEX = {"pgvector": "ivfflat_bq_rerank", "VectorChord": "vchordrq"}

METRIC_COLS = [
    ("QPS (peak)\nhigher is better", "max_qps"),
    ("Avg Single-Client Latency (ms)\nlower is better", "latency_avg_ms"),
    ("Recall %\nhigher is better", "recall"),
    ("Index Size (MB)\nlower is better", "index_size_mb"),
    ("Build RAM (MB)\nlower is better", "build_ram_mb"),
    ("Build Time (s)\nlower is better", "build_time_s"),
]
METRIC_HIGHER = {m[0]: (m[1] in ("max_qps", "recall")) for m in METRIC_COLS}
METRIC_ORDER = [m[0] for m in METRIC_COLS]

# ---------------------------------------------------------------------------
# Typography — single source of truth for all text sizes
# ---------------------------------------------------------------------------

FONT = {
    "title": 12,
    "subtitle": 9,
    "caption": 8,
    "strip": 9,         # facet panel titles
    "axis_label": 9,    # x/y axis titles
    "axis_tick": 8,     # tick labels on both axes
    "legend_title": 8,  # legend box titles (manual)
    "legend_entry": 7,  # legend entry labels (manual)
    "legend_marker": 3, # legend marker point size
    "top_legend": 9,    # top fill legend
    "bar_label": 10,    # rotated labels inside bars (e.g. recall %)
}

# Marker sizes for data points
MARKER = {
    "latency": 3,
    "latency_stroke": 0.5,
    "qps": 2.5,
    "qps_stroke": 0.4,
}


# --- Default equation helpers (must match config.py formulas) ---

def _pgvector_default_lists(N: int) -> int:
    return max(1, int(math.sqrt(N)))

def _pgvector_default_probes(N: int) -> int:
    return max(1, int(2.7 * math.sqrt(math.sqrt(N))))

def _vchordrq_default_lists(N: int) -> list[int]:
    base = N / 625
    return [max(1, int(math.sqrt(base))), max(1, int(base))]

def _vchordrq_default_probes(N: int) -> list[int]:
    base = math.sqrt(N / 625)
    return [max(1, int(base / 2)), max(1, int(2.7 * base))]


def _build_case_dim_map() -> dict[int, int]:
    """Parse CaseType enum from cases.py to build case_id → dimension map."""
    cases_py = Path(__file__).resolve().parent.parent / "vectordb_bench" / "backend" / "cases.py"
    mapping = {}
    if cases_py.exists():
        for m in re.finditer(r"(\w*?(\d+)D\w*?)\s*=\s*(\d+)", cases_py.read_text()):
            dim, case_id = int(m.group(2)), int(m.group(3))
            mapping[case_id] = dim
    return mapping

_CASE_DIM = _build_case_dim_map()


HIERARCHY_THRESHOLD = 5000  # matches src/ivfflat.h IVFFLAT_HIERARCHY_THRESHOLD


def _elkan_bytes(S, C, D, itemsize):
    """Flat Elkan k-means peak, mirroring src/ivfkmeans.c:501-515."""
    return (
        S * itemsize
        + 2 * C * itemsize
        + 4 * S * C
        + 4 * C * C
        + 4 * C * D
        + 8 * S
        + 12 * C
    )


def pgvector_ivf_build_ram(N, C, D, quant="bit", skew=2.0):
    """Estimate peak RAM (bytes) for pgvector IVF index build.

    For lists < HIERARCHY_THRESHOLD: flat Elkan k-means.
    For lists >= HIERARCHY_THRESHOLD: hierarchical (coarse + fine phases).
    """
    S = min(max(C * 50, 10_000), N)
    if quant == "bit":
        itemsize = ((D + 7) // 8 + 8 + 7) & ~7
    elif quant == "halfvec":
        itemsize = (8 + 2 * D + 7) & ~7
    else:  # float
        itemsize = 16 + 4 * D

    if C < HIERARCHY_THRESHOLD:
        peak = _elkan_bytes(S, C, D, itemsize)
    else:
        u = max(2, min(round(C ** 0.5), C))

        coarse_S = min(u * 256, S)
        coarse_peak = _elkan_bytes(coarse_S, u, D, itemsize)

        fine_S = int(skew * S / u)
        fine_C = int(skew * C / u)
        fine_peak = _elkan_bytes(fine_S, fine_C, D, itemsize)

        peak = max(coarse_peak, fine_peak)

    return peak * 1.1


def vchordrq_build_ram(N, C, D, F=256, T=1, D_k=None):
    """Estimate peak RAM (bytes) for vchordrq index build (Lloyd + RaBitQ k-means)."""
    if D_k is None:
        D_k = D
    S = min(C * F, N)
    samples = 4 * D_k * S
    centroids = 4 * D_k * C
    targets = 8 * S
    update = (T + 1) * (4 * D_k * C + 4 * C)
    reduction = 0
    if D_k < D:
        reduction = (T + 1) * (4 * D * C + 4 * C)
    return samples + centroids + targets + update + reduction


def classify_index(index_type: str) -> str:
    return "VectorChord" if index_type == "vchordrq" else "pgvector"


def parse_index_size_mb(s: str) -> float | None:
    if not s or not s.strip():
        return None
    parts = s.strip().split()
    val = float(parts[0])
    unit = parts[1].upper() if len(parts) > 1 else "MB"
    if unit == "GB":
        val *= 1024
    elif unit == "KB":
        val /= 1024
    return val


def size_label(n: int) -> str:
    if n in SIZE_LABELS:
        return SIZE_LABELS[n]
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}K"


def dense_breaks(limits, target=15):
    lo, hi = limits
    rng = hi - lo
    if rng <= 0:
        return [lo]
    mag = 10 ** np.floor(np.log10(max(rng / target, 1e-9)))
    nice = [1, 2, 2.5, 5, 10]
    step = mag * min(nice, key=lambda n: abs(rng / (n * mag) - target))
    start = np.floor(lo / step) * step
    ticks = np.arange(start, hi + step * 0.5, step)
    return ticks[(ticks >= lo - step * 0.1) & (ticks <= hi + step * 0.1)].tolist()


# ---------------------------------------------------------------------------
# Manual legend builder — used for both QPS concurrency and Latency legends
# ---------------------------------------------------------------------------

def build_manual_legend(
    metric_name: str,
    title: str,
    entries: list[dict],  # [{"label": "c=1", "shape": "^"}, ...]
    ordered_cats: list[str],
    max_y: float,
    y_offset_frac: float = 1.30,
    extend_y: bool = True,
    metric_order: list[str] | None = None,
    side: str = "right",
    width_scale: float = 1.0,
) -> tuple[list, pd.DataFrame]:
    """Build a manual legend box in the given facet panel.

    Returns (list_of_plotnine_layers, anchor_df_for_y_expansion).
    If extend_y is False, no geom_blank anchor is added (legend sits in
    natural empty space).
    """
    m_order = metric_order or METRIC_ORDER
    n_entries = len(entries)
    leg_step = max_y * 0.064
    leg_title_y = max_y * y_offset_frac + leg_step
    leg_top = max_y * y_offset_frac

    # Anchor to extend y-axis for legend room
    anchor = pd.DataFrame([{
        "dataset_size": ordered_cats[0],
        "value": leg_title_y + leg_step,
        "metric": metric_name,
    }])
    anchor["metric"] = pd.Categorical(anchor["metric"], categories=m_order, ordered=True)
    anchor["dataset_size"] = pd.Categorical(anchor["dataset_size"], categories=ordered_cats, ordered=True)

    leg_cat = ordered_cats[0] if side == "left" else ordered_cats[-1]
    n_cats = len(ordered_cats)
    leg_x = 1 if side == "left" else n_cats  # 1-indexed position

    # Entry rows
    leg_rows = []
    for i, entry in enumerate(entries):
        leg_rows.append({
            "metric": metric_name,
            "dataset_size": leg_cat,
            "y": leg_top - i * leg_step,
            "shape_key": entry["label"],
            "label": entry["label"],
        })
    leg_df = pd.DataFrame(leg_rows)
    leg_df["metric"] = pd.Categorical(leg_df["metric"], categories=m_order, ordered=True)
    leg_df["dataset_size"] = pd.Categorical(leg_df["dataset_size"], categories=ordered_cats, ordered=True)

    # Title row
    title_df = pd.DataFrame([{
        "metric": metric_name, "dataset_size": leg_cat,
        "y": leg_title_y, "label": title,
    }])
    title_df["metric"] = pd.Categorical(title_df["metric"], categories=m_order, ordered=True)
    title_df["dataset_size"] = pd.Categorical(title_df["dataset_size"], categories=ordered_cats, ordered=True)

    # Background rect
    rect_df = pd.DataFrame([{
        "metric": metric_name,
        "xmin": leg_x - 0.50 * width_scale, "xmax": leg_x + 0.58 * width_scale,
        "ymin": leg_top - (n_entries - 0.3) * leg_step,
        "ymax": leg_title_y + 0.7 * leg_step,
    }])
    rect_df["metric"] = pd.Categorical(rect_df["metric"], categories=m_order, ordered=True)

    # Shape map for this legend's markers
    shape_map = {e["label"]: e["shape"] for e in entries}

    layers = []
    if extend_y:
        layers.append(
            geom_blank(data=anchor, mapping=aes(x="dataset_size", y="value"), inherit_aes=False),
        )
    layers += [
        geom_rect(
            data=rect_df,
            mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
            inherit_aes=False, fill="#eeeeeecc", color="#bbbbbb", size=0.5,
            show_legend=False,
        ),
        geom_text(
            data=title_df,
            mapping=aes(x="dataset_size", y="y", label="label"),
            inherit_aes=False, ha="center", size=FONT["legend_title"], fontweight="bold",
            show_legend=False,
        ),
        geom_point(
            data=leg_df,
            mapping=aes(x="dataset_size", y="y", shape="shape_key"),
            inherit_aes=False, fill="#888888", color="black",
            size=FONT["legend_marker"], stroke=0.5,
            position=position_nudge(x=-0.12), show_legend=False,
        ),
        geom_text(
            data=leg_df,
            mapping=aes(x="dataset_size", y="y", label="label"),
            inherit_aes=False, ha="left", nudge_x=0.02, size=FONT["legend_entry"],
            show_legend=False,
        ),
    ]

    return layers, anchor, shape_map


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(data_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        for r in data["results"]:
            if r.get("label") != ":)":
                continue
            m = r["metrics"]
            tc = r["task_config"]
            idx = tc["db_case_config"].get(
                "index", tc["db_config"]["db_label"].split("-")[0]
            )
            storage = tc["db_case_config"].get("storage_type", "heap")
            if tc["db"] == "WarehousePG" and storage != "heap":
                continue
            qps = m.get("qps", 0)
            idx_size = parse_index_size_mb(m.get("index_size", ""))
            if qps == 0 or idx_size is None:
                continue

            rc = m.get("row_count") or CASE_ID_SIZE.get(tc["case_config"]["case_id"])
            if not rc:
                continue

            conc_avg = m.get("conc_latency_avg_list", [])
            conc_qps = m.get("conc_qps_list", [])
            conc_nums = m.get("conc_num_list", [])

            # Extract index-specific configuration params + estimate build RAM
            cfg = tc["db_case_config"]
            case_id = tc["case_config"]["case_id"]
            dim = _CASE_DIM.get(case_id, 1536)
            index_params = {}
            build_ram_mb = None
            if idx == "ivfflat_bq_rerank":
                lists_factor = cfg.get("rerank_lists_amplify_factor", 1)
                probe_factor = cfg.get("rerank_probe_amplify_factor", 1)
                cfg_lists = cfg.get("lists", 0)
                cfg_probes = cfg.get("probes", 0)
                actual_lists = (cfg_lists if cfg_lists else _pgvector_default_lists(rc)) * lists_factor
                actual_probes = (cfg_probes if cfg_probes else _pgvector_default_probes(rc)) * probe_factor
                index_params = {
                    "lists": actual_lists,
                    "probes": actual_probes,
                    "rerank_limit_factor": cfg.get("rerank_limit_amplify_factor"),
                }
                # Derive quantization from index type name
                if "bq" in idx:
                    quant = "bit"
                elif "halfvec" in idx:
                    quant = "halfvec"
                else:
                    quant = "float"
                build_ram_mb = pgvector_ivf_build_ram(rc, actual_lists, dim, quant=quant) / (1024 * 1024)
            elif idx == "vchordrq":
                cfg_lists = str(cfg.get("vchordrq_lists", "0")).strip()
                cfg_probes = str(cfg.get("vchordrq_probes", "0")).strip()
                # lists can be multi-level comma-separated, e.g. "10, 300"
                if cfg_lists and cfg_lists != "0":
                    actual_lists = cfg_lists  # keep as string
                else:
                    dl = _vchordrq_default_lists(rc)
                    actual_lists = ", ".join(str(x) for x in dl)
                if cfg_probes and cfg_probes != "0":
                    actual_probes = cfg_probes
                else:
                    dp = _vchordrq_default_probes(rc)
                    actual_probes = ", ".join(str(x) for x in dp)
                index_params = {
                    "lists": actual_lists,
                    "probes": actual_probes,
                    "epsilon": cfg.get("vchordrq_epsilon"),
                    "residual_quant": cfg.get("vchordrq_residual_quantization"),
                    "rerank_in_table": cfg.get("vchordrq_rerank_in_table"),
                    "spherical_centroids": cfg.get("vchordrq_spherical_centroids"),
                    "sampling_factor": cfg.get("vchordrq_sampling_factor"),
                }
                # C = last (leaf-level) cluster count
                leaf_lists = int(actual_lists.split(",")[-1].strip())
                sampling_factor = cfg.get("vchordrq_sampling_factor", 256)
                build_ram_mb = vchordrq_build_ram(rc, leaf_lists, dim, F=sampling_factor) / (1024 * 1024)

            rows.append(
                {
                    "system": tc["db"],
                    "index_type": idx,
                    "index_family": classify_index(idx),
                    "dataset_n": rc,
                    "dataset_size": size_label(rc),
                    "recall": m.get("recall", 0),
                    "max_qps": max(conc_qps) if conc_qps else qps,
                    "conc_qps_list": conc_qps,
                    "conc_num_list": conc_nums,
                    "latency_avg_ms": m.get("serial_latency_avg", 0) * 1000 or (conc_avg[0] * 1000 if conc_avg else None),
                    "latency_p95_ms": m.get("serial_latency_p95", 0) * 1000,
                    "latency_p99_ms": m.get("serial_latency_p99", 0) * 1000,
                    "index_size_mb": idx_size,
                    "build_time_s": m.get("optimize_duration", 0),
                    "source_file": p.name,
                    "index_params": index_params,
                    "build_ram_mb": build_ram_mb,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Deduplicate: keep run with highest QPS per (system, dataset_n, index_type)
    df = df.sort_values("max_qps", ascending=False).drop_duplicates(
        subset=["system", "dataset_n", "index_type"], keep="first"
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Filter to fixed index per family → long format for faceting
# ---------------------------------------------------------------------------


def filter_and_pivot(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the fixed index type for each family
    mask = df.apply(
        lambda r: r["index_type"] == FIXED_INDEX[r["index_family"]], axis=1
    )
    fixed = df[mask].copy()

    # Pivot to long format: one row per (system, dataset_size, family, metric)
    long_rows: list[dict] = []
    for _, row in fixed.iterrows():
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
# Overlay data builders
# ---------------------------------------------------------------------------


def _build_latency_overlay(sdf: pd.DataFrame, ordered: list[str], metric_order: list[str] | None = None):
    """Build p95/p99 point overlay for the Latency panel."""
    m_order = metric_order or METRIC_ORDER
    lat_mask = sdf["lat_p95"].notna()
    lat_pts = sdf[lat_mask].copy()
    pts_rows = []
    for _, r in lat_pts.iterrows():
        base = {"dataset_size": r["dataset_size"], "index_family": r["index_family"], "metric": r["metric"]}
        pts_rows.append({**base, "lat_y": r["lat_p95"], "percentile": "p95"})
        pts_rows.append({**base, "lat_y": r["lat_p99"], "percentile": "p99"})
    pts_df = pd.DataFrame(pts_rows)
    if not pts_df.empty:
        pts_df["metric"] = pd.Categorical(pts_df["metric"], categories=m_order, ordered=True)
        pts_df["dataset_size"] = pd.Categorical(pts_df["dataset_size"], categories=ordered, ordered=True)
        dodge_order = ["VectorChord_p95", "VectorChord_p99", "pgvector_p95", "pgvector_p99"]
        pts_df["dodge_group"] = pts_df["index_family"] + "_" + pts_df["percentile"]
        pts_df["dodge_group"] = pd.Categorical(pts_df["dodge_group"], categories=dodge_order, ordered=True)
    return pts_df


def _build_qps_overlay(sdf: pd.DataFrame, ordered: list[str], metric_order: list[str] | None = None):
    """Build concurrency-level point overlay for the QPS panel."""
    m_order = metric_order or METRIC_ORDER
    qps_mask = sdf["conc_qps_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    rows = []
    for _, r in sdf[qps_mask].iterrows():
        for cnum, cqps in zip(r["conc_num_list"], r["conc_qps_list"]):
            rows.append({
                "dataset_size": r["dataset_size"],
                "index_family": r["index_family"],
                "metric": r["metric"],
                "qps_y": cqps,
                "conc_label": f"c={cnum}",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["metric"] = pd.Categorical(df["metric"], categories=m_order, ordered=True)
        df["dataset_size"] = pd.Categorical(df["dataset_size"], categories=ordered, ordered=True)

        # Detect overlapping points per bar and add small x nudge
        df["x_nudge"] = 0.0
        y_range = df["qps_y"].max() - df["qps_y"].min()
        threshold = max(y_range * 0.04, 1)
        nudge_step = 0.06
        for _, grp in df.groupby(["dataset_size", "index_family"]):
            if len(grp) <= 1:
                continue
            sg = grp.sort_values("qps_y")
            ys = sg["qps_y"].values
            idxs = sg.index.tolist()
            # Cluster adjacent-by-y points that are too close
            clusters: list[list[int]] = [[0]]
            for j in range(1, len(ys)):
                if ys[j] - ys[j - 1] < threshold:
                    clusters[-1].append(j)
                else:
                    clusters.append([j])
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                n = len(cluster)
                offsets = np.linspace(-nudge_step * (n - 1) / 2,
                                      nudge_step * (n - 1) / 2, n)
                for k, off in zip(cluster, offsets):
                    df.loc[idxs[k], "x_nudge"] = off

        # Manual dodge offset (matches plotnine's position_dodge for 2 groups)
        families = sorted(df["index_family"].unique())
        n_fam = len(families)
        dodge_map = {fam: DODGE_W * (i - (n_fam - 1) / 2) / n_fam
                     for i, fam in enumerate(families)}
        df["total_nudge"] = df["index_family"].map(dodge_map) + df["x_nudge"]
    else:
        df = pd.DataFrame(columns=["dataset_size", "index_family", "metric",
                                    "qps_y", "conc_label", "x_nudge", "total_nudge"])
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def build_figure(long: pd.DataFrame, system: str, system_label: str):
    sdf = long[long["system"] == system].copy()
    if sdf.empty:
        print(f"No data for {system}, skipping.")
        return None

    ordered = sdf.drop_duplicates("dataset_n").sort_values("dataset_n")["dataset_size"].tolist()
    sdf["dataset_size"] = pd.Categorical(sdf["dataset_size"], categories=ordered, ordered=True)

    pgv = FIXED_INDEX["pgvector"]
    vc = FIXED_INDEX["VectorChord"]

    # Pre-scan concurrency levels to decide QPS panel label early
    qps_pts_df_tmp = _build_qps_overlay(sdf, ordered)
    single_conc = None
    if not qps_pts_df_tmp.empty:
        conc_labels = sorted(
            qps_pts_df_tmp["conc_label"].unique(),
            key=lambda s: int(s.split("=")[1]),
        )
        if len(conc_labels) == 1:
            single_conc = int(conc_labels[0].split("=")[1])

    # Determine the metric order (rename QPS panel when single concurrency)
    qps_metric = METRIC_ORDER[0]  # "QPS (peak)\nhigher is better"
    qps_base = qps_metric  # original name for renaming
    if single_conc is not None:
        conc_tag = "single client" if single_conc == 1 else f"c={single_conc}"
        qps_label = f"QPS ({conc_tag})\nhigher is better"
    else:
        qps_label = qps_metric
    metric_order = [qps_label if m == qps_base else m for m in METRIC_ORDER]

    # Rename QPS metric in data if needed
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

    # Build overlay data (pass metric_order so categoricals match)
    pts_df = _build_latency_overlay(sdf, ordered, metric_order)
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
        qps_pts_df = qps_pts_df.iloc[0:0]  # empty — no markers needed

    lat_shape_map = {"p95": "o", "p99": "D"}
    all_shape_map = {**lat_shape_map, **conc_shape_map}

    # --- Build manual legends with shared helper ---
    extra_layers = []

    # QPS concurrency legend (only when multiple concurrency levels)
    if conc_shape_map:
        qps_max = sdf[sdf["metric"] == qps_label]["value"].max()
        entries = [{"label": cl, "shape": conc_shape_map[cl]} for cl in conc_labels_sorted]
        layers, _, _ = build_manual_legend(
            qps_label, "Concurrency", entries, ordered, qps_max,
            y_offset_frac=0.90, extend_y=False, metric_order=metric_order,
            width_scale=1.65,
        )
        extra_layers.extend(layers)

    # Latency p95/p99 legend
    lat_metric = METRIC_ORDER[1]  # "Avg Latency (ms)\n..."
    if not pts_df.empty:
        lat_max = sdf[sdf["metric"] == lat_metric]["value"].max()
        # Also consider p95/p99 values for y range
        lat_overlay_max = pts_df["lat_y"].max()
        lat_max = max(lat_max, lat_overlay_max) if lat_overlay_max else lat_max
        entries = [
            {"label": "p95", "shape": lat_shape_map["p95"]},
            {"label": "p99", "shape": lat_shape_map["p99"]},
        ]
        layers, _, _ = build_manual_legend(
            lat_metric, "Latency", entries, ordered, lat_max,
            y_offset_frac=0.90, extend_y=False,
            metric_order=metric_order, side="left",
            width_scale=1.1,
        )
        extra_layers.extend(layers)

    # Recall bar labels — position at mid-height of each bar
    recall_metric = METRIC_ORDER[2]  # "Recall %"
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
        # Latency p95/p99 markers
        + geom_point(
            data=pts_df,
            mapping=aes(x="dataset_size", y="lat_y", fill="index_family",
                        shape="percentile", group="dodge_group"),
            position=position_dodge(width=DODGE_W),
            inherit_aes=False, color="black",
            size=MARKER["latency"], stroke=MARKER["latency_stroke"],
            show_legend=False,
        )
        # QPS concurrency markers added below via per-nudge layers
        + facet_wrap("~metric", scales="free_y", ncol=6)
        + scale_fill_manual(
            values=COLORS,
            labels={"pgvector": f"pgvector ({pgv})", "VectorChord": f"VectorChord ({vc})"},
        )
        + scale_shape_manual(values=all_shape_map)
        + scale_y_continuous(breaks=dense_breaks)
        + guides(
            fill=guide_legend(title="", position="top"),
            shape=False,  # all legends are manual now
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

    p = (
        p
        + labs(
            title=f"pgvector vs VectorChord \u2014 {system_label}",
                subtitle=(
                    f"Comparing bit-quantized pgvector ivfflat with half-vector reranking"
                    f" and bit-quantized (RaBitQ) VectorChord vchordrq with in-table reranking"
                    f" | Dataset: OpenAI 1536-dim cosine"
                ),
            caption="",
            x="-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   Dataset size (rows) -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  ",
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


def _within_pct(actual, expected, pct=1):
    """Check if actual is within pct% of expected."""
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) <= pct / 100


def _check_pgvector_eq(frows: pd.DataFrame) -> bool:
    """Check if all pgvector rows match the default equations within 1%."""
    for _, row in frows.iterrows():
        p = row["index_params"]
        if not p:
            return False
        N = row["dataset_n"]
        if not _within_pct(p["lists"], _pgvector_default_lists(N)):
            return False
        if not _within_pct(p["probes"], _pgvector_default_probes(N)):
            return False
    return True


def _check_vchordrq_eq(frows: pd.DataFrame) -> bool:
    """Check if all vchordrq rows match the default equations within 1%."""
    for _, row in frows.iterrows():
        p = row["index_params"]
        if not p:
            return False
        N = row["dataset_n"]
        actual_lists = [int(x.strip()) for x in str(p["lists"]).split(",")]
        actual_probes = [int(x.strip()) for x in str(p["probes"]).split(",")]
        exp_lists = _vchordrq_default_lists(N)
        exp_probes = _vchordrq_default_probes(N)
        if len(actual_lists) != len(exp_lists) or len(actual_probes) != len(exp_probes):
            return False
        if not all(_within_pct(a, e) for a, e in zip(actual_lists, exp_lists)):
            return False
        if not all(_within_pct(a, e) for a, e in zip(actual_probes, exp_probes)):
            return False
    return True


def _build_family_lines(df: pd.DataFrame, system: str, family: str, idx_type: str) -> list[tuple]:
    """Build caption lines for one index family. Each entry is (text, color[, highlight])."""
    default_color = "#333333"
    sdf = df[(df["system"] == system)].copy()
    frows = sdf[sdf["index_type"] == idx_type].sort_values("dataset_n")
    if frows.empty:
        return []
    lines: list[tuple] = []
    lines.append((f"{family} ({idx_type}):", default_color, COLORS[family]))

    # Check if actual params match default equations
    if idx_type == "ivfflat_bq_rerank" and _check_pgvector_eq(frows):
        lines.append((r"  eq: $lists\!=\!\sqrt{N},\ probes\!=\!2.7\!\cdot\!\sqrt{\sqrt{N}}$", default_color))
    elif idx_type == "vchordrq" and _check_vchordrq_eq(frows):
        lines.append((r"  eq: $lists\!=\![\sqrt{N/625},\ N/625]$", default_color))
        lines.append((r"  eq: $probes\!=\![0.5\!\cdot\!\sqrt{N/625},\ 2.7\!\cdot\!\sqrt{N/625}]$", default_color))

    # Per-size lists/probes
    for _, row in frows.iterrows():
        p = row["index_params"]
        if not p:
            continue
        lines.append((f"  {row['dataset_size']}: lists={p.get('lists')}, probes={p.get('probes')}", default_color))

    # Static params — one per line
    params = frows.iloc[0]["index_params"]
    for k, v in params.items():
        if k in ("lists", "probes") or v is None:
            continue
        if isinstance(v, bool):
            v = "yes" if v else "no"
        lines.append((f"  {k}={v}", default_color))

    return lines


SYSTEM_LABELS = {
    "VectorChord": "PostgreSQL 18",
    "WarehousePG": "WarehousePG (MPP Greenplum)",
}


# ---------------------------------------------------------------------------
# Slide-deck mode — progressive reveal
# ---------------------------------------------------------------------------

def build_slides(long: pd.DataFrame, system: str, system_label: str):
    """Generate a sequence of progressive-reveal figures for presentations."""
    sdf = long[long["system"] == system].copy()
    if sdf.empty:
        return []

    ordered = sdf.drop_duplicates("dataset_n").sort_values("dataset_n")["dataset_size"].tolist()
    sdf["dataset_size"] = pd.Categorical(sdf["dataset_size"], categories=ordered, ordered=True)

    pgv = FIXED_INDEX["pgvector"]
    vc = FIXED_INDEX["VectorChord"]
    m_order = list(METRIC_ORDER)
    qps_m, lat_m, rec_m = m_order[0], m_order[1], m_order[2]
    idx_m, ram_m, bt_m = m_order[3], m_order[4], m_order[5]

    # --- Extract serial (c=1) QPS values ---
    qps_rows = sdf[sdf["metric"] == qps_m].copy()
    for i, r in qps_rows.iterrows():
        cnl, cql = r.get("conc_num_list", []), r.get("conc_qps_list", [])
        if isinstance(cnl, list) and isinstance(cql, list) and 1 in cnl:
            qps_rows.at[i, "value"] = cql[cnl.index(1)]
    serial_qps_max = qps_rows["value"].max() if not qps_rows.empty else 0

    # --- Build overlays once ---
    pts_df = _build_latency_overlay(sdf, ordered, m_order)
    qps_pts_df = _build_qps_overlay(sdf, ordered, m_order)

    # Concurrency / latency shapes
    lat_shape_map = {"p95": "o", "p99": "D"}
    conc_shape_map = {}
    conc_labels_sorted = []
    _shapes = list("^svPX*ph")
    if not qps_pts_df.empty:
        conc_labels_sorted = sorted(
            qps_pts_df["conc_label"].unique(), key=lambda s: int(s.split("=")[1])
        )
        for i, cl in enumerate(conc_labels_sorted):
            conc_shape_map[cl] = _shapes[i % len(_shapes)]
    all_shape_map = {**lat_shape_map, **conc_shape_map}

    # --- Scale anchors: one set per QPS mode ---
    families = list(COLORS.keys())  # ["pgvector", "VectorChord"]

    def _anchors(qps_max_override=None):
        rows = []
        for m in m_order:
            md = sdf[sdf["metric"] == m]
            if md.empty:
                continue
            if m == qps_m and qps_max_override is not None:
                mx = qps_max_override
            elif m == rec_m:
                mx = 100
            elif m == lat_m and not pts_df.empty:
                mx = max(md["value"].max(), pts_df["lat_y"].max())
            else:
                mx = md["value"].max()
            # Include all dataset sizes × both families so legend always shows
            for ds in ordered:
                for fam in families:
                    rows.append({"dataset_size": ds, "value": 0, "metric": m, "index_family": fam})
            rows.append({"dataset_size": ordered[0], "value": mx, "metric": m, "index_family": families[0]})
        a = pd.DataFrame(rows)
        a["metric"] = pd.Categorical(a["metric"], categories=m_order, ordered=True)
        a["dataset_size"] = pd.Categorical(a["dataset_size"], categories=ordered, ordered=True)
        return a

    anchors_full = _anchors()
    anchors_serial = _anchors(qps_max_override=serial_qps_max * 1.1)

    # --- Common theme block ---
    def _theme_block():
        return (
            theme_minimal()
            + theme(
                figure_size=(16.5, 6.25),
                legend_position="top",
                plot_title=element_text(size=FONT["title"], weight="bold"),
                plot_subtitle=element_text(size=FONT["subtitle"], color="#555555"),
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

    # --- Build one slide ---
    def _slide(name, visible_metrics, qps_mode=None):
        """
        visible_metrics: set of metric names to show (excluding QPS, handled by qps_mode).
        qps_mode: None=hidden, "single_peak"=c=1 at peak scale,
                  "single"=c=1 rescaled, "full"=peak bars + markers.
        """
        parts = [sdf[sdf["metric"].isin(visible_metrics)]]
        if qps_mode in ("single_peak", "single"):
            parts.append(qps_rows)  # serial QPS values
        elif qps_mode == "full":
            parts.append(sdf[sdf["metric"] == qps_m])
        vis = pd.concat(parts, ignore_index=True) if parts else sdf.iloc[0:0].copy()
        vis["metric"] = pd.Categorical(vis["metric"], categories=m_order, ordered=True)
        vis["dataset_size"] = pd.Categorical(vis["dataset_size"], categories=ordered, ordered=True)

        anch = anchors_serial if qps_mode == "single" else anchors_full

        p = (
            ggplot(vis, aes(x="dataset_size", y="value", fill="index_family"))
            + geom_col(position=position_dodge(width=DODGE_W), width=0.6)
            + geom_blank(data=anch, mapping=aes(x="dataset_size", y="value", fill="index_family"), inherit_aes=False)
            + facet_wrap("~metric", scales="free_y", ncol=6)
            + scale_fill_manual(
                values=COLORS,
                labels={"pgvector": f"pgvector ({pgv})", "VectorChord": f"VectorChord ({vc})"},
            )
            + scale_shape_manual(values=all_shape_map)
            + scale_y_continuous(breaks=dense_breaks)
            + guides(fill=guide_legend(title="", position="top"), shape=False)
            + labs(
                title=f"pgvector vs VectorChord \u2014 {system_label}",
                subtitle=(
                    f"Comparing bit-quantized pgvector ivfflat with half-vector reranking"
                    f" and bit-quantized (RaBitQ) VectorChord vchordrq with in-table reranking"
                    f" | Dataset: OpenAI 1536-dim cosine"
                ),
                caption="", x="Dataset size (rows)", y="",
            )
            + _theme_block()
        )

        # Recall labels
        if rec_m in visible_metrics:
            rc = vis[vis["metric"] == rec_m].copy()
            rc["label"] = rc["value"].apply(lambda v: f"{v:.1f}%")
            rc["label_y"] = rc["value"] / 2
            p = p + geom_text(
                data=rc,
                mapping=aes(x="dataset_size", y="label_y", label="label", group="index_family"),
                position=position_dodge(width=DODGE_W),
                inherit_aes=False, va="center", ha="center", angle=90,
                size=FONT["bar_label"], color="black", show_legend=False,
            )

        # Latency markers
        if lat_m in visible_metrics and not pts_df.empty:
            p = p + geom_point(
                data=pts_df,
                mapping=aes(x="dataset_size", y="lat_y", fill="index_family",
                            shape="percentile", group="dodge_group"),
                position=position_dodge(width=DODGE_W),
                inherit_aes=False, color="black",
                size=MARKER["latency"], stroke=MARKER["latency_stroke"],
                show_legend=False,
            )
            # Latency legend
            lat_max = max(vis[vis["metric"] == lat_m]["value"].max(),
                          pts_df["lat_y"].max())
            entries = [{"label": "p95", "shape": "o"}, {"label": "p99", "shape": "D"}]
            layers, _, _ = build_manual_legend(
                lat_m, "Latency", entries, ordered, lat_max,
                y_offset_frac=0.90, extend_y=False, metric_order=m_order,
                side="left", width_scale=1.1,
            )
            for layer in layers:
                p = p + layer

        # QPS concurrency markers + legend
        if qps_mode == "full" and not qps_pts_df.empty:
            for nudge_val, sub_df in qps_pts_df.groupby("total_nudge"):
                p = p + geom_point(
                    data=sub_df,
                    mapping=aes(x="dataset_size", y="qps_y", fill="index_family", shape="conc_label"),
                    inherit_aes=False, color="black",
                    size=MARKER["qps"], stroke=MARKER["qps_stroke"],
                    position=position_nudge(x=nudge_val), show_legend=False,
                )
            if conc_shape_map:
                qps_max = sdf[sdf["metric"] == qps_m]["value"].max()
                entries = [{"label": cl, "shape": conc_shape_map[cl]} for cl in conc_labels_sorted]
                layers, _, _ = build_manual_legend(
                    qps_m, "Concurrency", entries, ordered, qps_max,
                    y_offset_frac=0.90, extend_y=False, metric_order=m_order,
                    width_scale=1.65,
                )
                for layer in layers:
                    p = p + layer

        return (name, p)

    # --- Slide sequence ---
    return [
        _slide("01_empty", set()),
        _slide("02_recall", {rec_m}),
        _slide("03_qps_single_wide", {rec_m}, "single_peak"),
        _slide("04_qps_single", {rec_m}, "single"),
        _slide("05_qps_single_full_scale", {rec_m}, "single_peak"),
        _slide("06_qps_full", {rec_m}, "full"),
        _slide("07_latency", {rec_m, lat_m}, "full"),
        _slide("08_index_size", {rec_m, lat_m, idx_m}, "full"),
        _slide("09_build_ram", {rec_m, lat_m, idx_m, ram_m}, "full"),
        _slide("10_build_time", {rec_m, lat_m, idx_m, ram_m, bt_m}, "full"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Open interactive window")
    parser.add_argument("--slides", action="store_true", help="Generate progressive-reveal slide PNGs")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    df = load_data(args.data_dir)
    if df.empty:
        print("No data found.")
        sys.exit(1)

    print(f"Loaded {len(df)} result rows from {df['source_file'].nunique()} files")
    print(f"Systems: {sorted(df['system'].unique())}")
    print(f"Dataset sizes: {sorted(df['dataset_n'].unique())}")
    print(f"Fixed indices: {FIXED_INDEX}")

    long = filter_and_pivot(df)
    summary = long[long["metric"] == METRIC_ORDER[0]][
        ["system", "dataset_size", "index_family", "index_type", "value", "recall"]
    ]
    print(f"\nQPS rows:\n{summary.to_string(index=False)}")

    cap_size = FONT["caption"]

    file_names = {"VectorChord": "pg18", "WarehousePG": "whpg"}
    for system, fname in file_names.items():
        label = SYSTEM_LABELS.get(system, system)
        result = build_figure(long, system, label)
        if result is None:
            continue
        p, multi_conc = result

        notes_lines = [
            "Plot notes:",
        ]
        if multi_conc:
            notes_lines.append("  QPS bars = peak; markers = other concurrency levels.")
        notes_lines.append("  Latency bars = avg; markers = p95 / p99.")
        notes_lines.append("  Build time = CREATE INDEX time.")
        caption_left = "\n".join(notes_lines)

        families = list(FIXED_INDEX.items())
        col_lines = [_build_family_lines(df, system, fam, idx) for fam, idx in families]
        mpl_fig = p.draw()
        mpl_fig.text(0.01, -0.02, caption_left, fontsize=cap_size,
                     ha="left", va="top", color="#333333")
        # Render index params: title + two side-by-side family columns
        fig_h = mpl_fig.get_size_inches()[1]
        line_h = (cap_size * 1.4) / 72 / fig_h
        params_x = 0.50
        col_xs = [params_x, 0.68]  # two columns, tightly grouped
        mpl_fig.text(params_x, -0.02, "Index params:", fontsize=cap_size,
                     ha="left", va="top", color="#333333")
        y_start = -0.02 - line_h
        n_rows = max(len(c) for c in col_lines) if col_lines else 0
        for row_i in range(n_rows):
            # Compute y: track cumulative height from tallest previous rows
            y = y_start
            for prev in range(row_i):
                # Check if any column had a LaTeX line at that row
                has_latex = any(
                    prev < len(c) and "$" in c[prev][0] for c in col_lines
                )
                y -= line_h * (1.8 if has_latex else 1.0)
            for col_i, lines in enumerate(col_lines):
                if row_i >= len(lines):
                    continue
                entry = lines[row_i]
                text, color = entry[0], entry[1]
                highlight = entry[2] if len(entry) > 2 else None
                kwargs = {}
                if highlight:
                    kwargs["bbox"] = dict(
                        facecolor=highlight, alpha=0.3, edgecolor="none",
                        pad=1.5, boxstyle="round,pad=0.15",
                    )
                mpl_fig.text(col_xs[col_i], y, text, fontsize=cap_size,
                             ha="left", va="top", color=color, **kwargs)
        for ext in ("png", "pdf"):
            out = args.output_dir / f"{fname}.{ext}"
            mpl_fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"Saved {out}")

    if args.slides:
        for system, fname in file_names.items():
            label = SYSTEM_LABELS.get(system, system)
            slides = build_slides(long, system, label)
            if not slides:
                continue
            slide_dir = args.output_dir / f"{fname}_slides"
            if slide_dir.exists():
                for old in slide_dir.glob("*.png"):
                    old.unlink()
            slide_dir.mkdir(exist_ok=True)
            for slide_name, slide_p in slides:
                fig = slide_p.draw()
                for ext in ("png",):
                    out = slide_dir / f"{slide_name}.{ext}"
                    fig.savefig(out, dpi=200, bbox_inches="tight")
                    print(f"Saved {out}")
                import matplotlib.pyplot as plt
                plt.close(fig)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
