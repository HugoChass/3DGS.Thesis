import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================
# CONFIG
# ==========================
MAIN_FOLDER = "/tudelft.net/staff-umbrella/hchassagnette/Workspace/output/ThesisResearch"
PLOT_DIR = os.path.join(MAIN_FOLDER, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

METRICS = {
    "psnr": "image_metrics/full/psnr",
    "ssim": "image_metrics/full/ssim",
    "lpips": "image_metrics/full/lpips",
    "miou": "image_metrics/full/miou",
}

RUNTIME_METRICS = {
    "total_time_h": "Total time (hours)",
    "sec_per_it": "Seconds per iteration",
}


# ==========================
# RUN TYPE FILTERING
# ==========================

# Option A: keep ONLY these run types (exact match)
INCLUDE_RUN_TYPES = None
# Example:
# INCLUDE_RUN_TYPES = [
#     "streetgsSemantic_CE_0,01",
#     "streetgsSemantic_Focal_0,3_N2",
# ]

# Option B: remove these run types (exact match)
EXCLUDE_RUN_TYPES = None
# Example:
# EXCLUDE_RUN_TYPES = ["vanilla005"]

# Option C: keep run types containing ANY of these substrings
INCLUDE_KEYWORDS = None
# Example:
# INCLUDE_KEYWORDS = ["Semantic", "CE"]

# Option D: remove run types containing ANY of these substrings
EXCLUDE_KEYWORDS = ['vanilla']
# Example:
# EXCLUDE_KEYWORDS = ["debug", "test"]


# ==========================
# HELPER FUNCTIONS
# ==========================
def extract_run_type(folder_name):
    """
    Removes the trailing _XXX scene id.
    Assumes scene id is always the last 3 characters.
    """
    return folder_name[:-4] if folder_name[-4] == "_" else folder_name


def load_metrics(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def run_type_allowed(run_type):
    if INCLUDE_RUN_TYPES is not None:
        if run_type not in INCLUDE_RUN_TYPES:
            return False

    if EXCLUDE_RUN_TYPES is not None:
        if run_type in EXCLUDE_RUN_TYPES:
            return False

    if INCLUDE_KEYWORDS is not None:
        if not any(k in run_type for k in INCLUDE_KEYWORDS):
            return False

    if EXCLUDE_KEYWORDS is not None:
        if any(k in run_type for k in EXCLUDE_KEYWORDS):
            return False

    return True

import re

TOTAL_TIME_RE = re.compile(
    r".*Total time:\s*(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)\s*"
    r"\(\s*(?P<spi>[0-9]*\.?[0-9]+)\s*s\s*/\s*it\s*\)",
    re.IGNORECASE
)

def parse_log_for_runtime(log_path):
    """
    Returns (total_time_seconds, seconds_per_iteration) or (None, None) if not found.
    """
    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                # Search, not match, so any prefix is fine
                if "total time" not in line.lower():
                    continue
                m = TOTAL_TIME_RE.search(line)
                if m:
                    h = int(m.group("h"))
                    mn = int(m.group("m"))
                    s = int(m.group("s"))
                    total_s = h * 3600 + mn * 60 + s
                    sec_per_it = float(m.group("spi"))
                    return total_s, sec_per_it
    except OSError:
        pass

    return None, None

from matplotlib.patches import Patch

# Metrics where "higher is better" (sorted descending). Others will be sorted ascending.
HIGHER_IS_BETTER = {
    "psnr", "ssim", "miou",
    # add more if you plot them
}
# e.g. LPIPS and runtime: lower is better -> ascending by default


def method_key_from_run_type(run_type: str) -> str:
    """
    Groups run types into method families.
    Examples:
      streetgsSemantic_CE_0,01 -> streetgsSemantic_CE
      streetgsSemantic_Focal_0,3_N2 -> streetgsSemantic_Focal
      vanilla005 -> vanilla005
    """
    parts = run_type.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return run_type


def build_method_color_map(run_types):
    """
    Map each method_key to a stable color using matplotlib's default color cycle.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

    methods = sorted({method_key_from_run_type(rt) for rt in run_types})
    return {m: cycle[i % len(cycle)] for i, m in enumerate(methods)}


# ==========================
# DATA COLLECTION
# ==========================
data = defaultdict(lambda: defaultdict(list))
runtime_data = defaultdict(lambda: defaultdict(list))

for subfolder in os.listdir(MAIN_FOLDER):
    subfolder_path = os.path.join(MAIN_FOLDER, subfolder)

    if not os.path.isdir(subfolder_path):
        continue

    run_type = extract_run_type(subfolder)

    if not run_type_allowed(run_type):
        continue

    metric_files = glob.glob(
        os.path.join(subfolder_path, "metrics", "images_full_*.json")
    )

    if metric_files:
        metrics = load_metrics(metric_files[0])

        for short_name, full_key in METRICS.items():
            if full_key in metrics:
                data[run_type][short_name].append(metrics[full_key])
    
    log_files = glob.glob(os.path.join(subfolder_path, "logs", "log_*.txt"))

    if log_files:
        # If multiple logs exist, take the most recent by modified time
        log_files.sort(key=os.path.getmtime, reverse=True)
        total_s, sec_per_it = parse_log_for_runtime(log_files[0])

        if total_s is not None:
            runtime_data[run_type]["total_time_h"].append(total_s / 3600.0)
        if sec_per_it is not None:
            runtime_data[run_type]["sec_per_it"].append(sec_per_it)

# Sort run types once for consistent plotting
run_types = sorted(data.keys())
print(run_types)
# ==========================
# PLOTTING
# ==========================
# ==========================
# PLOTTING (ORDERED + COLORED BY METHOD)
# ==========================
all_run_types = sorted(data.keys())
method_color = build_method_color_map(all_run_types)

for metric_name in METRICS.keys():

    # Only keep run types that have this metric
    run_types_metric = [
        rt for rt in all_run_types
        if metric_name in data[rt] and len(data[rt][metric_name]) > 0
    ]
    if not run_types_metric:
        print(f"[WARN] No data for metric: {metric_name}, skipping.")
        continue

    # Compute mean per run type
    means = {rt: float(np.mean(data[rt][metric_name])) for rt in run_types_metric}

    # Sort by mean (descending if higher-is-better, else ascending)
    reverse = metric_name in HIGHER_IS_BETTER
    run_types_sorted = sorted(run_types_metric, key=lambda rt: means[rt], reverse=reverse)

    # Colors per bar/box based on method family
    colors = [method_color[method_key_from_run_type(rt)] for rt in run_types_sorted]

    # Build legend (only methods present in this plot)
    present_methods = []
    for rt in run_types_sorted:
        mk = method_key_from_run_type(rt)
        if mk not in present_methods:
            present_methods.append(mk)
    legend_handles = [Patch(facecolor=method_color[m], label=m) for m in present_methods]

    # ---------- BAR PLOT (MEAN) ----------
    plt.figure(figsize=(12, 5))
    plt.bar(run_types_sorted, [means[rt] for rt in run_types_sorted], color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_name.upper())
    plt.title(f"Mean {metric_name.upper()} over scenes (sorted by mean)")
    plt.legend(handles=legend_handles, title="Method", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric_name}_mean_sorted_colored.png"), dpi=300)
    plt.close()

    # ---------- BOX PLOT (DISTRIBUTION) ----------
    box_data = [data[rt][metric_name] for rt in run_types_sorted]

    plt.figure(figsize=(12, 5))
    bp = plt.boxplot(box_data, labels=run_types_sorted, showfliers=True, patch_artist=True)

    # Color the boxes by method family
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} distribution over scenes (sorted by mean)")
    plt.legend(handles=legend_handles, title="Method", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric_name}_box_sorted_colored.png"), dpi=300)
    plt.close()

# ==========================
# RUNTIME PLOTTING
# ==========================

# For runtime metrics: lower is better -> ascending sort (reverse=False)
# If you ever have a runtime metric where higher is better, add it here.
RUNTIME_HIGHER_IS_BETTER = set()  # usually empty


runtime_run_types = sorted(runtime_data.keys())

# Use the same method-color mapping idea
runtime_method_color = build_method_color_map(runtime_run_types)

for metric_name, y_label in RUNTIME_METRICS.items():
    run_types_metric = [
        rt for rt in runtime_run_types
        if metric_name in runtime_data[rt] and len(runtime_data[rt][metric_name]) > 0
    ]

    if not run_types_metric:
        print(f"[WARN] No runtime data for: {metric_name}, skipping.")
        continue

    # Mean per run type
    means = {rt: float(np.mean(runtime_data[rt][metric_name])) for rt in run_types_metric}

    # Sort by mean (runtime: lower is better -> ascending)
    reverse = metric_name in RUNTIME_HIGHER_IS_BETTER
    run_types_sorted = sorted(run_types_metric, key=lambda rt: means[rt], reverse=reverse)

    # Colors per run type (by method family)
    colors = [runtime_method_color[method_key_from_run_type(rt)] for rt in run_types_sorted]

    # Legend handles (only methods present in this plot)
    present_methods = []
    for rt in run_types_sorted:
        mk = method_key_from_run_type(rt)
        if mk not in present_methods:
            present_methods.append(mk)
    legend_handles = [Patch(facecolor=runtime_method_color[m], label=m) for m in present_methods]

    # --- Mean bar plot ---
    plt.figure(figsize=(12, 5))
    plt.bar(run_types_sorted, [means[rt] for rt in run_types_sorted], color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y_label)
    plt.title(f"Mean {y_label} over scenes (sorted by mean)")
    plt.legend(handles=legend_handles, title="Method", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric_name}_mean_sorted_colored.png"), dpi=300)
    plt.close()

    # --- Box plot ---
    box_data = [runtime_data[rt][metric_name] for rt in run_types_sorted]

    plt.figure(figsize=(12, 5))
    bp = plt.boxplot(box_data, labels=run_types_sorted, showfliers=True, patch_artist=True)

    # Color boxes by method family
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y_label)
    plt.title(f"{y_label} distribution over scenes (sorted by mean)")
    plt.legend(handles=legend_handles, title="Method", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric_name}_box_sorted_colored.png"), dpi=300)
    plt.close()

print(f"Runtime plots saved to: {PLOT_DIR}")
