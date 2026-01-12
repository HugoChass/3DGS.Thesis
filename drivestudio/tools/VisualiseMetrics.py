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
    r"Total time:\s*(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)\s*\((?P<spi>[0-9]*\.?[0-9]+)\s*s\s*/\s*it\)"
)

def parse_log_for_runtime(log_path):
    """
    Returns (total_time_seconds, seconds_per_iteration) or (None, None) if not found.
    """
    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                if "Total time:" not in line:
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

    if not metric_files:
        continue

    metrics = load_metrics(metric_files[0])

    for short_name, full_key in METRICS.items():
        if full_key in metrics:
            data[run_type][short_name].append(metrics[full_key])

# --------- Parse runtime logs ----------
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
for metric_name in METRICS.keys():

    # ---------- BAR PLOT (MEAN) ----------
    means = [
        np.mean(data[rt][metric_name])
        for rt in run_types
        if metric_name in data[rt]
    ]

    plt.figure(figsize=(12, 5))
    plt.bar(run_types, means)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_name.upper())
    plt.title(f"Mean {metric_name.upper()} over scenes")
    plt.tight_layout()

    mean_path = os.path.join(PLOT_DIR, f"{metric_name}_mean.png")
    plt.savefig(mean_path, dpi=300)
    plt.close()

    # ---------- BOX PLOT (DISTRIBUTION) ----------
    box_data = [
        data[rt][metric_name]
        for rt in run_types
        if metric_name in data[rt]
    ]

    plt.figure(figsize=(12, 5))
    plt.boxplot(box_data, labels=run_types, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} distribution over scenes")
    plt.tight_layout()

    box_path = os.path.join(PLOT_DIR, f"{metric_name}_box.png")
    plt.savefig(box_path, dpi=300)
    plt.close()

# ==========================
# RUNTIME PLOTTING
# ==========================
runtime_run_types = sorted(runtime_data.keys())
print(runtime_data)
for metric_name, y_label in RUNTIME_METRICS.items():
    print(metric_name, y_label)
    run_types_metric = [
        rt for rt in runtime_run_types
        if metric_name in runtime_data[rt] and len(runtime_data[rt][metric_name]) > 0
    ]

    if not run_types_metric:
        print(f"[WARN] No runtime data for: {metric_name}, skipping.")
        continue

    means = [np.mean(runtime_data[rt][metric_name]) for rt in run_types_metric]

    # --- Mean bar plot ---
    plt.figure(figsize=(12, 5))
    plt.bar(run_types_metric, means)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y_label)
    plt.title(f"Mean {y_label} over scenes")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric_name}_mean.png"), dpi=300)
    plt.close()

    # --- Box plot ---
    box_data = [runtime_data[rt][metric_name] for rt in run_types_metric]

    plt.figure(figsize=(12, 5))
    plt.boxplot(box_data, labels=run_types_metric, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y_label)
    plt.title(f"{y_label} distribution over scenes")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric_name}_box.png"), dpi=300)
    plt.close()

print(f"Plots saved to: {PLOT_DIR}")
