#!/usr/bin/env python3
import os
import glob
import json
import re
import pandas as pd

# ==========================
# CONFIG
# ==========================
MAIN_FOLDER = "/tudelft.net/staff-umbrella/hchassagnette/Workspace/output/ThesisResearch"
OUT_DIR = os.path.join(MAIN_FOLDER, "tables")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "results_per_scene.csv")
OUT_XLSX = os.path.join(OUT_DIR, "results_per_scene.xlsx")

# JSON metric keys
METRIC_KEYS = {
    "PSNR": "image_metrics/full/psnr",
    "SSIM": "image_metrics/full/ssim",
    "LPIPS": "image_metrics/full/lpips",
    "MIOU": "image_metrics/full/miou",
}

# Runtime parsing
TOTAL_TIME_RE = re.compile(
    r"Total time:\s*(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)\s*"
    r"\(\s*(?P<spi>[0-9]*\.?[0-9]+)\s*s\s*/\s*it\s*\)",
    re.IGNORECASE
)

def extract_scene_number(folder_name: str):
    """Return last 3 digits if folder ends with _XXX, else None."""
    if len(folder_name) >= 4 and folder_name[-4] == "_" and folder_name[-3:].isdigit():
        return int(folder_name[-3:])
    return None

def extract_run_type(folder_name: str):
    """Remove trailing _XXX if present."""
    if len(folder_name) >= 4 and folder_name[-4] == "_" and folder_name[-3:].isdigit():
        return folder_name[:-4]
    return folder_name

def method_key_from_run_type(run_type: str) -> str:
    """
    Your method family grouping: first two underscore-separated tokens.
    Examples:
      streetgsSemantic_CE_0,01 -> streetgsSemantic_CE
      streetgsSemantic_Focal_0,001_N2 -> streetgsSemantic_Focal
      vanilla005 -> vanilla005
    """
    parts = run_type.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return run_type

def load_first_metrics_json(run_folder: str):
    """Return dict from first images_full_*.json, else None."""
    metric_files = glob.glob(os.path.join(run_folder, "metrics", "images_full_*.json"))
    if not metric_files:
        return None
    # if multiple exist, pick the most recent by mtime
    metric_files.sort(key=os.path.getmtime, reverse=True)
    try:
        with open(metric_files[0], "r") as f:
            return json.load(f)
    except OSError:
        return None

def parse_runtime_from_logs(run_folder: str):
    """
    Return (total_time_hours, sec_per_it) from the most recent log_*.txt,
    else (None, None).
    """
    log_files = glob.glob(os.path.join(run_folder, "logs", "log_*.txt"))
    if not log_files:
        return None, None

    log_files.sort(key=os.path.getmtime, reverse=True)
    log_path = log_files[0]

    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                if "total time" not in line.lower():
                    continue
                m = TOTAL_TIME_RE.search(line)
                if m:
                    h = int(m.group("h"))
                    mn = int(m.group("m"))
                    s = int(m.group("s"))
                    total_s = h * 3600 + mn * 60 + s
                    sec_per_it = float(m.group("spi"))
                    return total_s / 3600.0, sec_per_it
    except OSError:
        pass

    return None, None

# ==========================
# COLLECT ROWS
# ==========================
rows = []

for subfolder in os.listdir(MAIN_FOLDER):
    run_path = os.path.join(MAIN_FOLDER, subfolder)
    if not os.path.isdir(run_path):
        continue

    scene_number = extract_scene_number(subfolder)
    run_type = extract_run_type(subfolder)
    method = method_key_from_run_type(run_type)

    # Load metrics JSON (optional)
    metrics = load_first_metrics_json(run_path)

    psnr = ssim = lpips = miou = None
    if isinstance(metrics, dict):
        psnr = metrics.get(METRIC_KEYS["PSNR"])
        ssim = metrics.get(METRIC_KEYS["SSIM"])
        lpips = metrics.get(METRIC_KEYS["LPIPS"])
        miou = metrics.get(METRIC_KEYS["MIOU"])

    # Load runtime (optional)
    total_h, sec_per_it = parse_runtime_from_logs(run_path)

    # If you ONLY want rows that have at least something useful, enforce it here:
    if scene_number is None:
        # No scene suffix -> skip (you said runs are 000..009)
        continue

    if metrics is None and total_h is None and sec_per_it is None:
        # Completely empty -> probably failed run folder
        continue

    rows.append({
        "method": method,
        "scene_number": scene_number,
        "SSIM": ssim,
        "PSNR": psnr,
        "LPIPS": lpips,
        "MIOU": miou,
        "total_run_time_h": total_h,
        "sec_per_iteration": sec_per_it,
        "run_type_full": run_type,  # helpful extra column (you can remove if you want)
    })

# ==========================
# SAVE CSV / EXCEL
# ==========================
df = pd.DataFrame(rows)

# Sort for readability: by method then scene
if not df.empty:
    df = df.sort_values(["method", "scene_number", "run_type_full"], kind="stable")

# Your requested columns (plus run_type_full at end; remove if you don't want it)
cols = [
    "method", "scene_number",
    "SSIM", "PSNR", "LPIPS", "MIOU",
    "total_run_time_h", "sec_per_iteration",
    "run_type_full",
]
df = df[cols]

df.to_csv(OUT_CSV, index=False)
df.to_excel(OUT_XLSX, index=False)

print(f"Wrote {len(df)} rows")
print(f"CSV : {OUT_CSV}")
print(f"XLSX: {OUT_XLSX}")
