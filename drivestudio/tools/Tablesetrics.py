#!/usr/bin/env python3
import os
import glob
import json
import re
import pandas as pd

# ==========================
# CONFIG
# ==========================
MAIN_FOLDER = "/tudelft.net/staff-umbrella/hchassagnette/Workspace/output/ThesisResultsV2"
OUT_DIR = os.path.join(MAIN_FOLDER, "tables")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "results_per_scene.csv")

# JSON metric keys
METRIC_KEYS_FULL = {
    "PSNR": "image_metrics/full/psnr",
    "SSIM": "image_metrics/full/ssim",
    "LPIPS": "image_metrics/full/lpips",
    "MIOU": "image_metrics/full/miou",
}
METRIC_KEYS_TEST = {
    "PSNR": "image_metrics/test/psnr",
    "SSIM": "image_metrics/test/ssim",
    "LPIPS": "image_metrics/test/lpips",
    "MIOU": "image_metrics/test/miou",
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

def load_first_metrics_json(run_folder: str, prefix: str):
    """
    Return dict from the most recent metrics/{prefix}*.json, else None.
    Example prefixes: "images_full_" or "images_test_".
    """
    metric_files = glob.glob(os.path.join(run_folder, "metrics", f"{prefix}*.json"))
    if not metric_files:
        return None
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

def safe_get_miou(frame_metrics: dict):
    """
    Try common locations for mIoU in a per-frame metrics dict.
    Returns float or None.
    """
    if not isinstance(frame_metrics, dict):
        return None
    if "miou" in frame_metrics and frame_metrics["miou"] is not None:
        return frame_metrics["miou"]
    if "image_metrics/full/miou" in frame_metrics and frame_metrics["image_metrics/full/miou"] is not None:
        return frame_metrics["image_metrics/full/miou"]
    # sometimes nested
    maybe = frame_metrics.get("image_metrics", {})
    if isinstance(maybe, dict):
        full = maybe.get("full", {})
        if isinstance(full, dict) and "miou" in full:
            return full.get("miou")
    return None

def load_novel_view_miou(run_folder: str):
    """
    Loads videos/novel_30000/front_center_interp_metrics.json and returns
    the average mIoU across frames, or None if missing/unreadable.
    """
    path = os.path.join(run_folder, "videos", "novel_30000", "front_center_interp_metrics.json")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            obj = json.load(f)
    except OSError:
        return None

    frame_dicts = []

    # Common shapes:
    # 1) dict: {frame_id: {metrics...}, ...}
    # 2) list: [{metrics...}, {metrics...}, ...]
    if isinstance(obj, dict):
        if "frames" in obj and isinstance(obj["frames"], (list, dict)):
            frames = obj["frames"]
            if isinstance(frames, list):
                frame_dicts = frames
            elif isinstance(frames, dict):
                frame_dicts = list(frames.values())
        else:
            frame_dicts = list(obj.values())
    elif isinstance(obj, list):
        frame_dicts = obj

    miou_vals = []
    for fm in frame_dicts:
        v = safe_get_miou(fm)
        if v is not None:
            miou_vals.append(float(v))

    if len(miou_vals) == 0:
        return None

    return sum(miou_vals) / len(miou_vals)

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
    method = run_type

    if scene_number is None:
        continue

    # Load full-view metrics JSON (optional)
    metrics_full = load_first_metrics_json(run_path, prefix="images_full_")
    # Load test-view metrics JSON (optional)
    metrics_test = load_first_metrics_json(run_path, prefix="images_test_")

    psnr = ssim = lpips = miou = None
    psnr_test = ssim_test = lpips_test = miou_test = None

    if isinstance(metrics_full, dict):
        psnr = metrics_full.get(METRIC_KEYS_FULL["PSNR"])
        ssim = metrics_full.get(METRIC_KEYS_FULL["SSIM"])
        lpips = metrics_full.get(METRIC_KEYS_FULL["LPIPS"])
        miou = metrics_full.get(METRIC_KEYS_FULL["MIOU"])

    if isinstance(metrics_test, dict):
        psnr_test = metrics_test.get(METRIC_KEYS_TEST["PSNR"])
        ssim_test = metrics_test.get(METRIC_KEYS_TEST["SSIM"])
        lpips_test = metrics_test.get(METRIC_KEYS_TEST["LPIPS"])
        miou_test = metrics_test.get(METRIC_KEYS_TEST["MIOU"])

    # Load runtime (optional)
    total_h, sec_per_it = parse_runtime_from_logs(run_path)

    # Load novel-view mIoU (optional)
    novel_view_miou = load_novel_view_miou(run_path)

    # Skip totally empty runs (now includes test metrics too)
    if (
        metrics_full is None
        and metrics_test is None
        and total_h is None
        and sec_per_it is None
        and novel_view_miou is None
    ):
        continue

    rows.append({
        "method": method,
        "scene_number": scene_number,

        "SSIM": ssim,
        "PSNR": psnr,
        "LPIPS": lpips,
        "MIOU": miou,

        "SSIM_test": ssim_test,
        "PSNR_test": psnr_test,
        "LPIPS_test": lpips_test,
        "MIOU_test": miou_test,

        "novel_view_miou": novel_view_miou,
        "total_run_time_h": total_h,
        "sec_per_iteration": sec_per_it,
    })

# ==========================
# SAVE CSV / EXCEL
# ==========================
df = pd.DataFrame(rows)

if not df.empty:
    df = df.sort_values(["method", "scene_number"], kind="stable")

cols = [
    "method", "scene_number",
    "SSIM", "PSNR", "LPIPS", "MIOU",
    "SSIM_test", "PSNR_test", "LPIPS_test", "MIOU_test",
    "novel_view_miou",
    "total_run_time_h", "sec_per_iteration",
]
df = df[cols]

df.to_csv(OUT_CSV, index=False)

print(f"Wrote {len(df)} rows")
print(f"CSV : {OUT_CSV}")
